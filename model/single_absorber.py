from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import xarray as xr

from constants import P0_REF, T0_REF, VMR_REF
from functional_forms import FunctionalForm, functional_form_registry


def lnp(p, p0, **_ignored):
    return np.log(p / p0)


def lnp_withself(p, p0, vmr, vmr0, self_scaling):
    return np.log((p / p0) * (1.0 + vmr * self_scaling) / (1.0 + vmr0 * self_scaling))


def dT(T, T0):
    return T - T0


@dataclass(frozen=True)
class AbsorberConfig:
    species: str = ""
    frequency_grid: Optional[tuple[float, ...]] = (1012.2305,)


class SingleSpeciesModel(Protocol):
    """Protocol for species cross-section models."""

    config: "AbsorberConfig"

    def cross_section(
        self,
        pressure: np.ndarray,
        temperature: np.ndarray,
        vmr: np.ndarray,
    ) -> np.ndarray:
        """Return cross-section matrix with shape (levels, frequency)."""
        ...

    def train(self) -> None:
        """Build the model, e.g. by fitting coefficients."""
        ...

    def save(self, path: str) -> None:
        """Save the model to disk."""
        ...

    def load(self, path: str) -> None:
        """Load the model from disk."""
        ...

    def cross_section_from_atmds(self, atmds: xr.Dataset) -> xr.DataArray:
        xsec = xr.apply_ufunc(
            self.cross_section,
            atmds["pressure"],
            atmds["temperature"],
            atmds["vmr"].sel(species=self.config.species),
            input_core_dims=[[], [], []],  # all are (levels,)
            output_core_dims=[["frequency"]],  # output adds frequency dim
            dask="parallelized",  # works with dask arrays too
            output_dtypes=[float],
        )
        return xsec.assign_coords(frequency=self.config.frequency_grid)


@dataclass(frozen=True)
class FunctionalConfig(AbsorberConfig):
    p0: float = P0_REF
    T0: float = T0_REF
    vmr0: float = VMR_REF
    pressure_form_name: str = "Hinge"
    temperature_form_name: str = "Rational"
    self_scaling: int | float = 0  # for self-broadening effects


@dataclass
class FunctionalCoeffs:
    xsec0: Optional[np.ndarray] = None
    pressure_coeffs: Optional[np.ndarray] = None
    temperature_coeffs: Optional[np.ndarray] = None


class FunctionalAbsorber(SingleSpeciesModel):
    pressure_form: FunctionalForm
    temperature_form: FunctionalForm
    config: "FunctionalConfig"
    coeffs: "FunctionalCoeffs"

    def __init__(
        self,
        species: str,
        pressure_form_name: str,
        temperature_form_name: str,
        p0: float = P0_REF,
        T0: float = T0_REF,
        vmr0: float = VMR_REF,
        frequency_grid: Optional[tuple[float, ...]] = None,
        self_scaling: int | float = 0,
    ) -> None:
        self.pressure_form = functional_form_registry.get(pressure_form_name)
        self.temperature_form = functional_form_registry.get(temperature_form_name)
        self.config = FunctionalConfig(
            species=species,
            p0=p0,
            T0=T0,
            vmr0=vmr0,
            frequency_grid=frequency_grid,
            self_scaling=self_scaling,
        )
        self.coeffs = FunctionalCoeffs()

        self.pressure_var = (
            staticmethod(lnp_withself) if self_scaling != 0 else staticmethod(lnp)
        )
        self.temperature_var = staticmethod(dT)

    def cross_section(
        self,
        pressure: np.ndarray,
        temperature: np.ndarray,
        vmr: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return cross-section matrix with shape (levels, frequency)."""
        x_p = self.pressure_var(
            pressure,
            self.config.p0,
            vmr=vmr,
            vmr0=self.config.vmr0,
            self_scaling=self.config.self_scaling,
        )
        x_t = self.temperature_var(temperature, self.config.T0)

        p_scale = self.pressure_form.evaluate(x_p, self.coeffs.pressure_coeffs)
        t_scale = self.temperature_form.evaluate(x_t, self.coeffs.temperature_coeffs)

        return self.coeffs.xsec0 * np.exp(p_scale + t_scale)

    def train(
        self,
        reference_xsec: Optional[str] = None,
        max_iter: int = 3,
        **training_kwargs,
    ) -> None:
        """Fit coefficients."""

        sampling_kwargs = training_kwargs.get("sampling_kwargs", {})
        functional_config_kwargs = training_kwargs.get("functional_config_kwargs", {})

        for key, val in functional_config_kwargs.items():
            if hasattr(self.config, key) and val is not None:
                setattr(self.config, key, val)

        if reference_xsec is not None:
            reference_ds = self._validate_xsec_dataset(reference_xsec, freq_atol=1e-3)
        else:
            from utils import calulate_arts_reference, sample_atmospheres

            p_grid, t_grid = sample_atmospheres(**sampling_kwargs)

            p_grid = (
                np.append(p_grid, [self.config.p0])
                if self.config.p0 not in p_grid
                else p_grid
            )
            t_grid = (
                np.append(t_grid, [self.config.T0])
                if self.config.T0 not in t_grid
                else t_grid
            )

            reference_ds = calulate_arts_reference(
                self.config.species,
                self.config.frequency_grid,
                p_grid,
                t_grid,
                np.full_like(p_grid, self.config.vmr0),
            )

        x_p = self.pressure_var(
            reference_ds["pressure"].values,
            self.config.p0,
            vmr=reference_ds["vmr"].values,
            vmr0=self.config.vmr0,
            self_scaling=self.config.self_scaling,
        )
        x_t = self.temperature_var(reference_ds["temperature"].values, self.config.T0)

        self.coeffs.xsec0 = (
            reference_ds["xsec"]
            .sel(pressure=self.config.p0, temperature=self.config.T0)
            .values
        )

        reference_ds["norm_lnxsec"] = np.log(
            reference_ds["xsec"]
            / reference_ds["xsec"].sel(
                pressure=self.config.p0, temperature=self.config.T0
            )
        )

        # training using alternating least squares

        p_pred = t_pred = np.zeros_like(reference_ds["norm_lnxsec"].values)
        p_coeffs = t_coeffs = None

        prev_rss = np.inf

        for iteration in range(max_iter):

            # Fit T given P (lnxsec - P_effect ~ T_effect)
            t_coeffs = self.temperature_form.fit(
                x_t, reference_ds["norm_lnxsec"].values - p_pred
            )
            t_pred = self.temperature_form.evaluate(x_t, t_coeffs)

            # Fit P given T (lnxsec - T_effect ~ P_effect)
            p_coeffs = self.pressure_form.fit(
                x_p, reference_ds["norm_lnxsec"].values - t_pred
            )
            p_pred = self.pressure_form.evaluate(x_p, p_coeffs)

            rss = np.sum((reference_ds["norm_lnxsec"].values - p_pred - t_pred) ** 2)

            if iteration > 0 and (prev_rss - rss) / prev_rss < 1e-8:
                break
            prev_rss = rss

            self.coeffs.pressure_coeffs = p_coeffs
            self.coeffs.temperature_coeffs = t_coeffs

            return rss

    def save(self, path: str) -> None:
        """Save the full model (config + coefficients) to disk."""
        model_ds = xr.Dataset(
            data_vars={
                "pressure_coeffs": (["pressure_coeff"], self.coeffs.pressure_coeffs),
                "temperature_coeffs": (
                    ["temperature_coeff"],
                    self.coeffs.temperature_coeffs,
                ),
            },
            coords={
                "pressure_coeff": self.pressure_form.coefficient_names,
                "temperature_coeff": self.temperature_form.coefficient_names,
            },
        )
        model_ds.attrs.update(vars(self.config))
        model_ds.to_netcdf(path)

    def load(self, path: str) -> None:
        """Load the full model (config + coefficients) from disk."""
        model_ds = xr.open_dataset(path)
        self.config = FunctionalConfig(**model_ds.attrs)
        self.coeffs.pressure_coeffs = model_ds["pressure_coeffs"].values
        self.coeffs.temperature_coeffs = model_ds["temperature_coeffs"].values

    def _validate_xsec_dataset(
        self,
        source: xr.Dataset | str | Path,
        freq_atol: float = 1e-3,
    ) -> xr.Dataset:

        REQUIRED_VARS = {"xsec", "pressure", "temperature"}
        REQUIRED_DIMS = {"frequency", "case"}
        REQUIRED_COORDS = {"frequency", "case"}

        if isinstance(source, (str, Path)):
            source = xr.open_dataset(source)
        if not isinstance(source, xr.Dataset):
            raise TypeError(f"Expected xr.Dataset or path, got {type(source)}")

        # check for required variables, dimensions, and coordinates
        missing_vars = REQUIRED_VARS - set(source.data_vars)
        if missing_vars:
            raise ValueError(f"Missing variables: {missing_vars}")

        source_dims = {str(dim) for dim in source.dims}
        missing_dims = REQUIRED_DIMS - source_dims
        if missing_dims:
            raise ValueError(f"Missing dimensions: {missing_dims}")

        source_coords = {str(coord) for coord in source.coords}
        missing_coords = REQUIRED_COORDS - source_coords
        if missing_coords:
            raise ValueError(f"Missing coordinates: {missing_coords}")

        # check if frequency grid matches expected grid
        freq_grid = self.config.frequency_grid
        if freq_grid is not None:
            ds_freq = source.coords["frequency"].values
            if not np.allclose(ds_freq, freq_grid, atol=freq_atol):
                raise ValueError("Frequency grid does not match expected grid")

        # check if p0 and T0 are in the dataset
        if self.config.p0 not in source.coords["pressure"].values:
            raise ValueError(f"Reference pressure p0={self.config.p0} not in dataset")
        if self.config.T0 not in source.coords["temperature"].values:
            raise ValueError(
                f"Reference temperature T0={self.config.T0} not in dataset"
            )

        return source
