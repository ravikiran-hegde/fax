from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from faxsec.abstract_class import (
    ARRAYLIKE,
    AbsorberConfig,
    SavableModel,
    SingleSpeciesModel,
)
from faxsec.constants import REF_PRESSURE, REF_TEMPERATURE, REF_VMR
from faxsec.forms import FunctionalForm, functional_form_registry

logger = logging.getLogger(__name__)


def lnp(p, ref_pressure, **_ignored):
    return np.log(p / ref_pressure)


def lnp_withself(p, ref_pressure, vmr, ref_vmr, self_scaling):
    return np.log(
        (p / ref_pressure)
        * (1.0 + vmr * self_scaling)  # / (1.0 + ref_vmr * self_scaling)
    )


def dT(T, ref_temperature):
    return T - ref_temperature


def T_ratio(T, ref_temperature):
    return T / ref_temperature


@dataclass
class FunctionalConfig(AbsorberConfig):
    pressure_form_name: str = "Hinge"
    temperature_form_name: str = "Rational"
    self_scaling: int | float = 0  # for self-broadening effects
    ref_pressure: float = REF_PRESSURE
    ref_temperature: float = REF_TEMPERATURE
    ref_vmr: float = REF_VMR
    temperature_variable: str = "dT"  # "dT" or "T_ratio"


@dataclass
class FunctionalCoeffs:
    xsec0: Optional[np.ndarray] = None
    pressure_coeffs: Optional[np.ndarray] = None
    temperature_coeffs: Optional[np.ndarray] = None


class FunctionalAbsorber(SingleSpeciesModel, SavableModel):
    pressure_form: FunctionalForm
    temperature_form: FunctionalForm
    config: "FunctionalConfig"
    coeffs: "FunctionalCoeffs"

    def __init__(
        self,
        species: str,
        frequency_grid: ARRAYLIKE,
        pressure_form_name: str = "Hinge",
        temperature_form_name: str = "Rational",
        ref_pressure: float = REF_PRESSURE,
        ref_temperature: float = REF_TEMPERATURE,
        ref_vmr: float = REF_VMR,
        self_scaling: int | float = 0,
        temperature_variable: str = "dT",
    ) -> None:
        pressure_form = functional_form_registry.get(pressure_form_name)
        temperature_form = functional_form_registry.get(temperature_form_name)
        if pressure_form is None:
            raise ValueError(f"Unknown pressure form: {pressure_form_name}")
        if temperature_form is None:
            raise ValueError(f"Unknown temperature form: {temperature_form_name}")
        if temperature_variable not in ("dT", "T_ratio"):
            raise ValueError(
                f"Unknown temperature_variable: {temperature_variable}. "
                "Use 'dT' or 'T_ratio'."
            )

        self.pressure_form = pressure_form
        self.temperature_form = temperature_form
        self.config = FunctionalConfig(
            species=species,
            ref_pressure=ref_pressure,
            ref_temperature=ref_temperature,
            ref_vmr=ref_vmr,
            frequency_grid=frequency_grid,
            self_scaling=self_scaling,
            temperature_variable=temperature_variable,
        )
        self.coeffs = FunctionalCoeffs()

        self.pressure_var = (
            staticmethod(lnp_withself) if self_scaling != 0 else staticmethod(lnp)
        )
        self.temperature_var = staticmethod(
            T_ratio if temperature_variable == "T_ratio" else dT
        )

    def cross_section(
        self,
        pressure: np.ndarray,
        temperature: np.ndarray,
        vmr: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return cross-section matrix with shape (levels, frequency)."""
        x_p = self.pressure_var(
            pressure,
            self.config.ref_pressure,
            vmr=vmr,
            ref_vmr=self.config.ref_vmr,
            self_scaling=self.config.self_scaling,
        )
        x_t = self.temperature_var(temperature, self.config.ref_temperature)

        return self.cross_section_from_x_vars(x_p, x_t)

    def cross_section_from_x_vars(
        self,
        x_p: np.ndarray,
        x_t: np.ndarray,
    ) -> np.ndarray:
        """Return cross-section matrix with shape (levels, frequency) from pre-computed x_p and x_t."""
        p_scale = self.pressure_form.evaluate(x_p, self.coeffs.pressure_coeffs)
        t_scale = self.temperature_form.evaluate(x_t, self.coeffs.temperature_coeffs)

        xsec = self.coeffs.xsec0 * np.exp(p_scale + t_scale)
        return np.clip(np.nan_to_num(xsec, nan=0, posinf=0, neginf=0), 0, 1e10)

    def train(
        self,
        reference_xsec: Optional[str | Path] = None,
        max_iter: int = 4,
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
            from .utils import calulate_arts_reference, sample_atmospheres

            arts_reference_kwargs = training_kwargs.get("arts_reference_kwargs", {})

            p_grid, t_grid = sample_atmospheres(**sampling_kwargs)

            has_ref_case = np.any(
                np.isclose(p_grid, self.config.ref_pressure)
                & np.isclose(t_grid, self.config.ref_temperature)
            )
            if not has_ref_case:
                p_grid = np.append(p_grid, [self.config.ref_pressure])
                t_grid = np.append(t_grid, [self.config.ref_temperature])

            reference_ds = calulate_arts_reference(
                self.config.species,
                self.config.frequency_grid,
                p_grid,
                t_grid,
                np.full_like(p_grid, self.config.ref_vmr),
                **arts_reference_kwargs,
            )

            reference_ds = self._validate_xsec_dataset(reference_ds, freq_atol=1e-3)

        x_p = self.pressure_var(
            reference_ds["pressure"].values,
            self.config.ref_pressure,
            vmr=reference_ds["vmr"].values,
            ref_vmr=self.config.ref_vmr,
            self_scaling=self.config.self_scaling,
        )
        x_t = self.temperature_var(
            reference_ds["temperature"].values, self.config.ref_temperature
        )

        self.coeffs.xsec0 = (
            reference_ds["xsec"]
            .sel(
                pressure=self.config.ref_pressure,
                temperature=self.config.ref_temperature,
            )
            .values
        )

        reference_ds["norm_lnxsec"] = np.log(
            reference_ds["xsec"]
            / reference_ds["xsec"].sel(
                pressure=self.config.ref_pressure,
                temperature=self.config.ref_temperature,
            )
        )

        # training using alternating least squares

        p_pred = t_pred = np.zeros_like(reference_ds["norm_lnxsec"].values)
        p_coeffs = t_coeffs = None

        prev_rss = np.inf

        logger.info(
            "Training %s (%s x %s, self_scaling=%s): %d reference cases, max_iter=%d",
            self.config.species,
            self.config.pressure_form_name,
            self.config.temperature_form_name,
            self.config.self_scaling,
            reference_ds.sizes.get("case", 0),
            max_iter,
        )

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

            self.coeffs.pressure_coeffs = p_coeffs
            self.coeffs.temperature_coeffs = t_coeffs

            logger.debug("  iter %d/%d: rss=%.6g", iteration + 1, max_iter, rss)

            if iteration > 0 and (prev_rss - rss) / prev_rss < 1e-10:
                logger.info(
                    "Converged after %d iterations (rss=%.6g)", iteration + 1, rss
                )
                break
            prev_rss = rss
        else:
            logger.info("Reached max_iter=%d (rss=%.6g)", max_iter, rss)

        return rss

    def to_dataset(self) -> xr.Dataset:

        ds = xr.Dataset(
            {
                "xsec0": (("frequency",), self.coeffs.xsec0),
                "pressure_coeffs": (
                    ("p_order", "frequency"),
                    self.coeffs.pressure_coeffs,
                ),
                "temperature_coeffs": (
                    ("t_order", "frequency"),
                    self.coeffs.temperature_coeffs,
                ),
                "ref_pressure": ((), self.config.ref_pressure),
                "ref_temperature": ((), self.config.ref_temperature),
                "ref_vmr": ((), self.config.ref_vmr),
                "self_scaling": ((), float(self.config.self_scaling)),
            },
            coords={
                "frequency": self.config.frequency_grid,
                "p_order": (
                    self.pressure_form.coefficient_names()
                    if self.coeffs.pressure_coeffs is not None
                    else 0
                ),
                "t_order": (
                    self.temperature_form.coefficient_names()
                    if self.coeffs.temperature_coeffs is not None
                    else 0
                ),
                "species": self.config.species,
            },
            attrs={
                "pressure_form": self.config.pressure_form_name,
                "temperature_form": self.config.temperature_form_name,
                "temperature_variable": self.config.temperature_variable,
                "model_class": self.class_name,
            },
        )
        return ds.expand_dims("species")

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> FunctionalAbsorber:
        """Create a FunctionalAbsorber from an xarray Dataset."""
        config = FunctionalConfig(
            species=ds.coords["species"].values.item(),
            pressure_form_name=ds.attrs["pressure_form"],
            temperature_form_name=ds.attrs["temperature_form"],
            frequency_grid=ds.coords["frequency"].values,
            self_scaling=float(ds.self_scaling.values),
            ref_pressure=float(ds.ref_pressure.values),
            ref_temperature=float(ds.ref_temperature.values),
            ref_vmr=float(ds.ref_vmr.values),
            temperature_variable=ds.attrs.get("temperature_variable", "dT"),
        )
        coeffs = FunctionalCoeffs(
            xsec0=ds.xsec0.values,
            pressure_coeffs=ds.pressure_coeffs.values,
            temperature_coeffs=ds.temperature_coeffs.values,
        )
        absorber = cls(
            species=config.species,
            frequency_grid=config.frequency_grid,
            pressure_form_name=config.pressure_form_name,
            temperature_form_name=config.temperature_form_name,
            ref_pressure=config.ref_pressure,
            ref_temperature=config.ref_temperature,
            ref_vmr=config.ref_vmr,
            self_scaling=config.self_scaling,
            temperature_variable=config.temperature_variable,
        )
        absorber.coeffs = coeffs
        return absorber

    def save_data(self, path: str | Path) -> None:
        """Save the full model (config + coefficients) to disk."""
        model_ds = self.to_dataset()
        model_ds.to_netcdf(path)

    def load_data(self, path: str | Path) -> None:
        """Load the full model (config + coefficients) from disk."""
        model_ds = xr.open_dataset(path)
        absorber = self.from_dataset(model_ds)
        self.config = absorber.config
        self.coeffs = absorber.coeffs

    @property
    def file_name(self) -> str:
        return f"{self.config.species}_{self.class_name}.nc"

    @property
    def class_name(self) -> str:
        return f"{self.config.pressure_form_name}_{self.config.temperature_form_name}"

    def _validate_xsec_dataset(
        self,
        source: xr.Dataset | str | Path,
        freq_atol: float = 1e-3,
    ) -> xr.Dataset:

        REQUIRED_VARS = {
            "xsec",
        }
        REQUIRED_DIMS = {
            "frequency",
            "case",
        }
        REQUIRED_COORDS = {"frequency", "case", "pressure", "temperature"}

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
            if missing_coords.issubset(set(source.data_vars)):
                source = source.set_index(case=list(missing_coords))
            else:
                raise ValueError(f"Missing coordinates: {missing_coords}")

        # check if frequency grid matches expected grid
        freq_grid = self.config.frequency_grid
        if freq_grid is not None:
            ds_freq = source.coords["frequency"].values
            if not np.allclose(ds_freq, freq_grid, atol=freq_atol):
                raise ValueError("Frequency grid does not match expected grid")

        # check if ref_pressure and ref_temperature are in the dataset
        if self.config.ref_pressure not in source.coords["pressure"].values:
            raise ValueError(
                f"Reference pressure ref_pressure={self.config.ref_pressure} not in dataset"
            )
        if self.config.ref_temperature not in source.coords["temperature"].values:
            raise ValueError(
                f"Reference temperature ref_temperature={self.config.ref_temperature} not in dataset"
            )

        return source
