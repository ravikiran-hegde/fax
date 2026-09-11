from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
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

# Smallest value either factor of the product model may take while fitting, so
# the alternating fit never divides by a factor that has run through zero.
FACTOR_FLOOR = 1e-12


def lnp(p, ref_pressure, **_ignored):
    return np.log(p / ref_pressure)


def lnp_withself(p, ref_pressure, vmr, ref_vmr, self_scaling):
    return np.log(
        (p / ref_pressure)
        * (1.0 + vmr * self_scaling)  # / (1.0 + ref_vmr * self_scaling)
    )


def p_ratio(p, ref_pressure, **_ignored):
    return p / ref_pressure


def p_ratio_withself(p, ref_pressure, vmr, ref_vmr, self_scaling):
    return (p / ref_pressure) * (1.0 + vmr * self_scaling)


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
    xsec_floor: float = 1e-45  # reference values below this are underflow, not data


@dataclass
class FunctionalCoeffs:
    xsec0: Optional[np.ndarray] = None
    pressure_coeffs: Optional[np.ndarray] = None
    temperature_coeffs: Optional[np.ndarray] = None
    x_p_range: Optional[np.ndarray] = None
    x_t_range: Optional[np.ndarray] = None


class FunctionalAbsorber(SingleSpeciesModel, SavableModel):
    """xsec = xsec0 * exp(P(ln p/p0) + T(dT)), fitted additively in log space."""

    # (without self-broadening, with it) -- the subclass swaps the pair.
    pressure_variables = (lnp, lnp_withself)

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
        xsec_floor: float = 1e-45,
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
            pressure_form_name=pressure_form_name,
            temperature_form_name=temperature_form_name,
            ref_pressure=ref_pressure,
            ref_temperature=ref_temperature,
            ref_vmr=ref_vmr,
            frequency_grid=frequency_grid,
            self_scaling=self_scaling,
            temperature_variable=temperature_variable,
            xsec_floor=xsec_floor,
        )
        self.coeffs = FunctionalCoeffs()

        plain, with_self = type(self).pressure_variables
        self.pressure_var = staticmethod(with_self if self_scaling != 0 else plain)
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
        # The forms carry no information outside the range they were fitted on,
        # and a rational in particular grows fast there. Hold the boundary value
        # instead of extrapolating.
        if self.coeffs.x_p_range is not None:
            x_p = np.clip(x_p, *self.coeffs.x_p_range)
        if self.coeffs.x_t_range is not None:
            x_t = np.clip(x_t, *self.coeffs.x_t_range)

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

        # Copied because irrelevant frequencies are zeroed in it below, and
        # .values on a selection can be a view into the reference.
        self.coeffs.xsec0 = (
            reference_ds["xsec"]
            .sel(
                pressure=self.config.ref_pressure,
                temperature=self.config.ref_temperature,
            )
            .values.copy()
        )

        rss = self._fit_coefficients(reference_ds, x_p, x_t, max_iter)

        # A frequency whose cross-section stays under the relevance floor at
        # every training case cannot carry optical depth anywhere in a column,
        # so the fit there is unconstrained noise. Zeroing xsec0 makes the model
        # return exactly zero, and zeroing the temperature coefficients makes
        # the factored product zero too, for readers that never see xsec0. The
        # pressure coefficients stay as fitted: they are what keeps a reciprocal
        # form's denominator away from zero.
        dead = reference_ds["xsec"].values.max(axis=0) < self.config.xsec_floor
        if dead.any():
            self.coeffs.xsec0[dead] = 0.0
            self.coeffs.temperature_coeffs[:, dead] = 0.0
            logger.info(
                "%s: %d/%d frequencies below the relevance floor, set to zero",
                self.config.species,
                int(dead.sum()),
                dead.size,
            )

        # Self-broadening raises the effective pressure above anything in the
        # reference, which is built at ref_vmr, so the valid range must allow
        # for it or the correction would be clipped away at the surface.
        from .utils import COLUMN_VMR

        excess = COLUMN_VMR.get(self.config.species, 0.0) * self.config.self_scaling
        self.coeffs.x_p_range = np.array(
            [x_p.min(), self._broadened_x_p_max(x_p.max(), excess)]
        )
        self.coeffs.x_t_range = np.array([x_t.min(), x_t.max()])

        return rss

    def _broadened_x_p_max(self, x_p_max: float, excess: float) -> float:
        """Upper abscissa bound once self-broadening has raised the pressure."""
        return x_p_max + np.log1p(excess)

    def _log_training_start(self, reference_ds, max_iter) -> None:
        logger.info(
            "Training %s (%s x %s, self_scaling=%s): %d reference cases, max_iter=%d",
            self.config.species,
            self.config.pressure_form_name,
            self.config.temperature_form_name,
            self.config.self_scaling,
            reference_ds.sizes.get("case", 0),
            max_iter,
        )

    def _fit_coefficients(self, reference_ds, x_p, x_t, max_iter) -> float:
        """Alternating least squares on ln(xsec/xsec0), where the model is additive."""
        target = reference_ds["xsec"].values / self.coeffs.xsec0
        np.log(target, out=target)

        # Underflowed reference values are a constant placeholder, not data.
        weights = (
            target > np.log(self.config.xsec_floor) - np.log(self.coeffs.xsec0)
        ).astype(float)
        n_masked = int((weights == 0).sum())
        if n_masked:
            logger.info(
                "%s: masked %d/%d underflowed reference values",
                self.config.species,
                n_masked,
                weights.size,
            )

        residual = np.empty_like(target)  # scratch buffer, reused every iteration
        p_pred = t_pred = 0.0
        prev_rss = np.inf

        self._log_training_start(reference_ds, max_iter)

        for iteration in range(max_iter):

            # Fit T given P (lnxsec - P_effect ~ T_effect)
            np.subtract(target, p_pred, out=residual)
            t_coeffs = self.temperature_form.fit(x_t, residual, weights)
            t_pred = self.temperature_form.evaluate(x_t, t_coeffs)

            # Fit P given T (lnxsec - T_effect ~ P_effect)
            np.subtract(target, t_pred, out=residual)
            p_coeffs = self.pressure_form.fit(x_p, residual, weights)
            p_pred = self.pressure_form.evaluate(x_p, p_coeffs)

            residual -= p_pred  # residual is now target - p - t
            rss = np.einsum("ij,ij,ij->j", weights, residual, residual)

            self.coeffs.pressure_coeffs = p_coeffs
            self.coeffs.temperature_coeffs = t_coeffs

            logger.debug("  iter %d/%d: rss=%.6g", iteration + 1, max_iter, rss.sum())

            if iteration > 0 and np.all(prev_rss - rss < 1e-10 * prev_rss):
                logger.info(
                    "Converged after %d iterations (rss=%.6g)", iteration + 1, rss.sum()
                )
                break
            prev_rss = rss
        else:
            logger.info("Reached max_iter=%d (rss=%.6g)", max_iter, rss.sum())

        return float(rss.sum())

    def train_in_frequency_chunks(
        self,
        reference_xsec: str | Path,
        frequency_chunk: int = 2000,
        n_workers: Optional[int] = None,
        save_path: Optional[str | Path] = None,
        **train_kwargs,
    ) -> None:
        """Fit the frequency grid in chunks, one process per chunk.

        Frequencies are fitted independently, so each chunk is a full training
        run over its own slice of the reference. Chunks already present in
        ``save_path`` are not refitted, and the file is rewritten as each one
        arrives.
        """

        grid = np.asarray(self.config.frequency_grid, dtype=float)
        chunks = [
            slice(start, min(start + frequency_chunk, grid.size))
            for start in range(0, grid.size, frequency_chunk)
        ]
        concat = dict(
            dim="frequency", data_vars="minimal", coords="minimal", compat="override"
        )

        parts = []
        if save_path is not None and Path(save_path).exists():
            with xr.open_dataset(save_path) as saved:
                parts.append(saved.load())
            chunks = [
                chunk
                for chunk in chunks
                if not np.isin(grid[chunk], parts[0]["frequency"].values).all()
            ]

        logger.info(
            "Training %s in %d chunk(s) of %d frequencies on %s workers",
            self.config.species,
            len(chunks),
            frequency_chunk,
            n_workers or "all",
        )

        config = asdict(self.config)
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(
                    _train_frequency_chunk,
                    type(self),
                    {**config, "frequency_grid": grid[chunk]},
                    str(reference_xsec),
                    chunk,
                    train_kwargs,
                )
                for chunk in chunks
            ]
            for fitted, future in enumerate(as_completed(futures), 1):
                parts.append(future.result())
                logger.info(
                    "%s: %d/%d chunks fitted", self.config.species, fitted, len(chunks)
                )
                if save_path is not None:
                    tmp_path = Path(save_path).with_suffix(".partial")
                    xr.concat(parts, **concat).sortby("frequency").to_netcdf(tmp_path)
                    tmp_path.replace(save_path)

        trained = xr.concat(parts, **concat).sortby("frequency")
        self.coeffs = type(self).from_dataset(trained.isel(species=0)).coeffs

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
                "x_p_range": (("bound",), self.coeffs.x_p_range),
                "x_t_range": (("bound",), self.coeffs.x_t_range),
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
                "bound": ["min", "max"],
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
            x_p_range=ds["x_p_range"].values if "x_p_range" in ds else None,
            x_t_range=ds["x_t_range"].values if "x_t_range" in ds else None,
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


def _train_frequency_chunk(
    absorber_class: type,
    config: dict,
    reference_xsec: str,
    chunk: slice,
    train_kwargs: dict,
) -> xr.Dataset:
    """Fit one slice of the frequency grid, in a worker process."""

    absorber = absorber_class(**config)
    with xr.open_dataset(reference_xsec) as reference:
        absorber.train(
            reference_xsec=reference.isel(frequency=chunk).load(), **train_kwargs
        )
    return absorber.to_dataset()


class NoLogFunctionalAbsorber(FunctionalAbsorber):
    """xsec = xsec0 * P(p/p0) * T(dT), so a spectral point costs no transcendental.

    Same forms and the same stored layout as the log model; only the abscissa,
    the way the two factors combine, and the fit that follows from it differ.
    """

    pressure_variables = (p_ratio, p_ratio_withself)

    @property
    def class_name(self) -> str:
        return f"NoLog_{super().class_name}"

    def cross_section_from_x_vars(
        self,
        x_p: np.ndarray,
        x_t: np.ndarray,
    ) -> np.ndarray:
        """Return cross-section matrix with shape (levels, frequency) from pre-computed x_p and x_t."""
        if self.coeffs.x_p_range is not None:
            x_p = np.clip(x_p, *self.coeffs.x_p_range)
        if self.coeffs.x_t_range is not None:
            x_t = np.clip(x_t, *self.coeffs.x_t_range)

        # Both factors scale the cross-section, so a negative one is meaningless;
        # clipping them separately stops two negatives making a positive.
        p_scale = np.clip(
            self.pressure_form.evaluate(x_p, self.coeffs.pressure_coeffs), 0, None
        )
        t_scale = np.clip(
            self.temperature_form.evaluate(x_t, self.coeffs.temperature_coeffs), 0, None
        )

        xsec = self.coeffs.xsec0 * p_scale * t_scale
        return np.clip(np.nan_to_num(xsec, nan=0, posinf=0, neginf=0), 0, 1e10)

    def _broadened_x_p_max(self, x_p_max: float, excess: float) -> float:
        """Upper abscissa bound once self-broadening has raised the pressure."""
        return x_p_max * (1.0 + excess)

    def _fit_coefficients(self, reference_ds, x_p, x_t, max_iter) -> float:
        """Alternating least squares on xsec/xsec0, where the model is a product."""
        target = reference_ds["xsec"].values / self.coeffs.xsec0
        # target is 1 at the reference case, so holding both factors to 1 there
        # makes the model return the stored xsec0 exactly.
        i_ref = int(
            np.flatnonzero(
                (reference_ds["pressure"].values == self.config.ref_pressure)
                & (reference_ds["temperature"].values == self.config.ref_temperature)
            )[0]
        )

        # Underflowed reference values are a constant placeholder, not data.
        # The kept ones carry weight 1/y^2, which makes the fit minimise
        # relative rather than absolute error in the cross-section.
        keep = target > self.config.xsec_floor / self.coeffs.xsec0
        weights = np.zeros_like(target)
        np.divide(1.0, target, out=weights, where=keep)
        weights *= weights
        n_masked = int((weights == 0).sum())
        if n_masked:
            logger.info(
                "%s: masked %d/%d underflowed reference values",
                self.config.species,
                n_masked,
                weights.size,
            )

        # Holding one factor of the product fixed leaves a weighted least-squares
        # problem for the other: w (y - P T)^2 = w P^2 (y/P - T)^2.
        ratio = np.empty_like(target)  # scratch buffer, reused every iteration
        scaled_weights = np.empty_like(target)
        p_pred = np.ones_like(target)
        prev_rss = np.inf

        self._log_training_start(reference_ds, max_iter)

        for iteration in range(max_iter):

            # Fit T given P (xsec / xsec0 / P_effect ~ T_effect)
            np.divide(target, p_pred, out=ratio)
            np.square(p_pred, out=scaled_weights)
            scaled_weights *= weights
            t_coeffs = self.temperature_form.fit(
                x_t, ratio, scaled_weights, x_ref=x_t[i_ref]
            )
            t_pred = np.clip(
                self.temperature_form.evaluate(x_t, t_coeffs), FACTOR_FLOOR, None
            )

            # Fit P given T (xsec / xsec0 / T_effect ~ P_effect)
            np.divide(target, t_pred, out=ratio)
            np.square(t_pred, out=scaled_weights)
            scaled_weights *= weights
            p_coeffs = self.pressure_form.fit(
                x_p, ratio, scaled_weights, x_ref=x_p[i_ref]
            )
            p_pred = np.clip(
                self.pressure_form.evaluate(x_p, p_coeffs), FACTOR_FLOOR, None
            )

            np.multiply(p_pred, t_pred, out=ratio)
            ratio -= target  # ratio is now the model error on xsec / xsec0
            rss = np.einsum("ij,ij,ij->j", weights, ratio, ratio)

            self.coeffs.pressure_coeffs = p_coeffs
            self.coeffs.temperature_coeffs = t_coeffs

            logger.debug("  iter %d/%d: rss=%.6g", iteration + 1, max_iter, rss.sum())

            if iteration > 0 and np.all(prev_rss - rss < 1e-10 * prev_rss):
                logger.info(
                    "Converged after %d iterations (rss=%.6g)", iteration + 1, rss.sum()
                )
                break
            prev_rss = rss
        else:
            logger.info("Reached max_iter=%d (rss=%.6g)", max_iter, rss.sum())

        self._refine_jointly(target, weights, x_p, x_t, i_ref)
        return float(rss.sum())

    def _refine_jointly(self, target, weights, x_p, x_t, i_ref) -> None:
        """Fit both factors together per frequency, starting from the ALS result.

        Alternating leaves each factor absorbing the other's error; refining
        them together removes that, and normalising each to 1 at the reference
        keeps the model equal to xsec0 there. A refinement that drives either
        factor through zero anywhere in range is discarded: the product model
        has no way to represent a sign change, so it would evaluate to zero.
        """
        from scipy.optimize import least_squares

        n_p = self.coeffs.pressure_coeffs.shape[0]
        p_grid = np.geomspace(max(x_p.min(), 1e-30), x_p.max(), 512)
        t_grid = np.linspace(x_t.min(), x_t.max(), 512)
        logger.info("Refining %s jointly ...", self.config.species)
        n_kept = 0

        for fi in range(target.shape[1]):
            used = weights[:, fi] > 0
            if used.sum() < n_p + 6:
                continue
            y = np.where(used, target[:, fi], 1.0)
            w = used.astype(float)
            start = np.concatenate(
                [
                    self.coeffs.pressure_coeffs[:, fi],
                    self.coeffs.temperature_coeffs[:, fi],
                ]
            )

            def factors(par):
                return (
                    self.pressure_form.evaluate(x_p, par[:n_p, None])[:, 0],
                    self.temperature_form.evaluate(x_t, par[n_p:, None])[:, 0],
                )

            def residual(par):
                p_fac, t_fac = factors(par)
                anchor = p_fac[i_ref] * t_fac[i_ref]
                return (p_fac * t_fac / anchor / y - 1.0) * w

            # Refinement polishes the alternating fit, it does not restructure
            # it: holding each coefficient's sign keeps whatever the form's own
            # fit established, non-negativity of a pole-free factor included.
            lower = np.where(start >= 0, 0.0, -np.inf)
            upper = np.where(start >= 0, np.inf, 0.0)
            result = least_squares(
                residual,
                np.clip(start, lower, upper),
                bounds=(lower, upper),
                loss="soft_l1",
                max_nfev=600,
            )
            p_fac, t_fac = factors(result.x)
            if not (
                np.isfinite(p_fac[i_ref]) and p_fac[i_ref] > 0 and t_fac[i_ref] > 0
            ):
                continue
            p_range = self.pressure_form.evaluate(p_grid, result.x[:n_p, None])
            t_range = self.temperature_form.evaluate(t_grid, result.x[n_p:, None])
            if p_range.min() <= 0 or t_range.min() <= 0:
                continue
            n_kept += 1
            self.coeffs.pressure_coeffs[:, fi] = self.pressure_form.rescale(
                result.x[:n_p], 1.0 / p_fac[i_ref]
            )
            self.coeffs.temperature_coeffs[:, fi] = self.temperature_form.rescale(
                result.x[n_p:], 1.0 / t_fac[i_ref]
            )
        logger.info(
            "%s: joint refinement kept for %d/%d frequencies",
            self.config.species,
            n_kept,
            target.shape[1],
        )
