"""
Halocarbon model from ARTS

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyarts3
import xarray as xr

from .abstract_class import (
    T_1D_ARRAYLIKE,
    AbsorberConfig,
    SavableModel,
    SingleSpeciesModel,
)


@dataclass
class CrossFitConfig(AbsorberConfig):
    """Configuration for the continuum absorber."""

    halocarbon_model: str = "XFIT"
    data_source: Optional[str] = "./../data/halocarbon/CFC11_XFIT.xml"


class CrossFitAbsorber(SingleSpeciesModel, SavableModel):
    config: CrossFitConfig
    _halocarbon_ds: xr.Dataset

    def __init__(
        self,
        species: str,
        frequency_grid: T_1D_ARRAYLIKE,
        data_source: Optional[str] = None,
    ):
        self.config = CrossFitConfig(
            species=species,
            frequency_grid=frequency_grid,
            data_source=data_source,
        )

        self._required_data = ["ref_pres", "ref_temp"]
        self._prepare_data()

    def cross_section(
        self,
        pressure: np.ndarray,
        temperature: np.ndarray,
        vmr: np.ndarray,
    ) -> np.ndarray:
        p00 = self._halocarbon_ds["p00"].values
        p10 = self._halocarbon_ds["p10"].values
        p01 = self._halocarbon_ds["p01"].values
        p20 = self._halocarbon_ds["p20"].values

        xsec = (
            p00
            + p10 * temperature[:, None]
            + p20 * temperature[:, None] ** 2
            + p01 * pressure[:, None]
        )
        # # Check for negative values and remove them without introducing bias, meaning
        # # the integral over the spectrum must not change. Not necessary.
        # logic = xsec < 0
        # if np.sum(logic) > 0:

        #     # original sum over spectrum
        #     sumX_org = np.sum(xsec)

        #     # remove negative values
        #     xsec[logic] = 0

        #     if sumX_org >= 0:
        #         # estimate ratio between altered and original sum of spectrum
        #         w = sumX_org / np.sum(xsec)

        #         # scale altered spectrum
        #         xsec = xsec * w

        return xsec.clip(0, None)

    def _interpolate_ds_to_frequency_grid(
        self, halocarbon_data: xr.Dataset
    ) -> xr.Dataset:
        """
        Interpolate the halocarbon data to the model's frequency grid using
        nearest neighbor. Cubic or nearest neighbor?.
        """
        target_nu = np.asarray(self.config.frequency_grid, dtype=float)

        interpolated_ds = halocarbon_data.interp(
            frequency=target_nu,
            method="cubic",
            kwargs={"fill_value": 0.0},
        )
        return interpolated_ds

    @staticmethod
    def _extract_band_name(name: str) -> int:
        prefix = name.removesuffix("_coeffs")
        if not prefix.startswith("band"):
            raise ValueError(f"Unexpected band variable name: {name}")
        return int(prefix.removeprefix("band"))

    @staticmethod
    def _xsec_xml_to_dataset(xml_path: str | Path) -> xr.Dataset:
        """Load an ARTS XsecRecord XML file into an xarray Dataset.

        The XML is read with ``pyarts3.xml.load(...).to_xarray()`` and then
        flattened into a single monotonic frequency axis because the bands in the
        record are disjoint.
        """
        xml_path = Path(xml_path)
        source = pyarts3.xml.load(str(xml_path)).to_xarray()

        species = source.attrs.get("species")
        if species is None:
            raise ValueError(f"Species metadata missing in {xml_path}")

        band_names = [name for name in source.data_vars if name.endswith("_coeffs")]
        if not band_names:
            raise ValueError(f"No band coefficient variables found in {xml_path}")

        band_names = sorted(band_names, key=CrossFitAbsorber._extract_band_name)
        frequency_blocks: list[np.ndarray] = []
        coefficient_blocks: list[np.ndarray] = []
        n_frequency: list[int] = []

        band_start_frequency: list[float] = []
        band_end_frequency: list[float] = []
        min_pressures = np.asarray(source["fitminpressures"].values, dtype=float)
        max_pressures = np.asarray(source["fitmaxpressures"].values, dtype=float)
        min_temperatures = np.asarray(source["fitmintemperatures"].values, dtype=float)
        max_temperatures = np.asarray(source["fitmaxtemperatures"].values, dtype=float)
        coefficient_names = np.asarray(source[band_names[0]].coords["coeffs"].values)
        coefficient_vars = [str(name) for name in coefficient_names]

        previous_stop: float | None = None
        for band_name in band_names:
            band_frequency_dim = next(
                dim for dim in source[band_name].dims if dim != "coeffs"
            )
            band_frequency = np.asarray(
                source.coords[band_frequency_dim].values, dtype=float
            )
            band_coefficients = np.asarray(source[band_name].values, dtype=float)

            if band_frequency.size == 0:
                raise ValueError(f"Empty frequency grid in {band_name} of {xml_path}")
            if previous_stop is not None and band_frequency[0] <= previous_stop:
                raise ValueError(
                    "Fit coefficient bands overlap or are not strictly ordered"
                )
            previous_stop = float(band_frequency[-1])

            frequency_blocks.append(band_frequency)
            for coeff_index, coeff_name in enumerate(coefficient_vars):
                if len(coefficient_blocks) <= coeff_index:
                    coefficient_blocks.append([])
                coefficient_blocks[coeff_index].append(
                    band_coefficients[:, coeff_index]
                )
            n_frequency.append(band_frequency.size)
            band_start_frequency.append(float(band_frequency[0]))
            band_end_frequency.append(float(band_frequency[-1]))

        frequency = np.concatenate(frequency_blocks)
        band_id = np.concatenate(
            [np.full(size, idx, dtype=int) for idx, size in enumerate(n_frequency)]
        )

        flattened_coefficients = {
            coeff_name: np.concatenate(blocks, axis=0)
            for coeff_name, blocks in zip(
                coefficient_vars, coefficient_blocks, strict=True
            )
        }

        return xr.Dataset(
            {
                "band_id": (("frequency",), band_id),
                "min_pressure": (("band",), min_pressures),
                "max_pressure": (("band",), max_pressures),
                "min_temperature": (("band",), min_temperatures),
                "max_temperature": (("band",), max_temperatures),
                "band_start_frequency": (
                    ("band",),
                    np.asarray(band_start_frequency, dtype=float),
                ),
                "band_end_frequency": (
                    ("band",),
                    np.asarray(band_end_frequency, dtype=float),
                ),
                "n_frequency": (("band",), np.asarray(n_frequency, dtype=int)),
                **{
                    coeff_name: (("frequency",), coeff_values)
                    for coeff_name, coeff_values in flattened_coefficients.items()
                },
            },
            coords={
                "frequency": frequency,
                "band": np.arange(len(frequency_blocks)),
                "species": species,
            },
            attrs={
                "source": str(xml_path),
                "format": "ARTS XFIT",
            },
        )

    def _prepare_data(self):
        """Prepare the continuum data on frequency grid"""

        if self.config.data_source is None:
            raise ValueError("Halocarbon absorber requires a data_source")

        halocarbon_ds = self._xsec_xml_to_dataset(self.config.data_source)

        # Interpolate the coeffs to the frequency grid
        self._halocarbon_ds = self._interpolate_ds_to_frequency_grid(halocarbon_ds)

    def to_dataset(self) -> xr.Dataset:
        """Convert a continuum dataset to be merge-compatible with the functional dataset."""

        halo_ds = self._halocarbon_ds.copy()
        halo_ds = halo_ds.drop_dims("band")
        halo_ds = halo_ds.drop_vars("band_id", errors="ignore")
        halo_ds.attrs["halocarbon_model"] = self.config.halocarbon_model
        halo_ds.attrs["data_source"] = self.config.data_source
        halo_ds.attrs["model_class"] = self.class_name

        return halo_ds.expand_dims("species")

    def save_data(self, path: str | Path) -> None:
        """Save the model to disk."""
        self._halocarbon_ds.attrs.update(vars(self.config))
        self._halocarbon_ds.to_netcdf(path)

    def load_data(self, path: str | Path) -> None:
        """Load the model from disk."""
        self._halocarbon_ds = xr.open_dataset(path)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> "CrossFitAbsorber":
        """Create a CrossFitAbsorber from an xarray Dataset."""
        config = CrossFitConfig(
            species=ds.coords["species"].values.item(),
            frequency_grid=ds.coords["frequency"].values,
            data_source=ds.attrs.get("data_source", None),
        )
        absorber = cls(
            species=config.species,
            frequency_grid=config.frequency_grid,
            data_source=config.data_source,
        )
        # TODO: Currently as _prepapre_data is called in __init__, the data reqires reloading from datasource. We could optimize this by allowing to pass the dataset directly to the constructor or by adding a method to set the dataset after initialization.
        return absorber

    @property
    def file_name(self) -> str:
        return f"{self.config.species}_{self.class_name}.nc"

    @property
    def class_name(self) -> str:
        return f"{self.config.halocarbon_model.replace('.', '_')}"
