"""
MT_CKD Water Vapor Continuum

Implemented from  from Mlawer et al., JQSRT 2023 and Fortran90 code that contains the following statement:

!  --------------------------------------------------------------------------
! |  Copyright ©, Atmospheric and Environmental Research, Inc., 2022         |
! |                                                                          |
! |  All rights reserved. This source code was developed as part of the      |
! |  LBLRTM software and is designed for scientific and research purposes.   |
! |  Atmospheric and Environmental Research Inc. (AER) grants USER the right |
! |  to download, install, use and copy this software for scientific and     |
! |  research purposes only. This software may be redistributed as long as   |
! |  this copyright notice is reproduced on any copy made and appropriate    |
! |  acknowledgment is given to AER. This software or any modified version   |
! |  of this software may not be incorporated into proprietary software or   |
! |  commercial software offered for sale without the express written        |
! |  consent of AER.                                                         |
! |                                                                          |
! |  This software is provided as is without any express or implied          |
! |  warranties.                                                             |
!  --------------------------------------------------------------------------

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from faxsec.constants import CM_TO_M

from .abstract_class import (
    ARRAYLIKE,
    AbsorberConfig,
    SavableModel,
    SingleSpeciesModel,
)


@dataclass
class ContinuumConfig(AbsorberConfig):
    """Configuration for the continuum absorber."""

    continuum_type: str = ""  # self or foreign or both
    model: str = "MT_CKD_4.3"
    data_source: Optional[str] = "./../data/continuum/absco-ref_wv-mt-ckd.nc"


class ContinuumAbsorber(SingleSpeciesModel, SavableModel):
    config: ContinuumConfig
    _data: xr.Dataset

    def __init__(
        self,
        species: str,
        continuum_type: str,
        frequency_grid: ARRAYLIKE,
        data_source: Optional[str | xr.Dataset] = None,
    ):
        self._required_data = ["ref_pres", "ref_temp"]
        if data_source is not None and isinstance(data_source, xr.Dataset):
            self._data = data_source
            self.config = ContinuumConfig(
                species=species,
                frequency_grid=frequency_grid,
                continuum_type=continuum_type,
                data_source=self._data.attrs.get("data_source", None),
            )
        else:
            self.config = ContinuumConfig(
                species=species,
                frequency_grid=frequency_grid,
                continuum_type=continuum_type,
                data_source=data_source,
            )

            self._prepare_raw_data()

        self._validate_data()

    def cross_section(
        self,
        pressure: np.ndarray,
        temperature: np.ndarray,
        vmr: np.ndarray,
    ) -> np.ndarray: ...

    @staticmethod
    def interpolate_ds_to_frequency_grid(
        ds: xr.Dataset, output_frequency_grid
    ) -> xr.Dataset:
        """Interpolate the continuum data to the model's frequency grid using
        cubic interpolation (scipy-backed via xarray)."""
        from .utils import hz_to_kayser

        target_nu = np.asarray(hz_to_kayser(output_frequency_grid), dtype=float)

        interpolated_ds = ds.interp(
            wavenumbers=target_nu,
            method="cubic",
            kwargs={"fill_value": 0.0},
        )
        return interpolated_ds

    def _validate_data(self) -> None:
        """Validate that the required data is present in the dataset."""

        missing_vars = [
            var for var in self._required_data if var not in self._data.variables
        ]
        if missing_vars:
            raise ValueError(
                f"Missing required variables in continuum dataset: {missing_vars}"
            )

        if not np.allclose(self._data["frequency"].values, self.config.frequency_grid):
            raise ValueError(
                "Frequency grid in dataset does not match the model's frequency grid."
            )

    def _prepare_raw_data(self):
        """Prepare the continuum data on frequency grid"""

        if self.config.data_source is None:
            raise ValueError("Continuum absorber requires a data_source")

        continuum_ds = xr.open_dataset(self.config.data_source)

        # rename ref_press / ref_temp to match functional dataset's naming
        rename_map = {
            "ref_press": "ref_pressure",
            "ref_temp": "ref_temperature",
        }
        rename_map = {
            k: v for k, v in rename_map.items() if k in continuum_ds.variables
        }
        continuum_ds = continuum_ds.rename(rename_map)

        # Convert from mBar to Pa and update attributes
        continuum_ds["ref_pressure"] = continuum_ds["ref_pressure"] * 100
        continuum_ds["ref_pressure"].attrs["units"] = "Pa"

        # interpolate required variables to the model's frequency grid
        continuum_ds = self.interpolate_ds_to_frequency_grid(
            continuum_ds[self._required_data], self.config.frequency_grid
        )

        # move every variable on "wavenumbers" onto the existing "frequency" dim
        continuum_ds["frequency"] = ("wavenumbers", self.config.frequency_grid)

        if "wavenumbers" in continuum_ds.dims:
            continuum_ds = continuum_ds.swap_dims(
                {"wavenumbers": "frequency"}
            ).drop_vars("wavenumbers")
        for var in continuum_ds.data_vars:
            if continuum_ds[var].attrs.get("units", None) == "cm**2/molecule cm-1":
                continuum_ds[var] = (
                    continuum_ds[var] / CM_TO_M**2
                )  # Convert from cm^2 to m^2
                continuum_ds[var].attrs["units"] = "m**2/molecule cm-1"

        species_name = self.config.species
        continuum_ds.attrs["continuum_type"] = self.config.continuum_type
        continuum_ds.attrs["model"] = self.config.model
        continuum_ds.attrs["data_source"] = self.config.data_source
        continuum_ds.attrs["model_class"] = self.class_name

        continuum_ds = continuum_ds.expand_dims("species").assign_coords(
            species=[species_name]
        )

        # Interpolate the continuum data to the frequency grid
        self._data = continuum_ds

    def to_dataset(self) -> xr.Dataset:
        """Convert a continuum dataset to be merge-compatible with the functional dataset."""
        return self._data.copy()

    def save_data(self, path: str | Path) -> None:
        """Save the model to disk."""
        self._data.attrs.update(vars(self.config))
        self._data.to_netcdf(path)

    def load_data(self, path: str | Path) -> None:
        """Load the model from disk."""
        self._data = xr.open_dataset(path)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> "ContinuumAbsorber":
        """Create a ContinuumAbsorber from an xarray Dataset."""

        absorber = cls(
            species=ds.coords["species"].values.item(),
            frequency_grid=ds.coords["frequency"].values,
            continuum_type=ds.attrs.get("continuum_type", ""),
            data_source=ds,
        )

        return absorber

    @property
    def file_name(self) -> str:
        return f"{self.config.species}_{self.class_name}.nc"

    @property
    def class_name(self) -> str:
        return f"{self.config.continuum_type}_continuum_{self.config.model.replace('.', '_')}"


class H2OContinuum(ContinuumAbsorber):
    def __init__(
        self,
        frequency_grid: ARRAYLIKE,
        data_source: Optional[str] = None,
        **_ignored,  # for uniform api for class methods.
    ):
        self._self_continuum = SelfContinuumAbsorber(
            species="H2O", frequency_grid=frequency_grid, data_source=data_source
        )
        self._foreign_continuum = ForeignContinuumAbsorber(
            species="H2O", frequency_grid=frequency_grid, data_source=data_source
        )

        self.config = ContinuumConfig(
            species="H2O",
            frequency_grid=frequency_grid,
            continuum_type="both",
            data_source=data_source,
        )

        self._prepare_raw_data()

    def _prepare_raw_data(self):
        """Prepare the continuum data on frequency grid"""
        self._data = xr.merge(
            [self._self_continuum._data, self._foreign_continuum._data],
            compat="no_conflicts",
        )
        self._data.attrs["continuum_type"] = "both"
        self._data.attrs["model_class"] = self.class_name

    def cross_section(
        self,
        pressure: np.ndarray,
        temperature: np.ndarray,
        vmr: np.ndarray,
    ) -> np.ndarray:
        """Calculate the total continuum cross-section as the sum of self and foreign contributions."""
        xsec_self = self._self_continuum.cross_section(pressure, temperature, vmr)
        xsec_foreign = self._foreign_continuum.cross_section(pressure, temperature, vmr)
        return xsec_self + xsec_foreign


class SelfContinuumAbsorber(ContinuumAbsorber):
    def __init__(
        self,
        species: str,
        frequency_grid: ARRAYLIKE,
        data_source: Optional[str] = None,
    ):

        self._required_data = [
            "ref_pressure",
            "ref_temperature",
            "self_absco_ref",
            "self_texp",
        ]

        if data_source is not None and isinstance(data_source, xr.Dataset):
            self._data = data_source
            self.config = ContinuumConfig(
                species=species,
                frequency_grid=frequency_grid,
                continuum_type="self",
                data_source=self._data.attrs.get("data_source", None),
            )
        else:
            self.config = ContinuumConfig(
                species=species,
                frequency_grid=frequency_grid,
                continuum_type="self",
                data_source=data_source,
            )

            self._prepare_raw_data()

        self._validate_data()

        self.ref_pressure = float(self._data["ref_pressure"].values.squeeze())
        self.ref_temperature = float(self._data["ref_temperature"].values.squeeze())
        self.self_texp = self._data["self_texp"].values
        self.self_absco_ref = self._data["self_absco_ref"].values

    def cross_section(
        self,
        pressure: np.ndarray,
        temperature: np.ndarray,
        vmr: np.ndarray,
    ) -> np.ndarray:
        """Calculate the self-continuum cross-section using the interpolated dataset."""
        from .utils import rad_fun

        partial_p_ratio = (pressure * vmr / self.ref_pressure)[:, None]
        temperature_ratio = (self.ref_temperature / temperature)[:, None]

        xsec = (
            self.self_absco_ref
            * partial_p_ratio
            * (temperature_ratio ** (self.self_texp + 1.0))
            * rad_fun(self.config.frequency_grid, temperature[:, None])
        )

        return xsec


class ForeignContinuumAbsorber(ContinuumAbsorber):
    def __init__(
        self,
        species: str,
        frequency_grid: ARRAYLIKE,
        data_source: Optional[str | xr.Dataset] = None,
    ):
        self._required_data = ["ref_pressure", "ref_temperature", "for_absco_ref"]

        if data_source is not None and isinstance(data_source, xr.Dataset):
            self._data = data_source
            self.config = ContinuumConfig(
                species=species,
                frequency_grid=frequency_grid,
                continuum_type="foreign",
                data_source=self._data.attrs.get("data_source", None),
            )
        else:
            self.config = ContinuumConfig(
                species=species,
                frequency_grid=frequency_grid,
                continuum_type="foreign",
                data_source=data_source,
            )

            self._prepare_raw_data()

        self._validate_data()

        self.ref_pressure = float(self._data["ref_pressure"].values.squeeze())
        self.ref_temperature = float(self._data["ref_temperature"].values.squeeze())
        self.foreign_absco_ref = self._data["for_absco_ref"].values

    def cross_section(
        self,
        pressure: np.ndarray,
        temperature: np.ndarray,
        vmr: np.ndarray,
    ) -> np.ndarray:
        """Calculate the foreign-continuum cross-section using the interpolated dataset."""
        from .utils import rad_fun

        partial_p_ratio = (pressure * (1.0 - vmr) / self.ref_pressure)[:, None]
        temperature_ratio = (self.ref_temperature / temperature)[:, None]

        xsec = (
            self.foreign_absco_ref
            * partial_p_ratio
            * temperature_ratio
            * rad_fun(self.config.frequency_grid, temperature[:, None])
        )

        return xsec
