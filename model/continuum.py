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

from .abstract_class import (
    T_1D_ARRAYLIKE,
    AbsorberConfig,
    SavableModel,
    SingleSpeciesModel,
)


@dataclass
class ContinuumConfig(AbsorberConfig):
    """Configuration for the continuum absorber."""

    continuum_type: str = ""  # self or foreign or both
    continuum_model: str = "MT_CKD_4.3"
    data_source: Optional[str] = "./../data/continuum/absco-ref_wv-mt-ckd.nc"


class ContinuumAbsorber(SingleSpeciesModel, SavableModel):
    config: ContinuumConfig
    _continuum_ds: xr.Dataset

    def __init__(
        self,
        species: str,
        continuum_type: str,
        frequency_grid: T_1D_ARRAYLIKE,
        data_source: Optional[str] = None,
    ):
        self.config = ContinuumConfig(
            species=species,
            frequency_grid=frequency_grid,
            continuum_type=continuum_type,
            data_source=data_source,
        )

        self._required_data = ["ref_pres", "ref_temp"]
        self._prepare_data()

    def cross_section(
        self,
        pressure: np.ndarray,
        temperature: np.ndarray,
        vmr: np.ndarray,
    ) -> np.ndarray: ...

    def _interpolate_ds_to_frequency_grid(
        self, continuum_data: xr.Dataset
    ) -> xr.Dataset:
        """Interpolate the continuum data to the model's frequency grid using
        cubic interpolation (scipy-backed via xarray)."""
        from .utils import hz_to_kayser

        target_nu = np.asarray(hz_to_kayser(self.config.frequency_grid), dtype=float)

        interpolated_ds = continuum_data.interp(
            wavenumbers=target_nu,
            method="cubic",
            kwargs={"fill_value": 0.0},
        )
        return interpolated_ds

    def _prepare_data(self):
        """Prepare the continuum data on frequency grid"""

        if self.config.data_source is None:
            raise ValueError("Continuum absorber requires a data_source")

        continuum_ds = xr.open_dataset(self.config.data_source)[self._required_data]

        # Interpolate the continuum data to the frequency grid
        self._continuum_ds = self._interpolate_ds_to_frequency_grid(continuum_ds)

    def to_dataset(self) -> xr.Dataset:
        """Convert a continuum dataset to be merge-compatible with the functional dataset."""

        cont_ds = self._continuum_ds.copy()

        # rename ref_press / ref_temp to match functional dataset's naming
        rename_map = {
            "ref_press": "ref_pressure",
            "ref_temp": "ref_temperature",
        }
        rename_map = {k: v for k, v in rename_map.items() if k in cont_ds.variables}
        cont_ds = cont_ds.rename(rename_map)

        cont_ds["frequency"] = ("wavenumbers", self.config.frequency_grid)
        # move every variable on "wavenumbers" onto the existing "frequency" dim
        if "wavenumbers" in cont_ds.dims:
            cont_ds = cont_ds.swap_dims({"wavenumbers": "frequency"}).drop_vars(
                "wavenumbers"
            )

        species_name = self.config.species
        cont_ds.attrs["continuum_type"] = self.config.continuum_type
        cont_ds.attrs["continuum_model"] = self.config.continuum_model
        cont_ds.attrs["data_source"] = self.config.data_source
        cont_ds.attrs["model_class"] = self.class_name

        return cont_ds.expand_dims("species").assign_coords(species=[species_name])

    def save_data(self, path: str | Path) -> None:
        """Save the model to disk."""
        self._continuum_ds.attrs.update(vars(self.config))
        self._continuum_ds.to_netcdf(path)

    def load_data(self, path: str | Path) -> None:
        """Load the model from disk."""
        self._continuum_ds = xr.open_dataset(path)

    @property
    def file_name(self) -> str:
        return f"{self.config.species}_{self.class_name}.nc"

    @property
    def class_name(self) -> str:
        return f"{self.config.continuum_type}_continuum_{self.config.continuum_model.replace('.', '_')}"


class H2OContinuum(ContinuumAbsorber):
    def __init__(
        self,
        frequency_grid: T_1D_ARRAYLIKE,
        data_source: Optional[str] = None,
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

        self._prepare_data()

    def _prepare_data(self):
        """Prepare the continuum data on frequency grid"""
        self._continuum_ds = xr.merge(
            [self._self_continuum._continuum_ds, self._foreign_continuum._continuum_ds],
            compat="no_conflicts",
        )

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
        frequency_grid: T_1D_ARRAYLIKE,
        data_source: Optional[str] = None,
    ):
        self.config = ContinuumConfig(
            species=species,
            frequency_grid=frequency_grid,
            continuum_type="self",
            data_source=data_source,
        )

        self._required_data = ["ref_press", "ref_temp", "self_absco_ref", "self_texp"]
        self._prepare_data()
        self.ref_pressure = (
            float(self._continuum_ds["ref_press"].values) * 100
        )  # Convert from mBar to Pa
        self.ref_temperature = float(self._continuum_ds["ref_temp"].values)
        self.self_texp = self._continuum_ds["self_texp"].values
        self.self_absco_ref = self._continuum_ds["self_absco_ref"].values

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
            * 1e-4  # Convert from cm^2 to m^2
        )

        return xsec


class ForeignContinuumAbsorber(ContinuumAbsorber):
    def __init__(
        self,
        species: str,
        frequency_grid: T_1D_ARRAYLIKE,
        data_source: Optional[str] = None,
    ):
        self.config = ContinuumConfig(
            species=species,
            frequency_grid=frequency_grid,
            continuum_type="foreign",
            data_source=data_source,
        )

        self._required_data = ["ref_press", "ref_temp", "for_absco_ref"]
        self._prepare_data()
        self.ref_pressure = (
            float(self._continuum_ds["ref_press"].values) * 100
        )  # Convert from mBar to Pa
        self.ref_temperature = float(self._continuum_ds["ref_temp"].values)
        self.foreign_absco_ref = self._continuum_ds["for_absco_ref"].values

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
            * 1e-4  # Convert from cm^2 to m^2
        )

        return xsec
