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
from typing import Optional

import numpy as np
import xarray as xr

from .single_absorber import AbsorberConfig, SingleSpeciesModel


@dataclass(frozen=True)
class ContinuumConfig(AbsorberConfig):
    """Configuration for the continuum absorber."""

    continuum_type: str = ""  # self and/or foreign
    continuum_model: str = "MT_CKD_4.3"
    data_source: Optional[str] = "./../data/continuum/absco-ref_wv-mt-ckd.nc"


class ContinuumAbsorber(SingleSpeciesModel):
    config: ContinuumConfig

    def __init__(
        self,
        continuum_type: str,
        species: str = "H2O",
        frequency_grid: Optional[tuple[float, ...]] = None,
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

        continuum_ds = xr.open_dataset(self.config.data_source)[self._required_data]

        # Interpolate the continuum data to the frequency grid
        self._continuum_ds = self._interpolate_ds_to_frequency_grid(continuum_ds)

        # add frequency grid as a coordinate
        self._continuum_ds = self._continuum_ds.assign_coords(
            frequency=self.config.frequency_grid
        )


class H2OContinuum(ContinuumAbsorber):
    def __init__(
        self,
        frequency_grid: Optional[tuple[float, ...]] = None,
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
            continuum_type="self and foreign",
            data_source=data_source,
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
        species: str = "H2O",
        frequency_grid: Optional[tuple[float, ...]] = None,
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
            * rad_fun(self.config.frequency_grid, temperature)
            * 1e-4  # Convert from cm^2 to m^2
        )

        return xsec


class ForeignContinuumAbsorber(ContinuumAbsorber):
    def __init__(
        self,
        species: str = "H2O",
        frequency_grid: Optional[tuple[float, ...]] = None,
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
            * rad_fun(self.config.frequency_grid, temperature)
            * 1e-4  # Convert from cm^2 to m^2
        )

        return xsec
