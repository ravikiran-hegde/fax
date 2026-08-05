from dataclasses import dataclass
from typing import Dict, Tuple, Type

import numpy as np
import xarray as xr

from faxsec.abstract_class import ARRAYLIKE, SavableModel, SingleSpeciesModel
from faxsec.constants import BOLTZMANN
from faxsec.continuum import H2OContinuum
from faxsec.functional import FunctionalAbsorber
from faxsec.xfit import CrossFitAbsorber


@dataclass
class GasOpticsConfig:
    """Configuration for the gas optics model."""

    species: Tuple[str, ...]  # e.g., ("H2O", "CO2", "O3")
    frequency_grid: ARRAYLIKE


class GasOptics:
    _absorbers: Dict[str, SingleSpeciesModel]

    def __init__(
        self,
        species: Tuple[str, ...],
        frequency_grid: ARRAYLIKE = (1012.2305,),
    ) -> None:

        self.config = GasOpticsConfig(species=species, frequency_grid=frequency_grid)
        self._absorbers = {}

    @classmethod
    def from_absorbers(cls, absorbers: Dict[str, SingleSpeciesModel]) -> "GasOptics":
        """Create a GasOptics instance from a dictionary of absorbers."""
        if not absorbers:
            raise ValueError("No absorbers provided.")
        frequency_grid = absorbers[next(iter(absorbers))].config.frequency_grid
        species = tuple(absorber.config.species for absorber in absorbers.values())
        gas_optics = cls(species=species, frequency_grid=frequency_grid)
        gas_optics._absorbers = absorbers
        gas_optics.validate()
        return gas_optics

    @classmethod
    def from_datatree(cls, dt: xr.DataTree) -> "GasOptics":
        """Create a GasOptics instance from a datatree."""
        absorbers = {}
        for class_name in dt.keys():
            absorber_cls = absorber_registry.get(class_name, None)
            if absorber_cls is None:
                print(f"Unsupported faxsec class {class_name} in datatree. Skipping.")
            else:
                ds = dt[class_name]
                for sp in ds.species.values:
                    sp_ds = ds.sel(species=sp)
                    absorber = absorber_cls.from_dataset(sp_ds.to_dataset())
                    absorbers[str(sp) + "_" + class_name] = absorber
        return cls.from_absorbers(absorbers)

    def add_absorber(self, absorber: SingleSpeciesModel) -> None:
        """Add an absorber to the model."""
        self._absorbers[str(absorber.config.species) + "_" + absorber.class_name] = (
            absorber
        )
        self.config.species = (
            self.species
        )  # Update species list to include new absorber
        self.validate()
        print(f"Updated absrobers: {list(self._absorbers.keys())}")

    def validate(self) -> None:
        """Validate the configuration of the gas optics model."""
        if not self._absorbers:
            raise ValueError("No absorbers have been added to the model.")
        for absorber in self._absorbers.values():
            if not isinstance(absorber, SingleSpeciesModel):
                raise TypeError(
                    f"Absorber {absorber} is not an instance of SingleSpeciesModel."
                )
            if not np.allclose(
                absorber.config.frequency_grid, self.config.frequency_grid
            ):
                raise ValueError(
                    f"Frequency grid mismatch for absorber {absorber.config.species}. "
                    f"Expected {self.config.frequency_grid}, got {absorber.config.frequency_grid}."
                )
            if absorber.config.species not in self.config.species:
                raise ValueError(
                    f"Species {absorber.config.species} of absorber {absorber} "
                    f"is not in the configured species list {self.config.species}."
                )

    @staticmethod
    def _decode_species(species: str):
        """
        JUST AN IDEA STILL
        Decode the species string to determine the model type and parameters.

        species is in the format Speciesname:Model:Args1_Args2_...

        Model options:
        - ARTS or A: ARTS line-by-line model (requires ARTS data files)
            args: ARTS tags "H2O_H2O-ForeignContCKDMT400_H2O-SelfContCKDMT400")
        - Functional or F: Functional form model (e.g., with self-scaling)
            args: PressureFormName_TemperatureFormName_selfscaling (e.g., "Hinge_Rational_4.078")
        - Continuum or C: Continuum absorption model (e.g., MT_CKD)
            args: Continuumtype_Modelname (e.g., "Sel_MT_CKD_4.3")

        Eg.,    H2O:F:Hinge_Rational_4.078
                H2O:C:self_MT_CKD_4.3
                H2O:ARTS:H2O_H2O-ForeignContCKDMT400_H2O-SelfContCKDMT400

        Species is necessary and case insensitive. Model default is F:Hinge_Rational_0
        ------
        parts = species.strip().split(":")

        if len(parts) == 1:
            species_name = parts[0].strip().upper()
            model_type = "Functional"
            model_args = ("Hinge", "Rational", "0")
        else:
            species_name, model_type, *model_args = parts
            model_args = model_args[0].split("_") if model_args else ()

        if model_type in ("ARTS", "A"):
            from faxsec.arts import ARTSAbsorber

            return ARTSAbsorber(
                species=species_name,
                frequency_grid=self.config.frequency_grid,
                arts_tag=tuple(model_args),
            )

        elif model_type in ("Functional", "F"):

            from faxsec.functional import FunctionalAbsorber

            pressure_form_name, temperature_form_name, self_scaling = model_args
            return FunctionalAbsorber(
                species=species_name,
                pressure_form_name=pressure_form_name,
                temperature_form_name=temperature_form_name,
                frequency_grid=self.config.frequency_grid,
                self_scaling=float(self_scaling),
            )
        elif model_type in ("Continuum", "C"):
            from faxsec.continuum import ContinuumAbsorber

            continuum_type = model_args
            return ContinuumAbsorber(
                species=species_name,
                frequency_grid=self.config.frequency_grid,
                continuum_type=continuum_type,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        """

    def build(self, **kwargs) -> None:
        """Build all species models and m"""
        for absorber in self._absorbers.values():
            if isinstance(absorber, FunctionalAbsorber):
                absorber.train(**kwargs)

    def cross_section_from_ds(
        self,
        atmosphere_ds: xr.Dataset,
        pressure_var: str = "pressure_layer",
        temperature_var: str = "temperature_layer",
    ) -> xr.DataArray:
        """Calculate cross-section for each species and frequency."""
        xsec_list = []
        active_absorbers = self.active_absorbers(atmosphere_ds)
        for absorber in active_absorbers.values():
            xsec = absorber.cross_section_from_atmds(
                atmosphere_ds,
                pressure_var=pressure_var,
                temperature_var=temperature_var,
            )
            xsec_list.append(xsec)

        xsec_total = xr.concat(xsec_list, dim="species")

        return xsec_total.assign_coords(species=list(active_absorbers.keys()))

    def absorption_from_ds(
        self,
        atmosphere_ds: xr.Dataset,
        pressure_var: str = "pressure_layer",
        temperature_var: str = "temperature_layer",
    ) -> xr.DataArray:
        """Calculate absorption coefficients for each species and frequency."""
        n_total = atmosphere_ds[pressure_var] / (
            BOLTZMANN * atmosphere_ds[temperature_var]
        )
        abs_list = []
        active_absorbers = self.active_absorbers(atmosphere_ds)
        for absorber in active_absorbers.values():
            xsec = absorber.cross_section_from_atmds(
                atmosphere_ds,
                pressure_var=pressure_var,
                temperature_var=temperature_var,
            )

            abs_coef = xsec * atmosphere_ds[absorber.config.species] * n_total
            abs_list.append(abs_coef)
        abs_coef_all = xr.concat(abs_list, dim="species").assign_coords(
            species=list(active_absorbers.keys())
        )

        return xr.DataArray(
            abs_coef_all,
            attrs={"units": "m-1", "fullname": "absorption_coefficient"},
            name="abs_coef",
        )

    def optical_depth_from_ds(
        self,
        atmosphere_ds: xr.Dataset,
        pressure_var: str = "pressure_layer",
        temperature_var: str = "temperature_layer",
        N_per_m2_var: str = "N_per_m2_dry",
    ) -> xr.DataArray:
        """Calculate optical depth for each species and frequency."""

        tau_list = []
        active_absorbers = self.active_absorbers(atmosphere_ds)
        for absorber in active_absorbers.values():
            xsec = absorber.cross_section_from_atmds(
                atmosphere_ds,
                pressure_var=pressure_var,
                temperature_var=temperature_var,
            )

            tau = (
                xsec
                * atmosphere_ds[N_per_m2_var]
                * atmosphere_ds[absorber.config.species]
            )
            tau_list.append(tau)

        tau_all = xr.concat(tau_list, dim="species").assign_coords(
            species=list(active_absorbers.keys())
        )

        return xr.DataArray(
            tau_all,
            attrs={"units": "m/m", "fullname": "optical_depth"},
            name="tau",
        )

    def transmission_from_ds(
        self,
        atmosphere_ds: xr.Dataset,
    ) -> xr.DataArray:
        """Calculate transmission for each species and frequency."""
        tau = self.optical_depth_from_ds(atmosphere_ds=atmosphere_ds)

        transmission = np.exp(-tau)
        transmission.name = "transmission"
        transmission.attrs = {
            "units": "m/m",
            "long_name": "Transmissivity",
            "description": "exp(-tau), where tau is optical depth",
        }

        return transmission

    @property
    def absorbers(self) -> dict[str, SingleSpeciesModel]:
        """Return the configured absorbers keyed by species name."""

        return self._absorbers

    @property
    def species(self) -> tuple[str, ...]:
        """Return the normalized absorber species names."""

        return tuple(
            set(absorber.config.species for absorber in self._absorbers.values())
        )

    def active_absorbers(
        self, atmosphere_ds: xr.Dataset
    ) -> dict[str, SingleSpeciesModel]:
        """Return the absorbers that are present in the atmosphere dataset."""
        return {
            name: absorber
            for name, absorber in self._absorbers.items()
            if absorber.config.species in atmosphere_ds.variables
        }


absorber_registry: Dict[str, Type[SavableModel]] = {
    "XFIT": CrossFitAbsorber,
    "Hinge_Rational": FunctionalAbsorber,
    "both_continuum_MT_CKD_4_3": H2OContinuum,
}
