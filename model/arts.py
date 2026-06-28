"""ARTS-backed absorption adapter with the fastabs API.

This class allows using pyarts SingleSpeciesAbsorption within the same
`calculate_absorption` interface as the fast models, enabling drop-in
comparisons and RT runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pyarts3 as pyarts

from model.abstract_class import ARRAYLIKE, AbsorberConfig, SingleSpeciesModel

from .constants import EPS, REF_VMR

LINE_CUTOFF_HZ = 750e9
PYARTS_VERSION = "3.0.0dev8"


def species_from_tag(tag: str) -> str:
    """Return base species name from an absorption tag.

    Tags may include isotopologues or models (e.g. "H2O-161", "H2O,ForeignCont").
    """

    return tag.split(",")[0].split("-")[0]


@dataclass
class ARTSConfig(AbsorberConfig):
    arts_tag: tuple[str, ...] = ("",)
    cutoff_hz: float | None = LINE_CUTOFF_HZ
    use_self_broadening: bool = True
    vmr0: float = REF_VMR


class ARTSAbsorber(SingleSpeciesModel):
    """Adapter to provide calculate_absorption like fast models using ARTS.

    It uses pyarts SingleSpeciesAbsorption internally for each species tag.
    """

    config: ARTSConfig

    def __init__(
        self,
        species: str,
        frequency_grid: ARRAYLIKE,
        arts_tag: Optional[tuple[str, ...]] = None,
    ):
        arts_tag = arts_tag if arts_tag is not None else (species,)
        self.config = ARTSConfig(
            species=species,
            frequency_grid=frequency_grid,
            arts_tag=arts_tag,
        )

        import pyarts3

        pyarts3.data.download(version=PYARTS_VERSION)

        self._absorbers = self._create_absorbers(
            self.config.arts_tag, self.config.cutoff_hz
        )

    def _create_absorbers(
        self,
        absorption_tags: Sequence[str],
        cutoff_hz: Optional[float],
    ):
        """Factory for underlying ARTS absorbers."""
        from pyarts3.recipe import SingleSpeciesAbsorption

        return {
            t: SingleSpeciesAbsorption(species=t, cutoff=cutoff_hz)
            for t in absorption_tags
        }

    def cross_section(
        self,
        pressure: np.ndarray,
        temperature: np.ndarray,
        vmr: np.ndarray,
    ) -> np.ndarray:
        """Compute cross-sections using ARTS.

        Returns
        -------
        np.ndarray
            Cross-section array (level, frequency) in m^2
        """
        print("Calculating ARTS cross-sections with tags", self.config.arts_tag)
        N = pressure.size
        F = len(self.config.frequency_grid)
        S = len(self.config.arts_tag)
        xsec_stack = np.zeros((N, F))
        default_vmrs = {
            "N2": 0.7808,
            "O2": 0.2095,
            "CO2": 4.2e-4,
            "H2O": 0.001,
            "CH4": 1.9e-6,
        }
        atm = pyarts.arts.AtmPoint()
        for i in range(N):
            atm.pressure = float(pressure[i])
            atm.temperature = float(temperature[i])

            vmr_arts = (
                float(vmr[i]) if self.config.use_self_broadening else self.config.vmr0
            )
            if vmr_arts == 0.0:
                continue

            # set default VMRs for other species in the atmosphere; needed foir some CIA in SW
            for sp in default_vmrs:
                if sp != self.config.species:
                    atm[sp] = default_vmrs[sp]

            atm[self.config.species] = vmr_arts

            for tag in self.config.arts_tag:

                try:
                    xsec = self._absorbers[tag](
                        self.config.frequency_grid, atm
                    ) / atm.number_density(self.config.species)
                except Exception as e:
                    print(f"Error calculating cross-section for tag {tag}: {e}")
                    xsec = np.zeros(F)

                xsec = np.nan_to_num(xsec, nan=EPS, posinf=EPS, neginf=EPS)
                xsec = np.clip(xsec, EPS, 1e10)
                xsec_stack[i, :] += xsec

        return xsec_stack

    @property
    def class_name(self) -> str:
        return "ARTS_SingleSpeciesRecipe"
