"""ARTS-backed absorption adapter with the fastabs API.

This class allows using pyarts within the same `calculate_absorption`
interface as the fast models, enabling drop-in comparisons and RT runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pyarts3 as pyarts

from faxsec.abstract_class import ARRAYLIKE, AbsorberConfig, SingleSpeciesModel
from faxsec.constants import DEFAULT_VMR, EPS, REF_VMR

logger = logging.getLogger(__name__)

LINE_CUTOFF_HZ = 750e9
PYARTS_VERSION = "3.0.0dev10"


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

    Builds one Workspace per absorption tag and evaluates the whole stack of
    atmospheric points in a single ``spectral_propmat_pathFromPath`` call per
    tag, instead of looping point-by-point in Python.
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

        pyarts.data.download(version=PYARTS_VERSION)

        self._freq_grid = pyarts.arts.AscendingGrid(
            np.asarray(self.config.frequency_grid, dtype=float)
        )
        self._absorbers = self._create_absorbers(
            self.config.arts_tag, self.config.cutoff_hz
        )
        logger.debug(
            "ARTSAbsorber(%s) ready: tags=%s, %d frequency points",
            species,
            self.config.arts_tag,
            len(self.config.frequency_grid),
        )

    def _create_absorbers(
        self,
        absorption_tags: Sequence[str],
        cutoff_hz: Optional[float],
    ):
        """Factory for underlying ARTS Workspaces, one per tag."""
        absorbers = {}
        for tag in absorption_tags:
            ws = pyarts.Workspace()
            ws.WignerInit()
            ws.abs_speciesSet(species=[tag])
            ws.ReadCatalogData()
            if cutoff_hz is not None:
                for band in ws.abs_bands:
                    ws.abs_bands[band].cutoff = "ByLine"
                    ws.abs_bands[band].cutoff_value = cutoff_hz
            # ignore_errors turns e.g. a CIA temperature out of table range
            # into NaN for that point instead of raising; nan_to_num below
            # then floors it, same as skipping that point.
            ws.spectral_propmat_agendaAuto(ignore_errors=1)
            ws.jac_targetsOff()
            absorbers[tag] = ws
        return absorbers

    def _propmat_path(self, ws, tag: str, atm_path, F: int) -> np.ndarray:
        """Propagation matrix for a stack of atm points in one ARTS call."""
        N = len(atm_path)
        ray_path = pyarts.arts.ArrayOfPropagationPathPoint(
            [pyarts.arts.PropagationPathPoint()] * N
        )
        freq_grid_path = pyarts.arts.ArrayOfAscendingGrid([self._freq_grid] * N)
        ws.freq_wind_shift_jac_path = pyarts.arts.ArrayOfVector3(np.zeros((N, 3)))

        try:
            ws.spectral_propmat_pathFromPath(
                atm_path=atm_path,
                ray_path=ray_path,
                freq_grid_path=freq_grid_path,
            )
            return np.array([1.0 * v[:, 0] for v in ws.spectral_propmat_path])
        except Exception as e:
            logger.warning("Error calculating cross-section for tag %s: %s", tag, e)
            return np.zeros((N, F))

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
        pressure = np.asarray(pressure, dtype=float)
        temperature = np.asarray(temperature, dtype=float)
        vmr = np.asarray(vmr, dtype=float)
        N = pressure.size
        F = len(self.config.frequency_grid)

        logger.debug(
            "Calculating ARTS cross-sections for %s with tags %s (N=%d)",
            self.config.species,
            self.config.arts_tag,
            N,
        )

        vmr_arts = (
            vmr if self.config.use_self_broadening else np.full(N, self.config.vmr0)
        )

        atm_data = {
            "p": pyarts.arts.Vector(pressure),
            "t": pyarts.arts.Vector(temperature),
        }
        # set default VMRs for other species in the atmosphere; needed for some CIA in SW
        for sp, default_vmr in DEFAULT_VMR.items():
            if sp != self.config.species:
                atm_data[sp] = pyarts.arts.Vector(np.full(N, default_vmr))
        atm_data[self.config.species] = pyarts.arts.Vector(vmr_arts)
        atm_path = pyarts.arts.ArrayOfAtmPoint.from_dict(atm_data)

        number_density = np.array(
            [p.number_density(self.config.species) for p in atm_path]
        )
        safe_density = np.where(number_density > 0, number_density, 1.0)

        xsec_stack = np.zeros((N, F))
        for tag, ws in self._absorbers.items():
            propmat = self._propmat_path(ws, tag, atm_path, F)
            xsec = propmat / safe_density[:, None]
            xsec = np.nan_to_num(xsec, nan=EPS, posinf=EPS, neginf=EPS)
            xsec = np.clip(xsec, EPS, 1e10)
            xsec_stack += xsec

        xsec_stack[vmr_arts == 0.0, :] = 0.0

        return xsec_stack

    @property
    def class_name(self) -> str:
        return "ARTS_SingleSpeciesRecipe"
