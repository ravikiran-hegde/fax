#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from constants import EPS, VMR_REF
from single_absorber import AbsorberConfig, SingleSpeciesModel

LINE_CUTOFF_HZ = 750e9
PYARTS_VERSION = "3.0.0dev8"


def species_from_tag(tag: str) -> str:
    """Return base species name from an absorption tag.

    Tags may include isotopologues or models (e.g. "H2O-161", "H2O,ForeignCont").
    """

    return tag.split(",")[0].split("-")[0]


@dataclass(frozen=True)
class ARTSConfig(AbsorberConfig):
    arts_tag: tuple[str, ...] = ("",)
    cutoff_hz: float | None = LINE_CUTOFF_HZ
    use_self_broadening: bool = True
    vmr0: float = VMR_REF


class ARTSAbsorber(SingleSpeciesModel):
    """Adapter to provide calculate_absorption like fast models using ARTS.

    It uses pyarts SingleSpeciesAbsorption internally for each species tag.
    """

    config: ARTSConfig

    def __init__(
        self,
        species: str,
        frequency_grid: Optional[tuple[float, ...]] = None,
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
        self, absorption_tags: Sequence[str], cutoff_hz: Optional[float]
    ):
        """Factory for underlying ARTS absorbers.

        Subclasses (e.g. NoSelfARTSAbsorptionModel) can override to customize
        the SingleSpeciesAbsorption implementation while keeping all higher-
        level logic identical.
        """
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

        N = pressure.size
        F = self.config.frequency_grid.size
        S = len(self.config.arts_tag)
        xsec_stack = np.zeros((N, F))

        atm = pyarts.arts.AtmPoint()
        for i in range(N):
            atm.pressure = float(pressure[i])
            atm.temperature = float(temperature[i])

            vmr_arts = (
                float(vmr[i]) if self.config.use_self_broadening else self.config.vmr0
            )
            if vmr_arts == 0.0:
                continue
            atm[self.config.species] = vmr_arts

            for tag in self.config.arts_tag:

                xsec = self._absorbers[tag](
                    self.config.frequency_grid, atm
                ) / atm.number_density(self.config.species)
                xsec = np.nan_to_num(xsec, nan=EPS, posinf=EPS, neginf=EPS)
                xsec = np.clip(xsec, EPS, 1e10)
                xsec_stack[i, :] += xsec

        return xsec_stack

    def train(self) -> None:
        """No training needed for ARTS absorbers."""
        pass

    def save(self, path: str) -> None:
        """No model to save for ARTS absorbers."""
        pass

    def load(self, path: str) -> None:
        """No model to load for ARTS absorbers."""
        pass
