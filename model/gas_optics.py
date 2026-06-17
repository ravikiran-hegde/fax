from typing import Dict

import numpy as np
import xarray as xr

from .single_absorber import FunctionalAbsorber, SingleSpeciesModel


class GasOptics:
    def __init__(
        self,
        species: Dict[str, SingleSpeciesModel],
    ) -> None:
        self.species = species

        def build(self) -> None:
            """Build all species models."""
            for absorber in self.species.values():
                absorber.build()

        def load():
            """Load all species models from disk."""
            for absorber in self.species.values():
                absorber.train()

        def optical_depth(
            p: np.ndarray,
            t: np.ndarray,
            vmr: np.ndarray,
        ) -> xr.DataArray:
            """Calculate optical depth for each species and frequency."""
            tau = np.zeros((len(p), len(t), len(vmr)))
            for i, absorber in enumerate(self.species.values()):
                xsec = absorber.cross_section(p, t, vmr)
                tau[..., i] = xsec * vmr[..., None] * p[..., None]
            return xr.DataArray(
                tau,
                dims=["level", "species"],
                coords={"species": list(self.species.keys())},
            )

        def transmission(
            p: np.ndarray,
            t: np.ndarray,
            vmr: np.ndarray,
        ) -> xr.DataArray:
            """Calculate transmission for each species and frequency."""
            tau = optical_depth(p, t, vmr)
            return xr.DataArray(
                np.exp(-tau),
                dims=["level", "species"],
                coords={"species": list(self.species.keys())},
            )
