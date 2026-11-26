"""Train FAX line models against ARTS line-by-line references."""

from pathlib import Path

import numpy as np

from artsfax.constants import REF_VMR, REFERENCE_VMR, SELF_SCALING
from artsfax.functional import FunctionalAbsorber, NoLogFunctionalAbsorber
from artsfax.utils import (
    DEFAULT_REFERENCE_MEMORY_BUDGET,
    ensure_reference_dataset,
    reference_cache_path,
    xsec_relevance_floor,
)

# Named training configurations; a band may override any entry.
TRAINING_CONFIGS = {
    "atmospheric_nolog": {
        "ref_pressure": 1.0e4,
        "ref_temperature": 240.0,
        "temperature_variable": "dT",
        "formulation": "nolog",
        "pressure_form": "ShiftedReciprocalLaurent",
        "temperature_form": "Rational",
        "max_iter": 4,
        "sampling": {
            "method": "atmospheric",
            "p_range": [0.01, 1.1e5],
            "N_samples": 2000,
            "pressure_weight": 0.5,
        },
        # SW: equal air mass per stratum; LW: weighted towards the stratosphere
        "bands": {
            "LW": {"sampling": {"pressure_weight": 0.5}},
            "SW": {"sampling": {"pressure_weight": 1.0}},
        },
    },
}

FORMULATIONS = {"log": FunctionalAbsorber, "nolog": NoLogFunctionalAbsorber}


def band_config(config: dict, band: str) -> dict:
    """Configuration for one band, with any band overrides merged in."""
    resolved = {k: v for k, v in config.items() if k != "bands"}
    resolved["sampling"] = dict(resolved["sampling"])
    for key, value in config.get("bands", {}).get(band, {}).items():
        if key == "sampling":
            resolved["sampling"].update(value)
        else:
            resolved[key] = value
    return resolved


def train_fax(
    species: str,
    arts_tag: tuple[str, ...] | None,
    frequency_grid: np.ndarray,
    reference_dir: str | Path,
    config: dict,
    memory_budget: int = DEFAULT_REFERENCE_MEMORY_BUDGET,
    save_path: str | Path | None = None,
    frequency_chunk: int = 2000,
    n_workers: int | None = None,
) -> FunctionalAbsorber:
    """Train a FAX model for one species on a frequency grid in Hz.

    The reference in ``reference_dir`` is used if present and computed there
    otherwise; frequencies already in ``save_path`` are not refitted.
    """
    ref_vmr = REFERENCE_VMR.get(species, REF_VMR)
    reference = reference_cache_path(
        species=species, arts_tag=arts_tag, cache_dir=reference_dir
    )
    ensure_reference_dataset(
        species=species,
        frequency_grid=frequency_grid,
        arts_tag=arts_tag,
        cache_path=reference,
        ref_pressure=config["ref_pressure"],
        ref_temperature=config["ref_temperature"],
        ref_vmr=ref_vmr,
        sampling_kwargs=config["sampling"],
        arts_reference_kwargs={"memory_budget": memory_budget},
    )

    absorber = FORMULATIONS[config["formulation"]](
        species=species,
        pressure_form_name=config["pressure_form"],
        temperature_form_name=config["temperature_form"],
        frequency_grid=frequency_grid,
        self_scaling=SELF_SCALING.get(species, 0.0),
        xsec_floor=xsec_relevance_floor(species),
        temperature_variable=config["temperature_variable"],
        ref_pressure=config["ref_pressure"],
        ref_temperature=config["ref_temperature"],
        ref_vmr=ref_vmr,
    )

    if save_path is None and np.size(frequency_grid) <= frequency_chunk:
        absorber.train(reference_xsec=reference, max_iter=config["max_iter"])
    else:
        absorber.train_in_frequency_chunks(
            reference_xsec=reference,
            frequency_chunk=frequency_chunk,
            n_workers=n_workers,
            save_path=save_path,
            max_iter=config["max_iter"],
        )
    return absorber
