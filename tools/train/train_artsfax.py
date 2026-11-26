"""Train FAX line models only, on a given frequency grid and (p, T) range.

Each species' ARTS reference is read from --reference-dir, or computed there.

    python tools/train/train_artsfax.py --species H2O --species CO2:CO2,CO2-CKDMT252 \\
        --kayser 10 3250 2000 --p-range 1 1.1e5 --t-range 150 330 --out fax.nc
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from artsfax.constants import DATA_DIR
from artsfax.gas_optics import GasOptics, write_datatree
from artsfax.log_config import setup_logging
from artsfax.train import TRAINING_CONFIGS, band_config, train_fax
from artsfax.utils import ATMOSPHERIC_T_ENVELOPE, kayser_to_hz, relative_to_data

logger = logging.getLogger(__name__)


def species_spec(spec: str) -> tuple[str, tuple[str, ...] | None]:
    """Species and ARTS tags from NAME or NAME:TAG,TAG."""
    name, _, tags = spec.partition(":")
    return name, tuple(tags.split(",")) if tags else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--species",
        action="append",
        required=True,
        type=species_spec,
        help="NAME or NAME:TAG,TAG,... (ARTS tags; default the species itself). Repeatable.",
    )
    grid = parser.add_mutually_exclusive_group(required=True)
    grid.add_argument(
        "--frequency", type=Path, help="Frequency grid in Hz: text, one per line, or .npy"
    )
    grid.add_argument(
        "--kayser",
        type=float,
        nargs=3,
        metavar=("MIN", "MAX", "N"),
        help="Wavenumber grid in cm^-1",
    )
    parser.add_argument("--log", action="store_true", help="Log spacing for --kayser")
    parser.add_argument("--p-range", type=float, nargs=2, metavar=("PMIN", "PMAX"), help="Pa")
    parser.add_argument("--t-range", type=float, nargs=2, metavar=("TMIN", "TMAX"), help="K")
    parser.add_argument("--samples", type=int, help="Number of (p, T) training cases")
    parser.add_argument("--pressure-weight", type=float, help="See sample_atmospheres_atmospheric")
    parser.add_argument("--config", default="atmospheric_nolog", choices=TRAINING_CONFIGS)
    parser.add_argument("--out", type=Path, required=True, help="Trained datatree to write")
    parser.add_argument(
        "--reference-dir", type=Path, help="Default: data/reference/<name of --out>"
    )
    parser.add_argument("--workers", type=int, help="Fitting processes (default: all cores)")
    parser.add_argument("--frequency-chunk", type=int, default=2000)
    parser.add_argument(
        "--reference-memory", type=float, default=2.0, help="GB of ARTS memory per chunk"
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.frequency is not None:
        load = np.load if args.frequency.suffix == ".npy" else np.loadtxt
        grid = load(args.frequency)
    else:
        lo, hi, n = args.kayser
        spacing = np.geomspace if args.log else np.linspace
        grid = kayser_to_hz(spacing(lo, hi, int(n)))

    config = band_config(TRAINING_CONFIGS[args.config], band=None)
    sampling = config["sampling"]
    if args.p_range is not None:
        sampling["p_range"] = list(args.p_range)
    if args.samples is not None:
        sampling["N_samples"] = args.samples
    if args.pressure_weight is not None:
        sampling["pressure_weight"] = args.pressure_weight
    if args.t_range is not None:
        lo, hi = args.t_range
        if sampling["method"] == "atmospheric":
            # bound the temperature envelope to the range, keeping its shape in p
            sampling["envelope"] = tuple(
                (p, float(np.clip(t_min, lo, hi)), float(np.clip(t_max, lo, hi)))
                for p, t_min, t_max in ATMOSPHERIC_T_ENVELOPE
            )
        else:
            sampling["T_range"] = [lo, hi]

    reference_dir = args.reference_dir or DATA_DIR / "reference" / args.out.stem
    logger.info(
        "Training %d species on %d frequencies, sampling %s",
        len(args.species),
        grid.size,
        sampling,
    )

    absorbers = {}
    for species, tags in args.species:
        absorber = train_fax(
            species=species,
            arts_tag=tags,
            frequency_grid=grid,
            reference_dir=reference_dir,
            config=config,
            memory_budget=int(args.reference_memory * 1e9),
            frequency_chunk=args.frequency_chunk,
            n_workers=args.workers,
        )
        absorbers[f"{species}_{absorber.class_name}"] = absorber

    datatree = GasOptics.from_absorbers(absorbers).to_datatree()
    datatree.attrs.update(
        reference_cache=relative_to_data(reference_dir),
        training_config=args.config,
        config_detail=str(config),
        arts_tags=str({species: tags for species, tags in args.species}),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_datatree(datatree, args.out)
    logger.info("Saved %d absorbers -> %s", len(absorbers), args.out)


if __name__ == "__main__":
    main()
