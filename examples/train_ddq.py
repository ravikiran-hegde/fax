# %%
import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from artsfax.constants import DATA_DIR
from artsfax.gas_optics import GasOptics, write_datatree
from artsfax.log_config import setup_logging
from artsfax.train import TRAINING_CONFIGS, band_config, train_fax
from artsfax.utils import kayser_to_hz, rayleigh_xsec_stamnes_2017, relative_to_data

setup_logging()
logger = logging.getLogger(__name__)

DDQ_DIR = DATA_DIR / "ddq"
ALTERNATIVE_DDQ_DIR = DDQ_DIR / "Additional configurations"


def ddq_path(quadrature: str, band: str) -> Path:
    """The quadrature file for one band: the default set, or a numbered alternative."""
    if quadrature == "default":
        return DDQ_DIR / f"DDQ_{band}.h5"
    return ALTERNATIVE_DDQ_DIR / f"DDQ_{band}_{quadrature}.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suffix", default="", help="Suffix of the trained datatree to write"
    )
    parser.add_argument(
        "--reference-suffix",
        default="",
        help="Suffix of the reference cache directory (default: none)",
    )
    parser.add_argument("--bands", default="LW,SW")
    parser.add_argument(
        "--ddq",
        default="default",
        help="Quadrature set: 'default', or the index of one in "
        "data/ddq/Additional configurations",
    )
    parser.add_argument(
        "--config", default="atmospheric_nolog", choices=TRAINING_CONFIGS
    )
    return parser.parse_args()


args = parse_args()
suffix = args.suffix
base_config = TRAINING_CONFIGS[args.config]
reference_suffix = args.reference_suffix


# %% Train DDQ LW FAX models

lines = {
    "LW": {
        "H2O": None,
        "CO2": ("CO2", "CO2-CKDMT252"),
        "O3": None,
        "O2": (
            "O2",
            "O2-CIA-O2",
        ),
        "CH4": None,
        "N2O": None,
        "N2": (
            "N2",
            "N2-CIA-N2",
        ),
    },
    "SW": {
        "H2O": None,
        "CO2": ("CO2", "CO2-CKDMT252"),
        "O3": None,
        "O2": (
            "O2",
            "O2-CIAfunCKDMT100",
        ),
        "CH4": None,
        "N2O": None,
        "N2": (
            "N2",
            "N2-CIAfunCKDMT252",
            "N2-CIArotCKDMT252",
        ),
    },
}

halocarbons = {
    "SW": {
        "O3-XFIT": ("O3-XFIT",),
        "CFC11": ("CFC11-XFIT",),
        "CFC12": ("CFC12-XFIT",),
        # "HFC125": ("HFC125-XFIT",),
        # "HFC32": ("HFC32-XFIT",),
        # "CCL4": ("CCL4-XFIT",),
        # "HFC143A": ("HFC143A-XFIT",),
        # "HFC23": ("HFC23-XFIT",),
        # "CF4": ("CF4-XFIT",),
        # "CFC22": ("CFC22-XFIT",),
        # "HFC134A": ("HFC134A-XFIT",),
    },
    "LW": {
        "CFC11": ("CFC11-XFIT",),
        "CFC12": ("CFC12-XFIT",),
        # "HFC125": ("HFC125-XFIT",),
        # "HFC32": ("HFC32-XFIT",),
        # "CCL4": ("CCL4-XFIT",),
        # "HFC143A": ("HFC143A-XFIT",),
        # "HFC23": ("HFC23-XFIT",),
        # "CF4": ("CF4-XFIT",),
        # "CFC22": ("CFC22-XFIT",),
        # "HFC134A": ("HFC134A-XFIT",),
    },
}

continuum = {
    "H2O": (
        "H2O-ForeignContCKDMT430",
        "H2O-SelfContCKDMT430",
    ),
}

for band in args.bands.split(","):
    ddq_case = ddq_path(args.ddq, band)
    case_name = ddq_case.stem  # e.g. "DDQ_LW"

    kayser_quadrature = xr.load_dataset(ddq_case)
    kayser_grid = kayser_quadrature["S"].values
    kayser_weights = kayser_quadrature["W"].values
    frequency_grid = kayser_to_hz(kayser_grid)
    weights_hz = kayser_to_hz(kayser_weights)

    logger.info(
        "Training gas optics for %s (band=%s): %d frequency points",
        case_name,
        band,
        frequency_grid.size,
    )

    absorbers = {}
    config = band_config(base_config, band)
    reference_cache_dir = DATA_DIR / "reference" / f"{case_name}{reference_suffix}"

    # lines
    for sp in lines[band].keys():
        func_abs = train_fax(
            species=sp,
            arts_tag=lines[band][sp],
            frequency_grid=frequency_grid,
            reference_dir=reference_cache_dir,
            config=config,
        )
        absorbers[sp] = func_abs

    # xfit
    from artsfax.xfit import CrossFitAbsorber

    for sp in halocarbons[band].keys():
        func_abs = CrossFitAbsorber(
            species=sp,
            frequency_grid=frequency_grid,
            data_source=DATA_DIR / "halocarbon" / f"{sp.split('-')[0]}-XFIT.xml",
        )
        absorbers[sp] = func_abs

    # continuum
    from artsfax.continuum import H2OContinuum

    for sp in continuum.keys():
        absorbers[f"{sp}_continuum"] = H2OContinuum(frequency_grid=frequency_grid)

    # quadrature related data
    ddq = xr.Dataset(
        {
            "weights_hz": ("frequency", weights_hz),
        }
    )
    if band == "SW":
        import pyarts3

        # solar source
        solar_source_file = DATA_DIR / "solar_spectra" / "solar_spectrum_July_2008.xml"
        solar_source = (
            pyarts3.xml.load(
                str(solar_source_file),
            )
            .to_xarray()
            .rename({"Frequencys": "frequency"})
        )
        solar_source = xr.Dataset(
            {"spectral_solar_radiance": (("frequency",), solar_source.values[:, 0])},
            coords={"frequency": solar_source.frequency},
        ).interp(frequency=frequency_grid, method="cubic")

        total_solar_irradiance = 1361.0  # W/m^2
        ddq["spectral_solar_irradiance"] = (
            total_solar_irradiance
            * solar_source["spectral_solar_radiance"]
            / np.dot(solar_source["spectral_solar_radiance"].values, weights_hz)
        )
        ddq["spectral_solar_irradiance"].attrs["source"] = relative_to_data(solar_source_file)

        # rayleigh scattering cross-section
        ddq_rayleigh = rayleigh_xsec_stamnes_2017(frequency_grid)
        ddq["xsec_rayleigh"] = ("frequency", ddq_rayleigh)
        ddq["xsec_rayleigh"].attrs["source"] = "Stamnes et al. 2017"

    ddq.attrs["model_class"] = "DDQ"

    # all data together
    datatree = GasOptics.from_absorbers(absorbers).to_datatree()

    datatree["DDQ"] = ddq
    datatree.attrs.update(
        reference_cache=str(reference_cache_dir.relative_to(DATA_DIR)),
        suffix=suffix,
        training_config=args.config,
        config_detail=str(config),
    )

    output_path = DATA_DIR / "ff" / f"gas_optics_DDQ_{band}{suffix}.nc"
    write_datatree(datatree, output_path)
    logger.info("Saved %s: %d absorbers -> %s", case_name, len(absorbers), output_path)

# %%
