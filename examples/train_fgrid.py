# %%
import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from faxsec.constants import (
    REF_PRESSURE,
    REF_TEMPERATURE,
    REF_VMR,
    REFERENCE_VMR,
    SELF_SCALING,
)
from faxsec.functional import FunctionalAbsorber
from faxsec.log_config import setup_logging
from faxsec.utils import (
    ensure_reference_dataset,
    kayser_to_hz,
    rayleigh_xsec_stamnes_2017,
    reference_cache_path,
    xsec_relevance_floor,
)

setup_logging()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Named training configurations. Each gives the reference point the fit is
# anchored at and how the (p, T) training sample is drawn; bands may differ.
TRAINING_CONFIGS = {
    "legacy": {
        "ref_pressure": REF_PRESSURE,
        "ref_temperature": REF_TEMPERATURE,
        "temperature_variable": "dT",
        "sampling": {"method": "natural", "p_range": [0.01, 110000], "N_samples": 1000},
    },
    "atmospheric": {
        "ref_pressure": 1.0e4,
        "ref_temperature": 240.0,
        "temperature_variable": "dT",
        "sampling": {
            "method": "atmospheric",
            "p_range": [1.0, 1.1e5],
            "N_samples": 2000,
            "pressure_weight": 0.5,
        },
        # Shortwave surface flux is a column-transmission problem, so equal air
        # mass per stratum is the natural design. The longwave emits from every
        # layer and takes part of its OLR from the stratosphere, so it needs a
        # compromise.
        "bands": {
            "LW": {"sampling": {"pressure_weight": 0.5}},
            "SW": {"sampling": {"pressure_weight": 1.0}},
        },
    },
}


def band_config(config: dict, band: str) -> dict:
    """Configuration for one band, with any band overrides merged in."""
    resolved = {k: v for k, v in config.items() if k != "bands"}
    resolved["sampling"] = dict(resolved["sampling"])
    override = config.get("bands", {}).get(band, {})
    for key, value in override.items():
        if key == "sampling":
            resolved["sampling"].update(value)
        else:
            resolved[key] = value
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suffix", default="", help="Suffix of the trained datatree to write"
    )
    parser.add_argument(
        "--reference-suffix",
        default=None,
        help="Reference cache to train from (default: named after --config, "
        "since the sampling configuration is what determines it)",
    )
    parser.add_argument("--bands", default="LW,SW")
    parser.add_argument("--config", default="atmospheric", choices=TRAINING_CONFIGS)
    return parser.parse_args()


args = parse_args()
suffix = args.suffix
base_config = TRAINING_CONFIGS[args.config]
reference_suffix = (
    args.reference_suffix if args.reference_suffix is not None else f"_{args.config}"
)


def train_fax(
    species: str,
    arts_tag: tuple[str] | None,
    frequency_grid: np.ndarray,
    reference_cache_dir: str | Path,
    sampling_kwargs: dict | None = None,
    ref_pressure: float = REF_PRESSURE,
    ref_temperature: float = REF_TEMPERATURE,
    temperature_variable: str = "dT",
) -> FunctionalAbsorber:
    """Train a FAX model for a given species and frequency grid.

    Parameters
    ----------
    species : str
        The species to train the FAX model for.
    frequency_grid : np.ndarray
        The frequency grid in Hz.
    reference_cache_dir : Path | None, optional
        The directory to cache reference datasets, by default None.
    """

    ref_vmr = REFERENCE_VMR.get(species, REF_VMR)

    ensure_reference_dataset(
        species=species,
        frequency_grid=frequency_grid,
        arts_tag=arts_tag,
        cache_path=reference_cache_path(
            species=species,
            arts_tag=arts_tag,
            cache_dir=reference_cache_dir,
        ),
        ref_pressure=ref_pressure,
        ref_temperature=ref_temperature,
        ref_vmr=ref_vmr,
        sampling_kwargs=sampling_kwargs,
    )

    func_abs = FunctionalAbsorber(
        species=species,
        pressure_form_name="Hinge",
        temperature_form_name="Rational",
        frequency_grid=frequency_grid,
        self_scaling=SELF_SCALING.get(species, 0.0),
        xsec_floor=xsec_relevance_floor(species),
        temperature_variable=temperature_variable,
        ref_pressure=ref_pressure,
        ref_temperature=ref_temperature,
        ref_vmr=ref_vmr,
    )

    func_abs.train(
        reference_xsec=reference_cache_path(
            species=species,
            arts_tag=arts_tag,
            cache_dir=reference_cache_dir,
        ),
        max_iter=10,
        sampling_kwargs=sampling_kwargs,
    )

    return func_abs


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
    },
    "LW": {
        "CFC11": ("CFC11-XFIT",),
        "CFC12": ("CFC12-XFIT",),
    },
}

continuum = {
    "H2O": (
        "H2O-ForeignContCKDMT400",
        "H2O-SelfContCKDMT400",
    ),
}

# Define frequencies (wavenumbers) longwave
# wavenumber range taken from DDQ paper
wvn_min_lw = 10.0  # cm^-1
wvn_max_lw = 1 / 2e-6 / 100  # cm^-1
N_wvn_lw = 100_000
kayser_lw = np.linspace(wvn_min_lw, wvn_max_lw, N_wvn_lw)
f_grid_lw = kayser_to_hz(kayser_lw)

# Define frequencies (wavenumbers) shortwave
# wavenumber range taken from DDQ paper
# but using log spacing
wvn_min_sw = 1 / 1e-5 / 100
wvn_max_sw = 1e5
N_wvn_sw = 100_001
kayser_sw = np.logspace(np.log10(wvn_min_sw), np.log10(wvn_max_sw), N_wvn_sw)
f_grid_sw = kayser_to_hz(kayser_sw)

train_cases = {
    f"Highres_LW_{N_wvn_lw}": kayser_lw,
    f"Highres_SW_{N_wvn_sw}": kayser_sw,
}
for case_name, kayser_grid in train_cases.items():
    frequency_grid = kayser_to_hz(kayser_grid)

    absorbers = {}
    band = case_name.split("_")[1][:2]  # Extract the band (LW or SW) from the case name

    logger.info(
        "Training gas optics for %s (band=%s): %d frequency points",
        case_name,
        band,
        frequency_grid.size,
    )

    config = band_config(base_config, band)
    sampling_kwargs = config["sampling"]
    reference_cache_dir = DATA_DIR / "reference" / f"{case_name}{reference_suffix}"

    # lines
    for sp in lines[band].keys():
        func_abs = train_fax(
            species=sp,
            arts_tag=lines[band][sp],
            frequency_grid=frequency_grid,
            reference_cache_dir=reference_cache_dir,
            sampling_kwargs=sampling_kwargs,
            ref_pressure=config["ref_pressure"],
            ref_temperature=config["ref_temperature"],
            temperature_variable=config["temperature_variable"],
        )
        absorbers[sp] = func_abs

    # halocarbons
    from faxsec.xfit import CrossFitAbsorber

    for sp in halocarbons[band].keys():
        func_abs = CrossFitAbsorber(
            species=sp,
            frequency_grid=frequency_grid,
            data_source=DATA_DIR / "halocarbon" / f"{sp.split('-')[0]}-XFIT.xml",
        )
        absorbers[sp] = func_abs

    # continuum
    from faxsec.continuum import H2OContinuum

    for sp in continuum.keys():
        absorbers[f"{sp}_continuum"] = H2OContinuum(
            frequency_grid=frequency_grid,
            data_source=DATA_DIR / "continuum" / "absco-ref_wv-mt-ckd400.nc",
        )

    # quadrature related data
    other = xr.Dataset(
        {
            "kayser_grid": ("frequency", kayser_grid),
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
        other["spectral_solar_irradiance"] = (
            total_solar_irradiance
            * solar_source["spectral_solar_radiance"]
            / np.trapezoid(
                solar_source["spectral_solar_radiance"].values,
                x=solar_source.frequency.values,
            )
        )
        other["spectral_solar_irradiance"].attrs["source"] = str(solar_source_file)

        # rayleigh scattering cross-section
        rayleigh_xsec = rayleigh_xsec_stamnes_2017(frequency_grid)
        other["xsec_rayleigh"] = ("frequency", rayleigh_xsec)
        other["xsec_rayleigh"].attrs["source"] = "Stamnes et al. 2017"

    other.attrs["model_class"] = "Other"

    # all data together
    datatree = xr.DataTree()
    datasets = [absorber.to_dataset() for absorber in absorbers.values()]
    groups = {}
    for ds in datasets:
        key = ds.attrs["model_class"]
        if key in groups:
            groups[key] = xr.concat([groups[key], ds], dim="species")
        else:
            groups[key] = ds

    for key, ds in groups.items():
        datatree[key] = ds

    datatree["Other"] = other
    datatree.attrs.update(
        reference_cache=str(reference_cache_dir),
        suffix=suffix,
        training_config=args.config,
        config_detail=str(config),
    )
    output_path = DATA_DIR / "ff" / f"gas_optics_{case_name}{suffix}.nc"
    datatree.to_netcdf(output_path, mode="w")
    logger.info("Saved %s: %d absorbers -> %s", case_name, len(absorbers), output_path)

# %%
