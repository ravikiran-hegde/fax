# %%
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from faxsec.constants import SELF_SCALING
from faxsec.functional import FunctionalAbsorber
from faxsec.log_config import setup_logging
from faxsec.utils import (
    ensure_reference_dataset,
    kayser_to_hz,
    rayleigh_xsec_stamnes_2017,
    reference_cache_path,
)

setup_logging()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def train_fax(
    species: str,
    arts_tag: tuple[str] | None,
    frequency_grid: np.ndarray,
    reference_cache_dir: str | Path,
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

    ensure_reference_dataset(
        species=species,
        frequency_grid=frequency_grid,
        arts_tag=arts_tag,
        cache_path=reference_cache_path(
            species=species,
            arts_tag=arts_tag,
            cache_dir=reference_cache_dir,
        ),
    )

    func_abs = FunctionalAbsorber(
        species=species,
        pressure_form_name="Hinge",
        temperature_form_name="Rational",
        frequency_grid=frequency_grid,
        self_scaling=SELF_SCALING.get(species, 0.0),
    )

    func_abs.train(
        reference_xsec=reference_cache_path(
            species=species,
            arts_tag=arts_tag,
            cache_dir=reference_cache_dir,
        )
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
        "HFC125": ("HFC125-XFIT",),
        "HFC32": ("HFC32-XFIT",),
        "CCL4": ("CCL4-XFIT",),
        "HFC143A": ("HFC143A-XFIT",),
        "HFC23": ("HFC23-XFIT",),
        "CF4": ("CF4-XFIT",),
        # "CFC22": ("CFC22-XFIT",),
        "HFC134A": ("HFC134A-XFIT",),
    },
    "LW": {
        "CFC11": ("CFC11-XFIT",),
        "CFC12": ("CFC12-XFIT",),
        "HFC125": ("HFC125-XFIT",),
        "HFC32": ("HFC32-XFIT",),
        "CCL4": ("CCL4-XFIT",),
        "HFC143A": ("HFC143A-XFIT",),
        "HFC23": ("HFC23-XFIT",),
        "CF4": ("CF4-XFIT",),
        # "CFC22": ("CFC22-XFIT",),
        "HFC134A": ("HFC134A-XFIT",),
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
    # f"Highres_LW_{N_wvn_lw}": kayser_lw,
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

    # lines
    for sp in lines[band].keys():
        func_abs = train_fax(
            species=sp,
            arts_tag=lines[band][sp],
            frequency_grid=frequency_grid,
            reference_cache_dir=DATA_DIR / "reference" / case_name,
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

    output_path = DATA_DIR / "ff" / f"gas_optics_{case_name}.nc"
    datatree.to_netcdf(output_path, mode="w")
    logger.info("Saved %s: %d absorbers -> %s", case_name, len(absorbers), output_path)

# %%
