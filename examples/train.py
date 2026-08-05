# %%
from pathlib import Path

import numpy as np
import xarray as xr

from faxsec.constants import SELF_SCALING
from faxsec.functional import FunctionalAbsorber
from faxsec.utils import (
    ensure_reference_dataset,
    kayser_to_hz,
    rayleigh_xsec_stamnes_2017,
    reference_cache_path,
)


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
        reference_xsec=str(
            reference_cache_path(
                species=species,
                arts_tag=arts_tag,
                cache_dir=reference_cache_dir,
            )
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
    "SW": {"CFC11": ("CFC11-XFIT",), "CFC12": ("CFC12-XFIT",), "O3-XFIT": ("O3-XFIT",)},
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

ddq_loc = "../data/ddq/Additional configurations"
ddq_files = [
    f"{ddq_loc}/DDQ_{band}_{i}.h5" for band in ["LW", "SW"] for i in range(1, 9)
]
# ddq_files = [f"../data/ddq/DDQ_{band}.h5" for band in ["LW", "SW"]]

for ddq_case in ddq_files:
    kayser_quadrature = xr.load_dataset(ddq_case)
    kayser_grid = kayser_quadrature["S"].values
    kayser_weights = kayser_quadrature["W"].values
    frequency_grid = kayser_to_hz(kayser_grid)
    weights_hz = kayser_to_hz(kayser_weights)

    absorbers = {}
    band = ddq_case.split("_")[1][:2]  # Extract the band (LW or SW) from the filename
    case_name = ddq_case.split(".")[-2].split("/")[-1]
    # lines
    for sp in lines[band].keys():
        func_abs = train_fax(
            species=sp,
            arts_tag=lines[band][sp],
            frequency_grid=frequency_grid,
            reference_cache_dir="../data/reference/" + case_name,
        )
        absorbers[sp] = func_abs

    # halocarbons
    from faxsec.xfit import CrossFitAbsorber

    for sp in halocarbons[band].keys():
        func_abs = CrossFitAbsorber(
            species=sp,
            frequency_grid=frequency_grid,
            data_source=f"../data/halocarbon/{sp.split('-')[0]}-XFIT.xml",
        )
        absorbers[sp] = func_abs

    # continuum
    from faxsec.continuum import H2OContinuum

    for sp in continuum.keys():
        absorbers[f"{sp}_continuum"] = H2OContinuum(
            frequency_grid=frequency_grid,
            data_source="../data/continuum/absco-ref_wv-mt-ckd400.nc",
        )

    # quadrature related data
    ddq = xr.Dataset(
        {
            "weights_hz": ("frequency", weights_hz),
        }
    )
    if band == "SW":
        import pyarts3

        # solar source
        solar_source_file = Path("../data/solar_spectra/solar_spectrum_July_2008.xml")
        solar_source = (
            pyarts3.xml.load(
                solar_source_file,
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
        ddq["spectral_solar_irradiance"].attrs[
            "source"
        ] = "../data/solar_spectra/solar_spectrum_July_2008.xml"

        # rayleigh scattering cross-section
        ddq_rayleigh = rayleigh_xsec_stamnes_2017(frequency_grid)
        ddq["xsec_rayleigh"] = ("frequency", ddq_rayleigh)
        ddq["xsec_rayleigh"].attrs["source"] = "Stamnes et al. 2017"

    ddq.attrs["model_class"] = "DDQ"

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

    datatree["DDQ"] = ddq

    datatree.to_netcdf(f"../data/ff/gas_optics_{case_name}.nc", mode="w")

# %%
