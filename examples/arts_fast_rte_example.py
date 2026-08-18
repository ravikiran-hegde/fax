"""Run DDQ-frequency ARTS gas optics and RTE on a handful of RFMIP columns.

Same pipeline as ``arts_ddq_rte_example.py``, run on the first 5 RFMIP
columns of variant 0 as a quick smoke test.
"""

# %%

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import xarray as xr
from pyrte_rrtmgp import rte

from faxsec.arts import species_from_tag
from faxsec.constants import AVOGADRO, GRAVITY, MEAN_MOLAR_MASS_AIR, MEAN_MOLAR_MASS_H2O
from faxsec.gas_optics import GasOptics
from faxsec.utils import kayser_to_hz, planck_nu, rayleigh_xsec_stamnes_2017

_ = rte  # keep import for accessor registration


ROOT = Path(__file__).resolve().parents[1]

example_dir = ROOT / "data/rte_examples"
rfmip_file = example_dir / "rfmip-states.nc"

N_COLUMNS = 5
VARIANT = 0

rename_dict = {
    "pres_layer": "pressure_layer",
    "pres_level": "pressure_level",
    "temp_layer": "temperature_layer",
    "temp_level": "temperature_level",
}

lines = {
    "LW": {
        "H2O": None,
        "CO2": ("CO2", "CO2-CKDMT252"),
        "O3": None,
        "O2": ("O2", "O2-CIA-O2"),
        "CH4": None,
        "N2O": None,
        "N2": ("N2", "N2-CIA-N2"),
    },
    "SW": {
        "H2O": None,
        "CO2": ("CO2", "CO2-CKDMT252"),
        "O3": None,
        "O2": ("O2", "O2-CIAfunCKDMT100"),
        "CH4": None,
        "N2O": None,
        "N2": ("N2", "N2-CIAfunCKDMT252", "N2-CIArotCKDMT252"),
    },
}

halocarbons = {
    "SW": (
        "CFC11-XFIT",
        "CFC12-XFIT",
        "O3-XFIT",
    ),
    "LW": (
        "CFC11-XFIT",
        "CFC12-XFIT",
    ),
}

continuum_tags = ("H2O-ForeignContCKDMT350", "H2O-SelfContCKDMT350")


def aggregate_fluxes(fluxes: xr.Dataset, band: str) -> xr.Dataset:
    band = band.lower()

    spectral_flux_up = fluxes[f"{band}_flux_up"]
    spectral_flux_down = fluxes[f"{band}_flux_down"]

    fluxes[f"{band}_flux_up"] = spectral_flux_up.sum("frequency")
    fluxes[f"{band}_flux_down"] = spectral_flux_down.sum("frequency")
    fluxes[f"{band}_flux_net"] = fluxes[f"{band}_flux_down"] - fluxes[f"{band}_flux_up"]
    return fluxes


def transpose_rte_input(optical_props: xr.Dataset) -> xr.Dataset:
    if "frequency" not in optical_props.dims:
        return optical_props

    dims = tuple(dim for dim in optical_props.dims if dim != "frequency")
    return optical_props.transpose(*dims, "frequency")


def add_net_flux(fluxes: xr.Dataset) -> None:
    fluxes["flux_net"] = fluxes["lw_flux_net"] + fluxes["sw_flux_net"]
    fluxes["flux_up"] = fluxes["lw_flux_up"] + fluxes["sw_flux_up"]
    fluxes["flux_down"] = fluxes["lw_flux_down"] + fluxes["sw_flux_down"]


def _append_tag(species_tags: dict[str, list[str]], species: str, tag: str) -> None:
    if species not in species_tags:
        species_tags[species] = []
    if tag not in species_tags[species]:
        species_tags[species].append(tag)


def build_species_tag_map(band: str) -> dict[str, tuple[str, ...]]:
    band = band.upper()
    species_tags: dict[str, list[str]] = {}

    for species, tags in lines[band].items():
        line_tags = (species,) if tags is None else tags
        for tag in line_tags:
            _append_tag(species_tags, species, tag)

    for tag in halocarbons[band]:
        _append_tag(species_tags, species_from_tag(tag).upper(), tag)

    for tag in continuum_tags:
        _append_tag(species_tags, species_from_tag(tag).upper(), tag)

    return {species: tuple(tags) for species, tags in species_tags.items()}


def load_ddq_aux_data(band: str) -> xr.Dataset:
    band = band.upper()
    ddq_raw = xr.load_dataset(ROOT / f"data/ddq/DDQ_{band}.h5")

    frequency_grid = kayser_to_hz(ddq_raw["S"].values)
    weights_hz = kayser_to_hz(ddq_raw["W"].values)

    ddq = xr.Dataset(
        data_vars={"weights_hz": ("frequency", weights_hz)},
        coords={"frequency": frequency_grid},
    )

    if band == "SW":
        import pyarts3

        solar_source_file = ROOT / "data/solar_spectra/solar_spectrum_July_2008.xml"
        solar_source = (
            pyarts3.xml.load(str(solar_source_file))
            .to_xarray()
            .rename({"Frequencys": "frequency"})
        )
        solar_source = xr.Dataset(
            {"spectral_solar_radiance": (("frequency",), solar_source.values[:, 0])},
            coords={"frequency": solar_source.frequency},
        ).interp(frequency=frequency_grid, method="cubic")

        total_solar_irradiance = 1361.0
        ddq["spectral_solar_irradiance"] = (
            total_solar_irradiance
            * solar_source["spectral_solar_radiance"]
            / np.dot(solar_source["spectral_solar_radiance"].values, weights_hz)
        )
        ddq["spectral_solar_irradiance"].attrs["source"] = str(solar_source_file)

        ddq_rayleigh = rayleigh_xsec_stamnes_2017(frequency_grid)
        ddq["xsec_rayleigh"] = ("frequency", ddq_rayleigh)
        ddq["xsec_rayleigh"].attrs["source"] = "Stamnes et al. 2017"

    return ddq


def build_gas_optics_for_band(band: str, frequency_grid: np.ndarray) -> GasOptics:
    species_tags = build_species_tag_map(band)
    return GasOptics.from_arts_tags(species_tags, frequency_grid=frequency_grid)


def prepare_atmosphere(atm_ds: xr.Dataset, species: tuple[str, ...]) -> xr.Dataset:
    atm_ds = atm_ds.rename({k: v for k, v in rename_dict.items() if k in atm_ds})

    species_rename = {sp.lower(): sp for sp in species if sp.lower() in atm_ds}
    atm_ds = atm_ds.rename(species_rename)

    dp = (
        atm_ds["pressure_level"]
        .diff(dim="level", label="lower")
        .rename({"level": "layer"})
    )
    dp = xr.where(dp < 0, -dp, dp)
    vmr_h2o = atm_ds["H2O"] if "H2O" in atm_ds else xr.zeros_like(dp)
    m_air = (MEAN_MOLAR_MASS_AIR + MEAN_MOLAR_MASS_H2O * vmr_h2o) / (1.0 + vmr_h2o)
    atm_ds["N_per_m2_dry"] = dp / GRAVITY * AVOGADRO / (m_air * (1.0 + vmr_h2o))

    return atm_ds


def do_arts_fast_example(
    atm_ds: xr.Dataset,
    band: str,
    gas_optics: GasOptics,
    ddq_aux: xr.Dataset,
) -> xr.Dataset:
    band = band.lower()
    atm_ds = prepare_atmosphere(atm_ds.load(), gas_optics.species)

    flat_ds = atm_ds.stack(atm_points=list(atm_ds["temperature_layer"].dims))

    optical_props = (
        gas_optics.optical_depth_from_ds(atmosphere_ds=flat_ds)
        .unstack("atm_points")
        .sum(dim="species")
        .to_dataset(name="tau")
    )

    if band == "sw":
        optical_props["tau_rayleigh"] = (
            ddq_aux["xsec_rayleigh"] * atm_ds["N_per_m2_dry"]
        )
        optical_props["tau"] = optical_props["tau"] + optical_props["tau_rayleigh"]
        optical_props["ssa"] = optical_props["tau_rayleigh"] / optical_props["tau"]
        optical_props["g"] = xr.zeros_like(optical_props["tau"])

        optical_props["toa_source"] = (
            ddq_aux["spectral_solar_irradiance"] * ddq_aux["weights_hz"]
        )
        optical_props = optical_props.expand_dims({"gpt": 1}, axis=-1)
        optical_props["surface_albedo"] = atm_ds["surface_albedo"]
        optical_props["solar_zenith_angle"] = atm_ds["solar_zenith_angle"]
        optical_props["total_solar_irradiance"] = atm_ds["total_solar_irradiance"]

    elif band == "lw":
        weights = ddq_aux["weights_hz"]

        optical_props["layer_source"] = (
            xr.apply_ufunc(
                planck_nu,
                optical_props["frequency"],
                atm_ds["temperature_layer"],
                vectorize=True,
            )
            * weights
        )

        optical_props["level_source"] = (
            xr.apply_ufunc(
                planck_nu,
                optical_props["frequency"],
                atm_ds["temperature_level"],
                vectorize=True,
            )
            * weights
        )

        optical_props["surface_source"] = (
            xr.apply_ufunc(
                planck_nu,
                optical_props["frequency"],
                atm_ds["surface_temperature"],
                vectorize=True,
            )
            * weights
        )
        optical_props["surface_source_jacobian"] = xr.zeros_like(
            optical_props["surface_source"]
        )

        optical_props = optical_props.expand_dims({"gpt": 1}, axis=-1)
        optical_props["surface_emissivity"] = atm_ds["surface_emissivity"]
    else:
        raise ValueError(f"Invalid band: {band}")

    optical_props = transpose_rte_input(optical_props)

    top_delta = (
        atm_ds["pressure_layer"].isel(layer=0) - atm_ds["pressure_layer"].isel(layer=-1)
    ).mean()
    optical_props.attrs["top_at_1"] = bool(top_delta.values < 0)

    fluxes = optical_props.rte.solve(add_to_input=False)
    fluxes = aggregate_fluxes(fluxes, band)

    return fluxes


# %%
if __name__ == "__main__":
    output_dir = ROOT / "data/rte_examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    ddq_aux_lw = load_ddq_aux_data("LW")
    ddq_aux_sw = load_ddq_aux_data("SW")

    t_build0 = time.time()
    gas_optics_lw = build_gas_optics_for_band("LW", ddq_aux_lw["frequency"].values)
    gas_optics_sw = build_gas_optics_for_band("SW", ddq_aux_sw["frequency"].values)
    t_build1 = time.time()
    print(f"Built gas optics (LW+SW) in {t_build1 - t_build0:.2f}s")

    atm_ds = (
        xr.open_dataset(rfmip_file)
        .isel(variant=VARIANT, col=slice(0, N_COLUMNS))
    )

    t0 = time.time()
    lw_fluxes = do_arts_fast_example(atm_ds, "lw", gas_optics_lw, ddq_aux_lw)
    sw_fluxes = do_arts_fast_example(atm_ds, "sw", gas_optics_sw, ddq_aux_sw)
    t1 = time.time()
    print(f"Computed optical properties + RTE for {N_COLUMNS} columns in {t1 - t0:.2f}s")

    fluxes = xr.merge([lw_fluxes, sw_fluxes], compat="equals", join="outer")
    add_net_flux(fluxes)

    print(fluxes[["lw_flux_net", "sw_flux_net", "flux_net"]])
    print("TOA net flux [W/m^2]:", fluxes["flux_net"].isel(level=0).values)
    print("Surface net flux [W/m^2]:", fluxes["flux_net"].isel(level=-1).values)

    output_path = output_dir / f"pyarts_fast_fluxes_rfmip_v{VARIANT:03d}.nc"
    fluxes.to_netcdf(output_path)
    print(f"Saved {output_path}")
