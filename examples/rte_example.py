"""Compute LW+SW radiative fluxes for one atmosphere test case.

Supports three interchangeable gas-optics backends:

- ``fax``    -- the trained FAX functional-form model, loaded from a saved
                ``GasOptics`` datatree (``data/ff/gas_optics_DDQ_{LW,SW}*.nc``).
- ``arts``   -- the same DDQ frequency/weight quadrature, but with every
                absorber evaluated online through ARTS (no training stage).
- ``rrtmgp`` -- RRTMGP's own G-point gas optics, as an independent reference.

Run on a full case, or narrow it down to one variant and/or a slice of
columns for a quick check:

    python rte_example.py --case rfmip --gas-optics fax
    python rte_example.py --case rfmip --gas-optics arts --variant 0 --columns 0:5
    python rte_example.py --case rce --gas-optics rrtmgp --columns 3

This replaces the older arts_ddq_rte_example.py / arts_fast_rte_example.py /
arts_rte_by_variant.py, which duplicated most of this pipeline three times
over with only the case/variant/column selection differing.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import xarray as xr
from pyrte_rrtmgp import rte

from faxsec.arts import species_from_tag
from faxsec.constants import AVOGADRO, GRAVITY, MEAN_MOLAR_MASS_AIR, MEAN_MOLAR_MASS_H2O
from faxsec.gas_optics import GasOptics
from faxsec.log_config import setup_logging
from faxsec.utils import kayser_to_hz, planck_nu, rayleigh_xsec_stamnes_2017

_ = rte  # keep import for accessor registration

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXAMPLE_DIR = DATA_DIR / "rte_examples"

CASE_FILES = {
    "rfmip": EXAMPLE_DIR / "rfmip-states.nc",
    "ckdmip": EXAMPLE_DIR / "ckdmip-states.nc",
    "rce": EXAMPLE_DIR / "rce-states.nc",
}

RENAME_DICT = {
    "pres_layer": "pressure_layer",
    "pres_level": "pressure_level",
    "temp_layer": "temperature_layer",
    "temp_level": "temperature_level",
}

# ARTS absorption tags per band, used by the "arts" backend.
LINES = {
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
HALOCARBONS = {
    "SW": ("CFC11-XFIT", "CFC12-XFIT", "O3-XFIT"),
    "LW": ("CFC11-XFIT", "CFC12-XFIT"),
}
CONTINUUM_TAGS = ("H2O-ForeignContCKDMT350", "H2O-SelfContCKDMT350")


# -----------------------------------------------------------------------------
# Shared atmosphere / flux helpers
# -----------------------------------------------------------------------------


def prepare_atmosphere(atm_ds: xr.Dataset, species: tuple[str, ...]) -> xr.Dataset:
    """Rename to faxsec conventions and add the dry-air column density."""
    atm_ds = atm_ds.rename({k: v for k, v in RENAME_DICT.items() if k in atm_ds})

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


def transpose_rte_input(optical_props: xr.Dataset) -> xr.Dataset:
    if "frequency" not in optical_props.dims:
        return optical_props
    dims = tuple(dim for dim in optical_props.dims if dim != "frequency")
    return optical_props.transpose(*dims, "frequency")


def aggregate_fluxes(fluxes: xr.Dataset, band: str) -> xr.Dataset:
    band = band.lower()
    fluxes[f"{band}_flux_up"] = fluxes[f"{band}_flux_up"].sum("frequency")
    fluxes[f"{band}_flux_down"] = fluxes[f"{band}_flux_down"].sum("frequency")
    fluxes[f"{band}_flux_net"] = fluxes[f"{band}_flux_down"] - fluxes[f"{band}_flux_up"]
    return fluxes


def add_net_flux(fluxes: xr.Dataset) -> None:
    fluxes["flux_net"] = fluxes["lw_flux_net"] + fluxes["sw_flux_net"]
    fluxes["flux_up"] = fluxes["lw_flux_up"] + fluxes["sw_flux_up"]
    fluxes["flux_down"] = fluxes["lw_flux_down"] + fluxes["sw_flux_down"]


def profile_summary(atm_ds: xr.Dataset) -> str:
    """Brief '{dim: size, ...} profiles, N layers' description for logging."""
    profile_dims = {
        dim: size for dim, size in atm_ds.sizes.items() if dim not in ("layer", "level")
    }
    return f"{profile_dims or 1} profiles, {atm_ds.sizes['layer']} layers each"


# -----------------------------------------------------------------------------
# Gas-optics backends: "fax" (trained) and "arts" (online) both produce a
# faxsec GasOptics, so they share the same downstream flux computation.
# -----------------------------------------------------------------------------


def _append_tag(species_tags: dict[str, list[str]], species: str, tag: str) -> None:
    if species not in species_tags:
        species_tags[species] = []
    if tag not in species_tags[species]:
        species_tags[species].append(tag)


def build_species_tag_map(band: str) -> dict[str, tuple[str, ...]]:
    band = band.upper()
    species_tags: dict[str, list[str]] = {}

    for species, tags in LINES[band].items():
        line_tags = (species,) if tags is None else tags
        for tag in line_tags:
            _append_tag(species_tags, species, tag)

    for tag in HALOCARBONS[band]:
        _append_tag(species_tags, species_from_tag(tag).upper(), tag)

    for tag in CONTINUUM_TAGS:
        _append_tag(species_tags, species_from_tag(tag).upper(), tag)

    return {species: tuple(tags) for species, tags in species_tags.items()}


def load_ddq_aux_data(band: str) -> xr.Dataset:
    """DDQ frequency grid, quadrature weights, solar irradiance, Rayleigh xsec."""
    band = band.upper()
    ddq_raw = xr.load_dataset(DATA_DIR / "ddq" / f"DDQ_{band}.h5")

    frequency_grid = kayser_to_hz(ddq_raw["S"].values)
    weights_hz = kayser_to_hz(ddq_raw["W"].values)

    ddq = xr.Dataset(
        data_vars={"weights_hz": ("frequency", weights_hz)},
        coords={"frequency": frequency_grid},
    )

    if band == "SW":
        import pyarts3

        solar_source_file = DATA_DIR / "solar_spectra" / "solar_spectrum_July_2008.xml"
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


def build_arts_gas_optics(band: str) -> tuple[GasOptics, xr.Dataset]:
    """Build a GasOptics with every absorber evaluated online via ARTS."""
    ddq_aux = load_ddq_aux_data(band)
    species_tags = build_species_tag_map(band)
    gas_optics = GasOptics.from_arts_tags(
        species_tags, frequency_grid=ddq_aux["frequency"].values
    )
    return gas_optics, ddq_aux


def build_fax_gas_optics(band: str, suffix: str = "") -> tuple[GasOptics, xr.Dataset]:
    """Load a GasOptics trained with FAX functional forms from its datatree."""
    dt_path = DATA_DIR / "ff" / f"gas_optics_DDQ_{band.upper()}{suffix}.nc"
    gas_optics_dt = xr.open_datatree(dt_path)
    gas_optics = GasOptics.from_datatree(gas_optics_dt)
    return gas_optics, gas_optics_dt["DDQ"].to_dataset()


def build_gas_optics(
    backend: str, band: str, suffix: str = ""
) -> tuple[GasOptics, xr.Dataset]:
    if backend == "arts":
        return build_arts_gas_optics(band)
    if backend == "fax":
        return build_fax_gas_optics(band, suffix)
    raise ValueError(
        f"'{backend}' is not a faxsec.GasOptics backend (use 'fax' or 'arts')"
    )


def compute_fluxes(
    atm_ds: xr.Dataset,
    band: str,
    gas_optics: GasOptics,
    ddq_aux: xr.Dataset,
) -> xr.Dataset:
    """LW or SW fluxes from a faxsec GasOptics (shared by the fax/arts backends)."""
    band = band.lower()
    atm_ds = prepare_atmosphere(atm_ds, gas_optics.species)
    logger.info("%s: %s", band.upper(), profile_summary(atm_ds))

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
    return aggregate_fluxes(fluxes, band)


# -----------------------------------------------------------------------------
# RRTMGP backend: independent reference gas optics, its own variable naming.
# -----------------------------------------------------------------------------


def compute_rrtmgp_fluxes(atm_ds: xr.Dataset, band: str) -> xr.Dataset:
    from pyrte_rrtmgp.rrtmgp import GasOptics as RRTMGPGasOptics
    from pyrte_rrtmgp.rrtmgp.data_files import GasOpticsFiles

    band = band.lower()
    logger.info("%s RRTMGP optics: %s", band.upper(), profile_summary(atm_ds))

    if band == "lw":
        rrtmgp_optics = RRTMGPGasOptics(gas_optics_file=GasOpticsFiles.LW_G256)
        optical_props = rrtmgp_optics.compute(atm_ds, add_to_input=False)
        optical_props["surface_emissivity"] = atm_ds["surface_emissivity"]
    elif band == "sw":
        rrtmgp_optics = RRTMGPGasOptics(gas_optics_file=GasOpticsFiles.SW_G224)
        optical_props = rrtmgp_optics.compute(atm_ds, add_to_input=False)
        optical_props["total_solar_irradiance"] = atm_ds["total_solar_irradiance"]
        optical_props["surface_albedo"] = atm_ds["surface_albedo"]
        optical_props["solar_zenith_angle"] = atm_ds["solar_zenith_angle"]
    else:
        raise ValueError(f"Invalid band: {band}")

    fluxes = optical_props.rte.solve(add_to_input=False)
    fluxes[f"{band}_flux_net"] = fluxes[f"{band}_flux_down"] - fluxes[f"{band}_flux_up"]
    return fluxes


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_columns(spec: str) -> slice | int:
    """Parse '--columns' as a plain index ('3') or a slice ('start:stop')."""
    if ":" in spec:
        start, _, stop = spec.partition(":")
        return slice(int(start) if start else None, int(stop) if stop else None)
    return int(spec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASE_FILES), default="rfmip")
    parser.add_argument(
        "--gas-optics", choices=("fax", "arts", "rrtmgp"), default="fax"
    )
    parser.add_argument(
        "--variant",
        type=int,
        default=None,
        help="Select a single variant (default: all)",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default=None,
        help="Column index or 'start:stop' slice (default: all)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Filename suffix of the trained FAX datatree (only used by --gas-optics fax)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    atm_ds = xr.open_dataset(CASE_FILES[args.case])
    if args.variant is not None:
        atm_ds = atm_ds.isel(variant=args.variant)
    if args.columns is not None:
        atm_ds = atm_ds.isel(col=parse_columns(args.columns))
    atm_ds = atm_ds.load()

    non_vertical_dims = {
        dim: size for dim, size in atm_ds.sizes.items() if dim not in ("layer", "level")
    }
    logger.info(
        "Case '%s': backend=%s, %s", args.case, args.gas_optics, non_vertical_dims
    )

    t0 = time.time()
    if args.gas_optics == "rrtmgp":
        lw_fluxes = compute_rrtmgp_fluxes(atm_ds, "lw")
        sw_fluxes = compute_rrtmgp_fluxes(atm_ds, "sw")
    else:
        gas_optics_lw, ddq_aux_lw = build_gas_optics(args.gas_optics, "LW", args.suffix)
        gas_optics_sw, ddq_aux_sw = build_gas_optics(args.gas_optics, "SW", args.suffix)
        lw_fluxes = compute_fluxes(atm_ds, "lw", gas_optics_lw, ddq_aux_lw)
        sw_fluxes = compute_fluxes(atm_ds, "sw", gas_optics_sw, ddq_aux_sw)

    fluxes = xr.merge([lw_fluxes, sw_fluxes], compat="equals", join="outer")
    add_net_flux(fluxes)
    logger.info("Computed LW+SW fluxes in %.2fs", time.time() - t0)

    output_name = f"fluxes_{args.gas_optics}_{args.case}{args.suffix}"
    if args.variant is not None:
        output_name += f"_v{args.variant:03d}"
    if args.columns is not None:
        output_name += f"_c{args.columns.replace(':', '-')}"
    output_path = EXAMPLE_DIR / f"{output_name}.nc"
    fluxes.to_netcdf(output_path)
    logger.info("Saved fluxes -> %s", output_path)


if __name__ == "__main__":
    main()
