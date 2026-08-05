# %%
from pathlib import Path

import numpy as np
import xarray as xr
from pyrte_rrtmgp import rte

from faxsec.constants import AVOGADRO, GRAVITY, MEAN_MOLAR_MASS_AIR, MEAN_MOLAR_MASS_H2O
from faxsec.gas_optics import GasOptics
from faxsec.utils import planck_nu

rte  # so that autoformatters don't remove the import


# %%
example_dir = "/Users/rk/Work/rte-rrtmgp/build/rte-examples-data/"  # "/Users/rk/Work/rte-examples/"
example_files = [
    example_dir + file
    for file in ["ckdmip-states.nc", "rce-states.nc", "rfmip-states.nc"]
]

rename_dict = {
    "pres_layer": "pressure_layer",
    "pres_level": "pressure_level",
    "temp_layer": "temperature_layer",
    "temp_level": "temperature_level",
    "h2o": "H2O",
    "co2": "CO2",
    "o3": "O3",
    "ch4": "CH4",
    "n2o": "N2O",
    "co": "CO",
    "o2": "O2",
    "n2": "N2",
    "cfc11": "CFC11",
    "cfc12": "CFC12",
}


def aggregate_fluxes(fluxes, band):
    band = band.lower()

    fluxes[f"{band}_spectral_flux_up"] = fluxes[f"{band}_flux_up"]
    fluxes[f"{band}_spectral_flux_down"] = fluxes[f"{band}_flux_down"]

    fluxes[f"{band}_flux_up"] = fluxes[f"{band}_spectral_flux_up"].sum("frequency")
    fluxes[f"{band}_flux_down"] = fluxes[f"{band}_spectral_flux_down"].sum("frequency")
    fluxes[f"{band}_flux_net"] = fluxes[f"{band}_flux_down"] - fluxes[f"{band}_flux_up"]


def add_net_flux(fluxes):
    fluxes["flux_net"] = fluxes["lw_flux_net"] + fluxes["sw_flux_net"]
    fluxes["flux_up"] = fluxes["lw_flux_up"] + fluxes["sw_flux_up"]
    fluxes["flux_down"] = fluxes["lw_flux_down"] + fluxes["sw_flux_down"]


def do_ddq_example(file_path, band):
    atm_ds = xr.open_dataset(file_path)
    atm_ds = atm_ds.rename({k: v for k, v in rename_dict.items() if k in atm_ds})

    dp = np.abs(atm_ds["pressure_level"].diff(dim="level", label="lower")).rename(
        {"level": "layer"}
    )
    vmr_h2o = atm_ds["H2O"] if "H2O" in atm_ds else xr.zeros_like(dp)
    m_air = (MEAN_MOLAR_MASS_AIR + MEAN_MOLAR_MASS_H2O * vmr_h2o) / (1.0 + vmr_h2o)
    atm_ds["N_per_m2_dry"] = dp / GRAVITY * AVOGADRO / (m_air * (1.0 + vmr_h2o))

    flat_ds = atm_ds.stack(atm_points=("variant", "col", "layer"))

    gas_optics_dt = xr.open_datatree(f"../data/ff/gas_optics_DDQ_{band.upper()}.nc")
    gas_optics = GasOptics.from_datatree(gas_optics_dt)

    # Compute tau and other related fields for RTE
    optical_props = (
        gas_optics.optical_depth_from_ds(atmosphere_ds=flat_ds)
        .unstack("atm_points")
        .sum(dim="species")
        .to_dataset(name="tau")
    )

    # Additional band specific properties
    if band == "sw":
        optical_props["tau_rayleigh"] = (
            gas_optics_dt["DDQ"]["xsec_rayleigh"] * atm_ds["N_per_m2_dry"]
        )
        optical_props["tau"] = optical_props["tau"] + optical_props["tau_rayleigh"]
        optical_props["ssa"] = optical_props["tau_rayleigh"] / (optical_props["tau"])
        optical_props["g"] = xr.zeros_like(optical_props["tau"])

        ssi = gas_optics_dt["DDQ"]["spectral_solar_irradiance"]
        weights = gas_optics_dt["DDQ"]["weights_hz"]
        optical_props["toa_source"] = (
            ssi
            * (atm_ds["total_solar_irradiance"] / (ssi * weights).sum(dim="frequency"))
            * weights
        )
        optical_props = optical_props.expand_dims({"gpt": 1}, axis=-1)
        optical_props["surface_albedo"] = atm_ds["surface_albedo"]
        optical_props["solar_zenith_angle"] = atm_ds["solar_zenith_angle"]
        optical_props["total_solar_irradiance"] = atm_ds["total_solar_irradiance"]
    if band == "lw":

        weights = gas_optics_dt["DDQ"]["weights_hz"]

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

    # True if level 0 is toa.
    optical_props.attrs["top_at_1"] = (
        atm_ds["pressure_layer"].isel(layer=0) - atm_ds["pressure_layer"].isel(layer=-1)
    )[0] < 0

    # Solve RTE
    fluxes = optical_props.rte.solve(add_to_input=False)

    aggregate_fluxes(fluxes, band)

    if "profile_weight" in atm_ds:
        fluxes[f"global_{band}_surface_flux"] = (
            fluxes[f"{band}_flux_net"].isel(level=-1) * atm_ds["profile_weight"]
        ).sum(dim="site")
        fluxes[f"global_{band}_toa_flux"] = (
            fluxes[f"{band}_flux_net"].isel(level=0) * atm_ds["profile_weight"]
        ).sum(dim="site")

    return fluxes


def do_rrtgmp_example(file_path, band):

    from pyrte_rrtmgp.rrtmgp import GasOptics as R_GasOptics
    from pyrte_rrtmgp.rrtmgp.data_files import (
        GasOpticsFiles,
    )

    atm_ds = xr.load_dataset(file_path)

    if band == "lw":
        rrtmgp_optics = R_GasOptics(gas_optics_file=GasOpticsFiles.LW_G256)

        # atm_ds["pres_level"] = xr.ufuncs.maximum(
        #     rrtmgp_optics.press_min,
        #     atm_ds["pres_level"],
        # )
        optical_props = rrtmgp_optics.compute(
            atm_ds,
            add_to_input=False,
        )
        optical_props["surface_emissivity"] = atm_ds["surface_emissivity"]

    elif band == "sw":
        rrtmgp_optics = R_GasOptics(gas_optics_file=GasOpticsFiles.SW_G224)
        # atm_ds["pres_level"] = xr.ufuncs.maximum(
        #     rrtmgp_optics.press_min,
        #     atm_ds["pres_level"],
        # )
        optical_props = rrtmgp_optics.compute(
            atm_ds,
            add_to_input=False,
        )

        optical_props["total_solar_irradiance"] = atm_ds["total_solar_irradiance"]
        optical_props["surface_albedo"] = atm_ds["surface_albedo"]
        optical_props["solar_zenith_angle"] = atm_ds["solar_zenith_angle"]

    else:
        raise ValueError(f"Invalid band: {band}")

    fluxes = optical_props.rte.solve(
        add_to_input=False,
    )
    fluxes[f"{band}_flux_net"] = fluxes[f"{band}_flux_down"] - fluxes[f"{band}_flux_up"]

    if "profile_weight" in atm_ds:
        fluxes[f"global_{band}_surface_flux"] = (
            fluxes[f"{band}_flux_net"].isel(level=-1) * atm_ds["profile_weight"]
        ).sum(dim="site")
        fluxes[f"global_{band}_toa_flux"] = (
            fluxes[f"{band}_flux_net"].isel(level=0) * atm_ds["profile_weight"]
        ).sum(dim="site")

    return fluxes


# %% Run and save all examples
for example in range(len(example_files)):
    ddq_fluxes_lw = do_ddq_example(example_files[example], "lw")
    ddq_fluxes_sw = do_ddq_example(example_files[example], "sw")
    ddq_fluxes = xr.merge([ddq_fluxes_lw, ddq_fluxes_sw], compat="equals", join="outer")

    add_net_flux(ddq_fluxes)

    ddq_fluxes.to_netcdf(
        f"../data/rte_examples/pyddq_fluxes_{example_files[example].split('/')[-1].split('-')[0]}.nc"
    )

# for example in range(len(example_files)):
#     rrtmgp_fluxes_lw = do_rrtgmp_example(example_files[example], "lw")
#     rrtmgp_fluxes_sw = do_rrtgmp_example(example_files[example], "sw")
#     rrtmgp_fluxes = xr.merge(
#         [rrtmgp_fluxes_lw, rrtmgp_fluxes_sw], compat="equals", join="outer"
#     )
#     add_net_flux(rrtmgp_fluxes)
#     rrtmgp_fluxes.to_netcdf(
#         f"../data/rte_examples/pyrrtmgp_fluxes_{example_files[example].split('/')[-1].split('-')[0]}.nc"
#     )

# %%
