"""
Flux calculation using RTE
"""

# %%
from pathlib import Path

import numpy as np
import xarray as xr
from pyrte_rrtmgp import rte

from faxsec.constants import (
    BOLTZMANN,
    GRAVITY,
    LIGHT_SPEED,
    MEAN_MASS_AIR,
    PLANCK,
)
from faxsec.gas_optics import GasOptics

required_vars = [
    "pres_layer",
    "pres_level",
    "temp_layer",
    "temp_level",
    "surface_temperature",
    "surface_emissivity",
    "surface_albedo",
    "solar_zenith_angle",
    "profile_weight",
    "col",
    "variant",
]

species = [
    "H2O",
    "CO2",
    "O3",
    "CH4",
    "N2O",
    "CO",
    "O2",
    "N2",
    "CFC11",
    "CFC12",
]

required_vars = required_vars + species


def read_rfmip_profiles(
    file_path: (
        str | Path | None
    ) = "/Users/rk/Work/rfmip/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc",
    expt: int | list[int] | None = 0,
    site: int | list[int] | None = 0,
    include_species: list[str] | None = species,
    path: Path | None = None,
) -> xr.Dataset:
    """Read RFMIP and return canonical fastabs profile datasets."""

    rfmip_raw = xr.open_dataset(file_path)

    indexers = {}
    if expt is not None:
        indexers["expt"] = expt
    if site is not None:
        indexers["site"] = site

    rfmip = rfmip_raw.isel(**indexers).rename(
        {
            "water_vapor": "H2O",
            "carbon_dioxide_GM": "CO2",
            "ozone": "O3",
            "methane_GM": "CH4",
            "nitrous_oxide_GM": "N2O",
            "carbon_monoxide_GM": "CO",
            "oxygen_GM": "O2",
            "nitrogen_GM": "N2",
            "cfc11_GM": "CFC11",
            "cfc12_GM": "CFC12",
        }
    )

    rfmip = rfmip.drop_vars([v for v in rfmip.variables if v not in required_vars])

    for sp in include_species:
        rfmip[sp] = rfmip[sp] * float(rfmip[sp].attrs["units"])
        rfmip[sp].assign_attrs({"units": "1"})

    variable_map = {
        "pres_layer": "pressure_layer",
        "temp_layer": "temperature_layer",
        "pres_level": "pressure_level",
        "temp_level": "temperature_level",
    }

    atm_ds = rfmip.rename(variable_map)

    dp = np.abs(atm_ds["pressure_level"].diff(dim="level", label="lower")).rename(
        {"level": "layer"}
    )
    dry_vmr = 1.0 / (1.0 + atm_ds["H2O"]) if "H2O" in atm_ds else 1.0
    atm_ds["N_per_m2_dry"] = dp * dry_vmr / (GRAVITY * MEAN_MASS_AIR)

    return atm_ds


def planck_nu(f_grid_hz, temperature):
    """Return Planck Flux in W/m^2/Hz."""

    exponent = PLANCK * f_grid_hz / (BOLTZMANN * temperature)
    return (2 * PLANCK * f_grid_hz**3 / LIGHT_SPEED**2) / np.expm1(exponent)


# %% Load RFMIP Profiles
atm_ds = read_rfmip_profiles(site=None, expt=[0])
flat_ds = atm_ds.stack(atm_points=("expt", "site", "layer"))


# %% Instantiate a GasOptics
gas_optics_dt = xr.open_datatree("../data/ff/gas_optics_DDQ_LW.nc")
gas_optics = GasOptics.from_datatree(gas_optics_dt)


# %% Compute tau and other related fields for RTE
tau_da = gas_optics.optical_depth_from_ds(atmosphere_ds=flat_ds)
tau_da = tau_da.unstack("atm_points")


rte_input = tau_da.sum(dim="species").to_dataset(name="tau")


rte_input["layer_source"] = xr.apply_ufunc(
    planck_nu,
    rte_input["frequency"],
    atm_ds["temperature_layer"],
    vectorize=True,
)

rte_input["level_source"] = xr.apply_ufunc(
    planck_nu,
    rte_input["frequency"],
    atm_ds["temperature_level"],
    vectorize=True,
)

rte_input["surface_emissivity"] = atm_ds["surface_emissivity"]
rte_input["surface_source"] = xr.apply_ufunc(
    planck_nu,
    rte_input["frequency"],
    atm_ds["surface_temperature"],
    vectorize=True,
)

rte_input["surface_source_jacobian"] = xr.zeros_like(rte_input["surface_source"])
rte_input.attrs["top_at_1"] = True

rte_input = rte_input.expand_dims({"gpt": 1}, axis=-1)

# %%
rte
fluxes = rte_input.rte.solve(add_to_input=False)

fluxes["weights_hz"] = ("frequency", gas_optics_dt["DDQ"]["weights_hz"].values)

fluxes["brd_flux_up"] = (fluxes["lw_flux_up"] * fluxes["weights_hz"]).sum(
    dim="frequency"
)
fluxes["brd_flux_down"] = (fluxes["lw_flux_down"] * fluxes["weights_hz"]).sum(
    dim="frequency"
)
fluxes["brd_net_flux"] = fluxes["brd_flux_down"] - fluxes["brd_flux_up"]

fluxes["global_mean_toa_flux"] = (
    fluxes["brd_net_flux"].isel(level=0) * atm_ds["profile_weight"]
).sum(dim="site")

fluxes.sel(level=0)


# %% RRTMGP Example

from pyrte_rrtmgp.rrtmgp import GasOptics
from pyrte_rrtmgp.rrtmgp.data_files import (
    GasOpticsFiles,
)

gas_optics_lw = GasOptics(gas_optics_file=GasOpticsFiles.LW_G256)
atmosphere = xr.load_dataset(
    "/Users/rk/Work/rfmip/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
).isel(expt=[0])
atmosphere["pres_level"] = xr.ufuncs.maximum(
    gas_optics_lw.press_min,
    atmosphere["pres_level"],
)
gas_names = {
    "water_vapor": "h2o",
    "carbon_dioxide_GM": "co2",
    "ozone": "o3",
    "nitrous_oxide_GM": "n2o",
    "carbon_monoxide_GM": "co",
    "methane_GM": "ch4",
    "oxygen_GM": "o2",
    "nitrogen_GM": "n2",
    # "carbon_tetrachloride_GM": "ccl4",
    "cfc11_GM": "cfc11",
    "cfc12_GM": "cfc12",
    # "hcfc22_GM": "cfc22",
    # "hfc143a_GM": "hfc143a",
    # "hfc125_GM": "hfc125",
    # "hfc23_GM": "hfc23",
    # "hfc32_GM": "hfc32",
    # "hfc134a_GM": "hfc134a",
    # "cf4_GM": "cf4",
}

atmosphere = atmosphere.rename_vars(gas_names)
for g in gas_names.values():
    if hasattr(atmosphere[g], "units"):
        atmosphere[g] *= float(atmosphere[g].units)
        atmosphere[g].assign_attrs({"units": "1"})

optical_props = gas_optics_lw.compute(
    atmosphere,
    add_to_input=False,
)
optical_props["surface_emissivity"] = atmosphere.surface_emissivity


rrtmg_fluxes = optical_props.rte.solve(
    add_to_input=False,
)
rrtmg_fluxes["brd_net_flux"] = rrtmg_fluxes["lw_flux_down"] - rrtmg_fluxes["lw_flux_up"]

rrtmg_fluxes["global_mean_toa_flux"] = (
    rrtmg_fluxes["brd_net_flux"].isel(level=0) * atm_ds["profile_weight"]
).sum(dim="site")

rrtmg_fluxes.sel(level=0)
# %%
import matplotlib.pyplot as plt

toa_diff = (rrtmg_fluxes.brd_net_flux - fluxes.brd_net_flux).sel(expt=0, level=0)
# # plot only daytime sites
# toa_diff = toa_diff.where(site_mask, drop=True)

fig, ax = plt.subplots(
    1,
    2,
    figsize=(10, 5),
    sharey=True,
    gridspec_kw={"width_ratios": [4, 1], "wspace": 0.05},
)

ax[0].scatter(
    toa_diff.site.values,
    toa_diff.values,
    s=1000 * atm_ds["profile_weight"].sel(site=toa_diff.site).values,
    # alpha=0.7,
    edgecolor="k",
    linewidth=0.3,
)
ax[0].axhline(0, color="gray", linestyle="--", linewidth=1)
ax[0].set_xlabel("Site")
ax[0].set_ylabel("Difference in TOA flux (W/m^2)")
ax[0].grid(alpha=0.5)

ax[1].hist(
    toa_diff.values,
    bins=30,
    orientation="horizontal",
    color="steelblue",
    # alpha=0.7,
    edgecolor="k",
    linewidth=0.3,
)
ax[1].axhline(0, color="gray", linestyle="--", linewidth=1)
ax[1].set_xlabel("Count")
ax[1].grid(alpha=0.5)
ax[1].tick_params(labelleft=False)  # y labels only on the left plot

fig.suptitle("TOA Flux Differences (RRTMG - Reference)")
plt.tight_layout()
plt.show()

# %%
rrttmg_fluxes = rrtmg_fluxes.sel(site=toa_diff.site.values.squeeze())
fluxes = fluxes.sel(site=toa_diff.site.values.squeeze())
print(
    "Global daytime TOA flux difference (RRTMG - Reference):",
    (
        (rrtmg_fluxes.brd_net_flux - fluxes.brd_net_flux).sel(
            level=0,
        )
        * atm_ds["profile_weight"].sel(site=toa_diff.site.values.squeeze())
    )
    .sum(dim="site")
    .values,
)
# %%
