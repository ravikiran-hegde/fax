# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import xarray as xr
from pyrte_rrtmgp import rte

from model.constants import GRAVITY, MEAN_MASS_AIR
from model.gas_optics import GasOptics
from model.utils import planck_nu

rte


# %%
example_dir = "/Users/rk/Work/rte-examples/"
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


def aggregate_fluxes(fluxes, spectral_weights, band):
    band = band.lower()
    fluxes[f"{band}_spectral_flux_up"] = fluxes[f"{band}_flux_up"]
    fluxes[f"{band}_spectral_flux_down"] = fluxes[f"{band}_flux_down"]
    fluxes[f"{band}_flux_up"] = (
        fluxes[f"{band}_spectral_flux_up"] * spectral_weights
    ).sum(dim="frequency")
    fluxes[f"{band}_flux_down"] = (
        fluxes[f"{band}_spectral_flux_down"] * spectral_weights
    ).sum(dim="frequency")
    fluxes[f"{band}_flux_net"] = fluxes[f"{band}_flux_down"] - fluxes[f"{band}_flux_up"]


def do_ddq_example(file_path, band):
    atm_ds = xr.open_dataset(file_path)
    atm_ds = atm_ds.rename(rename_dict)
    dp = np.abs(atm_ds["pressure_level"].diff(dim="level", label="lower")).rename(
        {"level": "layer"}
    )
    dry_vmr = 1.0 / (1.0 + atm_ds["H2O"]) if "H2O" in atm_ds else 1.0
    atm_ds["N_per_m2_dry"] = dp * dry_vmr / (GRAVITY * MEAN_MASS_AIR)

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
        optical_props["toa_source"] = ssi * (
            atm_ds["total_solar_irradiance"] / (ssi * weights).sum(dim="frequency")
        )
        optical_props = optical_props.expand_dims({"gpt": 1}, axis=-1)
        optical_props["surface_albedo"] = atm_ds["surface_albedo"]
        optical_props["solar_zenith_angle"] = atm_ds["solar_zenith_angle"]

    if band == "lw":
        optical_props["surface_emissivity"] = atm_ds["surface_emissivity"]

        optical_props["layer_source"] = xr.apply_ufunc(
            planck_nu,
            optical_props["frequency"],
            atm_ds["temperature_layer"],
            vectorize=True,
        )

        optical_props["level_source"] = xr.apply_ufunc(
            planck_nu,
            optical_props["frequency"],
            atm_ds["temperature_level"],
            vectorize=True,
        )

        optical_props["surface_source"] = xr.apply_ufunc(
            planck_nu,
            optical_props["frequency"],
            atm_ds["surface_temperature"],
            vectorize=True,
        )
        optical_props["surface_source_jacobian"] = xr.zeros_like(
            optical_props["surface_source"]
        )

        optical_props = optical_props.expand_dims({"gpt": 1}, axis=-1)
        optical_props["surface_emissivity"] = atm_ds["surface_emissivity"]

    optical_props.attrs["top_at_1"] = True

    # Solve RTE
    fluxes = optical_props.rte.solve(add_to_input=False)

    aggregate_fluxes(fluxes, gas_optics_dt["DDQ"]["weights_hz"], band)

    if "profile_weight" in atm_ds:
        fluxes[f"global_{band}_surface_flux"] = (
            fluxes[f"{band}_net_flux"].isel(level=-1) * atm_ds["profile_weight"]
        ).sum(dim="site")
        fluxes[f"global_{band}_toa_flux"] = (
            fluxes[f"{band}_net_flux"].isel(level=0) * atm_ds["profile_weight"]
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


# %%
ddq_fluxes_lw = do_ddq_example(example_files[2], "lw")
ddq_fluxes_sw = do_ddq_example(example_files[2], "sw")
ddq_fluxes = xr.merge([ddq_fluxes_lw, ddq_fluxes_sw], compat="equals")

rrtmgp_fluxes_lw = do_rrtgmp_example(example_files[2], "lw")
rrtmgp_fluxes_sw = do_rrtgmp_example(example_files[2], "sw")
rrtmgp_fluxes = xr.merge([rrtmgp_fluxes_lw, rrtmgp_fluxes_sw], compat="equals")
# %%

atm_ds = xr.open_dataset(example_files[2])

# %%
import matplotlib.pyplot as plt

var = "sw_flux_net"
level = -1
variant = 0
toa_diff = (rrtmgp_fluxes[var] - ddq_fluxes[var]).sel(variant=variant, level=level)
# plot only daytime sites
site_mask = (rrtmgp_fluxes["sw_flux_down"].sel(variant=variant, level=0) > 0).squeeze()
toa_diff = toa_diff.where(site_mask, drop=True)

fig, ax = plt.subplots(
    1,
    2,
    figsize=(10, 5),
    sharey=True,
    gridspec_kw={"width_ratios": [4, 1], "wspace": 0.05},
)

scatter_sizes = (
    1000 * atm_ds["profile_weight"].sel(col=toa_diff.col).values
    if "profile_weight" in atm_ds
    else 10 * np.ones_like(toa_diff.col.values)
)
ax[0].scatter(
    toa_diff.col.values,
    toa_diff.values,
    s=scatter_sizes,
    # alpha=0.7,
    edgecolor="k",
    linewidth=0.3,
)
ax[0].axhline(0, color="gray", linestyle="--", linewidth=1)
ax[0].set_xlabel("Column")
ax[0].set_ylabel(f"Difference in {var} " +  r"/ W m$^{-2}$")
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

fig.suptitle(f"Flux Differences (RRTMG - DDQ), level = {level}, variant = {variant}")
plt.tight_layout()
plt.show()
# %%
rrtmgp_fluxes = rrtmgp_fluxes.sel(col=toa_diff.col.values.squeeze())
ddq_fluxes = ddq_fluxes.sel(col=toa_diff.col.values.squeeze())
# only for rfmip
#
print(
    f"Global daytime {var} difference (RRTMG - DDQ):",
    (
        (rrtmgp_fluxes[var] - ddq_fluxes[var]).sel(
            level=level,
        )
        * atm_ds["profile_weight"].sel(col=toa_diff.col.values.squeeze())
    )
    .sum(dim="col")
    .values,
)

# %%
