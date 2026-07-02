"""
Flux calculation using RTE
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import xarray as xr
from pyrte_rrtmgp import rte

from model.constants import GRAVITY, MEAN_MASS_AIR
from model.gas_optics import GasOptics

rte  # just so its used and not deleted during save

required_vars = [
    "pres_layer",
    "pres_level",
    "temp_layer",
    "temp_level",
    "surface_temperature",
    "surface_emissivity",
    "surface_albedo",
    "solar_zenith_angle",
    "total_solar_irradiance",
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
    # "CO",
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


def rayleigh_cross_section(kayser):
    """
    Rayleigh scattering cross section.

    Parameters
    ----------
    kayser : float or ndarray
        Spectroscopic wavenumber [cm^-1]

    Returns
    -------
    sigma : float or ndarray
        Rayleigh cross section [cm^2 molecule^-1]
    """

    # wavelength in microns
    lam_um = 1.0e4 / kayser

    # refractive index (Edlen/Bodhaine)
    n_minus1 = (
        8060.51
        + 2480990.0 / (132.274 - 1.0 / lam_um**2)
        + 17455.7 / (39.32957 - 1.0 / lam_um**2)
    ) * 1e-8

    n = 1.0 + n_minus1

    # King factor
    Fk = 1.034 + 3.17e-4 / lam_um**2

    # Loschmidt number [cm^-3]
    Ns = 2.546899e19

    sigma = (24 * np.pi**3) / (Ns**2) * kayser**4 * ((n**2 - 1) / (n**2 + 2)) ** 2 * Fk

    return sigma / 1e4  # convert from cm^2 to m^2


# %% Load RFMIP Profiles
atm_ds = read_rfmip_profiles(site=None, expt=[0])
flat_ds = atm_ds.stack(atm_points=("expt", "site", "layer"))


# %% Instantiate a GasOptics
gas_optics_dt = xr.open_datatree("../data/ff/test_3_sw.nc")
gas_optics = GasOptics.from_datatree(gas_optics_dt)

# %% Compute tau and other related fields for RTE
tau_da = gas_optics.optical_depth_from_ds(atmosphere_ds=flat_ds)
tau_da = tau_da.unstack("atm_points")


rte_input = tau_da.sum(dim="species").to_dataset(
    name="tau"
)  # .rename_dims({"frequency": "gpt"})

# rte_input["tau_rayleigh"] = (
#     gas_optics_dt["DDQ"]["xsec_rayleigh"] * atm_ds["N_per_m2_dry"] #* 1e2
# )
# rte_input["tau"] = rte_input["tau"] + rte_input["tau_rayleigh"]
# rte_input["ssa"] = rte_input["tau_rayleigh"] / (
#     rte_input["tau"] + rte_input["tau_rayleigh"]
# )

rte_input["ssa"] = xr.zeros_like(rte_input["tau"])
rte_input["g"] = xr.zeros_like(rte_input["tau"])


rte_input["weights_hz"] = ("frequency", gas_optics_dt["DDQ"]["weights_hz"].values)

rte_input["solar_spectral_irradiance"] = gas_optics_dt["DDQ"]["spectral_solar_irradiance"]

rte_input["mu0_solar"] = np.cos(np.radians(atm_ds["solar_zenith_angle"]))

rte_input["toa_source"] = rte_input["solar_spectral_irradiance"] * (
    atm_ds["total_solar_irradiance"]
    / (rte_input["solar_spectral_irradiance"] * rte_input["weights_hz"]).sum(
        dim="frequency"
    )
)
rte_input = rte_input.expand_dims({"gpt": 1}, axis=-1)

rte_input["surface_albedo"] = atm_ds["surface_albedo"]
rte_input["solar_zenith_angle"] = atm_ds["solar_zenith_angle"]
rte_input.attrs["top_at_1"] = True


# %%
fluxes = rte_input.rte.solve(add_to_input=False)

fluxes["weights_hz"] = ("frequency", gas_optics_dt["DDQ"]["weights_hz"].values)

fluxes["brd_flux_up"] = (fluxes["sw_flux_up"] * fluxes["weights_hz"]).sum(
    dim="frequency"
)
fluxes["brd_flux_down"] = (fluxes["sw_flux_down"] * fluxes["weights_hz"]).sum(
    dim="frequency"
)
fluxes["brd_net_flux"] = fluxes["brd_flux_down"] - fluxes["brd_flux_up"]

fluxes["global_mean_toa_flux"] = (
    fluxes["brd_net_flux"].isel(level=0) * atm_ds["profile_weight"]
).sum(dim="site")

fluxes.sel(level=0)


# %% RRTMGP Example

from pyrte_rrtmgp.examples import RFMIP_FILES, load_example_file
from pyrte_rrtmgp.rrtmgp import GasOptics
from pyrte_rrtmgp.rrtmgp.data_files import (
    GasOpticsFiles,
)

gas_optics_sw = GasOptics(gas_optics_file=GasOpticsFiles.SW_G224)
atmosphere = load_example_file(RFMIP_FILES.ATMOSPHERE).isel(expt=[0])
atmosphere["pres_level"] = xr.ufuncs.maximum(
    gas_optics_sw.press_min,
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

gas_optics_sw.compute(
    atmosphere,
    add_to_input=True,
)

# # discount for rayleigh scattering
atmosphere["tau"] = atmosphere["tau"] * (1.0 - atmosphere["ssa"])
atmosphere["ssa"] = xr.zeros_like(atmosphere["ssa"])

rrtmg_fluxes = atmosphere.rte.solve(
    add_to_input=False,
)
rrtmg_fluxes["brd_net_flux"] = rrtmg_fluxes["sw_flux_down"] - rrtmg_fluxes["sw_flux_up"]

rrtmg_fluxes["global_mean_toa_flux"] = (
    rrtmg_fluxes["brd_net_flux"].isel(level=0) * atm_ds["profile_weight"]
).sum(dim="site")

rrtmg_fluxes.sel(level=0)
# %%

# %%
import matplotlib.pyplot as plt

toa_diff = (rrtmg_fluxes.brd_net_flux - fluxes.brd_net_flux).sel(expt=0, level=0)
# plot only daytime sites
site_mask = (rrtmg_fluxes["sw_flux_down"].sel(expt=0, level=0) > 10).squeeze()
toa_diff = toa_diff.where(site_mask, drop=True)

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
