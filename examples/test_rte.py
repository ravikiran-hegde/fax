"""
Flux calculation using RTE
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import xarray as xr
from pyrte_rrtmgp.kernels.rte import lw_solver_2stream
from pyrte_rrtmgp import rte
from model.constants import BOLTZMANN, GRAVITY, LIGHT_SPEED, MEAN_MASS_AIR, PLANCK
from model.gas_optics import GasOptics
from model.utils import kayser_to_hz

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
    n_dry_per_m2 = dp * dry_vmr / (GRAVITY * MEAN_MASS_AIR)
    atm_ds["N_per_m2_dry"] = n_dry_per_m2

    return atm_ds


def planck_nu(f_grid_hz, temperature):
    """Return Planck Flux in W/m^2/Hz."""

    exponent = PLANCK * f_grid_hz / (BOLTZMANN * temperature)
    return (np.pi * PLANCK * f_grid_hz**3 / LIGHT_SPEED**2) / np.expm1(exponent)


def lw_fluxes_from_problem_ds(problem_ds: xr.Dataset) -> xr.Dataset:
    """Compute LW fluxes from a problem dataset."""
    nmus: int = 1
    top_at_1: bool = problem_ds.attrs["top_at_1"]

    if "incident_flux" not in problem_ds:
        incident_flux: xr.DataArray = xr.zeros_like(problem_ds["surface_source"])
    else:
        incident_flux = problem_ds["incident_flux"]

    non_default_dims = [
        d for d in problem_ds.dims if d not in ["layer", "level", "gpt", "bnd", "pair"]
    ]

    # Expand surface emissivity dimensions if needed
    _, problem_ds["surface_emissivity"] = xr.broadcast(
        problem_ds,
        problem_ds["surface_emissivity"],
        exclude=["layer", "level", "bnd", "pair"],
    )

    problem_ds = problem_ds.stack({"stacked_cols": non_default_dims})
    incident_flux = incident_flux.stack({"stacked_cols": non_default_dims})

    ssa: xr.DataArray = (
        problem_ds["ssa"] if "ssa" in problem_ds else xr.zeros_like(problem_ds["tau"])
    )
    g: xr.DataArray = (
        problem_ds["g"] if "g" in problem_ds else xr.zeros_like(problem_ds["tau"])
    )
    do_rescaling: bool = "ssa" in problem_ds and "g" in problem_ds

    (
        spectral_flux_up,
        spectral_flux_down,
    ) = xr.apply_ufunc(
        lw_solver_2stream,
        problem_ds.sizes["stacked_cols"],
        problem_ds.sizes["layer"],
        problem_ds.sizes["gpt"],
        problem_ds["tau"],
        ssa,
        g,
        problem_ds["layer_source"],
        problem_ds["level_source"],
        problem_ds["surface_emissivity"],
        problem_ds["surface_source"],
        incident_flux,
        kwargs={
            "top_at_1": top_at_1,
        },
        input_core_dims=[
            [],  # ncol
            [],  # nlay
            [],  # ngpt
            ["layer", "gpt"],  # tau
            ["layer", "gpt"],  # ssa
            ["layer", "gpt"],  # g
            ["layer", "gpt"],  # lay_source
            ["level", "gpt"],  # lev_source
            ["gpt"],  # sfc_emis
            ["gpt"],  # sfc_src
            ["gpt"],  # inc_flux
        ],
        output_core_dims=[
            ["level", "gpt"],  # solver_flux_up
            ["level", "gpt"],  # solver_flux_down
        ],
        output_dtypes=[np.float64, np.float64],
        dask="parallelized",
    )

    fluxes = xr.Dataset(
        {
            "flux_up": spectral_flux_up.unstack("stacked_cols"),
            "flux_down": spectral_flux_down.unstack("stacked_cols"),
        }
    )

    # transpose_order = non_default_dims + ["level"]

    lw_ddq_loc = "../data/ddq/DDQ_LW.h5"
    kayser_quadrature = xr.load_dataset(lw_ddq_loc)

    fluxes["weights_hz"] = ("frequency", kayser_to_hz(kayser_quadrature["W"].values))

    fluxes["lw_flux_up"] = (fluxes["flux_up"] * fluxes["weights_hz"]).sum(
        dim="frequency"
    )
    fluxes["lw_flux_down"] = (fluxes["flux_down"] * fluxes["weights_hz"]).sum(
        dim="frequency"
    )
    fluxes["lw_net_flux"] = fluxes["lw_flux_down"] - fluxes["lw_flux_up"]

    return fluxes  # .transpose(*transpose_order)


# %% Instantiate a GasOptics
gas_optics_dt = xr.open_datatree("../data/ff/test_2_lw.nc")
gas_optics = GasOptics.from_datatree(gas_optics_dt)

# %% Load RFMIP Profiles
atm_ds = read_rfmip_profiles(site=None, expt=None)

flat_ds = atm_ds.stack(levels=("expt", "site", "layer"))

# %% Compute tau and other related fields for RTE
tau_da = gas_optics.optical_depth_from_ds(atmosphere_ds=flat_ds)
tau_da = tau_da.unstack("levels")


rte_input = tau_da.sum(dim="species").to_dataset(
    name="tau"
)  # .rename_dims({"frequency": "gpt"})


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

rte_input["surface_source"] = xr.apply_ufunc(
    planck_nu,
    rte_input["frequency"],
    atm_ds["surface_temperature"],
    vectorize=True,
)
rte_input["surface_emissivity"] = atm_ds["surface_emissivity"]
rte_input["surface_source_jacobian"] = xr.zeros_like(rte_input["surface_source"])
rte_input.attrs["top_at_1"] = True

rte_input = rte_input.expand_dims({"gpt": 1}, axis=-1)

# %%
fluxes = rte_input.rte.solve(add_to_input=False)

lw_ddq_loc = "../data/ddq/DDQ_LW.h5"
kayser_quadrature = xr.load_dataset(lw_ddq_loc)

fluxes["weights_hz"] = ("frequency", kayser_to_hz(kayser_quadrature["W"].values))

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

from pyrte_rrtmgp.examples import RFMIP_FILES, load_example_file
from pyrte_rrtmgp.rrtmgp import GasOptics
from pyrte_rrtmgp.rrtmgp.data_files import (
    GasOpticsFiles,
)

gas_optics_lw = GasOptics(gas_optics_file=GasOpticsFiles.LW_G256)
gas_optics_sw = GasOptics(gas_optics_file=GasOpticsFiles.SW_G224)
atmosphere = load_example_file(RFMIP_FILES.ATMOSPHERE)
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

optical_props = gas_optics_lw.compute(
    atmosphere,
    add_to_input=False,
)
optical_props["surface_emissivity"] = atmosphere.surface_emissivity


lw_fluxes = optical_props.rte.solve(
    add_to_input=False,
)
lw_fluxes["brd_net_flux"] = lw_fluxes["lw_flux_down"] - lw_fluxes["lw_flux_up"]

lw_fluxes["global_mean_toa_flux"] = (
    lw_fluxes["brd_net_flux"].isel(level=0) * atm_ds["profile_weight"]
).sum(dim="site")

lw_fluxes.sel(level=0)
# %%
