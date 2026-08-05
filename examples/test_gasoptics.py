# %%
from pathlib import Path

import numpy as np
import xarray as xr

from faxsec.gas_optics import GasOptics
from faxsec.utils import (
    hz_to_kayser,
    kayser_to_hz,
    simple_vmr_profile,
)

# %% Initialise gas optics model with a frequency grid and species

lw_ddq_loc = "../data/ddq/DDQ_LW.h5"
kayser_quadrature_lw = xr.load_dataset(lw_ddq_loc)

kayser_quadrature = kayser_quadrature_lw

kayser_grid = kayser_quadrature["S"].values
kayser_weights = kayser_quadrature["W"].values

frequency_grid = kayser_to_hz(kayser_grid)

species = {
    "H2O": None,
    "CO2": ("CO2", "CO2-CKDMT252"),  # ,
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
    ),  # "N2-CIA-CH4" not available in pyarts3
}


absorbers = {}

dt = xr.open_datatree("../data/ff/test_2_lw.nc")

gas_optics = GasOptics.from_datatree(dt)
absorbers = gas_optics._absorbers

# dt_hr = dt["Hinge_Rational"]
# from faxsec.functional import FunctionalAbsorber
# for sp in species.keys():
#     ds = dt_hr.sel(species=sp)
#     func_abs = FunctionalAbsorber.from_dataset(ds=ds.to_dataset())
#     absorbers[sp] = func_abs

# dt_cont = dt["both_continuum_MT_CKD_4_3"]
# from faxsec.continuum import H2OContinuum
# absorbers["H2O" + "_both_Continuum"] = H2OContinuum.from_dataset(
#     ds=dt_cont.to_dataset()
# )


# dt_xfit = dt["XFIT"]
# from faxsec.xfit import CrossFitAbsorber
# for sp in ["CFC11", "CFC12"]:
#     ds = dt_xfit.sel(species=sp)
#     xfit_abs = CrossFitAbsorber.from_dataset(ds=ds.to_dataset())
#     absorbers[sp] = xfit_abs

# gas_optics = GasOptics.from_absorbers(
#     absorbers=absorbers,
# )


# %% Initialise an atmosphere dataset with pressure, temperature, and vmr for each species
unique_species = list(set(absorber.config.species for absorber in absorbers.values()))
pressure_levels = [1000e2, 900e2, 800e2, 700e2, 600e2, 500e2]  # in Pa
temperature_levels = [250.0, 260.0, 270.0, 280.0, 290.0, 300.0]  # in K

atmosphere_ds = xr.Dataset(
    {
        "pressure_layer": (("level"), pressure_levels),  # in Pa
        "temperature_layer": (("level"), temperature_levels),  # in K
    },
    coords={
        "level": np.arange(len(pressure_levels)),
    },
)
for sp in unique_species:
    vmr_profile = simple_vmr_profile(
        species=sp,
        pressure=np.array(pressure_levels),
        temperature=np.array(temperature_levels),
    )
    atmosphere_ds[sp] = (("level"), vmr_profile)

# %%

results = xr.Dataset()

results["abs_coef"] = gas_optics.absorption_from_ds(atmosphere_ds=atmosphere_ds)
results["xsec"] = gas_optics.cross_section_from_ds(atmosphere_ds=atmosphere_ds)


# %%
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

for sp in results.species.values:
    ax.scatter(
        hz_to_kayser(results["frequency"].values),
        results["xsec"].sel(species=sp, level=0).values,
        label=f"{sp}",
    )
ax.set_xlabel("Frequency (cm^-1)")
ax.set_ylabel("Cross Section (m^2)")
ax.set_yscale("log")
ax.legend()
ax.set_ylim(1e-30, 1e-20)
# %%
