# %%
import xarray as xr

from model.utils import kayser_to_hz

lw_ddq_loc = "./data/ddq/DDQ_LW.h5"
kayser_quadrature_lw = xr.load_dataset(lw_ddq_loc)


sw_ddq_loc = "./data/ddq/DDQ_SW.h5"
kayser_quadrature_sw = xr.load_dataset(sw_ddq_loc)

# kayser_quadrature = xr.concat([kayser_quadrature_lw, kayser_quadrature_sw], dim="S")
kayser_quadrature = kayser_quadrature_lw

kayser_grid = kayser_quadrature["S"].values


from model.single_absorber import FunctionalAbsorber

# %%
species = [
    "H2O",
    "CO2",
    "O3",
    "O2",
    # # "O2-CIA-O2",
    "CH4",
    "N2O",
    "N2",
    # # "N2-CIA-N2",
    # # "N2-CIA-CH4",
    # "CFC11-XFIT",
    # "CFC12-XFIT",
]


absorbers = {}
for sp in species:
    func_abs = FunctionalAbsorber(
        species=sp,
        pressure_form_name="Hinge",
        temperature_form_name="Rational",
        frequency_grid=kayser_to_hz(kayser_quadrature["S"].values),
    )

    func_abs.train()
    absorbers[sp] = func_abs

# %%
from model.continuum import H2OContinuum

h2o_cont = H2OContinuum(
    frequency_grid=kayser_to_hz(kayser_quadrature["S"].values),
    data_source="./data/continuum/absco-ref_wv-mt-ckd400.nc",
)
absorbers["H2O_continuum"] = h2o_cont
# %%

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

datatree.to_netcdf("./data/ff/test_2.nc", mode="w")
# %%
# Test loading data into functional absorber
from model.single_absorber import FunctionalAbsorber

dt = xr.open_datatree("./data/ff/test_2.nc")

# %%
dt_hr = dt["Hinge_Rational"]
absorbers = {}
for sp in dt_hr.species.values:
    ds = dt_hr.sel(species=sp)
    func_abs = FunctionalAbsorber.from_dataset(ds=ds.to_dataset())
    absorbers[sp] = func_abs

# %%

import matplotlib.pyplot as plt
import numpy as np

p = np.array([5e4])
t = np.array([250])
v = np.array([0.028])

for sp in absorbers:
    xsec = absorbers[sp].cross_section(p, t, v)
    plt.plot(
        kayser_grid,
        xsec[0],
        # s=kayser_quadrature["W"].values / 10,
        label=sp,
    )

plt.xlabel("Frequency (cm⁻¹)")
plt.ylabel("Cross-section (m²)")
plt.yscale("log")
plt.ylim(1e-32, 1e-20)
plt.legend()
plt.show()

# %%
