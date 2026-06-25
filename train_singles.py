# %%
import numpy as np
import xarray as xr

from model.utils import kayser_to_hz

lw_ddq_loc = "./data/ddq/DDQ_LW.h5"
kayser_quadrature_lw = xr.load_dataset(lw_ddq_loc)


sw_ddq_loc = "./data/ddq/DDQ_SW.h5"
kayser_quadrature_sw = xr.load_dataset(sw_ddq_loc)

# kayser_quadrature = xr.concat([kayser_quadrature_lw, kayser_quadrature_sw], dim="S")
kayser_quadrature = kayser_quadrature_lw

kayser_grid = kayser_quadrature["S"].values
kayser_weights = kayser_quadrature["W"].values

from model.arts import ARTSAbsorber
from model.single_absorber import FunctionalAbsorber

# def load_optimized_flux_quadrature(
#     quadrature_dir: (
#         str | None
#     ) = "/Users/rk/.cache/arts/arts-xml-data-3.0.0dev8/planets/Earth/Optimized-Flux-Frequencies",
#     band: str | None = "LW",
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """Load optimized flux frequencies and weights from ARTS XML files.

#     Returns
#     -------
#     tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
#         (f_grid_hz, kayser_grid, weights_hz, weights_kayser)
#     """

#     import pyarts3
#     from pathlib import Path
#     from model.utils import hz_to_kayser
#     from model.constants import CM_TO_M, LIGHT_SPEED
#     base = Path(quadrature_dir)
#     prefix = str(band).upper()
#     f_path = base / f"{prefix}-flux-optimized-f_grid.xml"
#     w_path = base / f"{prefix}-flux-optimized-quadrature_weights.xml"
#     # Keep explicit Python references and copy data into NumPy-owned memory.
#     # ARTS vectors can otherwise be views to temporary buffers.
#     f_vec = pyarts3.xml.load(str(f_path))
#     w_vec = pyarts3.xml.load(str(w_path))
#     f_grid_hz = np.array(f_vec, dtype=float, copy=True)
#     weights_hz = np.array(w_vec, dtype=float, copy=True)

#     if f_grid_hz.ndim != 1 or weights_hz.ndim != 1:
#         raise ValueError("Optimized quadrature f_grid and weights must be 1D vectors")
#     if f_grid_hz.shape != weights_hz.shape:
#         raise ValueError("Optimized quadrature frequencies and weights must match")

#     kayser_grid = hz_to_kayser(f_grid_hz)
#     weights_kayser = weights_hz / (CM_TO_M * LIGHT_SPEED)
#     return f_grid_hz, kayser_grid, weights_hz, weights_kayser

# _, kayser_grid, _, kayser_weights = load_optimized_flux_quadrature(band="LW")


# %%
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
    ),  # "N2-CIA-CH4" not available in ARTS
    # "CFC11": ("CFC11-XFIT",),
    # "CFC12": ("CFC12-XFIT",)
}


absorbers = {}
arts_absorbers = {}
for sp in species.keys():
    func_abs = FunctionalAbsorber(
        species=sp,
        pressure_form_name="Hinge",
        temperature_form_name="Rational",
        frequency_grid=kayser_to_hz(kayser_grid),
    )

    func_abs.train(arts_reference_kwargs={"arts_tag": species[sp]})
    absorbers[sp] = func_abs

    arts_absorbers[sp] = ARTSAbsorber(
        species=sp,
        frequency_grid=kayser_to_hz(kayser_grid),
        arts_tag=species[sp],
    )

# %%
from model.continuum import H2OContinuum

h2o_cont = H2OContinuum(
    frequency_grid=kayser_to_hz(kayser_grid),
    data_source="./data/continuum/absco-ref_wv-mt-ckd400.nc",
)
absorbers["H2O_continuum"] = h2o_cont

# %%
from model.halocarbon import HalocarbonAbsorber

cfc11_absorber = HalocarbonAbsorber(
    species="CFC11",
    frequency_grid=kayser_grid,
    data_source="./data/halocarbon/CFC11-XFIT.xml",
)

absorbers["CFC11"] = cfc11_absorber

cfc12_absorber = HalocarbonAbsorber(
    species="CFC12",
    frequency_grid=kayser_grid,
    data_source="./data/halocarbon/CFC12-XFIT.xml",
)

absorbers["CFC12"] = cfc12_absorber
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

datatree.to_netcdf("./data/ff/test_4.nc", mode="w")
# %%
# Test loading data into functional absorber
from model.single_absorber import FunctionalAbsorber

dt = xr.open_datatree("./data/ff/test_4.nc")

# %%
dt_hr = dt["Hinge_Rational"]
absorbers = {}
for sp in dt_hr.species.values:
    ds = dt_hr.sel(species=sp)
    func_abs = FunctionalAbsorber.from_dataset(ds=ds.to_dataset())
    absorbers[sp] = func_abs

# %%

import matplotlib.pyplot as plt

# p = np.array([5e4, 1000e2, 1e2])
# t = np.array([250, 273, 300])

p = [500e2]
t = [250]


def simple_vmr_profile(
    species: str, pressure: np.ndarray, temperature: np.ndarray
) -> np.ndarray:
    """Return a simple level-wise VMR profile for one species."""
    pressure = np.asarray(pressure, dtype=float)
    temperature = np.asarray(temperature, dtype=float)

    if  "H2O" in species:
        vmr = np.full_like(pressure, 0.001, dtype=float)
    elif species == "CO2":
        vmr = np.full_like(pressure, 4.2e-4, dtype=float)
    elif species == "O2":
        vmr = np.full_like(pressure, 0.2095, dtype=float)
    elif species == "CH4":
        vmr = np.full_like(pressure, 1.9e-6, dtype=float)
    elif species == "N2O":
        vmr = np.full_like(pressure, 3.3e-7, dtype=float)
    elif species == "N2":
        vmr = np.full_like(pressure, 0.7808, dtype=float)
    elif species == "CFC11":
        vmr = np.full_like(pressure, 2.5e-10, dtype=float)
    elif species == "CFC12":
        vmr = np.full_like(pressure, 5.0e-10, dtype=float)
    else:
        vmr = np.full_like(pressure, 1e-9, dtype=float)

    return vmr


species_order = list(absorbers.keys())
vmr_matrix = np.column_stack([simple_vmr_profile(sp, p, t) for sp in species_order])
species_vmr = {sp: vmr_matrix[:, i] for i, sp in enumerate(species_order)}

for i in range(len(p)):
    fig, ax = plt.subplots(figsize=(10, 12), nrows=2, sharex=True)

    for sp in absorbers:
        v = species_vmr[sp]
        pr = np.array([p[i]])
        tmp = np.array([t[i]])
        vmr = np.array([v[i]])
        xsec = absorbers[sp].cross_section(pr, tmp, vmr).clip(1e-30, 1e-20)
        xsec_arts = arts_absorbers[sp].cross_section(pr, tmp, vmr).clip(1e-30, 1e-20)
        ax[0].plot(
            kayser_grid,
            xsec[0],
            # s=kayser_quadrature["W"].values / 10,
            label=sp,
            ls="--",
            lw=0.001,
            marker="x",
        )

        ax[0].scatter(
            kayser_grid,
            xsec_arts[0],
            s=kayser_weights / 2,
            alpha=0.5,
        )

        ax[1].scatter(
            kayser_grid,
            (xsec[0] - xsec_arts[0]) * 100 / xsec_arts[0],
            label=sp,
            # s=kayser_weights / 2,
        )

    ax[0].set_ylabel("Cross-section (m²)")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-30, 1e-20)
    ax[0].legend()

    ax[1].legend()
    ax[1].set_xlabel("Frequency (cm⁻¹)")
    ax[1].set_ylabel("% Error")
    # ax[1].set_yscale("log", base=10)
    plt.suptitle(f"Pressure: {p[i]:.2e} Pa, Temperature: {t[i]:.1f} K")
    plt.tight_layout()
    plt.show()

# %%
