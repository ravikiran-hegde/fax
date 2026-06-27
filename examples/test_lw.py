# %%
from pathlib import Path

import numpy as np
import xarray as xr

from model.utils import ensure_reference_dataset, kayser_to_hz, reference_cache_path

lw_ddq_loc = "./data/ddq/DDQ_LW.h5"
kayser_quadrature_lw = xr.load_dataset(lw_ddq_loc)

kayser_quadrature = kayser_quadrature_lw

kayser_grid = kayser_quadrature["S"].values
kayser_weights = kayser_quadrature["W"].values


frequency_grid = kayser_to_hz(kayser_grid)

from model.arts import ARTSAbsorber
from model.functional import FunctionalAbsorber

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
    ),  # "N2-CIA-CH4" not available in pyarts3
    # "CFC11": ("CFC11-XFIT",),
    # "CFC12": ("CFC12-XFIT",)
}


absorbers = {}
arts_absorbers = {}
reference_cache_dir = Path("./data/reference_lw")

for sp in species.keys():
    ensure_reference_dataset(
        species=sp,
        frequency_grid=frequency_grid,
        arts_tag=species[sp],
        cache_path=reference_cache_path(
            species=sp,
            arts_tag=species[sp],
            cache_dir=reference_cache_dir,
        ),
    )

for sp in species.keys():
    func_abs = FunctionalAbsorber(
        species=sp,
        pressure_form_name="Hinge",
        temperature_form_name="Rational",
        frequency_grid=frequency_grid,
    )

    func_abs.train(
        reference_xsec=str(
            reference_cache_path(
                species=sp,
                arts_tag=species[sp],
                cache_dir=reference_cache_dir,
            )
        )
    )
    absorbers[sp] = func_abs

    arts_absorbers[sp] = ARTSAbsorber(
        species=sp,
        frequency_grid=frequency_grid,
        arts_tag=species[sp],
    )

# %%
from model.continuum import H2OContinuum

h2o_cont = H2OContinuum(
    frequency_grid=frequency_grid,
    data_source="./data/continuum/absco-ref_wv-mt-ckd400.nc",
)
absorbers["H2O_continuum"] = h2o_cont

arts_absorbers["H2O_continuum"] = ARTSAbsorber(
    species="H2O",
    frequency_grid=frequency_grid,
    arts_tag=(
        "H2O-ForeignContCKDMT400",
        "H2O-SelfContCKDMT400",
    ),
)
# %%
from model.xfit import CrossFitAbsorber

cfc11_absorber = CrossFitAbsorber(
    species="CFC11",
    frequency_grid=frequency_grid,
    data_source="./data/halocarbon/CFC11-XFIT.xml",
)

absorbers["CFC11"] = cfc11_absorber
# arts_absorbers["CFC11"] = ARTSAbsorber(
#     species="CFC11",
#     frequency_grid=frequency_grid,
#     arts_tag=("CFC11-XFIT",),
# )
cfc12_absorber = CrossFitAbsorber(
    species="CFC12",
    frequency_grid=frequency_grid,
    data_source="./data/halocarbon/CFC12-XFIT.xml",
)

absorbers["CFC12"] = cfc12_absorber
# arts_absorbers["CFC12"] = ARTSAbsorber(
#     species="CFC12",
#     frequency_grid=frequency_grid,
#     arts_tag=("CFC12-XFIT",),
# )
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

datatree.to_netcdf("./data/ff/test_2_lw.nc", mode="w")
# %%
# Test loading data into functional absorber
# from model.single_absorber import FunctionalAbsorber

# dt = xr.open_datatree("./data/ff/test_4.nc")

# dt_hr = dt["Hinge_Rational"]
# absorbers = {}
# for sp in dt_hr.species.values:
#     ds = dt_hr.sel(species=sp)
#     func_abs = FunctionalAbsorber.from_dataset(ds=ds.to_dataset())
#     absorbers[sp] = func_abs

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

    if "H2O" in species:
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
vmr_matrix = np.column_stack(
    [simple_vmr_profile(absorbers[sp].config.species, p, t) for sp in species_order]
)
species_vmr = {
    absorbers[sp].config.species: vmr_matrix[:, i] for i, sp in enumerate(species_order)
}

for i in range(len(p)):
    fig, ax = plt.subplots(figsize=(10, 12), nrows=2, sharex=True)

    for sp in absorbers:
        v = species_vmr[absorbers[sp].config.species]
        pr = np.array([p[i]])
        tmp = np.array([t[i]])
        vmr = np.array([v[i]])
        xsec = absorbers[sp].cross_section(pr, tmp, vmr)
        xsec_arts = (
            arts_absorbers[sp].cross_section(pr, tmp, vmr)
            if sp in arts_absorbers
            else np.zeros_like(xsec) * np.nan
        )
        ax[0].plot(
            kayser_grid,
            xsec_arts[0],
            # s=kayser_quadrature["W"].values / 10,
            ls="--",
            lw=0.001,
            marker="x",
        )

        ax[0].scatter(
            kayser_grid,
            xsec[0],
            s=kayser_weights / 2,
            alpha=0.5,
        )

        ax[1].scatter(
            kayser_grid,
            # (xsec[0] - xsec_arts[0]) * 100 / xsec_arts[0],
            np.log10(xsec[0] / xsec_arts[0]),
            label=sp,
            alpha=0.5,
            s=kayser_weights / 2,
        )

    ax[0].set_ylabel("Cross-section (m²)")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-30, 1e-20)
    # ax[0].legend(markerscale=1)

    ax[1].legend()
    ax[1].set_xlabel("Frequency (cm⁻¹)")
    ax[1].set_ylabel("Log10 ratio (model / ARTS)")

    # ax[1].set_yscale("log", base=10)
    plt.suptitle(f"Pressure: {p[i]:.2e} Pa, Temperature: {t[i]:.1f} K")
    plt.tight_layout()
    plt.show()


# %%
