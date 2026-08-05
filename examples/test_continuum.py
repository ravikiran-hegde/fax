# %%
from pathlib import Path

import xarray as xr

from model.continuum import H2OContinuum
from model.utils import kayser_to_hz

lw_ddq_loc = "../data/ddq/DDQ_LW.h5"
kayser_quadrature_lw = xr.load_dataset(lw_ddq_loc)

kayser_quadrature = kayser_quadrature_lw

kayser_grid = kayser_quadrature["S"].values
kayser_weights = kayser_quadrature["W"].values
frequency_grid = kayser_to_hz(kayser_grid)

absorbers = {}
arts_absorbers = {}


h2o_cont = H2OContinuum(
    frequency_grid=frequency_grid,
    data_source="../data/continuum/absco-ref_wv-mt-ckd400.nc",
)
absorbers["H2O_continuum"] = h2o_cont

from model.arts import PYARTS_VERSION, ARTSAbsorber

arts_absorbers["H2O_continuum"] = ARTSAbsorber(
    species="H2O",
    frequency_grid=frequency_grid,
    arts_tag=(
        "H2O-ForeignContCKDMT400",
        "H2O-SelfContCKDMT400",
    ),
)

import pyarts3
from pyarts3.recipe import SingleSpeciesAbsorption

pyarts3.data.download(version=PYARTS_VERSION)

cont = SingleSpeciesAbsorption(species="H2O-ForeignContCKDMT400, H2O-SelfContCKDMT400")
default_vmrs = {
    "N2": 0.7808,
    "O2": 0.2095,
    "CO2": 4.2e-4,
    "H2O": 6.29e-6,
    "CH4": 1.9e-6,
}

atm = pyarts3.arts.AtmPoint()
for sp in default_vmrs:
    atm[sp] = default_vmrs[sp]


# %%

import matplotlib.pyplot as plt
import numpy as np

p = np.array([5e4, 1000e2, 1e2])
t = np.array([150, 273, 300])

p = [10]
t = [230.8]


def simple_vmr_profile(
    species: str, pressure: np.ndarray, temperature: np.ndarray
) -> np.ndarray:
    """Return a simple level-wise VMR profile for one species."""
    pressure = np.asarray(pressure, dtype=float)
    temperature = np.asarray(temperature, dtype=float)

    if "H2O" in species:
        vmr = np.full_like(pressure, 6.29e-6, dtype=float)
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
    atm.pressure = float(p[i])
    atm.temperature = float(t[i])
    for sp in absorbers:
        v = species_vmr[absorbers[sp].config.species]
        pr = np.array([p[i]])
        tmp = np.array([t[i]])
        vmr = np.array([v[i]])
        xsec = absorbers[sp].cross_section(pr, tmp, vmr) * atm.number_density("H2O")
        xsec_arts = (
            arts_absorbers[sp].cross_section(pr, tmp, vmr)
            if sp in arts_absorbers
            else np.zeros_like(xsec) * np.nan
        ) * atm.number_density("H2O")

        abs = cont(frequency_grid, atm)
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
        ax[0].scatter(
            kayser_grid,
            abs,
            s=kayser_weights / 2,
            alpha=0.5,
            marker="^",
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
    # ax[0].set_ylim(1e-30, None)
    # ax[0].legend(markerscale=1)

    ax[1].legend()
    ax[1].set_xlabel("Frequency (cm⁻¹)")
    ax[1].set_ylabel("Log10 ratio (model / ARTS)")

    # ax[1].set_yscale("log", base=10)
    plt.suptitle(f"Pressure: {p[i]:.2e} Pa, Temperature: {t[i]:.1f} K")
    plt.tight_layout()
    plt.show()


# %%
