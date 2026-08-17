# %%

import xarray as xr

from faxsec.constants import DEFAULT_VMR
from faxsec.continuum import H2OContinuum
from faxsec.utils import kayser_to_hz, simple_vmr_profile

lw_ddq_loc = "/Users/rk/Work/faxsec/data/ddq/DDQ_LW.h5"
kayser_quadrature_lw = xr.load_dataset(lw_ddq_loc)

kayser_quadrature = kayser_quadrature_lw

kayser_grid = kayser_quadrature["S"].values
kayser_weights = kayser_quadrature["W"].values
frequency_grid = kayser_to_hz(kayser_grid)

absorbers = {}
arts_absorbers = {}


h2o_cont = H2OContinuum(
    frequency_grid=frequency_grid,
    data_source="/Users/rk/Work/faxsec/data/continuum/absco-ref_wv-mt-ckd.nc",
)
absorbers["H2O_continuum"] = h2o_cont

from faxsec.arts import PYARTS_VERSION, ARTSAbsorber

arts_absorbers["H2O_continuum"] = ARTSAbsorber(
    species="H2O",
    frequency_grid=frequency_grid,
    arts_tag=(
        "H2O-ForeignContCKDMT430",
        "H2O-SelfContCKDMT430",
    ),
)

import pyarts3
from pyarts3.recipe import SingleSpeciesAbsorption

pyarts3.data.download(version=PYARTS_VERSION)

cont = SingleSpeciesAbsorption(species="H2O-ForeignContCKDMT400")

atm = pyarts3.arts.AtmPoint()
for sp in DEFAULT_VMR:
    atm[sp] = DEFAULT_VMR[sp]


# %%

import matplotlib.pyplot as plt
import numpy as np

p = np.array([5e4, 1000e2, 1e2])
t = np.array([150, 273, 300])

p = [10]
t = [230.8]


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
        # ax[0].scatter(
        #     kayser_grid,
        #     abs,
        #     s=kayser_weights / 2,
        #     alpha=0.5,
        #     marker="^",
        # )

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
