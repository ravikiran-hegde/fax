# %%

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarts3
import xarray as xr
from matplotlib import pyplot as plt

from faxsec.utils import hz_to_kayser, kayser_to_hz

species = "O3"


def _extract_band_name(name: str) -> int:
    prefix = name.removesuffix("_coeffs")
    if not prefix.startswith("band"):
        raise ValueError(f"Unexpected band variable name: {name}")
    return int(prefix.removeprefix("band"))


def xsec_xml_to_dataset(xml_path: str | Path) -> xr.Dataset:
    """Load an ARTS XsecRecord XML file into an xarray Dataset.

    The XML is first read with ``pyarts3.xml.load(...).to_xarray()`` and then
    flattened into a single monotonic frequency axis because the bands in the
    record are disjoint.
    """

    xml_path = Path(xml_path)
    source = pyarts3.xml.load(str(xml_path)).to_xarray()

    species = source.attrs.get("species")
    if species is None:
        raise ValueError(f"Species metadata missing in {xml_path}")

    band_names = [name for name in source.data_vars if name.endswith("_coeffs")]
    if not band_names:
        raise ValueError(f"No band coefficient variables found in {xml_path}")

    band_names = sorted(band_names, key=_extract_band_name)
    frequency_blocks: list[np.ndarray] = []
    coefficient_blocks: list[np.ndarray] = []
    n_frequency: list[int] = []

    band_start_frequency: list[float] = []
    band_end_frequency: list[float] = []
    min_pressures = np.asarray(source["fitminpressures"].values, dtype=float)
    max_pressures = np.asarray(source["fitmaxpressures"].values, dtype=float)
    min_temperatures = np.asarray(source["fitmintemperatures"].values, dtype=float)
    max_temperatures = np.asarray(source["fitmaxtemperatures"].values, dtype=float)
    coefficient_names = np.asarray(source[band_names[0]].coords["coeffs"].values)

    previous_stop: float | None = None
    for band_name in band_names:
        band_index = _extract_band_name(band_name)
        band_frequency_dim = next(
            dim for dim in source[band_name].dims if dim != "coeffs"
        )
        band_frequency = np.asarray(
            source.coords[band_frequency_dim].values, dtype=float
        )
        band_coefficients = np.asarray(source[band_name].values, dtype=float)

        if band_frequency.size == 0:
            raise ValueError(f"Empty frequency grid in {band_name} of {xml_path}")
        if previous_stop is not None and band_frequency[0] <= previous_stop:
            raise ValueError(
                "Fit coefficient bands overlap or are not strictly ordered"
            )
        previous_stop = float(band_frequency[-1])

        frequency_blocks.append(band_frequency)
        coefficient_blocks.append(band_coefficients)
        n_frequency.append(band_frequency.size)
        band_start_frequency.append(float(band_frequency[0]))
        band_end_frequency.append(float(band_frequency[-1]))

    frequency = np.concatenate(frequency_blocks)
    fit_coefficients = np.concatenate(coefficient_blocks, axis=0)
    band_id = np.concatenate(
        [np.full(size, idx, dtype=int) for idx, size in enumerate(n_frequency)]
    )

    return xr.Dataset(
        {
            "fit_coefficients": (("frequency", "coefficient"), fit_coefficients),
            "band_id": (("frequency",), band_id),
            "min_pressure": (("band",), min_pressures),
            "max_pressure": (("band",), max_pressures),
            "min_temperature": (("band",), min_temperatures),
            "max_temperature": (("band",), max_temperatures),
            "band_start_frequency": (
                ("band",),
                np.asarray(band_start_frequency, dtype=float),
            ),
            "band_end_frequency": (
                ("band",),
                np.asarray(band_end_frequency, dtype=float),
            ),
            "n_frequency": (("band",), np.asarray(n_frequency, dtype=int)),
        },
        coords={
            "frequency": frequency,
            "band": np.arange(len(frequency_blocks)),
            "coefficient": coefficient_names,
            "species": species,
        },
        attrs={
            "source": str(xml_path),
            "format": "ARTS XsecRecord via pyarts3",
        },
    )


halocarbon = xsec_xml_to_dataset(
    f"/Users/rk/.cache/arts/arts-cat-data-3.0.0dev8/xsec/{species}-XFIT.xml"
)

# %%


def compute_xsec(ds, p, T):
    p00, p10, p01, p20 = ds.fit_coefficients.sel(
        coefficient=["p00", "p10", "p01", "p20"]
    ).T

    xsec = p00 + p10 * T + p20 * T**2 + p01 * p

    # Check for negative values and remove them without introducing bias, meaning
    # the integral over the spectrum must not change. Not necessary.
    logic = xsec < 0
    if np.sum(logic) > 0:

        # original sum over spectrum
        sumX_org = np.sum(xsec)

        # remove negative values
        xsec[logic] = 0

        if sumX_org >= 0:
            # estimate ratio between altered and original sum of spectrum
            w = sumX_org / np.sum(xsec)

            # scale altered spectrum
            xsec = xsec * w

    return xsec


# %%

from faxsec.arts import ARTSAbsorber
from faxsec.xfit import CrossFitAbsorber

lw_ddq_loc = "../data/ddq/DDQ_LW.h5"
kayser_quadrature_lw = xr.load_dataset(lw_ddq_loc)

sw_ddq_loc = "../data/ddq/DDQ_SW.h5"
kayser_quadrature_sw = xr.load_dataset(sw_ddq_loc)
# %%

from __future__ import annotations

from pathlib import Path

    species=species,
    frequency_grid=frequency_grid,
    data_source=f"../data/halocarbon/{species}-XFIT.xml",
)


plt.figure(figsize=(10, 6))


arts_abs = ARTSAbsorber(
    species=species,
    frequency_grid=halocarbon.frequency.values,
    arts_tag=(f"{species}-XFIT",),
)
p = [100, 500e2, 1e5]
T = [250, 273, 300]
import scipy

for i in range(len(p)):
    if i == 0:
        xsec = compute_xsec(halocarbon, p=p[i], T=T[i])
        interp = scipy.interpolate.interp1d(
            halocarbon.frequency.values,
            xsec,
            kind="cubic",
            fill_value=0,
            bounds_error=False,
        )
        xsec_intep = interp(frequency_grid)
        arts_xsec = arts_abs.cross_section(
            pressure=np.array([p[i]]),
            temperature=np.array([T[i]]),
            vmr=np.array([0.01]),
        )[0]
        model_xsec = halocarbon_absorber.cross_section(
            pressure=np.array([p[i]]),
            temperature=np.array([T[i]]),
            vmr=np.array([0.01]),
        )

        plt.scatter(
            hz_to_kayser(halocarbon.frequency.values),
            xsec,
            s=0.001,
            alpha=0.5,
            label="XFIT Data",
        )
        # plt.scatter(hz_to_kayser(arts_abs.config.frequency_grid), xsec_intep, s=100, alpha=1, label="XFIT Interp")
        plt.scatter(
            hz_to_kayser(arts_abs.config.frequency_grid),
            arts_xsec,
            s=0.01,
            alpha=0.5,
            label="ARTS XFIT",
        )
        plt.scatter(
            hz_to_kayser(halocarbon_absorber.config.frequency_grid),
            model_xsec,
            s=kayser_quadrature["W"].values / 10,
            alpha=0.5,
            label="Model",
        )

plt.yscale("log")
# plt.xlim(10,6500)
plt.xlabel("Frequency (cm^-1)")
plt.ylabel("Cross Section")
plt.ylim(1e-35, 1e-20)
plt.legend()
# %%


# %%


# %%
