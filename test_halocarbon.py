# %%

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarts3
import xarray as xr
from matplotlib import pyplot as plt

from model.utils import kayser_to_hz, hz_to_kayser

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

cfc11 = xsec_xml_to_dataset(
    "/Users/rk/.cache/arts/arts-cat-data-3.0.0dev8/xsec/CFC11-XFIT.xml"
)

# %%


def compute_xsec(ds, p, T):
    p00, p10, p01, p20 = ds.fit_coefficients.sel(
        coefficient=["p00", "p10", "p01", "p20"]
    ).T
    return p00 + p10 * T + p20 * T**2 + p01 * p


# %%

from model.arts import ARTSAbsorber
from model.utils import hz_to_kayser


from model.halocarbon import HalocarbonAbsorber

lw_ddq_loc = "./data/ddq/DDQ_LW.h5"
kayser_quadrature_lw = xr.load_dataset(lw_ddq_loc)

sw_ddq_loc = "./data/ddq/DDQ_SW.h5"
kayser_quadrature_sw = xr.load_dataset(sw_ddq_loc)

kayser_quadrature = xr.concat([kayser_quadrature_lw, kayser_quadrature_sw], dim="S")
# kayser_quadrature  = kayser_quadrature_lw
kayser_grid = kayser_to_hz(kayser_quadrature["S"].values)

cfc11_absorber = HalocarbonAbsorber(
    species="CFC11",
    frequency_grid=kayser_grid,
    data_source="./data/halocarbon/CFC11-XFIT.xml",)


plt.figure(figsize=(10, 6))


arts_abs = ARTSAbsorber(
    species="CFC11",
    frequency_grid=cfc11.frequency.values,
    arts_tag=("CFC11-XFIT",),
)
p = [100, 500e2, 1e5]
T = [250, 273, 300]
for i in range(len(p)):
    # if i == 0:
    xsec = compute_xsec(cfc11, p=p[i], T=T[i])
    arts_xsec = arts_abs.cross_section(
        pressure=np.array([p[i]]), temperature=np.array([T[i]]), vmr=np.array([0.01])
    )[0]
    model_xsec = cfc11_absorber.cross_section(
        pressure=np.array([p[i]]), temperature=np.array([T[i]]), vmr=np.array([0.01])
    )

    # plt.scatter(hz_to_kayser(cfc11.frequency.values), xsec , s = 0.001, alpha = 0.5)
    plt.scatter(hz_to_kayser(cfc11.frequency.values), arts_xsec, s=0.001, alpha=0.5)
    plt.scatter(hz_to_kayser(cfc11_absorber.config.frequency_grid), model_xsec, s=kayser_quadrature["W"].values/10, alpha=0.5)

plt.yscale("log")
plt.xlim(10,6500)
plt.xlabel("Frequency (cm^-1)")
plt.ylabel("Cross Section")
plt.ylim(1e-35, 1e-20)
# %%


# %%


# %%

