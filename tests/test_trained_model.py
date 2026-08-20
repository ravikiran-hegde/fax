"""Checks on a trained gas optics: the failure modes that stay silent.

A bad fit shows up as a slightly wrong flux and is easy to miss. These are the
ways it can be wrong catastrophically instead, or wrong in a way that cancels.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from faxsec.constants import REFERENCE_VMR
from faxsec.functional import FunctionalAbsorber
from faxsec.gas_optics import GasOptics
from faxsec.utils import atmospheric_t_range, hz_to_kayser

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BANDS = ("LW", "SW")

# Nothing in the atmosphere absorbs more strongly than this per molecule.
MAX_PHYSICAL_XSEC = 1e-18


def datatree_path(band: str) -> Path:
    return DATA_DIR / "ff" / f"gas_optics_DDQ_{band}.nc"


def open_datatree(band: str) -> xr.DataTree:
    path = datatree_path(band)
    if not path.exists():
        pytest.skip(f"no trained gas optics at {path}")
    with xr.open_datatree(path) as tree:
        return tree.load()


def load(band: str) -> GasOptics:
    return GasOptics.from_datatree(open_datatree(band))


def realizable_grid(n_p: int = 60, n_t: int = 40):
    p = np.geomspace(1.0, 1.05e5, n_p)
    t = np.linspace(160.0, 330.0, n_t)
    pp, tt = np.meshgrid(p, t, indexing="ij")
    t_min, t_max = atmospheric_t_range(p)
    inside = (tt >= t_min[:, None]) & (tt <= t_max[:, None])
    return pp[inside], tt[inside]


@pytest.mark.parametrize("band", BANDS)
def test_cross_sections_stay_physical(band):
    """A pole or a runaway extrapolation shows up as an absurd cross-section,
    which would black out a frequency rather than merely bias it."""
    gas_optics = load(band)
    p, t = realizable_grid()
    for name, absorber in gas_optics.absorbers.items():
        vmr = np.full(p.size, REFERENCE_VMR.get(absorber.config.species, 1e-9))
        xsec = absorber.cross_section(p, t, vmr)
        assert np.all(np.isfinite(xsec)), name
        assert np.all(xsec >= 0.0), name
        assert xsec.max() < MAX_PHYSICAL_XSEC, f"{name}: max {xsec.max():.3e}"


@pytest.mark.parametrize("band", BANDS)
def test_temperature_form_has_no_pole(band):
    """The rational denominator must not approach zero anywhere it is evaluated."""
    ds = open_datatree(band)["Hinge_Rational"].to_dataset()
    x = np.linspace(160.0, 330.0, 400) - float(ds["ref_temperature"].values.flat[0])
    for species in ds["species"].values:
        coeffs = ds["temperature_coeffs"].sel(species=species).values
        denominator = 1.0 + np.outer(x, coeffs[3]) + np.outer(x**2, coeffs[4])
        assert denominator.min() > 0.0, f"{band} {species}"


@pytest.mark.parametrize("band", BANDS)
def test_cross_sections_do_not_explode_outside_the_envelope(band):
    """Not required to be accurate there, but a model that returns an absurd
    value just outside its domain is a hazard in a host model."""
    gas_optics = load(band)
    p = np.geomspace(0.01, 1.1e5, 50)
    t = np.linspace(150.0, 350.0, 30)
    pp, tt = np.meshgrid(p, t, indexing="ij")
    for name, absorber in gas_optics.absorbers.items():
        vmr = np.full(pp.size, REFERENCE_VMR.get(absorber.config.species, 1e-9))
        xsec = absorber.cross_section(pp.ravel(), tt.ravel(), vmr)
        assert np.all(np.isfinite(xsec)), name
        assert xsec.max() < 1e-15, f"{name}: max {xsec.max():.3e}"


@pytest.mark.parametrize("band", BANDS)
def test_cia_species_are_referenced_at_atmospheric_vmr(band):
    """Regression: N2 and O2 were referenced at REF_VMR=1e-9, where their own
    collision-induced absorption is effectively switched off."""
    ds = open_datatree(band)["Hinge_Rational"].to_dataset()
    for species, expected in REFERENCE_VMR.items():
        if species not in ds["species"].values:
            continue
        assert float(ds["ref_vmr"].sel(species=species).values) == pytest.approx(
            expected
        ), f"{band} {species}"


@pytest.mark.parametrize("band", BANDS)
def test_collision_induced_absorption_is_present(band):
    """The consequence of the above: CIA must carry real optical depth."""
    gas_optics = load(band)
    p = np.array([9.0e4])
    t = np.array([280.0])
    for species in ("N2", "O2"):
        absorber = gas_optics.absorbers.get(f"{species}_Hinge_Rational")
        if absorber is None:
            continue
        vmr = REFERENCE_VMR[species]
        xsec = absorber.cross_section(p, t, np.array([vmr]))[0]
        column = xsec * vmr * 2.1e29
        assert column.max() > 1e-3, (
            f"{band} {species}: peak column optical depth {column.max():.2e}"
        )


@pytest.mark.parametrize("band", BANDS)
def test_dataset_roundtrip_preserves_cross_sections(band):
    gas_optics = load(band)
    p, t = realizable_grid(12, 8)
    for name, absorber in gas_optics.absorbers.items():
        if not isinstance(absorber, FunctionalAbsorber):
            continue
        vmr = np.full(p.size, 1e-9)
        restored = FunctionalAbsorber.from_dataset(
            absorber.to_dataset().isel(species=0)
        )
        assert np.allclose(
            absorber.cross_section(p, t, vmr), restored.cross_section(p, t, vmr)
        ), name


@pytest.mark.parametrize("band", BANDS)
def test_frequency_grid_matches_the_quadrature(band):
    trained = open_datatree(band)["Hinge_Rational"].coords["frequency"].values
    quadrature = xr.load_dataset(DATA_DIR / "ddq" / f"DDQ_{band}.h5")["S"].values
    assert np.allclose(hz_to_kayser(trained), quadrature, rtol=1e-6)
