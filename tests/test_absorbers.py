"""MT_CKD continuum, XFIT halocarbons and dataset-level optical depth."""

import numpy as np
import pytest
import xarray as xr

from artsfax import H2OContinuum, load_gas_optics
from artsfax.continuum import ForeignContinuumAbsorber, SelfContinuumAbsorber


@pytest.fixture(scope="module")
def shipped():
    return load_gas_optics("LW")


@pytest.fixture(scope="module")
def frequency(shipped):
    return np.asarray(next(iter(shipped.absorbers.values())).config.frequency_grid)


def test_self_continuum_scales_with_water_partial_pressure(frequency):
    absorber = SelfContinuumAbsorber("H2O", frequency)
    p, t = np.array([8.0e4, 8.0e4]), np.array([280.0, 280.0])
    xsec = absorber.cross_section(p, t, np.array([0.005, 0.010]))
    np.testing.assert_allclose(xsec[1], 2.0 * xsec[0], rtol=1e-12)


def test_foreign_continuum_scales_with_dry_partial_pressure(frequency):
    absorber = ForeignContinuumAbsorber("H2O", frequency)
    t = np.array([280.0, 280.0])
    xsec = absorber.cross_section(np.array([4.0e4, 8.0e4]), t, np.array([0.0, 0.0]))
    np.testing.assert_allclose(xsec[1], 2.0 * xsec[0], rtol=1e-12)


def test_continuum_is_self_plus_foreign_and_positive(frequency):
    p, t, vmr = np.array([1.0e5, 5.0e4]), np.array([290.0, 250.0]), np.array([0.02, 0.002])
    total = H2OContinuum(frequency_grid=frequency).cross_section(p, t, vmr)
    parts = SelfContinuumAbsorber("H2O", frequency).cross_section(
        p, t, vmr
    ) + ForeignContinuumAbsorber("H2O", frequency).cross_section(p, t, vmr)
    np.testing.assert_allclose(total, parts, rtol=1e-12)
    assert np.all(np.isfinite(total)) and np.all(total >= 0.0) and total.max() > 0.0


@pytest.mark.parametrize("species", ["CFC11", "CFC12"])
def test_halocarbon_fits_are_physical_and_band_limited(shipped, species):
    absorber = shipped.absorbers[f"{species}_XFIT"]
    p = np.array([1.0e2, 1.0e4, 1.0e5])
    t = np.array([210.0, 230.0, 290.0])
    xsec = absorber.cross_section(p, t, np.full(p.size, 5e-10))
    assert np.all(np.isfinite(xsec)) and np.all(xsec >= 0.0)
    absorbing = xsec.max(axis=0) > 0.0
    assert absorbing.any() and not absorbing.all(), "absorbs in its bands only"


def three_layer_atmosphere(cfc11=2.5e-10):
    return xr.Dataset(
        {
            "pressure_layer": ("layer", [2.0e3, 3.0e4, 9.0e4]),
            "temperature_layer": ("layer", [220.0, 240.0, 285.0]),
            "N_per_m2_dry": ("layer", [4.0e27, 1.5e28, 3.0e28]),
            "H2O": ("layer", [5e-6, 5e-4, 1e-2]),
            "CO2": ("layer", [4.2e-4] * 3),
            "CFC11": ("layer", [cfc11] * 3),
        }
    )


def test_optical_depth_covers_only_gases_in_the_atmosphere(shipped):
    tau = shipped.optical_depth_from_ds(three_layer_atmosphere())
    names = set(tau["species"].values)
    assert any(n.startswith("CO2_") for n in names)
    assert not any(n.startswith("CH4_") for n in names), "CH4 is not in the atmosphere"
    assert tau.sizes["layer"] == 3
    assert np.all(np.isfinite(tau)) and np.all(tau >= 0.0)


def test_optical_depth_is_proportional_to_amount(shipped):
    tau = shipped.optical_depth_from_ds(three_layer_atmosphere(2.5e-10))
    doubled = shipped.optical_depth_from_ds(three_layer_atmosphere(5.0e-10))
    np.testing.assert_allclose(
        doubled.sel(species="CFC11_XFIT"), 2.0 * tau.sel(species="CFC11_XFIT"), rtol=1e-12
    )
