"""Create, train, save and reload a gas optics, without ARTS."""

import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

import faxsec
from faxsec import (
    FunctionalAbsorber,
    GasOptics,
    H2OContinuum,
    NoLogFunctionalAbsorber,
    load_gas_optics,
)
from faxsec.gas_optics import write_datatree

REF_P, REF_T = 1.0e4, 240.0


@pytest.fixture(scope="module")
def halocarbon():
    """Shipped XFIT absorber; its grid is used throughout."""
    return load_gas_optics("LW").absorbers["CFC11_XFIT"]


@pytest.fixture(scope="module")
def frequency(halocarbon):
    return np.asarray(halocarbon.config.frequency_grid)


def pressure_power_law(x, n_freq):
    """Linear in ln p: in the log formulation's family."""
    return x[:, None] ** np.linspace(0.3, 1.0, n_freq)


def pressure_lorentz(x, n_freq):
    """Lorentz line at distance d, 1 at x = 1: in the nolog formulation's family."""
    d2 = np.geomspace(0.05, 5.0, n_freq) ** 2
    return (d2 + 1.0) * x[:, None] / (d2 + x[:, None] ** 2)


def write_reference(path, frequency, pressure_shape):
    rng = np.random.default_rng(0)
    n_case = 120
    pressure = np.geomspace(10.0, 1.0e5, n_case)
    temperature = rng.uniform(190.0, 300.0, n_case)
    pressure[0], temperature[0] = REF_P, REF_T
    strength = np.geomspace(1e-26, 1e-22, frequency.size)
    xsec = (
        strength
        * pressure_shape(pressure / REF_P, frequency.size)
        * np.exp(-(temperature[:, None] - REF_T) / 80.0)
    )
    xr.Dataset(
        {
            "xsec": (("case", "frequency"), xsec),
            "pressure": ("case", pressure),
            "temperature": ("case", temperature),
            "vmr": ("case", np.full(n_case, 1e-9)),
        },
        coords={"case": np.arange(n_case), "frequency": frequency},
    ).to_netcdf(path)
    return path


@pytest.fixture(scope="module")
def references(tmp_path_factory, frequency):
    folder = tmp_path_factory.mktemp("reference")
    return {
        "H2O_Hinge_Rational": write_reference(
            folder / "log.nc", frequency, pressure_power_law
        ),
        "H2O_ShiftedReciprocalLaurent_Rational": write_reference(
            folder / "nolog.nc", frequency, pressure_lorentz
        ),
    }


def trained(cls, frequency, reference, **forms):
    absorber = cls(
        species="H2O",
        frequency_grid=frequency,
        ref_pressure=REF_P,
        ref_temperature=REF_T,
        ref_vmr=1e-9,
        **forms,
    )
    absorber.train(reference_xsec=reference, max_iter=4)
    return absorber


@pytest.fixture(scope="module")
def gas_optics(frequency, references, halocarbon):
    absorbers = [
        trained(FunctionalAbsorber, frequency, references["H2O_Hinge_Rational"]),
        trained(
            NoLogFunctionalAbsorber,
            frequency,
            references["H2O_ShiftedReciprocalLaurent_Rational"],
            pressure_form_name="ShiftedReciprocalLaurent",
        ),
        H2OContinuum(frequency_grid=frequency),
        halocarbon,
    ]
    return GasOptics.from_absorbers(
        {f"{a.config.species}_{a.class_name}": a for a in absorbers}
    )


def evaluation_points():
    p = np.array([50.0, 3.0e3, 2.0e4, 8.0e4])
    t = np.array([210.0, 225.0, 250.0, 290.0])
    vmr = np.array([1e-6, 1e-5, 1e-3, 1e-2])
    return p, t, vmr


def test_training_follows_the_reference(gas_optics, references):
    for name, path in references.items():
        ref = xr.open_dataset(path)
        p, t, y = (ref[k].values for k in ("pressure", "temperature", "xsec"))
        model = gas_optics.absorbers[name].cross_section(p, t, np.full(p.size, 1e-9))
        rel = np.abs(model / y - 1.0)
        # the rational T form is about 2 % off the exponential at the range ends
        assert np.percentile(rel, 95) < 0.03, f"{name}: p95 {np.percentile(rel, 95):.4f}"


def test_save_and_reload_is_exact(gas_optics, tmp_path):
    path = tmp_path / "gas_optics.nc"
    write_datatree(gas_optics.to_datatree(), path)
    with xr.open_datatree(path) as tree:
        reloaded = GasOptics.from_datatree(tree.load())

    assert reloaded.absorbers.keys() == gas_optics.absorbers.keys()
    p, t, vmr = evaluation_points()
    for name, original in gas_optics.absorbers.items():
        again = reloaded.absorbers[name]
        assert type(again) is type(original), name
        np.testing.assert_array_equal(
            again.cross_section(p, t, vmr), original.cross_section(p, t, vmr), name
        )


def test_shipped_model_loads_without_arts():
    code = (
        "import sys; sys.modules['pyarts3'] = None\n"
        "import faxsec\n"
        "print(len(faxsec.load_gas_optics('SW').absorbers))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert int(result.stdout.split()[-1]) > 0


def test_public_names_are_exported():
    for name in faxsec.__all__:
        assert hasattr(faxsec, name), name
