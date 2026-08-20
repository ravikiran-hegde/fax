"""Training-sample generation, and that its settings actually take effect."""

import numpy as np
import pytest

from faxsec.utils import (
    ATMOSPHERIC_T_ENVELOPE,
    atmospheric_t_range,
    ensure_reference_dataset,
    sample_atmospheres,
    xsec_relevance_floor,
)

P_RANGE = [1.0, 1.05e5]


def test_atmospheric_samples_stay_inside_the_envelope():
    p, t = sample_atmospheres(
        p_range=P_RANGE, N_samples=2000, method="atmospheric", seed=0
    )
    assert p.min() >= P_RANGE[0] and p.max() <= P_RANGE[1] * (1 + 1e-9)
    t_min, t_max = atmospheric_t_range(p)
    assert np.all(t >= t_min - 1e-9)
    assert np.all(t <= t_max + 1e-9)


def test_pressure_weight_moves_samples_towards_the_mass():
    fractions = []
    for weight in (0.0, 0.35, 0.7):
        p, _ = sample_atmospheres(
            p_range=P_RANGE,
            N_samples=2000,
            method="atmospheric",
            seed=0,
            pressure_weight=weight,
        )
        fractions.append(float(np.mean(p > 5e4)))
    assert fractions[0] < fractions[1] < fractions[2]


def test_envelope_is_interpolated_not_extrapolated_wildly():
    knots = np.asarray(ATMOSPHERIC_T_ENVELOPE, dtype=float)
    t_min, t_max = atmospheric_t_range(knots[:, 0])
    assert np.allclose(t_min, knots[:, 1])
    assert np.allclose(t_max, knots[:, 2])
    # outside the knot range the envelope must clamp, not run away
    lo, hi = atmospheric_t_range(np.array([1e-3, 1e7]))
    assert lo.min() > 100.0 and hi.max() < 400.0


def test_relevance_floor_scales_with_how_much_of_the_species_there_is():
    """A rarer gas needs a larger cross-section to matter, so a higher floor."""
    assert xsec_relevance_floor("N2O") > xsec_relevance_floor("CO2")
    assert xsec_relevance_floor("CO2") > xsec_relevance_floor("H2O")
    assert xsec_relevance_floor("H2O") > 0.0


def test_sampling_kwargs_reach_the_reference_generator(tmp_path, monkeypatch):
    """Regression: the sampling settings were silently ignored, so every
    reference was built with the defaults."""
    seen = {}

    def fake_reference(species, frequency_grid, pressure, temperature, vmr, **kwargs):
        import xarray as xr

        seen["pressure"] = pressure
        return xr.Dataset(
            {
                "xsec": (("case", "frequency"), np.ones((pressure.size, 1))),
                "pressure": (("case",), pressure),
                "temperature": (("case",), temperature),
                "vmr": (("case",), vmr),
            },
            coords={"case": np.arange(pressure.size), "frequency": frequency_grid},
        )

    monkeypatch.setattr("faxsec.utils.calulate_arts_reference", fake_reference)

    for n_samples, name in ((200, "a"), (800, "b")):
        ensure_reference_dataset(
            species="H2O",
            frequency_grid=np.array([1e13]),
            cache_path=tmp_path / f"{name}.nc",
            sampling_kwargs={"N_samples": n_samples, "p_range": P_RANGE},
        )
        assert abs(seen["pressure"].size - n_samples) < n_samples * 0.25, name


def test_stale_reference_cache_is_flagged(tmp_path, monkeypatch, caplog):
    """A cached reference built under different sampling must not be reused silently."""
    import xarray as xr

    def fake_reference(species, frequency_grid, pressure, temperature, vmr, **kwargs):
        return xr.Dataset(
            {
                "xsec": (("case", "frequency"), np.ones((pressure.size, 1))),
                "pressure": (("case",), pressure),
                "temperature": (("case",), temperature),
                "vmr": (("case",), vmr),
            },
            coords={"case": np.arange(pressure.size), "frequency": frequency_grid},
        )

    monkeypatch.setattr("faxsec.utils.calulate_arts_reference", fake_reference)
    cache = tmp_path / "ref.nc"
    common = dict(species="H2O", frequency_grid=np.array([1e13]), cache_path=cache)

    ensure_reference_dataset(**common, sampling_kwargs={"N_samples": 200})
    with caplog.at_level("WARNING"):
        ensure_reference_dataset(**common, sampling_kwargs={"N_samples": 900})
    assert any("sampling" in r.message for r in caplog.records)
