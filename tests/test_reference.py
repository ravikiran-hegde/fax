"""Blocked, streamed generation of the ARTS training reference."""

import numpy as np
import xarray as xr

from faxsec.utils import (
    _ARTS_BYTES_PER_CASE_FREQ,
    DEFAULT_REFERENCE_MEMORY_BUDGET,
    _reference_dataset,
    calulate_arts_reference,
    ensure_reference_dataset,
)


class StubAbsorber:
    """Stands in for ARTSAbsorber: a smooth, reproducible cross-section."""

    def __init__(self, species, frequency_grid, arts_tag=None):
        self.frequency_grid = np.asarray(frequency_grid, dtype=float)
        StubAbsorber.calls = []

    def cross_section(self, pressure, temperature, vmr):
        StubAbsorber.calls.append(pressure.size)
        return 1e-25 * np.exp(
            np.log(pressure / 1e4)[:, None] + 0.0 * self.frequency_grid
        ) * (temperature[:, None] / 250.0)


def stub_arts(monkeypatch):
    """Route the reference chunks through StubAbsorber instead of ARTS."""
    import faxsec.arts

    monkeypatch.setattr(faxsec.arts, "ARTSAbsorber", StubAbsorber)


def test_case_chunk_fits_the_memory_budget():
    def chunk_for(n_freq, n_cases, budget=DEFAULT_REFERENCE_MEMORY_BUDGET):
        ds = _reference_dataset(
            "H2O", np.linspace(1e13, 2e13, n_freq), np.ones(n_cases),
            np.full(n_cases, 250.0), np.full(n_cases, 1e-4), memory_budget=budget,
        )
        return max(ds["xsec"].chunksizes["case"])

    chunk = chunk_for(100_000, 2001)
    assert 1 < chunk < 2001
    assert (
        chunk * 100_000 * _ARTS_BYTES_PER_CASE_FREQ
        <= DEFAULT_REFERENCE_MEMORY_BUDGET
    )
    # Never split further than there are cases to compute.
    assert chunk_for(10, 8) == 8


def test_chunked_reference_matches_a_single_call(monkeypatch):
    stub_arts(monkeypatch)
    freq = np.linspace(1e13, 2e13, 32)
    p = np.logspace(2, 5, 20)
    t = np.linspace(200.0, 300.0, 20)
    vmr = np.full_like(p, 1e-4)

    chunked = calulate_arts_reference("H2O", freq, p, t, vmr, case_chunk=6)
    assert sorted(StubAbsorber.calls) == [2, 6, 6, 6]

    whole = calulate_arts_reference("H2O", freq, p, t, vmr, case_chunk=len(p))
    assert chunked["xsec"].dtype == np.float64
    np.testing.assert_array_equal(chunked["xsec"].values, whole["xsec"].values)


def test_reference_is_streamed_chunk_by_chunk(tmp_path, monkeypatch):
    stub_arts(monkeypatch)
    cache = tmp_path / "H2O.nc"

    ensure_reference_dataset(
        species="H2O",
        frequency_grid=np.linspace(1e13, 2e13, 64),
        cache_path=cache,
        sampling_kwargs={"N_samples": 40, "p_range": [1.0, 1.05e5]},
        arts_reference_kwargs={"case_chunk": 8},
        ref_pressure=1.0e4,
        ref_temperature=240.0,
    )

    assert max(StubAbsorber.calls) <= 8
    assert not list(tmp_path.glob("*.partial"))
    with xr.open_dataset(cache) as ds:
        assert ds["xsec"].dtype == np.float64
        assert np.isfinite(np.log(ds["xsec"].values)).all()
        # The fit anchors on the reference point, so it must be present.
        assert np.isclose(ds["pressure"].values, 1.0e4).any()


def test_a_failed_write_leaves_no_cache(tmp_path, monkeypatch):
    stub_arts(monkeypatch)
    cache = tmp_path / "H2O.nc"

    def boom(*args, **kwargs):
        raise RuntimeError("ARTS died")

    monkeypatch.setattr(StubAbsorber, "cross_section", boom)
    try:
        ensure_reference_dataset(
            species="H2O",
            frequency_grid=np.linspace(1e13, 2e13, 8),
            cache_path=cache,
            sampling_kwargs={"N_samples": 10, "p_range": [1.0, 1.05e5]},
        )
    except RuntimeError:
        pass
    assert not cache.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_a_chunked_atmosphere_is_computed_chunk_by_chunk():
    """`dask="parallelized"` only chunks if the caller chunks the atmosphere,
    and it needs the size of the frequency dim it creates."""
    from faxsec.abstract_class import NullAbsorberModel

    calls = []

    class Counting(NullAbsorberModel):
        def cross_section(self, pressure, temperature, vmr):
            calls.append(pressure.size)
            return np.tile(pressure[:, None], (1, len(self.config.frequency_grid)))

    model = Counting(name="CO2", frequency_grid=np.linspace(1e13, 2e13, 16))
    n = 40
    atm = xr.Dataset(
        {
            "pressure_layer": ("case", np.logspace(2, 5, n)),
            "temperature_layer": ("case", np.linspace(200.0, 300.0, n)),
            "CO2": ("case", np.full(n, 4e-4)),
        }
    )

    calls.clear()
    eager = model.cross_section_from_atmds(atm)
    assert calls == [n]

    calls.clear()
    chunked = model.cross_section_from_atmds(atm.chunk({"case": 8}))
    assert chunked.chunks is not None  # nothing computed yet
    assert not calls
    np.testing.assert_array_equal(
        chunked.compute(scheduler="synchronous").values, eager.values
    )
    assert calls == [8] * 5


def test_frequency_chunks_train_to_the_same_fit(tmp_path):
    """Chunked training must reproduce whole-grid training exactly."""
    from faxsec.functional import FunctionalAbsorber

    n_case, n_freq = 60, 8
    rng = np.random.default_rng(0)
    pressure = np.logspace(1, 5, n_case)
    temperature = rng.uniform(200.0, 300.0, n_case)
    frequency = np.linspace(1e13, 2e13, n_freq)
    strength = np.linspace(1e-24, 1e-22, n_freq)
    xsec = (
        strength
        * (pressure[:, None] / 1e4) ** 0.6
        * np.exp(-(temperature[:, None] - 240.0) / 80.0)
    )
    reference = tmp_path / "ref.nc"
    xr.Dataset(
        {
            "xsec": (("case", "frequency"), xsec),
            "pressure": ("case", pressure),
            "temperature": ("case", temperature),
            "vmr": ("case", np.full(n_case, 1e-4)),
        },
        coords={"case": np.arange(n_case), "frequency": frequency},
    ).to_netcdf(reference)

    def model():
        return FunctionalAbsorber(
            species="H2O", frequency_grid=frequency, ref_pressure=pressure[0],
            ref_temperature=temperature[0], ref_vmr=1e-4, xsec_floor=1e-40,
        )

    whole = model()
    whole.train(reference_xsec=reference, max_iter=4)

    save_path = tmp_path / "chunks.nc"
    chunked = model()
    chunked.train_in_frequency_chunks(
        reference, frequency_chunk=3, n_workers=2, save_path=save_path, max_iter=4
    )

    np.testing.assert_array_equal(chunked.coeffs.xsec0, whole.coeffs.xsec0)
    np.testing.assert_array_equal(
        chunked.coeffs.pressure_coeffs, whole.coeffs.pressure_coeffs
    )
    np.testing.assert_array_equal(
        chunked.coeffs.temperature_coeffs, whole.coeffs.temperature_coeffs
    )

    # The saved file holds the whole grid, and a rerun refits nothing.
    with xr.open_dataset(save_path) as saved:
        np.testing.assert_allclose(saved["frequency"].values, frequency)
    resumed = model()
    resumed.train_in_frequency_chunks(
        reference, frequency_chunk=3, n_workers=2, save_path=save_path, max_iter=4
    )
    np.testing.assert_array_equal(
        resumed.coeffs.pressure_coeffs, whole.coeffs.pressure_coeffs
    )
