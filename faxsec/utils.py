from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from faxsec.constants import (
    BOLTZMANN,
    DATA_DIR,
    DEFAULT_VMR,
    LIGHT_SPEED,
    PLANCK,
    REF_PRESSURE,
    REF_TEMPERATURE,
    REF_VMR,
)

from .constants import KAYSER_TO_HZ

logger = logging.getLogger(__name__)


def kayser_to_hz(kaysers):
    """Convert wavenumbers in kaysers (cm^-1) to frequencies in Hz."""
    return kaysers * KAYSER_TO_HZ


def hz_to_kayser(hz):
    """Convert frequencies in Hz to wavenumbers in kaysers (cm^-1)."""
    return hz / KAYSER_TO_HZ


def rad_fun(nu, T):
    """
    Computes the FASCOD3/LBLRTM Radiation Function: nu * tanh(nu * h*c^2/ (2 *kB * T))

    Parameters:
    -----------
    nu : frequency (in Hz)
    T  : Temperature (in Kelvin)

    Returns:
    --------
    cm float or numpy.ndarray
        The calculated radiation function factor.
    """

    from .constants import CM_TO_M, RADCN2
    from .utils import hz_to_kayser

    nu = np.asarray(nu, dtype=float)
    T = np.asarray(T, dtype=float)

    kayser_nu = hz_to_kayser(nu)
    xviokt = (kayser_nu * CM_TO_M * RADCN2) / T  # broadcasts nu/T shapes -> bshape

    bshape = xviokt.shape
    kayser_nu_b = np.broadcast_to(kayser_nu, bshape)

    result = np.zeros(bshape)

    # Low frequency / Rayleigh-Jeans limit
    m1 = xviokt <= 0.01
    result[m1] = 0.5 * xviokt[m1] * kayser_nu_b[m1]

    # Mid-range operational branch
    m2 = (xviokt > 0.01) & (xviokt <= 10)
    expvkt = np.expm1(-xviokt[m2])
    result[m2] = -kayser_nu_b[m2] * expvkt / (2.0 + expvkt)

    # High frequency / Wien limit
    m3 = xviokt > 10
    result[m3] = kayser_nu_b[m3]

    return result.item() if result.ndim == 0 else result


def rayleigh_xsec_stamnes_2017(frequency_hz: np.ndarray) -> np.ndarray:
    # TODO: implement as a absorber?
    from faxsec.constants import LIGHT_SPEED

    # Convert frequency to wavelength in microns
    wavelength = LIGHT_SPEED / frequency_hz * 1e6  # microns

    # Coefficients for the polynomial
    a = np.array([3.9729066, 4.6547659e-2, 4.5055995e-4, 2.3229848e-5])

    # Calculate the Rayleigh scattering cross-section
    rayleigh_xsec = (
        np.polyval(a[::-1], wavelength ** (-2)) * 1e-28 * 1e-4 / wavelength**4  # m2
    )

    return rayleigh_xsec


# Molecules per square metre in a whole air column at 1013 hPa, and the
# representative maximum column-mean VMR of each species.
AIR_COLUMN = 2.1e29
COLUMN_VMR = {
    "H2O": 4.0e-2,
    "CO2": 2.0e-3,
    "O3": 1.0e-5,
    "N2O": 5.0e-7,
    "CH4": 4.0e-6,
    "O2": 2.095e-1,
    "N2": 7.808e-1,
}


def xsec_relevance_floor(species: str, tau_min: float = 1e-6) -> float:
    """Cross-section below which a species cannot reach ``tau_min`` in a column.

    Reference values below this are not worth fitting: they carry no optical
    depth, but in log space they span many e-folds and would otherwise steer
    the fit.
    """
    vmr = COLUMN_VMR.get(species, REF_VMR)
    return tau_min / (vmr * AIR_COLUMN)


def simple_vmr_profile(
    species: str,
    pressure: np.ndarray,
    temperature: Optional[np.ndarray] = None,
    default_vmrs: dict[str, float] = DEFAULT_VMR,
) -> np.ndarray:
    """Return a simple level-wise VMR profile for one species."""
    pressure = np.asarray(pressure, dtype=float)
    temperature = (
        np.asarray(temperature, dtype=float) if temperature is not None else None
    )
    if species in default_vmrs:
        vmr = np.full_like(pressure, default_vmrs[species], dtype=float)
    else:
        vmr = np.full_like(pressure, REF_VMR, dtype=float)

    return vmr


# -----------------------------
# ARTS related utilities and adapters
# -----------------------------
def calulate_arts_reference(
    species: str,
    frequency_grid: ArrayLike,
    pressure: np.ndarray,
    temperature: np.ndarray,
    vmr: np.ndarray,
    arts_tag: tuple[str, ...] | None = None,
) -> xr.Dataset:
    """Calculate reference cross-section dataset using ARTS."""

    from .arts import ARTSAbsorber

    absorber = ARTSAbsorber(
        species=species,
        frequency_grid=np.asarray(frequency_grid, dtype=float),
        arts_tag=arts_tag,
    )

    xsec = absorber.cross_section(pressure=pressure, temperature=temperature, vmr=vmr)
    return xr.Dataset(
        {
            "xsec": (("case", "frequency"), xsec),
            "pressure": (("case",), pressure),
            "temperature": (("case",), temperature),
            "vmr": (("case",), vmr),
        },
        coords={
            "case": np.arange(pressure.size),
            "frequency": frequency_grid,
        },
    )


def _slugify(value: Any) -> str:
    text = (
        "default"
        if value is None
        else "__".join(map(str, value)) if isinstance(value, tuple) else str(value)
    )
    text = text.replace("/", "-").replace(" ", "-")
    return "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "-" for char in text
    ).strip("-")


def reference_cache_path(
    species: str,
    arts_tag: tuple[str, ...] | None = None,
    cache_dir: str | Path = DATA_DIR / "reference_cache",
) -> Path:
    """Return the on-disk cache path for a reference dataset."""

    cache_dir = Path(cache_dir)
    tag = _slugify(arts_tag)
    return cache_dir / f"{species}__{tag}.nc"


def ensure_reference_dataset(
    species: str,
    frequency_grid: np.ndarray,
    arts_tag: tuple[str, ...] | None = None,
    cache_path: str | Path | None = None,
    sampling_kwargs: dict[str, Any] | None = None,
    arts_reference_kwargs: dict[str, Any] | None = None,
    ref_pressure: float = REF_PRESSURE,
    ref_temperature: float = REF_TEMPERATURE,
    ref_vmr: float = REF_VMR,
    force: bool = False,
) -> Path:
    """Create or load the reference dataset required by FunctionalAbsorber.train()."""

    if cache_path is None:
        cache_path = reference_cache_path(species=species, arts_tag=arts_tag)
    cache_path = Path(cache_path)

    sampling_kwargs = {} if sampling_kwargs is None else dict(sampling_kwargs)

    if cache_path.exists() and not force:
        cached = xr.open_dataset(cache_path).attrs.get("sampling_kwargs")
        if cached is not None and cached != str(sampling_kwargs):
            logger.warning(
                "Cached reference for %s was built with sampling %s, not %s; "
                "delete %s to rebuild it",
                species,
                cached,
                sampling_kwargs,
                cache_path,
            )
        logger.debug("Using cached ARTS reference for %s: %s", species, cache_path)
        return cache_path

    logger.info("Computing ARTS reference for %s (tags=%s)", species, arts_tag)

    arts_reference_kwargs = (
        {} if arts_reference_kwargs is None else dict(arts_reference_kwargs)
    )

    p_grid, t_grid = sample_atmospheres(**sampling_kwargs)

    has_reference_case = np.any(
        np.isclose(p_grid, ref_pressure) & np.isclose(t_grid, ref_temperature)
    )
    if not has_reference_case:
        p_grid = np.append(p_grid, ref_pressure)
        t_grid = np.append(t_grid, ref_temperature)

    reference_ds = calulate_arts_reference(
        species,
        frequency_grid,
        p_grid,
        t_grid,
        np.full_like(p_grid, ref_vmr, dtype=float),
        arts_tag=arts_tag,
        **arts_reference_kwargs,
    )

    reference_ds.attrs["sampling_kwargs"] = str(sampling_kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    reference_ds.to_netcdf(cache_path)
    logger.info(
        "Cached ARTS reference for %s: %d cases -> %s",
        species,
        reference_ds.sizes.get("case", 0),
        cache_path,
    )

    return cache_path


# -----------------------------
# Atmosphere sampling utilities
# -----------------------------


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _default_n_p_n_t(n_samples: int) -> tuple[int, int]:
    """Return near-square (n_p, n_T) defaults from total sample count."""
    n_p = max(2, int(np.sqrt(max(1, n_samples))))
    n_T = max(2, int(np.ceil(max(1, n_samples) / n_p)))
    return n_p, n_T


def _resolve_grid_shape_max(
    n_samples: int,
    n_p: int | None = None,
    n_t: int | None = None,
    samples_per_cell: int = 1,
) -> tuple[int, int]:
    """Resolve grid sizes to maximize used samples for partial inputs.

    For partial/default inputs, choose dimensions such that
    ``n_p * n_t * samples_per_cell <= n_samples`` and as large as possible.
    If both dimensions are explicitly provided, they are used as-is.
    """
    spc = max(1, int(samples_per_cell))
    budget = max(1, int(n_samples) // spc)

    if n_p is not None and n_t is not None:
        return int(n_p), int(n_t)

    if n_p is not None:
        p_count = max(1, int(n_p))
        t_count = max(1, budget // p_count)
        return p_count, t_count

    if n_t is not None:
        t_count = max(1, int(n_t))
        p_count = max(1, budget // t_count)
        return p_count, t_count

    # Default: near-square, then nudge to maximize usage under budget.
    p0, t0 = _default_n_p_n_t(budget)
    best_p, best_t = max(1, p0), max(1, t0)
    if best_p * best_t > budget:
        best_t = max(1, budget // best_p)

    best_prod = best_p * best_t
    p_start = max(1, int(np.sqrt(budget)) - 4)
    p_end = int(np.sqrt(budget)) + 5
    for p_count in range(p_start, p_end):
        if p_count <= 0:
            continue
        t_count = max(1, budget // p_count)
        prod = p_count * t_count
        if prod > best_prod:
            best_p, best_t, best_prod = p_count, t_count, prod

    return best_p, best_t


SamplingResult = Tuple[np.ndarray, np.ndarray]


def sample_atmospheres(
    p_range: Sequence[float] = [0.01, 110000],
    T_range: Sequence[float] = [150.0, 350.0],
    N_samples: int = 1000,
    seed: int | None = 42,
    method: str = "natural",
    **kwargs,
) -> SamplingResult:
    """Draw samples using one of the available atmosphere sampling methods."""
    if method in ("random", "uniform"):
        return sample_uniform(p_range, T_range, N_samples, seed)
    if method == "atmospheric":
        return sample_atmospheres_atmospheric(
            p_range,
            N_samples,
            seed=seed,
            **kwargs,
        )
    if method == "natural":
        return sample_atmospheres_natural(
            p_range,
            T_range,
            N_samples,
            seed=seed,
            **kwargs,
        )
    if method == "factorial":
        return sample_atmospheres_factorial(
            p_range,
            T_range,
            N_samples,
            n_p=kwargs.get("n_p"),
            n_T=kwargs.get("n_T"),
            seed=seed,
        )
    if method == "latin_hypercube":
        return sample_atmospheres_latin_hypercube(
            p_range,
            T_range,
            N_samples,
            seed=seed,
        )
    if method == "stratified":
        return sample_atmospheres_stratified(
            p_range,
            T_range,
            N_samples,
            n_p_strata=kwargs.get("n_p_strata"),
            n_T_strata=kwargs.get("n_T_strata"),
            samples_per_cell=int(kwargs.get("samples_per_cell", 1)),
            seed=seed,
        )
    if method == "pressure_fixed":
        # Default to geometric-midpoint pressure when no fixed pressure is given.
        p_default = np.sqrt(p_range[0] * p_range[1])
        return sample_atmospheres_pressure_fixed(
            p_range,
            T_range,
            N_samples,
            p=kwargs.get("p", p_default),
            seed=seed,
        )
    if method == "temperature_fixed":
        # Default to midpoint temperature when no fixed temperature is given.
        T_default = 0.5 * (T_range[0] + T_range[1])
        return sample_atmospheres_temperature_fixed(
            p_range,
            T_range,
            N_samples,
            T=float(kwargs.get("T", T_default)),
            seed=seed,
        )
    raise ValueError(f"Unknown sampling method: {method}")


def sample_uniform(
    p_range: Sequence[float],
    T_range: Sequence[float],
    N_samples: int,
    seed: int | None = None,
) -> SamplingResult:
    rng = _rng(seed)
    lnp = rng.uniform(
        low=np.log(p_range[0]),
        high=np.log(p_range[1]),
        size=N_samples,
    )
    p = np.exp(lnp)
    T = rng.uniform(low=T_range[0], high=T_range[1], size=N_samples)
    return p, T


def sample_atmospheres_natural(
    p_range: Sequence[float],
    T_range: Sequence[float],
    N_samples: int,
    seed: int | None = None,
    pressure_power: float = 4.0,
    n_p_strata: int | None = None,
    n_T_strata: int | None = None,
    samples_per_cell: int = 1,
) -> SamplingResult:
    """Stratified p-T sampling with high-pressure bias.

    Special parameter:
        pressure_power: bias strength (>1 biases toward high pressure).
        n_p_strata: pressure strata count (default inferred from N_samples).
        n_T_strata: temperature strata count (default inferred from N_samples).
        samples_per_cell: random samples per grid cell (default 1).
    """
    rng = _rng(seed)
    n_p_use, n_T_use = _resolve_grid_shape_max(
        N_samples,
        n_p=n_p_strata,
        n_t=n_T_strata,
        samples_per_cell=samples_per_cell,
    )

    lnp_min = np.log(p_range[0])
    lnp_max = np.log(p_range[1])
    base = np.linspace(0.0, 1.0, n_p_use + 1)
    power = max(float(pressure_power), 1.0)
    lnp_edges = lnp_min + (lnp_max - lnp_min) * (1.0 - (1.0 - base) ** power)
    t_edges = np.linspace(T_range[0], T_range[1], n_T_use + 1)

    lnp_samples = []
    t_samples = []
    for i_p in range(n_p_use):
        for i_t in range(n_T_use):
            for _ in range(samples_per_cell):
                lnp_samples.append(rng.uniform(lnp_edges[i_p], lnp_edges[i_p + 1]))
                t_samples.append(rng.uniform(t_edges[i_t], t_edges[i_t + 1]))

    T = np.asarray(t_samples, dtype=float)
    p = np.exp(lnp_samples)
    return p, T


# Temperature range the atmosphere actually occupies, as
# (pressure [Pa], T_min [K], T_max [K]) knots interpolated in log-pressure.
ATMOSPHERIC_T_ENVELOPE = (
    (1.0, 165.0, 245.0),
    (10.0, 190.0, 290.0),
    (100.0, 185.0, 315.0),
    (1.0e3, 160.0, 290.0),
    (1.0e4, 165.0, 260.0),
    (3.0e4, 170.0, 280.0),
    (1.0e5, 195.0, 330.0),
)


def atmospheric_t_range(
    pressure: ArrayLike,
    envelope: Sequence[Sequence[float]] = ATMOSPHERIC_T_ENVELOPE,
    pad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Occupied temperature range at each pressure, interpolated in log-p."""
    knots = np.asarray(envelope, dtype=float)
    lnp = np.log(np.asarray(pressure, dtype=float))
    lnp_knots = np.log(knots[:, 0])
    t_min = np.interp(lnp, lnp_knots, knots[:, 1]) - pad
    t_max = np.interp(lnp, lnp_knots, knots[:, 2]) + pad
    return t_min, t_max


def sample_atmospheres_atmospheric(
    p_range: Sequence[float],
    N_samples: int,
    seed: int | None = None,
    n_p_strata: int | None = None,
    pressure_weight: float = 0.5,
    envelope: Sequence[Sequence[float]] = ATMOSPHERIC_T_ENVELOPE,
    pad: float = 0.0,
) -> SamplingResult:
    """Stratified sampling of the (p, T) region the atmosphere actually occupies.

    Temperature is drawn conditionally on pressure, from the occupied range at
    that pressure, so no sample falls in a corner that cannot exist.

    ``pressure_weight`` sets how strata are spaced: 0 is uniform in log
    pressure, 1 is uniform in pressure, i.e. equal air mass per stratum.
    Intermediate values trade accuracy aloft against accuracy where most of
    the absorbing mass is.
    """
    rng = _rng(seed)
    n_p_use, n_T_use = _resolve_grid_shape_max(N_samples, n_p=n_p_strata)

    if pressure_weight <= 0:
        edges = np.linspace(np.log(p_range[0]), np.log(p_range[1]), n_p_use + 1)
        to_p = np.exp
    else:
        k = float(pressure_weight)
        edges = np.linspace(p_range[0] ** k, p_range[1] ** k, n_p_use + 1)
        to_p = lambda u: u ** (1.0 / k)  # noqa: E731

    frac_edges = np.linspace(0.0, 1.0, n_T_use + 1)
    drawn = rng.uniform(edges[:-1], edges[1:], size=(n_T_use, n_p_use)).T
    frac = rng.uniform(frac_edges[:-1], frac_edges[1:], size=(n_p_use, n_T_use))

    p = to_p(drawn).ravel()
    t_min, t_max = atmospheric_t_range(p, envelope, pad)
    T = t_min + frac.ravel() * (t_max - t_min)
    return p, T


def sample_atmospheres_factorial(
    p_range: Sequence[float],
    T_range: Sequence[float],
    N_samples: int,
    n_p: int | None = None,
    n_T: int | None = None,
    seed: int | None = None,
) -> SamplingResult:
    """Factorial p-T grid sampling.

    Special parameters:
        n_p: pressure grid size.
        n_T: temperature grid size.

    Behavior:
        - If both n_p and n_T are given, they are used as-is.
        - If one or both are missing, missing values are initialized so
          n_p*n_T is maximized under the N_samples budget.
    """
    n_p_use, n_T_use = _resolve_grid_shape_max(
        N_samples,
        n_p=n_p,
        n_t=n_T,
        samples_per_cell=1,
    )

    p_samples = np.logspace(np.log10(p_range[0]), np.log10(p_range[1]), n_p_use)
    T_samples = np.linspace(T_range[0], T_range[1], n_T_use)

    p_grid, T_grid = np.meshgrid(p_samples, T_samples, indexing="xy")
    p_flat = p_grid.ravel()
    T_flat = T_grid.ravel()

    return p_flat, T_flat


def sample_atmospheres_latin_hypercube(
    p_range: Sequence[float],
    T_range: Sequence[float],
    N_samples: int,
    seed: int | None = None,
) -> SamplingResult:
    """Latin hypercube p-T sampling."""
    rng = _rng(seed)

    # Generate LHS: stratified permutation for each dimension
    # This ensures each percentile is sampled exactly once
    p_percentiles = (
        rng.permutation(N_samples) + rng.uniform(0, 1, N_samples)
    ) / N_samples
    T_percentiles = (
        rng.permutation(N_samples) + rng.uniform(0, 1, N_samples)
    ) / N_samples

    # Map percentiles to actual ranges
    p = np.exp(p_percentiles * np.log(p_range[1] / p_range[0]) + np.log(p_range[0]))
    T = T_percentiles * (T_range[1] - T_range[0]) + T_range[0]

    return p, T


def sample_atmospheres_stratified(
    p_range: Sequence[float],
    T_range: Sequence[float],
    N_samples: int,
    n_p_strata: int | None = None,
    n_T_strata: int | None = None,
    samples_per_cell: int = 1,
    seed: int | None = None,
) -> SamplingResult:
    """Regular-cell stratified p-T sampling.

    Special parameters:
                n_p_strata: pressure strata count.
                n_T_strata: temperature strata count.
                samples_per_cell: random samples per grid cell (default 1).

        Behavior:
                - If n_p_strata, n_T_strata, and samples_per_cell are all given,
                    they are used as-is.
                - For partial inputs, missing strata counts are initialized so
                    n_p_strata*n_T_strata*samples_per_cell is maximized under N_samples.
    """
    rng = _rng(seed)

    n_p_use, n_T_use = _resolve_grid_shape_max(
        N_samples,
        n_p=n_p_strata,
        n_t=n_T_strata,
        samples_per_cell=samples_per_cell,
    )

    # Create cell boundaries (log-scale for pressure)
    lnp_min, lnp_max = np.log(p_range[0]), np.log(p_range[1])
    lnp_edges = np.linspace(lnp_min, lnp_max, n_p_use + 1)
    T_edges = np.linspace(T_range[0], T_range[1], n_T_use + 1)

    # Collect samples
    lnp_samples = []
    T_samples = []

    for i_p in range(n_p_use):
        for i_T in range(n_T_use):
            # Sample uniformly within this cell
            for _ in range(samples_per_cell):
                lnp_cell = rng.uniform(lnp_edges[i_p], lnp_edges[i_p + 1])
                T_cell = rng.uniform(T_edges[i_T], T_edges[i_T + 1])
                lnp_samples.append(lnp_cell)
                T_samples.append(T_cell)

    lnp = np.array(lnp_samples)
    p = np.exp(lnp)
    T = np.array(T_samples)

    return p, T


def sample_atmospheres_pressure_fixed(
    p_range: Sequence[float],
    T_range: Sequence[float],
    N_samples: int,
    p: float | np.ndarray | None = None,
    seed: int | None = None,
) -> SamplingResult:
    """Pressure-fixed temperature sampling.

    Special parameter:
        p: fixed pressure (default geometric midpoint of p_range).
    """
    rng = _rng(seed)

    if p is None:
        p_val = float(np.sqrt(p_range[0] * p_range[1]))
    elif isinstance(p, np.ndarray):
        p_val = np.mean(p)
    else:
        p_val = float(p)

    T = rng.uniform(T_range[0], T_range[1], N_samples)

    p_out = np.full(N_samples, p_val)

    return p_out, T


def sample_atmospheres_temperature_fixed(
    p_range: Sequence[float],
    T_range: Sequence[float],
    N_samples: int,
    T: float | None = None,
    seed: int | None = None,
) -> SamplingResult:
    """Temperature-fixed pressure sampling.

    Special parameter:
        T: fixed temperature (default midpoint of T_range).
    """
    rng = _rng(seed)

    lnp = rng.uniform(
        low=np.log(p_range[0]),
        high=np.log(p_range[1]),
        size=N_samples,
    )
    p = np.exp(lnp)
    T_val = 0.5 * (T_range[0] + T_range[1]) if T is None else float(T)
    T_out = np.full(N_samples, T_val)

    return p, T_out


def rayleigh_cross_section(kayser):
    """
    Rayleigh scattering cross section.

    Parameters
    ----------
    kayser : float or ndarray
        Spectroscopic wavenumber [cm^-1]

    Returns
    -------
    sigma : float or ndarray
        Rayleigh cross section [cm^2 molecule^-1]
    """

    # wavelength in microns
    lam_um = 1.0e4 / kayser

    # refractive index (Edlen/Bodhaine)
    n_minus1 = (
        8060.51
        + 2480990.0 / (132.274 - 1.0 / lam_um**2)
        + 17455.7 / (39.32957 - 1.0 / lam_um**2)
    ) * 1e-8

    n = 1.0 + n_minus1

    # King factor
    Fk = 1.034 + 3.17e-4 / lam_um**2

    # Loschmidt number [cm^-3]
    Ns = 2.546899e19

    sigma = (24 * np.pi**3) / (Ns**2) * kayser**4 * ((n**2 - 1) / (n**2 + 2)) ** 2 * Fk

    return sigma / 1e4  # convert from cm^2 to m^2


def planck_nu(f_grid_hz, temperature):
    """Return Planck Flux in W/m^2/Hz."""

    exponent = PLANCK * f_grid_hz / (BOLTZMANN * temperature)
    return (2 * PLANCK * f_grid_hz**3 / LIGHT_SPEED**2) / np.expm1(exponent)
