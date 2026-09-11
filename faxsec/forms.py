from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

FIT_EPS = 1e-300  # keeps a reweighting division finite
BIG_WEIGHT = 1e3  # turns the scale-fixing row of a fit into a constraint


class FunctionalForm(ABC):
    """Base class for generic functional forms f(x)."""

    @abstractmethod
    def evaluate(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate function at given x values."""

    @abstractmethod
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
        x_ref: float | None = None,
    ) -> np.ndarray:
        """Fit coefficients to data y = f(x).

        ``weights`` is an optional (N, F) array of least-squares weights;
        zero weight drops a sample from that frequency's fit. ``x_ref``, when
        given, constrains the fit to f(x_ref) = 1.
        """

    @abstractmethod
    def coefficient_names(self) -> List[str]:
        """Return ordered list of coefficient names."""

    def rescale(self, coeffs: np.ndarray, factor: np.ndarray) -> np.ndarray:
        """Return coefficients whose evaluation is ``factor`` times this one's."""
        raise NotImplementedError(f"{type(self).__name__} cannot be rescaled")


def anchored_lstsq(
    X: np.ndarray, t: np.ndarray, w: np.ndarray, ref_row: np.ndarray
) -> np.ndarray:
    """Weighted least squares constrained so ``ref_row @ coeffs == 1``."""
    j = int(np.argmax(np.abs(ref_row)))
    reduced = X - np.outer(X[:, j], ref_row) / ref_row[j]
    coeffs, *_ = np.linalg.lstsq(
        reduced * w[:, None], (t - X[:, j] / ref_row[j]) * w, rcond=None
    )
    coeffs[j] = 0.0
    coeffs[j] = (1.0 - ref_row @ coeffs) / ref_row[j]
    return coeffs


# ============================================================================
# Generic Functional Forms
# ============================================================================


class PolynomialForm(FunctionalForm):
    """Polynomial form: c0 + c1*x + ... + cn*x^n."""

    def __init__(self, order: int = 1, include_bias: bool = True):
        self.order = order
        self.include_bias = include_bias

    def _vandermode(self, x: np.ndarray) -> np.ndarray:
        """Construct Vandermonde matrix for polynomial evaluation."""
        start = 0 if self.include_bias else 1
        return np.polynomial.polynomial.polyvander(np.ravel(x), self.order)[:, start:]

    def evaluate(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate polynomial function c0 + c1*x + ... + cn*x^n.

        Parameters
        ----------
        x : np.ndarray (N,)
            The input data points at which to evaluate the polynomial.
        coeffs : np.ndarray (order+1, F)
            The coefficients of the polynomial, where coeffs[i] corresponds to the coefficient of x^i

        Returns
        -------
        np.ndarray (N, F)
            The evaluated polynomial values at each x for each frequency.

        """

        V = self._vandermode(x)  # (N, order+1)

        return V @ coeffs  # (N, deg) @ (deg, F) -> (N, F)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
        x_ref: float | None = None,
    ) -> np.ndarray:
        V = self._vandermode(x)
        if weights is None:
            coeffs, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
            return coeffs
        coeffs = np.zeros((V.shape[1], y.shape[1]))
        for fi in range(y.shape[1]):
            s = np.sqrt(np.clip(weights[:, fi], 0.0, None))
            coeffs[:, fi], *_ = np.linalg.lstsq(
                V * s[:, None], y[:, fi] * s, rcond=None
            )
        return coeffs

    def coefficient_names(self) -> List[str]:
        start = 0 if self.include_bias else 1
        return [f"p{i}" for i in range(start, self.order + 1)]


class HingeForm(FunctionalForm):
    """Piecewise-linear form with hinge point: c0 + c1*min(x, xb) + c2*max(x-xb, 0)."""

    def __init__(self, include_bias: bool = True):
        self.include_bias = include_bias

    def _hinge_terms(self, x_col: np.ndarray, xb: ArrayLike) -> tuple:
        """Basis terms (bias, min(x, xb), max(x - xb, 0)), shared by evaluate and fit."""
        above = np.maximum(x_col - xb, 0.0)
        return (1.0 if self.include_bias else 0.0), x_col - above, above

    def evaluate(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray (N,)
             The input data points at which to evaluate the function.
        coeffs : np.ndarray (4, F)
             c0, c1, c2, xb.

        Returns
        -------
        np.ndarray (N, F)
             The evaluated function values at each x for each frequency.
        """

        c0, c1, c2, xb = coeffs
        bias, below, above = self._hinge_terms(np.ravel(x)[:, None], xb)

        below *= c1
        above *= c2
        below += above
        below += bias * c0
        return below

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
        x_ref: float | None = None,
    ) -> np.ndarray:
        """Fit hinge coefficients optimizing breakpoint xb per frequency."""

        from scipy.optimize import minimize_scalar

        n_freq = y.shape[1]
        x_col = x[:, None]

        fit = np.zeros((4, n_freq))  # c0, c1, c2, xb

        def loss(xb, y_col, w):
            # Design matrix: [1, min(x, xb), max(x-xb, 0)]
            bias, below, above = self._hinge_terms(x_col, xb)
            X = np.hstack([np.full_like(below, bias), below, above])  # (N, 3)

            if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y_col)):
                return np.inf, np.zeros(X.shape[1])

            if x_ref is None:
                coeffs, *_ = np.linalg.lstsq(X * w[:, None], y_col * w, rcond=None)
            else:
                ref_bias, ref_below, ref_above = self._hinge_terms(
                    np.array([[x_ref]]), xb
                )
                coeffs = anchored_lstsq(
                    X,
                    y_col,
                    w,
                    np.array([ref_bias, ref_below.item(), ref_above.item()]),
                )
            pred = X @ coeffs
            return np.sum(((pred - y_col) * w) ** 2), coeffs

        # Optimization per frequency
        logger.info("Fitting Hinge model...")

        for fi in range(n_freq):
            y_col = y[:, fi]
            w = (
                np.ones_like(y_col)
                if weights is None
                else np.sqrt(np.clip(weights[:, fi], 0.0, None))
            )
            used = w > 0
            if used.sum() < 4:
                continue
            bounds = (float(np.min(x[used])), float(np.max(x[used])))

            # find optimal breakpoint for this frequency
            result = minimize_scalar(
                lambda xb: loss(xb, y_col, w)[0],
                bounds=bounds,
                method="bounded",
            )

            xb_opt = float(result.x)

            #  evaluate coefficients at optimal breakpoint
            _, coeffs = loss(xb_opt, y_col, w)

            fit[:3, fi] = coeffs
            fit[3, fi] = xb_opt

            if fi % max(1, n_freq // 10) == 0:
                logger.debug("Hinge fit progress: %d/%d", fi + 1, n_freq)

        return fit  # (4, F)

    def rescale(self, coeffs: np.ndarray, factor: np.ndarray) -> np.ndarray:
        """Return coefficients whose evaluation is ``factor`` times this one's."""
        # the breakpoint is a position in x and does not scale
        scaled = coeffs.copy()
        scaled[:3] *= factor
        return scaled

    def coefficient_names(self) -> List[str]:
        keys = ["h1", "h2", "h_break"]
        if self.include_bias:
            keys.insert(0, "h0")
        return keys


class ShiftedReciprocalLaurentForm(FunctionalForm):
    """1 / (c_m1/w + c_0 + c_1*w) + c_lin*w, with w = x + shift.

    Defined for x > 0. Non-negative coefficients keep the reciprocal's
    denominator away from zero, so the form has no pole and never changes sign.
    The shift floors the abscissa, flattening the form below it smoothly.
    """

    def __init__(self, n_breaks: int = 20, n_reweight: int = 6):
        self.n_breaks = n_breaks
        self.n_reweight = n_reweight

    def _factors(self, w: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Value of the form at abscissa w, shared by evaluate and fit."""
        reciprocal = coeffs[0] / w + coeffs[1] + coeffs[2] * w
        return 1.0 / np.where(np.abs(reciprocal) < 1e-300, 1e-300, reciprocal) + (
            coeffs[3] * w
        )

    def evaluate(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray (N,)
        coeffs : np.ndarray (5, F)
            Rows: the three reciprocal coefficients, the linear one, the shift.

        Returns
        -------
        np.ndarray (N, F)
        """
        w = np.ravel(x)[:, None] + coeffs[4][None, :]
        return self._factors(np.maximum(w, 1e-30), coeffs[:4])

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
        x_ref: float | None = None,
    ) -> np.ndarray:
        """Fit each frequency over a grid of shifts, keeping the best."""
        from scipy.optimize import least_squares, nnls

        x = np.ravel(x)
        n_freq = y.shape[1]
        coeffs = np.zeros((5, n_freq))
        breaks = np.geomspace(max(x.min(), 1e-30), x.max() * 0.1, self.n_breaks)
        x_ref = float(x.max()) if x_ref is None else float(x_ref)

        logger.info(
            "Fitting ShiftedReciprocalLaurent model (%d shifts)...", breaks.size
        )

        for fi in range(n_freq):
            y_col = y[:, fi]
            usable = np.isfinite(y_col) & (y_col > 0)
            w_col = np.ones_like(y_col) if weights is None else weights[:, fi]
            w_col = np.sqrt(np.clip(np.where(usable, w_col, 0.0), 0.0, None))
            if (w_col > 0).sum() < 5:
                continue
            y_safe = np.where(usable, y_col, 1.0)

            best, best_rss = None, np.inf
            for shift in breaks:
                w = x + shift
                terms = np.stack([1.0 / w, np.ones_like(w), w], axis=1)
                # A least-squares weight in y becomes 1/Q on the reciprocal.
                z = 1.0 / y_safe
                q = np.ones_like(z)
                for _ in range(self.n_reweight):
                    row = w_col / np.maximum(np.abs(q), FIT_EPS)
                    seed, _ = nnls(terms * row[:, None], z * row)
                    q = terms @ seed

                start = np.append(seed, 0.0)
                result = least_squares(
                    lambda c: w_col * (self._factors(w, c) - y_safe),
                    start,
                    bounds=(0.0, np.inf),
                    max_nfev=400,
                )
                at_ref = float(self._factors(np.array([x_ref + shift]), result.x)[0])
                if not at_ref > 0:
                    continue
                rss = float(np.sum(result.fun**2))
                if rss < best_rss:
                    best_rss = rss
                    best = np.append(self.rescale(result.x, 1.0 / at_ref), shift)
            if best is not None:
                coeffs[:, fi] = best

            if fi % max(1, n_freq // 10) == 0:
                logger.debug("ShiftedReciprocalLaurent progress: %d/%d", fi + 1, n_freq)

        return coeffs

    def rescale(self, coeffs: np.ndarray, factor: np.ndarray) -> np.ndarray:
        """Return coefficients whose evaluation is ``factor`` times this one's."""
        scaled = coeffs.copy()
        scaled[:3] /= factor
        scaled[3] *= factor
        return scaled

    def coefficient_names(self) -> List[str]:
        return ["lm1", "l0", "l1", "lin", "shift"]


class SmoothHingeForm(HingeForm):
    """Smooth hinge form: c0 + c1*x + (c2 - c1)*(1/beta)*softplus(beta*(x-x_break)).

    Uses the same `fit()` but a different `_hinge_matrix`
    """

    def __init__(self, beta: float = 4.0, include_bias: bool = True):
        super().__init__(include_bias=include_bias)
        self.beta = beta  # controls smoothness of transition (higher = sharper)

    def _hinge_matrix(self, x: np.ndarray, xb: ArrayLike) -> np.ndarray:
        """Construct smooth hinge matrix for evaluation.
        [1, x, softplus(beta*(x-xb))]
        """
        x_col = np.ravel(x)[:, None]  # (N, 1)
        xb = np.ravel(xb)[None, :]  # (1, F)

        H = np.empty((len(x), len(xb.ravel()), 3))
        H[:, :, 0] = 1.0 if self.include_bias else 0.0
        H[:, :, 1] = x_col  # linear term (N, F)
        H[:, :, 2] = (1.0 / self.beta) * np.logaddexp(
            0.0, self.beta * (x_col - xb)
        )  # smooth hinge (N, F)
        return H

    def evaluate(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        X = self._hinge_matrix(x, coeffs[-1])  # (N, F, 3)

        return np.einsum("NFK,KF->NF", X, coeffs[:-1])  # (N, F, 3) @ (3, F) -> (N, F)


class RationalForm(FunctionalForm):
    """Rational function: (a0 + a1*x + ... + an*x^n) / (1 + b1*x + ... + bm*x^m)."""

    def __init__(self, numerator_order: int = 2, denominator_order: int = 2):
        self.numerator_order = numerator_order
        self.denominator_order = denominator_order

    # helpers

    @property
    def _n_a(self) -> int:
        return self.numerator_order + 1

    @property
    def _n_b(self) -> int:
        return self.denominator_order

    @property
    def _n_params(self) -> int:
        return self._n_a + self._n_b

    # basis matrices

    def _vandermonde_num(self, x: np.ndarray) -> np.ndarray:
        """(N, n_a)"""
        return np.polynomial.polynomial.polyvander(np.ravel(x), self.numerator_order)

    def _vandermonde_den(self, x: np.ndarray) -> np.ndarray:
        """(N, n_b)  — excludes leading 1"""
        return np.polynomial.polynomial.polyvander(np.ravel(x), self.denominator_order)[
            :, 1:
        ]

    # ----------------------------------------------------------------------------

    def evaluate(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x      : (N,)
        coeffs : (n_params, F)  — rows: [a0..an, b1..bm]

        Returns
        -------
        (N, F)
        """
        Vn = self._vandermonde_num(x)  # (N, n_a)
        Vd = self._vandermonde_den(x)  # (N, n_b)

        num = Vn @ coeffs[: self._n_a]  # (N, F)
        den = Vd @ coeffs[self._n_a :] + 1.0  # (N, F)

        # den = np.where(np.abs(den) < 1e-12, np.copysign(1e-12, den), den)

        return num / den  # (N, F)

    @dataclass
    class FitConfig:
        regularization: float = 1e-2
        den_floor: float = 0.1
        collocation_margin: float = 0.05
        max_nfev: int = 800

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
        x_ref: float | None = None,
    ) -> np.ndarray:
        """Fit rational function per frequency.

        The denominator is required to stay positive across the evaluation
        range: a root inside it is a pole, and the cross-section diverges.
        Frequencies whose fits have poles are refitted with coefficient bounds.

        Returns
        -------
        coeffs : (n_params, F)
        """
        from scipy.optimize import least_squares

        self.fit_config = self.FitConfig()

        x = np.ravel(x)
        y = np.atleast_2d(y)
        n_freq = y.shape[1]
        coeffs = np.zeros((self._n_params, n_freq))

        x_min, x_max = x.min(), x.max()
        x_absmax = max(abs(x_min), abs(x_max), 1e-12)
        margin = self.fit_config.collocation_margin * (x_max - x_min)
        x_check = np.linspace(x_min - margin, x_max + margin, 512)

        # Fit against x/x_absmax so every power of the abscissa is order one and
        # the coefficients share a scale; they are divided back out at the end.
        Vn = self._vandermonde_num(x / x_absmax)
        Vd = self._vandermonde_den(x / x_absmax)
        Vd_check = self._vandermonde_den(x_check / x_absmax)

        b_scale = np.ones(self._n_b)
        floor = self.fit_config.den_floor
        # Sufficient bound: each denominator term is limited so their sum can
        # never pull 1 + sum(b_i x^i) below the floor.
        b_bound = (1.0 - floor) / (self._n_b * np.maximum(b_scale, 1e-30))

        def _den(b, Vd_):
            den = Vd_ @ b + 1.0
            return np.where(np.abs(den) < 1e-12, np.copysign(1e-12, den), den)

        # a0 is set by the anchor rather than fitted, so the form is 1 at x_ref.
        ref = (x_ref if x_ref is not None else 0.0) / x_absmax
        vn_ref = self._vandermonde_num(np.array([ref]))[0]
        vd_ref = self._vandermonde_den(np.array([ref]))[0]

        def _expand(params):
            if x_ref is None:
                return params[: self._n_a], params[self._n_a :]
            a_rest, b = params[: self._n_a - 1], params[self._n_a - 1 :]
            a0 = (1.0 + vd_ref @ b - vn_ref[1:] @ a_rest) / vn_ref[0]
            return np.concatenate([[a0], a_rest]), b

        def _residual(params, y_col, w):
            a, b = _expand(params)
            fit_res = ((Vn @ a) / _den(b, Vd) - y_col) * w
            reg_res = np.sqrt(self.fit_config.regularization) * b * b_scale
            return np.concatenate([fit_res, reg_res])

        n_params = self._n_params - (1 if x_ref is not None else 0)

        logger.info(
            "Fitting RationalForm (num=%d, den=%d) ...",
            self.numerator_order,
            self.denominator_order,
        )

        n_bounded = 0
        for fi in range(n_freq):
            y_col = y[:, fi]
            if not np.all(np.isfinite(y_col)):
                continue
            w = (
                np.ones_like(y_col)
                if weights is None
                else np.sqrt(np.clip(weights[:, fi], 0.0, None))
            )
            used = w > 0
            if used.sum() < n_params + 1:
                continue

            x0 = np.zeros(n_params)
            try:
                slope, intercept = np.polyfit(x[used], y_col[used], 1)
            except Exception:
                slope, intercept = 0.0, float(np.mean(y_col[used]))
            if x_ref is None:
                x0[0] = intercept
                if self._n_a > 1:
                    x0[1] = slope
            elif self._n_a > 1:
                x0[0] = slope

            result = least_squares(
                _residual,
                x0,
                args=(y_col, w),
                loss="soft_l1",
                max_nfev=int(self.fit_config.max_nfev),
            )

            if _den(_expand(result.x)[1], Vd_check).min() < floor:
                n_free_a = n_params - self._n_b
                lo = np.concatenate([np.full(n_free_a, -np.inf), -b_bound])
                hi = np.concatenate([np.full(n_free_a, np.inf), b_bound])
                result = least_squares(
                    _residual,
                    np.clip(x0, lo, hi),
                    args=(y_col, w),
                    bounds=(lo, hi),
                    max_nfev=int(self.fit_config.max_nfev),
                )
                n_bounded += 1

            unscale = x_absmax ** np.concatenate(
                [np.arange(self._n_a), np.arange(1, self._n_b + 1)]
            )
            coeffs[:, fi] = np.concatenate(_expand(result.x)) / unscale

        if n_bounded:
            logger.info("  %d/%d frequencies refitted pole-free", n_bounded, n_freq)

        return coeffs  # (n_params, F)

    def rescale(self, coeffs: np.ndarray, factor: np.ndarray) -> np.ndarray:
        """Return coefficients whose evaluation is ``factor`` times this one's."""
        scaled = coeffs.copy()
        scaled[: self._n_a] *= factor
        return scaled

    def coefficient_names(self) -> List[str]:
        return [f"rn{i}" for i in range(self._n_a)] + [
            f"rd{i}" for i in range(1, self._n_b + 1)
        ]


class PowerLawForm(FunctionalForm):
    """Power law form: c0 * x^c1 + c2."""

    def evaluate(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        x_safe = np.maximum(np.ravel(x), 1e-12)[:, None]  # avoid power of zero/negative
        c0, c1, c2 = coeffs[0], coeffs[1], coeffs[2]
        return c0 * (x_safe**c1) + c2

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
        x_ref: float | None = None,
    ) -> np.ndarray:
        from scipy.optimize import least_squares

        x_safe = np.maximum(np.ravel(x), 1e-12)
        n_freq = y.shape[1]
        coeffs = np.zeros((3, n_freq))

        logger.info("Fitting PowerLaw model...")
        for fi in range(n_freq):
            y_col = y[:, fi]
            if not np.all(np.isfinite(y_col)):
                continue

            y_min = np.min(y_col)
            y_shift = y_col - y_min + 1e-3
            try:
                p_init = np.polyfit(np.log(x_safe), np.log(y_shift), 1)
                c1_init = p_init[0]
                c0_init = np.exp(p_init[1])
            except Exception:
                c1_init, c0_init = 1.0, 1e-3

            x0 = [c0_init, c1_init, y_min]

            def residual(p):
                return (p[0] * (x_safe ** p[1]) + p[2]) - y_col

            try:
                result = least_squares(residual, x0, loss="soft_l1")
                p_opt = result.x
            except Exception:
                p_opt = x0

            coeffs[:, fi] = p_opt

            if fi % max(1, n_freq // 10) == 0:
                logger.debug("  PowerLaw fit progress: %d/%d", fi + 1, n_freq)

        return coeffs

    def coefficient_names(self) -> List[str]:
        return ["c0", "c1", "c2"]


class NullForm(FunctionalForm):
    """A null form that evaluates to explicitly 0.0 with no fitting variables."""

    def evaluate(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        return np.zeros((np.atleast_1d(x).shape[0], coeffs.shape[1]))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
        x_ref: float | None = None,
    ) -> np.ndarray:
        f_size = y.shape[1] if y.ndim > 1 else 1
        return np.zeros((1, f_size))

    def coefficient_names(self) -> List[str]:
        return ["_dummy"]


# ============================================================================
# Registry of functional forms
# ============================================================================

functional_form_registry: dict[str, FunctionalForm] = {
    "Polynomial": PolynomialForm(),
    "Hinge": HingeForm(),
    "ShiftedReciprocalLaurent": ShiftedReciprocalLaurentForm(),
    "SmoothHinge": SmoothHingeForm(),
    "Rational": RationalForm(),
    "Powerlaw": PowerLawForm(),
    "Null": NullForm(),
}
