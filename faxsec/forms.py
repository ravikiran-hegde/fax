from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


class FunctionalForm(ABC):
    """Base class for generic functional forms f(x)."""

    @abstractmethod
    def evaluate(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate function at given x values."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit coefficients to data y = f(x)."""

    @abstractmethod
    def coefficient_names(self) -> List[str]:
        """Return ordered list of coefficient names."""


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

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        V = self._vandermode(x)
        coeffs, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
        return coeffs

    def coefficient_names(self) -> List[str]:
        start = 0 if self.include_bias else 1
        return [f"p{i}" for i in range(start, self.order + 1)]


class HingeForm(FunctionalForm):
    """Piecewise-linear form with hinge point: c0 + c1*min(x, xb) + c2*max(x-xb, 0)."""

    def __init__(self, include_bias: bool = True):
        self.include_bias = include_bias

    def _hinge_matrix(self, x: np.ndarray, xb: ArrayLike) -> np.ndarray:
        """Construct hinge matrix for evaluation.

        Parameters        ----------
        x : np.ndarray (N,)
            The input data points at which to evaluate the function.
        xb : ArrayLike (F,)
            The breakpoints for the hinge function, one per frequency.

        Returns
        -------
        np.ndarray (N, F, 3)
            The hinge matrix for evaluation.
        """
        x_col = np.ravel(x)[:, None]  # (N, 1)
        xb = np.ravel(xb)[None, :]  # (1, F)

        H = np.empty((len(x), len(xb.ravel()), 3))
        H[:, :, 0] = 1.0 if self.include_bias else 0.0
        H[:, :, 1] = np.minimum(x_col, xb)  # below breakpoint (N, F)
        H[:, :, 2] = np.maximum(x_col - xb, 0.0)  # above breakpoint (N, F)
        return H

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
        X = self._hinge_matrix(x, coeffs[-1])  # (N, F, 3)
        return np.einsum("NFK,KF->NF", X, coeffs[:-1])  # (N, F, 3) @ (3, F) -> (N, F)

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit hinge coefficients optimizing breakpoint xb per frequency."""

        from scipy.optimize import minimize_scalar

        n_freq = y.shape[1]
        x_col = x[:, None]

        fit = np.zeros((4, n_freq))  # c0, c1, c2, xb

        def loss(xb, y_col):
            # Design matrix: [1, min(x, xb), max(x-xb, 0)]
            X = self._hinge_matrix(x_col, xb)[:, 0]  # (N, 3)

            if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y_col)):
                return np.inf, np.zeros(X.shape[1])

            coeffs, res, _, _ = np.linalg.lstsq(X, y_col, rcond=None)
            pred = X @ coeffs
            return np.sum((pred - y_col) ** 2), coeffs

        # Optimization per frequency
        logger.info("Fitting Hinge model...")

        bounds = (np.min(x_col), np.max(x_col))

        for fi in range(n_freq):
            y_col = y[:, fi]

            # find optimal breakpoint for this frequency
            result = minimize_scalar(
                lambda xb: loss(xb, y_col)[0],
                bounds=bounds,
                method="bounded",
            )

            xb_opt = float(result.x)

            #  evaluate coefficients at optimal breakpoint
            _, coeffs = loss(xb_opt, y_col)

            fit[:3, fi] = coeffs
            fit[3, fi] = xb_opt

            if fi % max(1, n_freq // 10) == 0:
                logger.debug("Hinge fit progress: %d/%d", fi + 1, n_freq)

        return fit  # (4, F)

    def coefficient_names(self) -> List[str]:
        keys = ["h1", "h2", "h_break"]
        if self.include_bias:
            keys.insert(0, "h0")
        return keys


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
        stability_weight: float = 1.0
        den_floor: float = 0.1
        n_collocation: int = 25
        collocation_margin: float = 0.05
        max_nfev: int = 800

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit rational function per frequency.

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
        x_span = x_max - x_min

        if x_span > 0:
            margin = self.fit_config.collocation_margin * x_span
            xg = np.linspace(
                x_min - margin, x_max + margin, int(self.fit_config.n_collocation)
            )
        else:
            xg = np.array([x_min])

        # Precompute basis matrices
        Vn = self._vandermonde_num(x)  # (N, n_a)
        Vd = self._vandermonde_den(x)  # (N, n_b)
        Vd_g = self._vandermonde_den(xg)  # (G, n_b)

        b_scale = np.array([x_absmax**i for i in range(1, self._n_b + 1)])

        def _den(b, Vd_):
            den = Vd_ @ b + 1.0
            return np.where(np.abs(den) < 1e-12, np.copysign(1e-12, den), den)

        def _residual(
            params: np.ndarray, y_col: np.ndarray, y_scale: float
        ) -> np.ndarray:
            a = params[: self._n_a]
            b = params[self._n_a :]

            fit_res = (Vn @ a) / _den(b, Vd) - y_col

            reg_res = np.sqrt(self.fit_config.regularization) * b * b_scale

            small = (
                np.maximum(0.0, self.fit_config.den_floor - np.abs(_den(b, Vd_g)))
                / self.fit_config.den_floor
            )
            stab_res = np.sqrt(self.fit_config.stability_weight) * y_scale * small

            return np.concatenate([fit_res, reg_res, stab_res])

        logger.info(
            "Fitting RationalForm (num=%d, den=%d) ...",
            self.numerator_order,
            self.denominator_order,
        )

        for fi in range(n_freq):
            y_col = y[:, fi]
            if not np.all(np.isfinite(y_col)):
                continue

            y_scale = max(float(np.ptp(y_col)), 1e-6)

            x0 = np.zeros(self._n_params)
            try:
                slope, intercept = np.polyfit(x, y_col, 1)
            except Exception:
                slope, intercept = 0.0, float(np.mean(y_col))
            x0[0] = intercept
            if self._n_a > 1:
                x0[1] = slope

            result = least_squares(
                _residual,
                x0,
                args=(y_col, y_scale),
                loss="soft_l1",
                max_nfev=int(self.fit_config.max_nfev),
            )

            coeffs[:, fi] = result.x

            if fi % max(1, n_freq // 10) == 0:
                logger.debug("  Rational fit progress: %d/%d", fi + 1, n_freq)

        return coeffs  # (n_params, F)

    def coefficient_names(self) -> List[str]:
        return [f"rn{i}" for i in range(self._n_a)] + [
            f"rd{i}" for i in range(1, self._n_b + 1)
        ]


class NullForm(FunctionalForm):
    """A null form that evaluates to explicitly 0.0 with no fitting variables."""

    def evaluate(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        return np.zeros((np.atleast_1d(x).shape[0], coeffs.shape[1]))

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
    "SmoothHinge": SmoothHingeForm(),
    "Rational": RationalForm(),
    "Null": NullForm(),
}
