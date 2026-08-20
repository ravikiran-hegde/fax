"""Functional forms: weighting, and the guarantees the fitted model must keep."""

import numpy as np
import pytest

from faxsec.forms import HingeForm, PolynomialForm, RationalForm, functional_form_registry


def test_registry_forms_roundtrip_shapes():
    x = np.linspace(-2.0, 2.0, 50)
    y = np.stack([x * 0.5 + 1.0, -x + 2.0], axis=1)
    for name, form in functional_form_registry.items():
        coeffs = form.fit(x, y)
        out = form.evaluate(x, coeffs)
        assert out.shape == y.shape, name
        assert np.all(np.isfinite(out)), name
        assert len(form.coefficient_names()) == coeffs.shape[0] or name == "Hinge"


@pytest.mark.parametrize("form", [PolynomialForm(order=2), HingeForm(), RationalForm()])
def test_zero_weight_samples_are_ignored(form):
    """A sample given zero weight must not influence the fit at all."""
    x = np.linspace(-3.0, 3.0, 80)
    truth = 0.4 * x + 1.5
    y = truth.copy()
    corrupt = np.zeros_like(x, dtype=bool)
    corrupt[::7] = True
    y[corrupt] = -500.0  # stand-in for underflowed reference values

    weights = (~corrupt).astype(float)[:, None]
    fitted = form.evaluate(x, form.fit(x, y[:, None], weights))[:, 0]

    assert np.max(np.abs(fitted[~corrupt] - truth[~corrupt])) < 1e-6


def test_rational_denominator_has_no_root_in_range():
    """A pole inside the evaluation range makes the cross-section diverge."""
    rng = np.random.default_rng(0)
    x = np.linspace(-90.0, 90.0, 60)
    # Curvature plus noise is what previously drove the denominator through zero.
    y = np.stack(
        [8.0 + 0.05 * x - 3e-4 * x**2 + rng.normal(0, 0.3, x.size) for _ in range(12)],
        axis=1,
    )
    form = RationalForm()
    coeffs = form.fit(x, y)

    n_a = form._n_a
    x_check = np.linspace(x.min() * 1.1, x.max() * 1.1, 400)
    powers = np.stack([x_check**i for i in range(1, form._n_b + 1)], axis=1)
    denominator = powers @ coeffs[n_a:] + 1.0
    assert denominator.min() > 0.0

    values = form.evaluate(x_check, coeffs)
    assert np.all(np.isfinite(values))
    assert np.abs(values).max() < 1e3


def test_hinge_breakpoint_stays_where_there_is_data():
    """With the low-pressure end masked out the kink must not go there."""
    x = np.linspace(0.0, 10.0, 60)
    y = np.where(x < 6.0, x, 6.0 + 3.0 * (x - 6.0))[:, None]
    weights = (x > 4.0).astype(float)[:, None]
    coeffs = HingeForm().fit(x, y, weights)
    assert coeffs[3, 0] >= 4.0
