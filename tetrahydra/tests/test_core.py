"""Test core functions."""

import pytest
import numpy as np
from tetrahydra.core import closure, perturb, power


def test_closure():
    """Test closure operator."""
    data = np.random.random([2, 3])
    assert np.sum(closure(data), axis=1) == pytest.approx(np.ones(2))


def test_perturb():
    """Test perturbation operator."""
    data = np.random.random([2, 3])
    p_vals = np.array([1., 2., 3.])  # perturbation values
    assert perturb(data, p_vals, reclose=False) == pytest.approx(data * p_vals)


def test_power():
    """Test powering operator."""
    data = np.random.random([2, 3])
    assert power(data, np.pi, reclose=False) == pytest.approx(data**np.pi)
