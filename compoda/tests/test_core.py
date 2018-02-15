"""Test core functions."""

import pytest
import numpy as np
from compoda.core import closure, perturb, power


def test_closure():
    """Test closure operator."""
    # Given
    data = np.random.random([2, 3])
    expected = np.ones(2)
    # When
    output = np.sum(closure(data), axis=1)
    # Then
    assert output == pytest.approx(expected)


def test_perturb():
    """Test perturbation operator."""
    # Given
    data = np.random.random([2, 3])
    p_vals = np.array([1., 2., 3.])  # perturbation values
    expected = data * p_vals
    # When
    output = perturb(data, p_vals, reclose=False)
    # Then
    assert np.all(output == expected)


def test_power():
    """Test powering operator."""
    # Given
    data = np.random.random([2, 3])
    expected = data**np.pi
    # When
    output = power(data, np.pi, reclose=False)
    # Then
    assert np.all(output == expected)
