"""Test core functions."""

import pytest
import numpy as np
from compoda.core import closure, perturb, power
from compoda.core import alr_transformation, inverse_alr_transformation
from compoda.core import clr_transformation, inverse_clr_transformation
from compoda.core import ilr_transformation, inverse_ilr_transformation


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


def test_alr():
    """Test alr transformations."""
    # Given
    data = np.random.random([2, 3])
    # When
    data = closure(data)
    output_1 = alr_transformation(data)
    output_2 = inverse_alr_transformation(output_1)
    # Then
    assert output_2 == pytest.approx(data)


def test_clr():
    """Test clr transformations."""
    # Given
    data = np.random.random([2, 3])
    # When
    data = closure(data)
    output_1 = clr_transformation(data)
    output_2 = inverse_clr_transformation(output_1)
    # Then
    assert output_2 == pytest.approx(data)


def test_ilr():
    """Test ilr transformations."""
    # Given
    data = np.random.random([2, 3])
    # When
    data = closure(data)
    output_1 = ilr_transformation(data)
    output_2 = inverse_ilr_transformation(output_1)
    # Then
    assert output_2 == pytest.approx(data)
