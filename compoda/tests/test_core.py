"""Test core functions."""

import pytest
import numpy as np
from compoda.core import closure, perturb, power
from compoda.core import alr_transformation, inverse_alr_transformation
from compoda.core import clr_transformation, inverse_clr_transformation
from compoda.core import ilr_transformation, inverse_ilr_transformation
from compoda.core import aitchison_dist, aitchison_norm, sample_center
from compoda.core import aitchison_inner_product


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


def test_aitchison_norm():
    """Test Aitchison norm."""
    # Given
    data = np.array([1/3, 1/3, 1/3])
    # When
    output = aitchison_norm(data[None, :])
    # Then
    assert output == pytest.approx(0)


def test_aitchison_distance():
    """Test Aitchison distance."""
    # Given
    data = np.array([[1/5, 2/5, 2/5], [2/5, 1/5, 2/5], [2/5, 2/5, 1/5]])
    # When
    output = aitchison_dist(data[0:2, :], data[1:3, :])
    # Then
    assert output[0] == output[1]


def test_sample_center():
    """Test compositional sample center."""
    # Given
    data = np.array([[1/5, 2/5, 2/5], [2/5, 1/5, 2/5], [2/5, 2/5, 1/5]])
    # When
    output = sample_center(data)
    # Then
    assert output[0] == pytest.approx(np.array([1/3, 1/3, 1/3]))


def test_aitchison_inner_product():
    """Test Aitchison inner product."""
    # Given
    data = np.array([[1/5, 2/5, 2/5], [2/5, 1/5, 2/5], [2/5, 2/5, 1/5]])
    # When
    output = aitchison_inner_product(data[0:2, :], data[1:3, :])
    assert output[0] == output[1]
