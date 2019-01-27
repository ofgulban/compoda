"""Test core functions."""

import pytest
import numpy as np
import compoda.core as coda


def test_closure():
    """Test closure operator."""
    # Given
    data = np.random.random([2, 3])
    expected = np.ones(2)
    # When
    output = np.sum(coda.closure(data), axis=1)
    # Then
    assert output == pytest.approx(expected)


def test_perturb():
    """Test perturbation operator."""
    # Given
    data = np.random.random([2, 3])
    p_vals = np.array([1., 2., 3.])  # perturbation values
    expected = data * p_vals
    # When
    output = coda.perturb(data, p_vals, reclose=False)
    # Then
    assert np.all(output == expected)


def test_power():
    """Test powering operator."""
    # Given
    data = np.random.random([2, 3])
    expected = data**np.pi
    # When
    output = coda.power(data, np.pi, reclose=False)
    # Then
    assert np.all(output == expected)


def test_alr():
    """Test alr transformations."""
    # Given
    data = np.random.random([2, 3])
    # When
    data = coda.closure(data)
    output_1 = coda.alr_transformation(data)
    output_2 = coda.inverse_alr_transformation(output_1)
    # Then
    assert output_2 == pytest.approx(data)


def test_clr():
    """Test clr transformations."""
    # Given
    data = np.random.random([2, 3])
    # When
    data = coda.closure(data)
    output_1 = coda.clr_transformation(data)
    output_2 = coda.inverse_clr_transformation(output_1)
    # Then
    assert output_2 == pytest.approx(data)


def test_ilr():
    """Test ilr transformations."""
    # Given
    data = np.random.random([2, 3])
    # When
    data = coda.closure(data)
    output_1 = coda.ilr_transformation(data)
    output_2 = coda.inverse_ilr_transformation(output_1)
    # Then
    assert output_2 == pytest.approx(data)


def test_aitchison_norm():
    """Test Aitchison norm."""
    # Given
    data = np.array([1/3, 1/3, 1/3])
    # When
    output = coda.aitchison_norm(data[None, :])
    # Then
    assert output == pytest.approx(0)


def test_aitchison_distance():
    """Test Aitchison distance."""
    # Given
    data = np.array([[1/5, 2/5, 2/5], [2/5, 1/5, 2/5], [2/5, 2/5, 1/5]])
    # When
    output = coda.aitchison_dist(data[0:2, :], data[1:3, :])
    # Then
    assert output[0] == output[1]


def test_sample_center():
    """Test compositional sample center."""
    # Given
    data = np.array([[1/5, 2/5, 2/5],
                     [2/5, 1/5, 2/5],
                     [2/5, 2/5, 1/5]])
    # When
    output = coda.sample_center(data)
    # Then
    assert output[0] == pytest.approx(np.array([1/3, 1/3, 1/3]))


def test_aitchison_inner_product():
    """Test Aitchison inner product."""
    # Given
    data = np.array([[1/5, 2/5, 2/5],
                     [2/5, 1/5, 2/5],
                     [2/5, 2/5, 1/5]])
    # When
    output = coda.aitchison_inner_product(data[0:2, :], data[1:3, :])
    assert output[0] == output[1]
