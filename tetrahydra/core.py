"""Core functions used in compositional data analysis."""

from __future__ import division
import numpy as np
from scipy.linalg import helmert


def closure(data, k=1.0):
    """Apply closure to data, sample-wise.

    Parameters
    ----------
    data : 2d numpy array, shape [n_samples, n_measurements]
        Data to be closed to a certain constant.
    k : float, positive
        Sum of the measurements will be equal to this number.

    Returns
    -------
    data : 2d numpy array, shape [n_samples, n_measurements]
        Closed data.

    """
    data_sum = np.sum(data, axis=1)
    for i in range(data.shape[1]):
        data[:, i] /= data_sum
    return data


def aitchison_inner_product(x, y):
    """Aitchison inner product of vectors in D dimensional simplex space [1].

    Parameters
    ----------
    x, y : 2d numpy array, shape = [n_samples, n_measurements]
        A vector in simplex space with barycentric coordinates.

    Returns
    -------
    ip_xy : 1d numpy array, shape = [n_samples, ]
        Aitchison inner product of samples.

    Reference
    ---------
    [1] Pawlowsky-Glahn, V., Egozcue, J. J., & Tolosana-Delgado, R.
        (2015). Modelling and Analysis of Compositional Data, pg. 26.
        Chichester, UK: John Wiley & Sons, Ltd.
        DOI: 10.1002/9781119003144

    """
    dims = x.shape
    temp = np.zeros(dims[0])
    for i in range(dims[1]):
        for j in range(dims[1]):
            temp += np.log(x[:, i]/x[:, j]) * np.log(y[:, i]/y[:, j])
    ip_xy = 1.0/(2.0*dims[1])*temp
    return ip_xy


def aitchison_norm(x):
    """Aitchison norm of vectors in D dimensional simplex space [1].

    Parameters
    ----------
    x : 2d numpy array, shape = [n_samples, n_measurements]
        A vector in simplex space with barycentric coordinates.

    Returns
    -------
    x_a : 1d numpy array, shape = [n_samples, ]
        Aitchison norm of samples.

    Reference
    ---------
    [1] Pawlowsky-Glahn, V., Egozcue, J. J., & Tolosana-Delgado, R.
        (2015). Modelling and Analysis of Compositional Data, pg. 26.
        Chichester, UK: John Wiley & Sons, Ltd.
        DOI: 10.1002/9781119003144

    """
    dims = x.shape
    temp = np.zeros(dims[0])
    for i in range(dims[1]):
        for j in range(dims[1]):
            temp += np.log(x[:, i] / x[:, j])**2
    x_a = np.sqrt(1.0/(2.0*dims[1])*temp)
    return x_a


def aitchison_dist(x, y):
    """Aitchison distance between vectors in D dimensional simplex space [1].

    Parameters
    ----------
    x, y : 2d numpy array, shape = [n_samples, n_measurements]
        Vectors in simplex space with barycentric coordinates.

    Returns
    -------
    d_xy : 1d numpy array, shape = [n_samples, ]
        Aitchison distance between vectors.

    Reference
    ---------
    [1] Pawlowsky-Glahn, V., Egozcue, J. J., & Tolosana-Delgado, R.
        (2015). Modelling and Analysis of Compositional Data, pg. 26.
        Chichester, UK: John Wiley & Sons, Ltd.
        DOI: 10.1002/9781119003144

    """
    dims = x.shape
    temp = np.zeros(dims[0])
    for i in range(dims[1]):
        for j in range(dims[1]):
            temp += (np.log(x[:, i]/x[:, j]) - np.log(y[:, i]/y[:, j]))**2
    d_xy = np.sqrt(1.0/(2.0*dims[1])*temp)
    return d_xy


def geometric_mean(data):
    """Geometric mean of 2d data.

    Parameters
    ----------
    data : 2d numpy array, shape [n_samples, n_measurements]
        Some data in the specified format.

    Returns
    -------
    data : 2d numpy array, shape [n_samples, n_coordinates-1]
        Sample-wise geometric mean of the data.

    """
    dims = data.shape
    gmean = np.ones(dims[0])
    for i in range(dims[1]):
        gmean *= data[:, i]
    return np.power(gmean, 1.0/dims[1])


def alr_transforation(data):
    """Additive logratio transformation.

    Parameters
    ----------
    data : 2d numpy array, shape [n_samples, n_coordinates]
        Barycentric coordinates (closed) of data in simplex space.

    Returns
    -------
    out : 2d numpy array, shape [n_samples, n_coordinates-1]
        Coordinates in real space.

    """
    dims = data.shape
    out = np.zeros([dims[0], dims[1]-1])
    for i in range(dims[1]-1):
        out[:, i] = data[:, i]/dims[1]
    return out


def clr_transforation(data):
    """Centered logratio transformation.

    Parameters
    ----------
    data : 2d numpy array, shape [n_samples, n_coordinates]
        Barycentric coordinates (closed) of data in simplex space.

    Returns
    -------
    out : 2d numpy array, shape [n_samples, n_coordinates-1]
        Coordinates in real space.

    """
    dims = data.shape
    out = np.zeros([dims[0], dims[1]-1])
    g = geometric_mean(data)
    for i in range(dims[1]-1):
        out[:, i] = np.log(data[:, i]/g)
    return out


def ilr_transformation(data):
    """Isometric logratio transformation (not vectorized).

    Parameters
    ----------
    data : 2d numpy array, shape [n_samples, n_coordinates]
        Barycentric coordinates (closed) of data in simplex space.

    Returns
    -------
    out : 2d numpy array, shape [n_samples, n_coordinates-1]
        Coordinates in real space.

    """
    dims = data.shape
    out = np.zeros((dims[0], dims[1]-1))
    helmertian = helmert(dims[1]).T
    for i in range(data.shape[0]):
        out[i, :] = np.dot(np.log(data[i, :]), helmertian)
    return out
