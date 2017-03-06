"""Core functions used in compositional data analysis."""

from __future__ import division
import numpy as np
from scipy.linalg import helmert


def closure(data, k=1.0):
    """Apply closure to data, sample-wise.

    Parameters
    ----------
    data : 2d numpy array, shape [n_samples, n_measurements]
        Data to be closed to a certain constant. Do not forget to deal with
        zeros in the data before this operation.
    k : float, positive
        Sum of the measurements will be equal to this number.

    Returns
    -------
    data : 2d numpy array, shape [n_samples, n_measurements]
        Closed data.

    Reference
    ---------
    [1] Pawlowsky-Glahn, V., Egozcue, J. J., & Tolosana-Delgado, R.
        (2015). Modelling and Analysis of Compositional Data, pg. 9.
        Chichester, UK: John Wiley & Sons, Ltd.
        DOI: 10.1002/9781119003144

    """
    data_sum = np.sum(data, axis=1)
    for i in range(data.shape[1]):
        data[:, i] /= data_sum
    return data


def perturb(x, y):
    """Perturbation (analogous to addition in real space).

    Parameters
    ----------
    x, y: 2d numpy array, shape [n_samples, n_measurements]
        Input x will be perturbed by y. Use y**-1 as the second input for
        perturbation difference (analogous to subtraction in real space).

    Returns
    -------
    out : 2d numpy array, shape [n_samples, n_measurements]
        Perturbed x.

    Reference
    ---------
    [1] Pawlowsky-Glahn, V., Egozcue, J. J., & Tolosana-Delgado, R.
        (2015). Modelling and Analysis of Compositional Data, pg. 24.
        Chichester, UK: John Wiley & Sons, Ltd.
        DOI: 10.1002/9781119003144

    """
    out = closure(x * y)
    return out


def power(x, a):
    """Powering transformation (analogous to multiplication in real space).

    Parameters
    ----------
    x : 2d numpy array, shape [n_samples, n_measurements]
        Input x will be powered by a.
    a : float
        Constant, real number.

    Returns
    -------
    out : 2d numpy array, shape [n_samples, n_measurements]
        Power transformed x.

    Reference
    ---------
    [1] Pawlowsky-Glahn, V., Egozcue, J. J., & Tolosana-Delgado, R.
        (2015). Modelling and Analysis of Compositional Data, pg. 24.
        Chichester, UK: John Wiley & Sons, Ltd.
        DOI: 10.1002/9781119003144

    """
    out = closure(np.power(x, a))
    return out


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


def inverse_ilr_transformation(data):
    """Inverse isometric logratio transformation (not vectorized).

    Parameters
    ----------
    data : 2d numpy array, shape [n_samples, n_coordinates]
        Coordinates in real space.

    Returns
    -------
    out : 2d numpy array, shape [n_samples, n_coordinates-1]
        Barycentric coordinates (closed) of data in simplex space.

    """
    dims = data.shape
    out = np.zeros((dims[0], dims[1]+1))
    helmertian = helmert(dims[1]+1)
    for i in range(data.shape[0]):
        out[i, :] = np.exp(np.dot(data[i, :], helmertian))
    return closure(out)


def sample_center(data):
    """Sample center.

    Parameters
    ----------
    data : 2d numpy array, shape [n_samples, n_coordinates]
        Barycentric coordinates (closed) of data in simplex space.

    Returns
    -------
        center : 2d numpy array, shape [1, n_coordinates]
        Central tendency of a compositional sample.

    Reference
    ---------
    [1] Pawlowsky-Glahn, V., Egozcue, J. J., & Tolosana-Delgado, R.
        (2015). Modelling and Analysis of Compositional Data, pg. 66.
        Chichester, UK: John Wiley & Sons, Ltd.
        DOI: 10.1002/9781119003144

    """
    dims = data.shape
    center = np.prod(np.power(data, 1./dims[0]), axis=0)
    return closure(center[None, :])


def sample_total_variance(data, center):
    """Sample total variance.

    Parameters
    ----------
    data : 2d numpy array, shape [n_samples, n_coordinates]
        Barycentric coordinates (closed) of data in simplex space.

    center : 2d numpy array, shape [1, n_coordinates]
        Central tendency of a compositional sample.

    Returns
    -------
        tot_var : float
        Global dispersion of compositional sample.

    Reference
    ---------
    [1] Pawlowsky-Glahn, V., Egozcue, J. J., & Tolosana-Delgado, R.
        (2015). Modelling and Analysis of Compositional Data, pg. 67.
        Chichester, UK: John Wiley & Sons, Ltd.
        DOI: 10.1002/9781119003144

    """
    dims = data.shape
    temp = 0
    center = np.ones(dims) * center
    temp = np.sum(aitchison_dist(data, center)**2, axis=0)
    tot_var = 1./dims[0] * temp
    return tot_var


def sample_sstd(data):
    """Sample simplicial standard deviation.

    Parameters
    ----------
    data : 2d numpy array, shape [n_samples, n_coordinates]
        Barycentric coordinates (closed) of data in simplex space.

    Returns
    -------
        sstd : float
        Simplicial standard deviation of the sample.

    Reference
    ---------
    [1] Pawlowsky-Glahn, V., Egozcue, J. J., & Tolosana-Delgado, R.
        (2015). Modelling and Analysis of Compositional Data, pg. 111.
        Chichester, UK: John Wiley & Sons, Ltd.
        DOI: 10.1002/9781119003144

    """
    dims = data.shape
    totvar = sample_total_variance(data, sample_center(data))
    sstd = np.sqrt(totvar/(dims[1]-1))
    return sstd
