"""Utility functions used in compositional data analysis."""

import numpy as np
from scipy.linalg import helmert


def truncate_and_scale(data, percMin=0.01, percMax=99.9, zeroTo=1.0):
    """Truncate and scale the data as a preprocessing step.

    Parameters
    ----------
    data : nd numpy array
        Data/image to be truncated and scaled.
    percMin : float, positive
        Minimum percentile to be truncated.
    percMax : float, positive
        Maximum percentile to be truncated.
    zeroTo : float
        Data will be returned in the range from 0 to this number.

    Returns
    -------
    data : nd numpy array
        Truncated and scaled data/image.

    """
    # adjust minimum
    percDataMin = np.percentile(data, percMin)
    data[np.where(data < percDataMin)] = percDataMin
    data = data - data.min()

    # adjust maximum
    percDataMax = np.percentile(data, percMax)
    data[np.where(data > percDataMax)] = percDataMax
    data = 1./data.max() * data
    return data * zeroTo


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


def rad_to_deg(radian):
    """Radian to degree conversion."""
    return radian * 360./(2.*np.pi)


def deg_to_rad(degree):
    """Degree to radian conversion."""
    return degree * (2.*np.pi)/360.


def cart_to_quasipolar(cart_coord):
    """Convert n-dimensional cartesian to quasipolar coordinates [1].

    Parameters
    ----------
    cart_coord : 2d numpy array, shape [n_samples, n_coordinates]
        Cartesian coordinates.

    Returns
    -------
    qpol_coord : 2d numpy array, shape [n_samples, n_coordinates]
        Quasi-polar coordinates. First coordinate is always 'r'
        calculated with euclidean norm.

    Reference
    ---------
    [1] Nguyen, T. M. (2014). N-Dimensional Quasipolar Coordinates -
        Theory and Application, pg 13. University of Las Vegas. URL:
        http://digitalscholarship.unlv.edu/thesesdissertations/2125

    """
    dims = cart_coord.shape
    qpol_coord = np.zeros(dims)
    r = np.linalg.norm(cart_coord, axis=1)
    qpol_coord[:, 0] = r

    def rec_mult(qpol_inter, n, nr_mult):
        """Iterative multiplication of sines [1]."""
        temp = 1.0
        for count in range(nr_mult):
            temp *= np.sin(qpol_inter[:, n+1+count])
        return temp

    # This for loop is the literal implementation of [1].
    # It starts from last quasi-polar (qpol) coordinate and finds n-1 thetas.
    # In other words, it starts to fill the qpol coordinates end to beginning.
    # Multiplication by 2 is done for range correction.
    for i, n in enumerate(range(dims[1]-1, 0, -1)):
        if n == dims[1]-1:
            qpol_coord[:, n] = np.arccos(cart_coord[:, n] / r)
        elif n == 1:
            qpol_coord[:, n] = np.arcsin(cart_coord[:, n]
                                         / (r * rec_mult(qpol_coord, n, i)))
        else:
            qpol_coord[:, n] = np.arccos(cart_coord[:, n]
                                         / (r * rec_mult(qpol_coord, n, i)))
    return qpol_coord


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
