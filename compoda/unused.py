"""Currently unused functions are archived here."""

import sys
import numpy as np


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


def progress_output(input1, input2, text=''):
    """Handy progress output procedure for command line.

    Parameters
    ----------
    input1, input2 : Int
        input1 out of input2

    """
    sys.stdout.write(("\r%i/%i " + text) % (input1, input2))
    sys.stdout.flush()


def isometric_projection(data, alpha=45, beta=35.5):
    """Do isometric projection.

    Parameters
    ----------
    TODO

    """
    alpha, beta = np.deg2rad(alpha), np.deg2rad(beta)
    # trans = (data.min() + data.max()) / 2.
    # data = data - trans
    r_1 = np.array([[1., 0., 0.],
                    [0., np.cos(alpha), np.sin(alpha)],
                    [0., -np.sin(alpha), np.cos(alpha)]])
    r_2 = np.array([[np.cos(beta), 0., -np.sin(beta)],
                    [0., 1., 0.],
                    [np.sin(beta), 0., np.cos(beta)]])
    rot = np.dot(r_1, r_2)
    projection = np.array([[1., 0., 0.], [0., 1., 0.]]).T
    dims = data.shape
    out = np.zeros((dims[0], dims[1]-1))
    temp = np.zeros(dims)
    for i in range(data.shape[0]):
        temp[i, :] = 1./np.sqrt(6) * np.dot(data[i, :], rot)
    for i in range(data.shape[0]):
        out[i, :] = np.dot(temp[i, :], projection)
    return out
