"""Utility functions."""

# Part of Tetrahydra library
# Copyright (C) 2016-2017  Omer Faruk Gulban
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
import sys
import numpy as np


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


def progress_output(input1, input2, text=''):
    """Handy progress output procedure for command line.

    Parameters
    ----------
    input1, input2 : Int
        input1 out of input2

    """
    sys.stdout.write(("\r%i/%i " + text) % (input1, input2))
    sys.stdout.flush()


def bloch_mh(M_0=800, TI=np.arange(80, 10000), T_1=1000, TR=7000, alpha=0.95):
    """(WIP) Bloch T1 equation.

    Parameters
    ----------
    M_0 : float
    TI : np.array, shape 1d
        Time intervals in milliseconds.
    T_1 : float
        T1 value of a certain tissue (eg. white matter).
    TR : float
        Repetition time in milliseconds.
    alpha : float

    Returns
    -------
    signal : np.array, shape 1d, float

    """
    signal = M_0 * (1. - 2. * alpha * np.exp(-(TI/T_1) + np.exp(-TR/T_1)))
    return signal


def kundu2012_eq1(S_0, TE_n, T_2s):
    """Kundu et al. 2012, equation 1.

    Parameters
    ----------
    S_0 : float
    TE_n : float
        Time in millisecons.
    T_2s : float

    Returns
    -------
    signal : float

    """
    signal = S_0 * np.exp(-TE_n/T_2s)
    return signal


def hagberg2014_eq5(M_0, TE_n, T_2s, phi):
    """Hagberg et al. 2014, equation 5.

    Parameters
    ----------
    M_0 : float
    TE_n : float
        Time in millisecons.
    T_2s : float
    phi : float
        Phase term in radians.

    Returns
    -------
    signal : complex

    """
    signal = M_0 * np.exp(-TE_n/T_2s) * np.exp(-np.complex(0, 1)*phi)
    return signal


def shan2014_eq6(A=[-0.2, 1.8, -1.8, 0.2], T=[0.1, 4, 10, 20],
                 D=[8, 1, 1, 1.2], t=range(4)):
    """Shan et al. 2014, equation 6.

    Parameters
    ----------
    A : float
        Height and direction of HRf.
    T : float
        Shift center of HRF.
    D : float
        Slope of HRF.
    t : float
        Timepoints

    Returns
    -------
    signal : float

    """
    A, T, D, t = np.asarray(A), np.asarray(T), np.asarray(D), np.asarray(t)
    signal = np.zeros(t.shape)
    for i in t:
        signal[i] = signal[i-1] + A[i]/(1. + np.exp((t[i]-T[i]) / D[i]))
    return signal


def posse1999_eq1(S_0, TE_n, T_2s, g=0, h=0):
    """Transverse (or spin-spin) magnetization decay (Posse et al. 1999, eq. 1).

    Parameters
    ----------
    S_0 : float
        The initial signal amplitude. May vary from measurement to
        measurement due to hardware instabilities or flow-related
        saturation effects.
    TE_n : float
        Echo time in milliseconds.
    T_2s : float
        T2* is the stimulus-dependent relaxation (decay) time constant.
    g : float
        White noise (thermal noise and hardware instabilities).
    h : float
        More slowly varying noise which reflects physiologica
        mechanisms, such as heart-beat-related brain pulsation and
        stimulus independent vasomotor activity.

    Returns
    -------
    signal : float

    """
    signal = S_0 * np.exp(-TE_n/T_2s) + g + h
    return signal


def bloch_long_relax(M_0, t, T_1):
    """Longitudinal (or spin-lattice) relaxation.

    Parameters
    ----------
    M_0 : float
        The initial signal amplitude. May vary from measurement to
        measurement due to hardware instabilities or flow-related
        saturation effects.
    T_1 : float
        T1 is the tissue dependent relaxation time constant.
    t : float
        Time in millisecons.

    Returns
    -------
    signal : float

    """
    signal = M_0 * (1 - np.exp(-t/T_1))
    return signal
