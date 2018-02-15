"""Some Bloch equations which might be useful in the future."""

import numpy as np


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


def shan2014_eq6(A=np.array([-0.2, 1.8, -1.8, 0.2]),
                 T=np.array([0.1, 4, 10, 20]), D=np.array([8, 1, 1, 1.2]),
                 t=range(4)):
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
