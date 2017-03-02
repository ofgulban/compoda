"""Simulations of milti-echo MRI signal."""

from __future__ import division
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import tetrahydra.core as tet
from tetrahydra.utils import posse1999_eq1

S_0 = 100
T_2s = [40, 50, 60]
TE_n = np.arange(0, 100)
nr_measurements = len(T_2s)
nr_TE = len(TE_n)
selected_TE = [15, 30, 45]

# The loop is explicitly following descriptions of Posse et al. 1999
signal = np.zeros((nr_measurements, nr_TE))
for r in range(nr_measurements):
    signal_r = posse1999_eq1(S_0, TE_n, T_2s[r])
    signal[r, :] = signal_r

# create HRF-like timeseries by appending reversed signal
timeseries = np.zeros((len(selected_TE), 6))
for r in range(len(selected_TE)):
    timeseries[r, :] = np.hstack((signal[:, selected_TE[r]],
                                  signal[::-1, selected_TE[r]]))

# Compositional descriptive (Aitchison norm) of signal
bary = tet.closure(np.copy(signal.T))
anorm = tet.aitchison_norm(bary)

# Plots -----------------------------------------------------------------------
fig = plt.figure(1)
fig.suptitle('Posse 1999, eq. 1', fontsize=14)

# Across TEs
ax_1 = fig.add_subplot(131)
ax_1.set_title('(A) T2 decay curves across measurements \n\
    of fluctuating T2*')
ax_1.set_xlabel('Time (echo domain)')
for r in range(nr_measurements):
    slc = ax_1.plot(signal[r, :], lw=3, label='T2* %i' % T_2s[r])
ax_1.legend(loc='upper right')

# Across measurements (hrf-like representation)
ax_2 = fig.add_subplot(132)
ax_2.set_title('(B) BOLD-like signal due to T2* fluctuations \n\
    at certain TEs across measurements')
ax_2.set_xlabel('Time (measurement domain)')
for r in range(len(selected_TE)):
    slc = ax_2.plot(timeseries[r, :], linewidth=3,
                    label='TE %i' % selected_TE[r])
ax_2.legend(loc='upper left')

# S0 invariant simplex space
ax_3 = fig.add_subplot(133)
ax_3.set_title('(C) Aitchison norm of B')
ax_3.set_xlabel('Time (??? domain)')
slc = ax_3.plot(anorm, linewidth=3)

plt.show()
