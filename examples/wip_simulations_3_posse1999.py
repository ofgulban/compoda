"""Simulations of multi-echo MRI signal_T2s."""

from __future__ import division
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import tetrahydra.core as tet
from tetrahydra.utils import posse1999_eq1

S_0 = [120, 112, 100]
T_2s = [40, 50, 60]
TE_n = np.arange(0, 100)
nr_measurements = len(T_2s)
nr_TE = len(TE_n)
selected_TE = np.array([15, 30, 45])
# quick HRF-like fMRI signal
selected_T2s = np.array([40, 40, 50, 60, 60, 50, 40, 37, 40, 40])
selected_S0s = np.array([100, 100, 112, 120, 120, 112, 100, 95, 100, 100])

# Considering T2* effects -----------------------------------------------------
# The loop is explicitly following descriptions of Posse et al. 1999
signal_T2s = np.zeros((nr_measurements, nr_TE))
for r in range(nr_measurements):
    signal_r = posse1999_eq1(S_0[-1], TE_n, T_2s[r])  # S0 constant
    signal_T2s[r, :] = signal_r

# create HRF-like timeseries by appending reversed signal_T2s
timeseries_T2s = np.zeros((len(selected_TE), len(selected_T2s)))
for r in range(len(selected_TE)):
    timeseries_T2s[r, :] = posse1999_eq1(S_0[-1], selected_TE[r], selected_T2s)

# Compositional descriptive (Aitchison norm) of signal_T2s
bary_T2s = tet.closure(np.copy(timeseries_T2s.T))
anorm_T2s = tet.aitchison_norm(bary_T2s)

# Considering S0 effects ------------------------------------------------------
signal_S0 = np.zeros((nr_measurements, nr_TE))
for r in range(nr_measurements):
    signal_r = posse1999_eq1(S_0[r], TE_n, T_2s[0])  # T2* constant
    signal_S0[r, :] = signal_r

# create HRF-like timeseries by appending reversed signal_T2s
timeseries_S0 = np.zeros((len(selected_TE), len(selected_S0s)))
for r in range(len(selected_TE)):
    timeseries_S0[r, :] = posse1999_eq1(selected_S0s, selected_TE[r], T_2s[0])

# Compositional descriptive (Aitchison norm) of signal_T2s
bary_S0 = tet.closure(np.copy(timeseries_S0.T))
anorm_S0 = tet.aitchison_norm(bary_S0)

# Plots -----------------------------------------------------------------------

fig = plt.figure(1)
fig.suptitle(r'$S(t_r, TE_n) = S_0(t_r) exp{(-TE_n/T_2^*(t_r))} \
    + g_{nr} + h_r$',
             fontsize=14, x=0.01, y=0.01,
             horizontalalignment='left', verticalalignment='bottom')

# Considering T2* effects -----------------------------------------------------
# Across TEs
ax_1 = fig.add_subplot(231)
ax_1.set_title('(A) Decay of transverse magnetization across measurements,\n\
    Row 1: fluctuating $T_2^*$, Row 2: fluctuating $S_0$')
ax_1.set_xlabel('Time (echo domain)')
ax_1.set_ylabel('signal (IV: $TE$)')
ax_1.set_ylim((0, 120))
# darker red for deoxygenated blood
colors = [(0.5, 0, 0), (1, 0, 0)]
for r in [0, -1]:  # only the smallest and biggest value for simplicity
    slc = ax_1.plot(signal_T2s[r, :], lw=2, color=colors[r],
                    label='$T_2^*=%i, S_0=%i$' % (T_2s[r], S_0[-1]))
ax_1.legend(loc='upper right')

# Across measurements (hrf-like representation)
ax_2 = fig.add_subplot(232)
ax_2.set_title('(B) Multi-echo HRF-like fMRI signal due to\n\
    Row 1: fluctuating $T_2^*$, Row 2: fluctuating $S_0$')
ax_2.set_xlabel('Time (measurement domain)')
ax_2.set_ylabel('signal (IV: $T_2^*$)')
ax_2.set_ylim(0, 90)
# darker grey to signify drop in signal magnitude
colors = ['0.75', '0.4', '0.1']
for r in range(len(selected_TE)):
    slc = ax_2.plot(timeseries_T2s[r, :], linewidth=5, color=colors[r],
                    label='$TE=%i$' % selected_TE[r])
ax_2.legend(loc='lower center')

# S0 invariant simplex space
ax_3 = fig.add_subplot(233)
ax_3.set_title('(C) Aitchison norm of Multi-echo HRF-like fMRI signal\n\
    (compositional descriptive in n-simplex space)')
ax_3.set_xlabel('Time (measurement domain)')
ax_3.set_ylabel('Aitchison norm (IV: $T_2^*$)')
ax_3.set_ylim(0, 1)
slc = ax_3.plot(anorm_T2s, lw=3, color='green')

# Considering S0 effects ------------------------------------------------------
# Across TEs
ax_4 = fig.add_subplot(234)
ax_4.set_xlabel('Time (echo domain)')
ax_4.set_ylabel('signal (IV: $TE$)')
ax_4.set_ylim((0, 120))
# Blues for comparison to T2* curves
colors = [(0, 0, 1), (0.5, 0, 0)]
for r in [-1, 0]:  # only the smallest and biggest value for simplicity
    slc = ax_4.plot(signal_S0[r, :], lw=2, color=colors[r],
                    label='$T_2^*=%i, S_0=%i$' % (T_2s[0], S_0[r]))
ax_4.legend(loc='upper right')

# Across measurements (hrf-like representation)
ax_5 = fig.add_subplot(235)
ax_5.set_xlabel('Time (measurement domain)')
ax_5.set_ylabel('signal (IV: $S_0$)')
ax_5.set_ylim(0, 90)
# darker grey to signify drop in signal magnitude
colors = ['0.75', '0.4', '0.1']
for r in range(len(selected_TE)):
    slc = ax_5.plot(timeseries_S0[r, :], linewidth=5, color=colors[r],
                    label='$TE=%i$' % selected_TE[r])
ax_5.legend(loc='lower center')

# S0 invariant simplex space
ax_6 = fig.add_subplot(236)
ax_6.set_xlabel('Time (measurement domain)')
ax_6.set_ylabel('Aitchison norm (IV: $S_0$)')
ax_6.set_ylim(0, 1)
slc = ax_6.plot(anorm_S0, lw=3, color='orange')

plt.show()
