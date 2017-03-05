"""Simulations of longitudinal relaxation TIs."""

from __future__ import division
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import tetrahydra.core as tet
from tetrahydra.utils import bloch_long_relax

M_0 = [80, 90, 100]
T_1 = [830, 1100, 1330]
TI = np.arange(0, 4000)
nr_measurements = len(T_1)
nr_TI = len(TI)
selected_TI = [500, 900, 1300, 1700]
# quick WMGM-interface-like sampling
selected_T1s = np.array([1400, 1000, 800])
selected_M0s = np.array([80, 95, 100])

# Considering T1 effects -----------------------------------------------------
signal_T1 = np.zeros((nr_measurements, nr_TI))
for r in range(nr_measurements):
    signal_r = bloch_long_relax(M_0[-1], TI, T_1[r])  # M0 constant
    signal_T1[r, :] = signal_r

# create cross tissue sampling
timeseries_T1 = np.zeros((len(selected_TI), len(selected_T1s)))
for r in range(len(selected_TI)):
    timeseries_T1[r, :] = bloch_long_relax(M_0[-1], selected_TI[r],
                                           selected_T1s)

# Compositional descriptive (Aitchison norm) of signal_T1
bary_T1 = tet.closure(np.copy(timeseries_T1.T))
anorm_T1 = tet.aitchison_norm(bary_T1)

# Considering M0 effects ------------------------------------------------------
signal_M0 = np.zeros((nr_measurements, nr_TI))
for r in range(nr_measurements):
    signal_r = bloch_long_relax(M_0[r], TI, T_1[0])  # T1 constant
    signal_M0[r, :] = signal_r

# create cross tissue sampling
timeseries_M0 = np.zeros((len(selected_TI), len(selected_M0s)))
for r in range(len(selected_TI)):
    timeseries_M0[r, :] = bloch_long_relax(selected_M0s, selected_TI[r],
                                           T_1[0])

# Compositional descriptive (Aitchison norm) of signal_T1
bary_M0 = tet.closure(np.copy(timeseries_M0.T))
anorm_M0 = tet.aitchison_norm(bary_M0)

# Plots -----------------------------------------------------------------------
fig = plt.figure(1)
fig.suptitle('$ M_z(t) = M_0(1-e^{(-t/T_1)})$',
             fontsize=14, x=0.01, y=0.01,
             horizontalalignment='left', verticalalignment='bottom')

# Considering M0 effects -----------------------------------------------------
# Across TIs
ax_1 = fig.add_subplot(231)
ax_1.set_title('(A) Relaxation of longitudinal magnetization,\n\
    Row 1: changing $T_1$, Row 2: changing $M_0$')
ax_1.set_xlabel('Time (echo domain)')
ax_1.set_ylabel('signal (IV: $TI$)')
ax_1.set_ylim((0, 100))
# darker gray for gray matter
colors = ['0.6', '0.25']
for r in [0, -1]:  # only the smallest and biggest value for simplicity
    slc = ax_1.plot(signal_T1[r, :], lw=2, color=colors[r],
                    markevery=selected_TI, marker='h',
                    label='$T_1=%i,\ M_0=%i$' % (T_1[r], M_0[-1]))
ax_1.legend(loc='lower right')

# Across voxels ("WM to GM transition" representation)
ax_2 = fig.add_subplot(232)
ax_2.set_title('(B) MRI signal across voxels (eg.  "WM to GM-like" transition)\n\
    Row 1: changing $T_1$, Row 2: changing $M_0$')
ax_2.set_xlabel('Voxels (spatial domain)')
ax_2.set_xticks(range(len(selected_T1s)))
ax_2.set_ylabel('signal (IV: $T_1=[1400, 1000, 800]$)')
ax_2.set_ylim(0, 90)
# darker grey to signify drop in signal magnitude
colors = ['0.8', '0.6', '0.4', '0.2']
for r in range(len(selected_TI)):
    slc = ax_2.plot(timeseries_T1[r, :], linewidth=5, color=colors[r],
                    label='$TI=%i$' % selected_TI[r])
ax_2.legend(loc='lower left')

# M0 invariant simplex space
ax_3 = fig.add_subplot(233)
ax_3.set_title('(C) Aitchison norm of MRI signal across voxels in B\n\
    (compositional descriptive in n-simplex space)')
ax_3.set_xlabel('Voxels (spatial domain)')
ax_3.set_ylabel('Aitchison norm (IV: $T_1$)')
ax_3.set_ylim(0, 1)
slc = ax_3.plot(anorm_T1, lw=3, color='green')

# Considering M0 effects ------------------------------------------------------
# Across TIs
ax_4 = fig.add_subplot(234)
ax_4.set_xlabel('Time (echo domain)')
ax_4.set_ylabel('signal (IV: $TI$)')
ax_4.set_ylim((0, 100))
# Blues for comparison to T1 curves
colors = [[0.2, 0.2, 1], '0.6']
for r in [-1, 0]:  # only the smallest and biggest value for simplicity
    slc = ax_4.plot(signal_M0[r, :], lw=2, color=colors[r],
                    markevery=selected_TI, marker='h',
                    label='$T_1=%i,\ M_0=%i$' % (T_1[0], M_0[r]))
ax_4.legend(loc='lower right')

# Across voxels ("WM to GM transition" representation)
ax_5 = fig.add_subplot(235)
ax_5.set_xlabel('Voxels (spatial domain)')
ax_5.set_xticks(range(len(selected_M0s)))
ax_5.set_ylabel('signal (IV: $M_0 [80, 95, 100]$)')
ax_5.set_ylim(0, 90)
# darker grey to signify drop in signal magnitude
colors = ['0.8', '0.6', '0.4', '0.2']
for r in range(len(selected_TI)):
    slc = ax_5.plot(timeseries_M0[r, :], linewidth=5, color=colors[r],
                    label='$TI=%i$' % selected_TI[r])
ax_5.legend(loc='lower left')

# M0 invariant simplex space
ax_6 = fig.add_subplot(236)
ax_6.set_xlabel('Voxels (spatial domain)')
ax_6.set_ylabel('Aitchison norm (IV: $M_0$)')
ax_6.set_ylim(0, 1)
slc = ax_6.plot(anorm_M0, lw=3, color='orange')

plt.show()
