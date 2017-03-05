"""(WIP) 3D line plots of longitudinal and and transversal magnetization."""

from __future__ import division
import seaborn
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import tetrahydra.core as tet
from tetrahydra.utils import bloch_long_relax, posse1999_eq1

t = np.arange(0, 1000)
S_0 = [100, 100, 180, 180]
T_2s = [90, 50, 50, 50]
T_1 = [1300, 1300, 1300, 800]
IVs = ['T_2^*', 'S_0', 'T_1']

# Plots -----------------------------------------------------------------------
fig = plt.figure(1)

# Magnetization ---------------------------------------------------------------
ax_1 = fig.add_subplot(221, projection='3d')
ax_1.set_xlabel('$time\ (echo domain)$')
ax_1.set_ylabel('$M_{xy}$')
ax_1.set_zlabel('$M_{z}$')

# Aitchison norm of transversal magnetization
ax_3 = fig.add_subplot(223)
ax_3.set_title('Aitchison norm of transversal magnetization')
ax_3.set_xlabel('$time\ (echo domain)$')
ax_3.set_ylabel('Anorm')

# Aitchison norm of longitudinal magnetization
ax_4 = fig.add_subplot(224)
ax_4.set_title('Aitchison norm of longitudinal magnetization')
ax_4.set_xlabel('$time\ (echo domain)$')
ax_4.set_ylabel('Anorm')

# axis 1 data
M_xy, M_z = [], []
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]
for i in range(len(T_2s)):
    M_xy.append(posse1999_eq1(S_0=S_0[i], TE_n=t, T_2s=T_2s[i]))
    M_z.append(bloch_long_relax(M_0=S_0[i], t=t, T_1=T_1[i]))
    ax_1.plot(t, M_xy[i], M_z[i], lw=3, alpha=0.5, color=colors[i])
    # (optional) plot 2D projections
    # ax_1.plot(t, M_xy, zdir='z', lw=1, color=[1, 0, 0])
    # ax_1.plot(t, M_z, zdir='y', lw=1, color=[0, 0, 1])

# axis 3 data
for i in range(len(T_2s)-1):
    bary = tet.closure(np.vstack((M_xy[i], M_xy[i+1])).T)
    anorm = tet.aitchison_norm(bary)
    ax_3.plot(anorm, lw='3', alpha=1,
              label='$IV: '+IVs[i]+'$')
ax_3.set_ylim(-1, 10)
ax_3.legend(loc='upper left')

# axis 4 data
for i in range(len(T_2s)-1):
    bary = tet.closure(np.vstack((M_z[i], M_z[i+1])).T)
    anorm = tet.aitchison_norm(bary)
    ax_4.plot(anorm, lw='3', alpha=1,
              label='$IV: '+IVs[i]+'$')
ax_4.set_ylim(-0.1, 1)
ax_4.legend(loc='upper left')

plt.show()
