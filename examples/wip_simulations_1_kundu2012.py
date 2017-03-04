"""Simulations of multi-echo MRI signal."""

from __future__ import division
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import tetrahydra.core as tet
from tetrahydra.utils import kundu2012_eq1

S_0 = [100., 50.]
T_2s = [20., 10.]

curve_1 = kundu2012_eq1(S_0=S_0[0], TE_n=np.arange(1, 100), T_2s=T_2s[0])
curve_2 = kundu2012_eq1(S_0=S_0[1], TE_n=np.arange(1, 100), T_2s=T_2s[0])
curve_3 = kundu2012_eq1(S_0=S_0[0], TE_n=np.arange(1, 100), T_2s=T_2s[1])

# mono exponential decays
fig1 = plt.figure(1)
ax_1 = fig1.add_subplot(121)
ax_1.set_title('Transversal decay (Kundu et al. 2012, eq 1)')
ax_1.set_xlabel('$time (echo domain)$')
ax_1.set_ylabel('$M_z$')
slc = ax_1.plot(curve_1, linewidth=3,
              label='B: S0 %i, T2* %i' % (S_0[0], T_2s[0]))
slc = ax_1.plot(curve_2, linewidth=3,
              label='G: S0 %i, T2* %i' % (S_0[1], T_2s[0]))
slc = ax_1.plot(curve_3, linewidth=3,
              label='R: S0 %i, T2* %i' % (S_0[0], T_2s[1]))
ax_1.legend(loc='upper right')

# barycentric coordinates of S_0 change or R_2s change
bary_1 = tet.closure(np.vstack((curve_1, curve_2)).T)
bary_2 = tet.closure(np.vstack((curve_1, curve_3)).T)
bary_3 = tet.closure(np.vstack((curve_2, curve_3)).T)
anorm_1 = tet.aitchison_norm(bary_1)
anorm_2 = tet.aitchison_norm(bary_2)
anorm_3 = tet.aitchison_norm(bary_3)

ax_2 = fig1.add_subplot(122)
ax_2.set_title('Aitchison norm of compositions')
ax_2.set_xlabel('$time (echo domain)$')
ax_2.set_ylabel(r'$\|x\|_\alpha,\ x\ \epsilon\ S^D$')
sll = ax_2.plot(anorm_1, linewidth=3, color='orange', label='B and G')
sll = ax_2.plot(anorm_2, linewidth=3, color='magenta', label='B and R')
sll = ax_2.plot(anorm_3, linewidth=1, color='black', label='G and R')
ax_2.legend(loc='upper left')

plt.show()
