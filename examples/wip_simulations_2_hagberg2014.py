"""(WIP) Simulations of multi-echo MRI signal."""

from __future__ import division
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import tetrahydra.core as tet
from tetrahydra.utils import hagberg2014_eq5


M_0 = [100., 50.]
T_2s = [20., 10.]
phi = np.linspace(0, np.pi, 200)


a = hagberg2014_eq5(100, 50, 20, 0)
b = hagberg2014_eq5(100, 50, 20, -np.pi/8)

np.absolute(a)
np.absolute(b)

np.angle(a, deg=True)
np.angle(b, deg=True)

np.exp(-np.complex(0, 1)*phi)

curve_1 = hagberg2014_eq5(M_0[0], np.arange(0, 100), T_2s[0], phi[0:100]*0)
curve_2 = hagberg2014_eq5(M_0[0], np.arange(0, 100), T_2s[0], phi[0:100])
curve_3 = hagberg2014_eq5(M_0[0], np.arange(0, 100), T_2s[0], phi[50:150])

curve_1_mag, curve_1_phi = np.absolute(curve_1), np.angle(curve_1, deg=True)
curve_2_mag, curve_2_phi = np.absolute(curve_2), np.angle(curve_2, deg=True)
curve_3_mag, curve_3_phi = np.absolute(curve_3), np.angle(curve_3, deg=True)

# handle signal magnitude
fig1 = plt.figure(1)
ax = fig1.add_subplot(221)
ax.set_title('Hagberg et al. 2014, eq 5')
slc = ax.plot(curve_1_phi, linewidth=3,
              label='B: M0 %i, T2* %i, phi %i'
              % (M_0[0], T_2s[0], np.rad2deg(phi[0])))
slc = ax.plot(curve_2_phi, linewidth=2,
              label='G: M0 %i, T2* %i, phi %i'
              % (M_0[1], T_2s[0], np.rad2deg(phi[0])))
slc = ax.plot(curve_3_phi, linewidth=1,
              label='R: M0 %i, T2* %i, phi %i'
              % (M_0[0], T_2s[1], np.rad2deg(phi[0])))
ax.legend(loc='upper right')

plt.show()
