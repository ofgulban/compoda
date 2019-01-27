"""Create a figure for cube to almost hexcone transformation."""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn
from compoda.core import closure, ilr_transformation

# create euclidean 3D lattice
parcels = 10
data = np.zeros([parcels**3, 3], dtype=float)
coords = np.linspace(1000, 2000, parcels)
x, y, z = np.meshgrid(coords, coords, coords)
data[:, 0] = x.flatten()
data[:, 1] = y.flatten()
data[:, 2] = z.flatten()

# isometric logratio transformation
bary = closure(np.copy(data))
ilr = ilr_transformation(bary)

# calculate intensity to demonstrate hexcone interpretation
inten = (np.max(data, axis=1) + np.min(data, axis=1)) / 2.  # intensity
hexc = np.zeros(data.shape)
hexc[:, 0:2], hexc[:, 2] = ilr, inten

# Plots
fig = plt.figure()
ax_1 = plt.subplot(131,  projection='3d')
ax_1.set_title('Euclidean space')
ax_1.set_aspect('equal')
ax_1.scatter(data[:, 0], data[:, 1], data[:, 2], color='k', s=3)

ax_2 = plt.subplot(132)
ax_2.scatter(ilr[:, 0], ilr[:, 1], color='k', s=3)
ax_2.set_xlim(np.percentile(ilr, 0), np.percentile(ilr, 100))
ax_2.set_ylim(np.percentile(ilr, 0), np.percentile(ilr, 100))
ax_2.set_title('ilr transformed')
ax_2.set_aspect('equal')

ax_3 = plt.subplot(133, projection='3d')
ax_3.scatter(hexc[:, 0], hexc[:, 1], hexc[:, 2], color='k', s=3)
ax_3.set_title('Hexcone')
ax_3.set_aspect('equal')

plt.show()
