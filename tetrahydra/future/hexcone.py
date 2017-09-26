"""Exercise different logratio trasnformations.

TODO: Turn this script into trasnformation tests.

"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tetrahydra.core import closure, ilr_transformation

# create euclidean 3D lattice
parcels = 5
data = np.zeros([parcels**3, 3], dtype=float)
coords = np.linspace(1, 2, parcels)
x, y, z = np.meshgrid(coords, coords, coords)
data[:, 0] = x.flatten()
data[:, 1] = y.flatten()
data[:, 2] = z.flatten()

# isometric logratio transformation
bary = closure(np.copy(data))
ilr = ilr_transformation(bary)

# arrive at hexcone
inten = np.mean(data, axis=1)  # intensity
hexc = np.zeros(data.shape)
hexc[:, 0:2], hexc[:, 2] = ilr, inten


# isometric projection
def isometric_projection(data):
    """Do isometric projection."""
    a, b = np.deg2rad(45), np.deg2rad(35.5)
    # trans = (data.min() + data.max()) / 2.
    # data = data - trans
    r_1 = np.array([[1., 0., 0.],
                    [0., np.cos(a), np.sin(a)],
                    [0., -np.sin(a), np.cos(a)]])
    r_2 = np.array([[np.cos(b), 0., -np.sin(b)],
                    [0., 1., 0.],
                    [np.sin(b), 0., np.cos(b)]])
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

isomet = isometric_projection(data)

# Plots
fig = plt.figure()
ax_1 = plt.subplot(131,  projection='3d')
ax_1.set_title('Euclidean space')
ax_1.set_aspect('equal')
ax_1.scatter(data[:, 0], data[:, 1], data[:, 2], color='red', s=5)

ax_2 = plt.subplot(132)
ax_2.scatter(ilr[:, 0], ilr[:, 1], color='green', s=5)
ax_2.set_title('ilr transformed')
ax_2.set_aspect('equal')

ax_3 = plt.subplot(133)
ax_3.scatter(isomet[:, 0], isomet[:, 1], color='blue', s=5)
ax_3.set_title('isometric projection')
ax_3.set_aspect('equal')

plt.show()
