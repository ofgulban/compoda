"""Convert barycentric coordinates to cartesian using regular simplices.

Assumes the following preprocessing steps are already done:
    1) Imputing, swap zeroes with minimum possible measurement.
    2) Normalization, 0 to 1.
    3) Reshaping, 2d numpy array with shape [n_samples, n_measurements]

NOTE: Sth is wrong. Numbers are not looking correct

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from simplex_coordinates1 import simplex_coordinates1
from simplex_coordinates2 import simplex_coordinates2
from utils import closure, cart_to_quasipolar, rad_to_deg, deg_to_rad

data = np.ones([6, 3])
data[0, 0] *= 100
data[1, 0:2] *= 100
data[2, 1] *= 100
data[3, 1:] *= 100
data[4, 2] *= 100
data[5, 0::2] *= 100


dims = data.shape

# cartesian coordinates of a regular simplex, shape [n_vertices, n_coords]
cart_simplex_coords = simplex_coordinates2(dims[1]-1).T
fig, ax = plt.subplots()
pol = Polygon(cart_simplex_coords, alpha=0.25)
ax.add_patch(pol)
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_aspect('equal')
plt.show()

# convert cartesian to quasi-polar coordinates
simplex_qpol_coords = cart_to_quasipolar(cart_simplex_coords)
simplex_qpol_coords[:, 1:] = rad_to_deg(simplex_qpol_coords[:, 1:])

print simplex_qpol_coords

# # close data to 1
# data = closure(data)
#
# # find cartesian coordinates of measurements in the simplex (TODO: function)
# cart_simplex_coords = np.array([cart_simplex_coords, ] * dims[0])
# data_cart_coords = np.zeros([dims[0], dims[1], cart_simplex_coords.shape[2]])
# for m in range(dims[1]):  # for every measurement(vertex)
#     for c in range(cart_simplex_coords.shape[2]):  # for every coordinate
#         data_cart_coords[:, m, c] = cart_simplex_coords[:, m, c] * data[:, m]
#
# # find center of mass
# data_com_coords = np.sum(data_cart_coords, axis=1)/data_cart_coords.shape[1]

# # convert cartesian to quasi-polar coordinates
# data_qpol_coords = cart_to_quasipolar(data_com_coords)
# data_qpol_coords[:, 1:] = rad_to_deg(data_qpol_coords[:, 1:])
#
# print data_qpol_coords
