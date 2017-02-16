"""(WIP) Barycentric to carthesian for simplices."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import distance
from tetrahydra.future import simplex_coordinates1, simplex_coordinates2
from tetrahydra.utils import cart_to_quasipolar, rad_to_deg

fig, ax = plt.subplots()

# reference regular simplex
tri = simplex_coordinates2(2).T
com = np.sum(tri, axis=0)/tri.shape[0]
print 'Barycenter_r:', com

# bary_multipliers forces applied (not totally sure about this part)
rgb = np.array([1, 100, 100])
rgb = 3*rgb/np.sum(rgb)  # 3 is the number of vertices

tri2 = (tri.T * rgb).T

com2 = np.sum(tri2, axis=0)/tri2.shape[0]
print 'Barycenter_1:', com2

# distance between reference barycenter and bary_multipliers barycenter
dist = distance.pdist([com, com2])
print 'Center to center distance:', dist

dist2 = distance.pdist([com, com2], 'euclidean')
# print 'Euclidean distance:', dist2

# quasi-polar coordinates of com2
temp = np.zeros((2, 3))
temp[1, :] = com2
qpol = cart_to_quasipolar(temp)
print rad_to_deg(qpol[:, 1:])

# plot
pol = Polygon(tri, alpha=0.25)
ax.add_patch(pol)
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_aspect('equal')

# plot tri 2
pol2 = Polygon(tri2, alpha=0.25, color='red')
ax.add_patch(pol2)
plt.show()

# -----------------------------------------------------------------------------

# import os
# from nibabel import load, save, Nifti1Image
# from AutoScale import AutoScale
# from __future__ import division
# np.seterr(divide='ignore', invalid='ignore')
#
#
# vol1 = load('/home/faruk/Data/retinex_tests/michelle/T1_ANISO.nii.gz')
# vol2 = load('/home/faruk/Data/retinex_tests/michelle/T2s_ANISO.nii.gz')
# vol3 = load('/home/faruk/Data/retinex_tests/michelle/PD_ANISO.nii.gz')
# basename = vol1.get_filename().split(os.extsep, 1)[0]
# dirname = os.path.dirname(vol1.get_filename())
# niiHeader, niiAffine = vol1.header, vol1.affine
# shape = vol1.shape + (3,)
#
# # Preprocess
# vol1 = AutoScale(vol1.get_data(), percMin=0.1, percMax=99.0, zeroTo=1.0)
# vol2 = AutoScale(vol2.get_data(), percMin=0.1, percMax=99.0, zeroTo=1.0)
# vol3 = AutoScale(vol3.get_data(), percMin=0.1, percMax=99.0, zeroTo=1.0)
#
# rgb = np.zeros(shape)
# rgb[:, :, :, 0] = vol1
# del vol1
# rgb[:, :, :, 1] = vol2
# del vol2
# rgb[:, :, :, 2] = vol3
# del vol3
#
# flat = rgb.reshape(shape[0]*shape[1]*shape[2], shape[3])
# flat.shape
#
# # normalize channels
# sum = np.sum(flat, axis=1)
# sum = np.tile(sum, [3, 1]).T
# bary_multipliers = 3.0*flat/sum  # 3 is the number of vertices
# del sum
# bary_multipliers = np.nan_to_num(bary_multipliers)
# flat[100000, :]
# bary_multipliers[100000, :]
#
# tri.shape
# del flat
# tri
# bary_multipliers[100000, :] * tri.T
#
# # find vertices of data simplices
# spx = [(bary_multipliers[i, :] * tri.T) for i in range(bary_multipliers.shape[0])]
# spx = np.asarray(spx)
# spx.shape
# spx = np.transpose(spx, axes=[0, 2, 1])
# spx.shape
#
# # find center of mass of simplices
# spx_com = np.sum(spx, axis=1)/spx.shape[1]
#
# com_tile = np.tile(com, [spx_com.shape[0], 1])
# com_tile[100000, :]
# spx_com[100000, :]
#
# com_tile.shape
# spx_com.shape
#
# bal = [distance.pdist([com_tile[i, :], spx_com[i, :]]) for i in range(com_tile.shape[0])]
# bal = np.asarray(bal)
# bal = np.squeeze(bal)
# bal = np.reshape(bal, shape[0:3])
#
# out = Nifti1Image(bal, header=niiHeader, affine=niiAffine)
# save(out, basename + '_bal.nii.gz')
# print 'Simplex stuff is done.'
