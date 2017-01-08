"""Pseudo-color test for mri data."""

from __future__ import division
import os
import numpy as np
from nibabel import load, save, Nifti1Image
# from conversions import rgb2hsl, hsl2rgb, AutoScale
from AutoScale import AutoScale
np.seterr(divide='ignore', invalid='ignore')

"""Load Data"""
#
vol1 = load('/home/faruk/Data/retinex_tests/com_michelle/T1.nii.gz')
vol2 = load('/home/faruk/Data/retinex_tests/com_michelle/T2s.nii.gz')
vol3 = load('/home/faruk/Data/retinex_tests/com_michelle/PD.nii.gz')

basename = vol1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(vol1.get_filename())
niiHeader, niiAffine = vol1.header, vol1.affine
shape = vol1.shape + (3,)

# Preprocess
vol1 = AutoScale(vol1.get_data(), percMin=0, percMax=100)
vol2 = AutoScale(vol2.get_data(), percMin=0, percMax=100)
vol3 = AutoScale(vol3.get_data(), percMin=0, percMax=100)

rgb = np.zeros(shape)
rgb[:, :, :, 0] = vol1
del vol1
rgb[:, :, :, 1] = vol2
del vol2
rgb[:, :, :, 2] = vol3
del vol3

# flatten for easier imagination since this is a voxel-wise operation.
flat = rgb.reshape(shape[0]*shape[1]*shape[2], shape[3])

# The center of mass of a simplex homogenous material is the average of
# vertices of the simplex.
simplex = np.zeros([shape[0]*shape[1]*shape[2], 3, 3])
# rgb representation to simplex vertex coordinates (using diagonal matrix)
simplex[:, 0, 0] = flat[:, 0]
simplex[:, 1, 1] = flat[:, 1]
simplex[:, 2, 2] = flat[:, 2]
center_of_mass = np.sum(simplex, axis=1)
center_of_mass = 1.0 / (3.0) * center_of_mass

# L2 norm (Euclidean distance) of the center of mass vector
norm = np.linalg.norm(center_of_mass, axis=1)

# norm needs to be normalized with range
normalizer = np.max(flat, axis=1) + np.min(flat, axis=1)
sat_ndim = norm / normalizer

sat_ndim = sat_ndim.reshape(shape[0], shape[1], shape[2])
# swap nans for graceful rendition (better use of the dynamic range)
min = np.nanmin(sat_ndim)
sat_ndim[np.isnan(sat_ndim)] = min
sat_ndim = AutoScale(sat_ndim, percMin=0, percMax=100)  # normalize

# save as nifti
out = Nifti1Image(sat_ndim, header=niiHeader, affine=niiAffine)
save(out, os.path.join(dirname, 'TEST_sat_ndim.nii.gz'))

print 'ndim saturation is done.'

del center_of_mass, norm, normalizer, sat_ndim

# Saturation from extrema -----------------------------------------------------
maxima, minima = np.max(flat, axis=1), np.min(flat, axis=1)
center_of_extrema = np.zeros(maxima.shape + (2,))
center_of_extrema[:, 0] = np.copy(maxima)
center_of_extrema[:, 1] = np.copy(minima)
center_of_extrema = 1.0 / (2.0) * center_of_extrema

# L2 norm (Euclidean distance) of the center of mass of extrema vector
norm = np.linalg.norm(center_of_extrema, axis=1)

# norm needs to be normalized with lightness
# plane = np.zeros(maxima.shape + (2,))
# plane[:, 0] = minima
# plane[:, 1] = np.copy(norm)

# !!! I need to find cartesian coordinates of simplex vertices to find center of mass
plane = np.zeros(maxima.shape + (3,))
plane[:, 0] = 1
plane[:, 1] = 1
plane[:, 2] = 1
epicenter = np.sum(plane, axis=1)
epicenter = 1.0 / (3.0) * epicenter

normalizer = np.linalg.norm(plane, axis=1)

# # normalizer = maxima + minima
sat_extr = norm  # / normalizer

sat_extr = sat_extr.reshape(shape[0], shape[1], shape[2])
# swap nans for graceful rendition (better use of the dynamic range)
min = np.nanmin(sat_extr)
sat_extr[np.isnan(sat_extr)] = min
sat_extr = AutoScale(sat_extr, percMin=0, percMax=100)  # normalize

# save as nifti
out = Nifti1Image(sat_extr, header=niiHeader, affine=niiAffine)
save(out, os.path.join(dirname, 'TEST_sat_extr.nii.gz'))

print 'extrema saturation is done.'
