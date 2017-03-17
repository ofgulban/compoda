"""Simplex space Aitchison metrics for MR images."""

import os
import numpy as np
from __future__ import division
from nibabel import Nifti1Image, load, save
from tetrahydra.core import aitchison_norm, aitchison_dist, closure
from tetrahydra.utils import truncate_and_scale


np.seterr(divide='ignore', invalid='ignore')

# Load data
vol1 = load('/media/Data_Drive/Benchmark_Data/compositional_data/Valentin/GRE/05deg.nii.gz')
vol2 = load('/media/Data_Drive/Benchmark_Data/compositional_data/Valentin/GRE/10deg.nii.gz')
vol3 = load('/media/Data_Drive/Benchmark_Data/compositional_data/Valentin/GRE/15deg.nii.gz')

nr_vol = 3  # TODO: get rid of this

# Prepare headers.
basename = vol1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(vol1.get_filename())
niiHeader, niiAffine = vol1.header, vol1.affine
shape = vol1.shape + (nr_vol,)

# hold the data in a list for preprocessing
volumes = []
volumes.append(vol1.get_data())
volumes.append(vol2.get_data())
volumes.append(vol3.get_data())
# volumes.append(vol4.get_data())


# Imputing, considering uint16 precision, replace 0 with 1
for i in range(len(volumes)):
    volumes[i][volumes[i] == 0] = 1.

# (optional) Normalize(0-1)
# for i in range(len(volumes)):
#     volumes[i] = truncate_and_scale(volumes[i], percMin=0, percMax=100)

# reshape data as 4D images (spatial indices of voxels and type of image)
rgb = np.asarray(volumes, dtype="float")
rgb = np.transpose(rgb, axes=[1, 2, 3, 0])

# flatten for more intuitive format because voxel-wise nature of operations.
rgb = rgb.reshape(shape[0]*shape[1]*shape[2], shape[3])

# closure (to 1)
closure(rgb)

# compute aitchison norm
test = aitchison_norm(rgb)
test = np.nan_to_num(test)
test = test.reshape(shape[0:3])

# save aitchison norm
out = Nifti1Image(test, header=niiHeader, affine=niiAffine)
save(out, os.path.join(dirname, 'TEST_anorm.nii.gz'))

# # compute compositional gradient magnitudes
# rgb = rgb.reshape(shape[0], shape[1], shape[2], shape[3])
#
# # x
# shape_x = [(shape[0]-2)*shape[1]*shape[2], shape[3]]
# shape_x_inv = [(shape[0]-2), shape[1], shape[2]]
# source_x = rgb[1:-1, :, :, :].reshape(shape_x)
# shift_x_f = rgb[2:, :, :].reshape(shape_x)
# shift_x_b = rgb[0:-2, :, :].reshape(shape_x)
# # y
# shape_y = [shape[0]*(shape[1]-2)*shape[2], shape[3]]
# shape_y_inv = [shape[0], (shape[1]-2), shape[2]]
# source_y = rgb[:, 1:-1, :, :].reshape(shape_y)
# shift_y_f = rgb[:, 2:, :].reshape(shape_y)
# shift_y_b = rgb[:, 0:-2, :].reshape(shape_y)
# # z
# shape_z = [shape[0]*shape[1]*(shape[2]-2), shape[3]]
# shape_z_inv = [shape[0], shape[1], (shape[2]-2)]
# source_z = rgb[:, :, 1:-1, :].reshape(shape_z)
# shift_z_f = rgb[:, :, 2:].reshape(shape_z)
# shift_z_b = rgb[:, :, 0:-2].reshape(shape_z)
#
# comp_gra_mag = []
# comp_gra_mag.append(aitchison_dist(source_x, shift_x_f).reshape(shape_x_inv))
# comp_gra_mag.append(aitchison_dist(source_x, shift_x_b).reshape(shape_x_inv))
# comp_gra_mag.append(aitchison_dist(source_y, shift_y_f).reshape(shape_y_inv))
# comp_gra_mag.append(aitchison_dist(source_y, shift_y_b).reshape(shape_y_inv))
# comp_gra_mag.append(aitchison_dist(source_z, shift_z_f).reshape(shape_z_inv))
# comp_gra_mag.append(aitchison_dist(source_z, shift_z_b).reshape(shape_z_inv))
#
# ali = np.zeros([shape[0], shape[1], shape[2], 6])
# ali[1:-1, :, :, 0] = comp_gra_mag[0]
# ali[1:-1, :, :, 1] = comp_gra_mag[1]
# ali[:, 1:-1, :, 2] = comp_gra_mag[2]
# ali[:, 1:-1, :, 3] = comp_gra_mag[3]
# ali[:, :, 1:-1, 4] = comp_gra_mag[4]
# ali[:, :, 1:-1, 5] = comp_gra_mag[5]
# ali = np.nan_to_num(ali)
#
#
# # save compositional gradients
# out = Nifti1Image(ali, affine=niiAffine)
# save(out, os.path.join(dirname, 'TEST_comp_gra.nii.gz'))
#
# veli = np.sqrt(np.sum(ali**2.0, axis=3))
# out = Nifti1Image(veli, affine=niiAffine)
# save(out, os.path.join(dirname, 'TEST_comp_gra_mag.nii.gz'))
