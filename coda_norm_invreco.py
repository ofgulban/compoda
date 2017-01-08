"""Simplex space Aitchison metrics for MR images."""

import os
import numpy as np
from __future__ import division
from nibabel import Nifti1Image, load, save
from utils import truncate_and_scale, aitchison_norm, aitchison_dist

np.seterr(divide='ignore', invalid='ignore')

# Load data
vol1 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0000_ANISO.nii.gz')
vol2 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0001_ANISO.nii.gz')
vol3 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0002_ANISO.nii.gz')
vol4 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0003_ANISO.nii.gz')
vol5 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0004_ANISO.nii.gz')
vol6 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0005_ANISO.nii.gz')
vol7 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0006_ANISO.nii.gz')
vol8 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0007_ANISO.nii.gz')
vol9 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0008_ANISO.nii.gz')
vol10 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0009_ANISO.nii.gz')
vol11 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0010_ANISO.nii.gz')
vol12 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0011_ANISO.nii.gz')
vol13 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0012_ANISO.nii.gz')
vol14 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0013_ANISO.nii.gz')
vol15 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0014_ANISO.nii.gz')
vol16 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0015_ANISO.nii.gz')
vol17 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0016_ANISO.nii.gz')
vol18 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0017_ANISO.nii.gz')
vol19 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0018_ANISO.nii.gz')
vol20 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0019_ANISO.nii.gz')
vol21 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0020_ANISO.nii.gz')
vol22 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0021_ANISO.nii.gz')
vol23 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0022_ANISO.nii.gz')
vol24 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0023_ANISO.nii.gz')
vol25 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0024_ANISO.nii.gz')
vol26 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0025_ANISO.nii.gz')
vol27 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0026_ANISO.nii.gz')
vol28 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0027_ANISO.nii.gz')
vol29 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0028_ANISO.nii.gz')
vol30 = load('/home/faruk/Data/Ingo/ir/aniso/AVG0029_ANISO.nii.gz')

# Prepare headers.
basename = vol1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(vol1.get_filename())
niiHeader, niiAffine = vol1.header, vol1.affine
shape = vol1.shape + (30,)

# hold the data in a list for preprocessing
volumes = []
volumes.append(vol1.get_data())
volumes.append(vol2.get_data())
volumes.append(vol3.get_data())
volumes.append(vol4.get_data())
volumes.append(vol5.get_data())
volumes.append(vol6.get_data())
volumes.append(vol7.get_data())
volumes.append(vol8.get_data())
volumes.append(vol9.get_data())
volumes.append(vol10.get_data())
volumes.append(vol11.get_data())
volumes.append(vol12.get_data())
volumes.append(vol13.get_data())
volumes.append(vol14.get_data())
volumes.append(vol15.get_data())
volumes.append(vol16.get_data())
volumes.append(vol17.get_data())
volumes.append(vol18.get_data())
volumes.append(vol19.get_data())
volumes.append(vol20.get_data())
volumes.append(vol21.get_data())
volumes.append(vol22.get_data())
volumes.append(vol23.get_data())
volumes.append(vol24.get_data())
volumes.append(vol25.get_data())
volumes.append(vol26.get_data())
volumes.append(vol27.get_data())
volumes.append(vol28.get_data())
volumes.append(vol29.get_data())
volumes.append(vol30.get_data())

# Imputing, considering uint16 precision, replace 0 with 1
for i in range(len(volumes)):
    volumes[i][volumes[i] == 0] = 1

# Normalize(0-1), truncate if needed
for i in range(len(volumes)):
    volumes[i] = truncate_and_scale(volumes[i], percMin=0, percMax=100)

# reshape data as 4D images (spatial indices of voxels and type of image)
rgb = np.asarray(volumes)
rgb = np.transpose(rgb, axes=[1, 2, 3, 0])

# flatten for more intuitive format because voxel-wise nature of operations.
rgb = rgb.reshape(shape[0]*shape[1]*shape[2], shape[3])

# closure (to 1)
rgb_sum = np.sum(rgb, axis=1)
for dim in range(rgb.shape[1]):
    rgb[:, dim] = rgb[:, dim] / rgb_sum

# compute aitchison norm
test = aitchison_norm(rgb)
test = np.nan_to_num(test)
test = test.reshape(shape[0:3])

# save aitchison norm
out = Nifti1Image(test, header=niiHeader, affine=niiAffine)
save(out, os.path.join(dirname, 'TEST_anorm_full.nii.gz'))

# compute compositional gradient magnitudes
rgb = rgb.reshape(shape[0], shape[1], shape[2], shape[3])

# x
shape_x = [(shape[0]-2)*shape[1]*shape[2], shape[3]]
shape_x_inv = [(shape[0]-2), shape[1], shape[2]]
source_x = rgb[1:-1, :, :, :].reshape(shape_x)
shift_x_f = rgb[2:, :, :].reshape(shape_x)
shift_x_b = rgb[0:-2, :, :].reshape(shape_x)
# y
shape_y = [shape[0]*(shape[1]-2)*shape[2], shape[3]]
shape_y_inv = [shape[0], (shape[1]-2), shape[2]]
source_y = rgb[:, 1:-1, :, :].reshape(shape_y)
shift_y_f = rgb[:, 2:, :].reshape(shape_y)
shift_y_b = rgb[:, 0:-2, :].reshape(shape_y)
# z
shape_z = [shape[0]*shape[1]*(shape[2]-2), shape[3]]
shape_z_inv = [shape[0], shape[1], (shape[2]-2)]
source_z = rgb[:, :, 1:-1, :].reshape(shape_z)
shift_z_f = rgb[:, :, 2:].reshape(shape_z)
shift_z_b = rgb[:, :, 0:-2].reshape(shape_z)

comp_gra_mag = []
comp_gra_mag.append(aitchison_dist(source_x, shift_x_f).reshape(shape_x_inv))
comp_gra_mag.append(aitchison_dist(source_x, shift_x_b).reshape(shape_x_inv))
comp_gra_mag.append(aitchison_dist(source_y, shift_y_f).reshape(shape_y_inv))
comp_gra_mag.append(aitchison_dist(source_y, shift_y_b).reshape(shape_y_inv))
comp_gra_mag.append(aitchison_dist(source_z, shift_z_f).reshape(shape_z_inv))
comp_gra_mag.append(aitchison_dist(source_z, shift_z_b).reshape(shape_z_inv))

ali = np.zeros([shape[0], shape[1], shape[2], 6])
ali[1:-1, :, :, 0] = comp_gra_mag[0]
ali[1:-1, :, :, 1] = comp_gra_mag[1]
ali[:, 1:-1, :, 2] = comp_gra_mag[2]
ali[:, 1:-1, :, 3] = comp_gra_mag[3]
ali[:, :, 1:-1, 4] = comp_gra_mag[4]
ali[:, :, 1:-1, 5] = comp_gra_mag[5]
ali = np.nan_to_num(ali)


# save compositional gradients
out = Nifti1Image(ali, affine=niiAffine)
save(out, os.path.join(dirname, 'TEST_comp_gra.nii.gz'))

veli = np.sqrt(np.sum(ali**2.0, axis=3))
out = Nifti1Image(veli, affine=niiAffine)
save(out, os.path.join(dirname, 'TEST_comp_gra_mag.nii.gz'))
