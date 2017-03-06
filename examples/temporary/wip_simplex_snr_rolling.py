"""Experimental stuff."""

from __future__ import division
import os
import numpy as np
import time
from nibabel import load, save, Nifti1Image
from retinex_for_mri.utils import truncate_and_scale
from tetrahydra.core import closure, aitchison_dist

# Load nifti
nii = load('/media/Data_Drive/Benchmark_Data/compositional_data/Ingo/20160215_high_quality/func_pRF.nii.gz')
basename_nii = nii.get_filename().split(os.extsep, 1)[0]

data = np.asarray(nii.get_data(), dtype=float)
dims = data.shape


# barycentric coordinates to preserve n-color information
bary = data.reshape(dims[0]*dims[1]*dims[2], dims[3])
bary = np.nan_to_num(bary)
bary[bary == 0] = 1  # impute (swap zeros with minimum measurable value)
bary = closure(bary, k=100.)

print "Rolling aitchison distances..."
start = time.time()

# rolling aitchison distance along time
n_simplex = 3
roll_tem = np.zeros((dims[0]*dims[1]*dims[2], dims[3]))
for i in range(dims[3]-n_simplex-1):
    print i
    comp_time_1 = bary[:, i:i+3]
    comp_time_2 = bary[:, i+1:i+1+3]
    # temporal compositional difference
    roll_tem[:, i] = aitchison_dist(comp_time_1, comp_time_2)


# spatial compositional distance
roll_spa = np.zeros((dims[0], dims[1], dims[2], dims[3]))
for i in range(dims[3]-n_simplex):
    print i
    comp_time = bary[:, i:i+3]
    comp_time = comp_time.reshape(dims[0], dims[1], dims[2], n_simplex+1)
    c_dims = comp_time.shape
    # x
    shape_x = [(c_dims[0]-2)*c_dims[1]*c_dims[2], c_dims[3]]
    shape_x_inv = [(c_dims[0]-2), c_dims[1], c_dims[2]]
    source_x = comp_time[1:-1, :, :, :].reshape(shape_x)
    shift_x_f = comp_time[2:, :, :].reshape(shape_x)
    shift_x_b = comp_time[0:-2, :, :].reshape(shape_x)
    # y
    shape_y = [c_dims[0]*(c_dims[1]-2)*c_dims[2], c_dims[3]]
    shape_y_inv = [c_dims[0], (c_dims[1]-2), c_dims[2]]
    source_y = comp_time[:, 1:-1, :, :].reshape(shape_y)
    shift_y_f = comp_time[:, 2:, :].reshape(shape_y)
    shift_y_b = comp_time[:, 0:-2, :].reshape(shape_y)
    # z
    shape_z = [c_dims[0]*c_dims[1]*(c_dims[2]-2), c_dims[3]]
    shape_z_inv = [c_dims[0], c_dims[1], (c_dims[2]-2)]
    source_z = comp_time[:, :, 1:-1, :].reshape(shape_z)
    shift_z_f = comp_time[:, :, 2:].reshape(shape_z)
    shift_z_b = comp_time[:, :, 0:-2].reshape(shape_z)

    comp_gra_mag = []
    comp_gra_mag.append(aitchison_dist(source_x, shift_x_f).reshape(shape_x_inv))
    comp_gra_mag.append(aitchison_dist(source_x, shift_x_b).reshape(shape_x_inv))
    comp_gra_mag.append(aitchison_dist(source_y, shift_y_f).reshape(shape_y_inv))
    comp_gra_mag.append(aitchison_dist(source_y, shift_y_b).reshape(shape_y_inv))
    comp_gra_mag.append(aitchison_dist(source_z, shift_z_f).reshape(shape_z_inv))
    comp_gra_mag.append(aitchison_dist(source_z, shift_z_b).reshape(shape_z_inv))

    comp_disT_spa = np.zeros([c_dims[0], c_dims[1], c_dims[2], 6])
    comp_disT_spa[1:-1, :, :, 0] = comp_gra_mag[0]
    comp_disT_spa[1:-1, :, :, 1] = comp_gra_mag[1]
    comp_disT_spa[:, 1:-1, :, 2] = comp_gra_mag[2]
    comp_disT_spa[:, 1:-1, :, 3] = comp_gra_mag[3]
    comp_disT_spa[:, :, 1:-1, 4] = comp_gra_mag[4]
    comp_disT_spa[:, :, 1:-1, 5] = comp_gra_mag[5]
    comp_disT_spa = np.nan_to_num(comp_disT_spa)
    roll_spa[..., i] = np.sqrt(np.sum(comp_disT_spa**2.0, axis=3))

end = time.time()
print 'Done in:', (end - start), 'seconds'

# temporal comp difference
roll_tem = roll_tem.reshape(dims[0], dims[1], dims[2], dims[3])
out = Nifti1Image(np.sum(roll_tem, axis=3), affine=np.eye(4))
save(out, basename_nii + '_roll_tem_mag.nii.gz')

# spatial comp difference
out = Nifti1Image(roll_spa, affine=np.eye(4))
save(out, basename_nii + '_roll_spa.nii.gz')

# spatio-temporal comp difference
roll_spatem = np.sqrt(np.sum(roll_tem**2 + roll_spa**2, axis=3))
out = Nifti1Image(roll_spatem, affine=np.eye(4))
save(out, basename_nii + '_roll_spatem_mag.nii.gz')
print 'Finished.'
