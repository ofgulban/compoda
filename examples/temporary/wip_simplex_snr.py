"""SNR alternative by using n-simplex as the sampling space of mri data."""

from __future__ import division
import os
import numpy as np
import time
from nibabel import load, save, Nifti1Image
from retinex_for_mri.utils import truncate_and_scale
from tetrahydra.core import closure, aitchison_dist
from scipy.ndimage import gaussian_filter

# Load nifti
nii = load('/media/Data_Drive/Benchmark_Data/compositional_data/Ingo/20160215_high_quality/func_pRF_25.nii.gz')
basename_nii = nii.get_filename().split(os.extsep, 1)[0]

data = np.asarray(nii.get_data(), dtype=float)

# # Z score
# zscore = zscore(data, axis=3, ddof=2)
# out = Nifti1Image(zscore, affine=np.eye(4))
# save(out, basename_nii + '_zscore.nii.gz')
#
# # std
# out = Nifti1Image(np.std(data, axis=3), affine=np.eye(4))
# save(out, basename_nii + '_std.nii.gz')

# save average image too
out = Nifti1Image(np.mean(data, axis=3), affine=np.eye(4))
save(out, basename_nii + '_mean.nii.gz')

dims = data.shape
# for i in range(dims[3]):
#     data[..., i] = truncate_and_scale(data[..., i], percMin=0., percMax=100.)

# barycentric coordinates to preserve n-color information
bary = data.reshape(dims[0]*dims[1]*dims[2], dims[3])
bary = closure(bary)

bary = bary.reshape(dims[0], dims[1], dims[2], dims[3])
out = Nifti1Image(bary, affine=np.eye(4))
save(out, basename_nii + '_bary.nii.gz')

# save aitchison distance gradient magnitude for fun (TODO: modularize)
start = time.time()
print 'Calculating aitchison gradient magnitude, may take some time...'
bary_2 = bary.reshape(dims[0], dims[1], dims[2], dims[3])

# x
shape_x = [(dims[0]-2)*dims[1]*dims[2], dims[3]]
shape_x_inv = [(dims[0]-2), dims[1], dims[2]]
source_x = bary_2[1:-1, :, :, :].reshape(shape_x)
shift_x_f = bary_2[2:, :, :].reshape(shape_x)
shift_x_b = bary_2[0:-2, :, :].reshape(shape_x)
print '.'
# y
shape_y = [dims[0]*(dims[1]-2)*dims[2], dims[3]]
shape_y_inv = [dims[0], (dims[1]-2), dims[2]]
source_y = bary_2[:, 1:-1, :, :].reshape(shape_y)
shift_y_f = bary_2[:, 2:, :].reshape(shape_y)
shift_y_b = bary_2[:, 0:-2, :].reshape(shape_y)
print '.'
# z
shape_z = [dims[0]*dims[1]*(dims[2]-2), dims[3]]
shape_z_inv = [dims[0], dims[1], (dims[2]-2)]
source_z = bary_2[:, :, 1:-1, :].reshape(shape_z)
shift_z_f = bary_2[:, :, 2:].reshape(shape_z)
shift_z_b = bary_2[:, :, 0:-2].reshape(shape_z)
print '.'

comp_gra_mag = []
comp_gra_mag.append(aitchison_dist(source_x, shift_x_f).reshape(shape_x_inv))
comp_gra_mag.append(aitchison_dist(source_x, shift_x_b).reshape(shape_x_inv))
comp_gra_mag.append(aitchison_dist(source_y, shift_y_f).reshape(shape_y_inv))
comp_gra_mag.append(aitchison_dist(source_y, shift_y_b).reshape(shape_y_inv))
comp_gra_mag.append(aitchison_dist(source_z, shift_z_f).reshape(shape_z_inv))
comp_gra_mag.append(aitchison_dist(source_z, shift_z_b).reshape(shape_z_inv))

ali = np.zeros([dims[0], dims[1], dims[2], 6])
ali[1:-1, :, :, 0] = comp_gra_mag[0]
ali[1:-1, :, :, 1] = comp_gra_mag[1]
ali[:, 1:-1, :, 2] = comp_gra_mag[2]
ali[:, 1:-1, :, 3] = comp_gra_mag[3]
ali[:, :, 1:-1, 4] = comp_gra_mag[4]
ali[:, :, 1:-1, 5] = comp_gra_mag[5]
ali = np.nan_to_num(ali)

# # save compositional gradients
# out = Nifti1Image(ali, affine=np.eye(4))
# save(out, basename_nii + '_adist.nii.gz')

veli = np.sqrt(np.sum(ali**2.0, axis=3))
out = Nifti1Image(veli, affine=np.eye(4))
save(out, basename_nii + '_adist_sos.nii.gz')

veli = gaussian_filter(veli, 1)
out = Nifti1Image(veli, affine=np.eye(4))
save(out, basename_nii + '_adist_sos_gauss.nii.gz')

end = time.time()
print 'Done in:', (end - start), 'seconds'
print 'Finished.'
