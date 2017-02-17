"""Example for epi inhomogeneity correction together with n-dim MSRCP."""

from __future__ import division
import os
import numpy as np
from nibabel import load, save, Nifti1Image
from retinex_for_mri.core import multi_scale_retinex_3d
from retinex_for_mri.filters import anisodiff3
from retinex_for_mri.utils import truncate_and_scale
from scipy.ndimage import zoom, gaussian_filter
from tetrahydra.core import closure, aitchison_dist

# Load nifti
nii = load('/media/Data_Drive/Benchmark_Data/Segmentator_Data/Marian/longTR.nii.gz')
basename_nii = nii.get_filename().split(os.extsep, 1)[0]

data = np.asarray(nii.get_data(), dtype=float)

# # interpolate data (optional, for testing)
# data = zoom(data, 2, order=0)
# data = data[..., 0::2]  # remove redundant channels after interpolation
# print 'Interpolation is done.'

dims = data.shape
for i in range(dims[3]):
    # data[..., i] = gaussian_filter(data[..., i], 2)
    data[..., i] = truncate_and_scale(data[..., i])

# calculate luminosity
luma = (np.min(data, axis=3) + np.max(data, axis=3)) / dims[3]
luma = multi_scale_retinex_3d(luma, scales=[1, 5, 20])
luma = truncate_and_scale(luma)  # a bit arbitrary

out = Nifti1Image(luma, affine=np.eye(4))
save(out, basename_nii + '_luma.nii.gz')
print "MSR on luminosity is done."

# barycentric coordinates to preserve n-color information
bary = data.reshape(dims[0]*dims[1]*dims[2], dims[3])
bary = closure(bary)
bary = bary.reshape(dims[0], dims[1], dims[2], dims[3])

# plug back in the repaired luminosity
luma = luma.reshape(dims[0], dims[1], dims[2], 1)
ncol = bary * luma  # ncol: n-dimensional color space

for i in range(dims[3]):
    ncol[..., i] = anisodiff3(np.squeeze(ncol[..., i]), niter=2, kappa=100, gamma=0.1, option=1)

out = Nifti1Image(ncol, affine=np.eye(4))
save(out, basename_nii + '_msrsp.nii.gz')
print "MSRSP is done."

# save average image too
out = Nifti1Image(np.mean(ncol, axis=3), affine=np.eye(4))
save(out, basename_nii + '_msrsp_mean.nii.gz')
print "Mean of MSRSP is done."

# save aitchison distance gradient magnitude for fun (TODO: modularize)
print 'Calculating aitchison gradient magnitude, may take some time...'
bary_2 = bary.reshape(dims[0], dims[1], dims[2], dims[3])

# x
shape_x = [(dims[0]-2)*dims[1]*dims[2], dims[3]]
shape_x_inv = [(dims[0]-2), dims[1], dims[2]]
source_x = bary_2[1:-1, :, :, :].reshape(shape_x)
shift_x_f = bary_2[2:, :, :].reshape(shape_x)
shift_x_b = bary_2[0:-2, :, :].reshape(shape_x)
# y
shape_y = [dims[0]*(dims[1]-2)*dims[2], dims[3]]
shape_y_inv = [dims[0], (dims[1]-2), dims[2]]
source_y = bary_2[:, 1:-1, :, :].reshape(shape_y)
shift_y_f = bary_2[:, 2:, :].reshape(shape_y)
shift_y_b = bary_2[:, 0:-2, :].reshape(shape_y)
# z
shape_z = [dims[0]*dims[1]*(dims[2]-2), dims[3]]
shape_z_inv = [dims[0], dims[1], (dims[2]-2)]
source_z = bary_2[:, :, 1:-1, :].reshape(shape_z)
shift_z_f = bary_2[:, :, 2:].reshape(shape_z)
shift_z_b = bary_2[:, :, 0:-2].reshape(shape_z)

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
veli = gaussian_filter(veli, 1)
out = Nifti1Image(veli, affine=np.eye(4))
save(out, basename_nii + '_adist_sos.nii.gz')

print 'Finished.'
