"""Simplex based tool for combining channels."""

import os
import numpy as np
from nibabel import load, save, Nifti1Image
from tetrahydra.core import closure, aitchison_norm
from tetrahydra.utils import truncate_and_scale
from retinex_for_mri.core import multi_scale_retinex_3d
from retinex_for_mri.filters import anisodiff3
from scipy.ndimage import zoom, gaussian_filter

# load nifti
file_path = '/home/faruk/Data/retinex_tests/pebre/kT/merged.nii.gz'
dir_name = os.path.dirname(file_path)
nii = load(file_path)
basename = nii.get_filename().split(os.extsep, 1)[0]

# get data in the voxels by channels format
data = nii.get_data()
# interpolate data (optional, for testing)
data = zoom(data, 2, order=0)
data = data[..., 0::2]  # remove redundant channels after interpolation
print 'Interpolation is done.'

# padding to mitigate gaussian effects in retinex
data = np.pad(data, ((20, 20), (20, 20), (20, 20), (0, 0)),
              mode='constant', constant_values=0)
dims = data.shape

# traditional combination
comb = np.sqrt(np.sum(data**2.0, axis=3))
# save combined image
out = Nifti1Image(comb, affine=np.eye(4))
save(out, basename + '_comb.nii.gz')

# normalize (simple color balance)
for i in range(dims[3]):
    data[..., i] = gaussian_filter(data[..., i], 1, mode="constant", cval=0.0)
    # data[..., i] = anisodiff3(data[..., i], niter=1, kappa=100, gamma=0.1, option=1)
    data[..., i] = truncate_and_scale(data[..., i], percMin=1, percMax=100)

# save smoothed combination
comb = np.sqrt(np.sum(data**2.0, axis=3))
# save combined image
out = Nifti1Image(comb, affine=np.eye(4))
save(out, basename + '_comb_smth.nii.gz')


# calculate luminosity(lightness)
luma = (np.min(data, axis=3) + np.max(data, axis=3)) / dims[3]

# save luminosity as nifti
out = Nifti1Image(luma, affine=np.eye(4))
save(out, basename + '_lum.nii.gz')

# apply retinex to luminosity
luma_rex = multi_scale_retinex_3d(luma+1, scales=[1, 5, 10])
luma_rex = truncate_and_scale(luma_rex, percMin=0, percMax=100)  # for edges

# save retinex-ed luminosity as nifti
out = Nifti1Image(luma_rex, affine=np.eye(4))
save(out, basename + '_lum_rex.nii.gz')

# simplex part
data = data.reshape(dims[0] * dims[1] * dims[2], dims[3])
bary = closure(data)

# amalgamate
bary = np.sort(bary, axis=1)
for i in range(dims[3]):
    bary[..., i] = gaussian_filter(bary[..., i], 2, mode="constant", cval=0.0)

# handle new shape
bary = bary.reshape(dims[0], dims[1], dims[2], dims[3])
bary = np.delete(bary, (0, 1, 46, 47), axis=3)  # (optional) remove channels
dims = bary.shape
bary = bary.reshape(dims[0] * dims[1] * dims[2], dims[3])

# plug back in the repaired luminosity
luma_rex = luma_rex.reshape(dims[0] * dims[1] * dims[2], 1)
ncol = bary * luma_rex  # ncol: n-dimensional color space

# save msrcp channels
img = ncol.reshape(dims[0], dims[1], dims[2], dims[3])
out = Nifti1Image(img, affine=np.eye(4))
save(out, basename + '_sixrex.nii.gz')


# traditional combination
comb = np.sqrt(np.sum(ncol**2.0, axis=1))

# threshold based on luminosity (optional)
luma = luma.reshape(dims[0] * dims[1] * dims[2])
luma[luma < 0.003] = 0
luma[luma >= 0.003] = 1
comb = comb * luma

# save combined image
img = comb.reshape(dims[0], dims[1], dims[2])
out = Nifti1Image(img, affine=np.eye(4))
save(out, basename + '_sixrex_comb.nii.gz')

print 'Finished.'
