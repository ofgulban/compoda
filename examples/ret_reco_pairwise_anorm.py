"""Simplex based tool for combining channels pairwise."""

import os
import numpy as np
from nibabel import load, save, Nifti1Image
from tetrahydra.core import closure, aitchison_norm
from tetrahydra.utils import truncate_and_scale
from retinex_for_mri.core import multi_scale_retinex_3d
from retinex_for_mri.filters import anisodiff3
from scipy.ndimage import zoom, gaussian_filter

# load nifti
file_path_1 = '/media/Data_Drive/Benchmark_Data/compositional_data/Pebre/SE_kT.nii.gz'
file_path_2 = '/media/Data_Drive/Benchmark_Data/compositional_data/Pebre/STE_kT.nii.gz'
dir_name = os.path.dirname(file_path_1)

nii_1 = load(file_path_1)
nii_2 = load(file_path_2)

img_1 = nii_1.get_data()
img_2 = nii_2.get_data()

# preprocess images
for i in range(24):
    img_1[..., i] = truncate_and_scale(img_1[..., i], percMin=40, percMax=100)
    img_2[..., i] = truncate_and_scale(img_2[..., i], percMin=40, percMax=100)

pair = np.zeros(img_1.shape + (2,))
pair[..., 0] = img_1
pair[..., 1] = img_2

dims = pair.shape
pair = pair.reshape(dims[0]*dims[1]*dims[2]*dims[3], dims[4])

bary = closure(pair)
anorm = aitchison_norm(bary)
# anorm = (bary[:, 0] - bary[:, 1])/(bary[:, 0] + bary[:, 1])

img = anorm.reshape(dims[0], dims[1], dims[2], dims[3])
out = Nifti1Image(img, affine=np.eye(4))
save(out, os.path.join(dir_name, 'pairwise_anorm.nii.gz'))

# traditional combination
comb = anorm.reshape(dims[0], dims[1], dims[2], dims[3])
comb = np.sqrt(np.sum(comb**2.0, axis=3))
comb = np.nan_to_num(comb)
# save combined image
out = Nifti1Image(comb, affine=np.eye(4))
save(out, os.path.join(dir_name, 'pairwise_anorm_comb.nii.gz'))



print 'Finished.'
