"""Export aitchison inner product."""

import os
import numpy as np
import tetrahydra.core as tet
from tetrahydra.utils import truncate_range, scale_range
from nibabel import load, save, Nifti1Image

# Load data
nii1 = load('/media/Data_Drive/ISILON/006_SEGMENTATION/VESSELS/S01/coda/S01_SES1_T1.nii.gz')
nii2 = load('/media/Data_Drive/ISILON/006_SEGMENTATION/VESSELS/S01/coda/S01_SES1_PD.nii.gz')
nii3 = load('/media/Data_Drive/ISILON/006_SEGMENTATION/VESSELS/S01/coda/S01_T2s.nii.gz')

mask = load('/media/Data_Drive/ISILON/006_SEGMENTATION/VESSELS/S01/coda/S01_brain_mask.nii.gz').get_data()
mask[mask > 0] = 1.  # binarize
idx_mask = mask > 0

basename = nii1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(nii1.get_filename())

vol1 = nii1.get_data()
vol2 = nii2.get_data()
vol3 = nii3.get_data()
dims = (np.sum(idx_mask), 3)

# only work on voxels within mask to not blow up the memory
comp = np.zeros(dims)
comp[..., 0] = vol1[idx_mask]
comp[..., 1] = vol2[idx_mask]
comp[..., 2] = vol3[idx_mask]

# (optional) truncate and rescale
# for i in range(comp.shape[1]):
#     temp = comp[:, i]
#     temp = truncate_range(temp)
#     temp = scale_range(temp, scale_factor=1000)
#     comp[:, i] = temp

# Impute
comp[comp == 0] = 1.

# Closure
comp = tet.closure(comp)

# Aitchison inner product
ref = np.ones(comp.shape)
ref[:, 1] = ref[:, 1] * 10
ref = tet.closure(ref)
ip = tet.aitchison_inner_product(comp, ref)

cos_theta = ip / (tet.aitchison_norm(comp) * tet.aitchison_norm(ref))
rad = np.arccos(cos_theta)
deg = rad * (360 / (2*np.pi))

# Determine sector
idx_sector2 = comp[:, 1] > comp[:, 2]
deg[idx_sector2] = 360. - deg[idx_sector2]

# Create output
out = np.zeros(nii1.shape)
out[idx_mask] = deg
img = Nifti1Image(out, affine=nii1.affine)
save(img, os.path.join(dirname, 'theta2_fullrange.nii.gz'))
