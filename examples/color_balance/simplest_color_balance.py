"""Color balance using simplest color balance method."""

import os
import numpy as np
from tetrahydra.utils import truncate_range
from nibabel import load, save, Nifti1Image

# Load data
nii1 = load('/home/faruk/gdrive/Segmentator/data/faruk/pt7/T1.nii.gz')
nii2 = load('/home/faruk/gdrive/Segmentator/data/faruk/pt7/PD.nii.gz')
nii3 = load('/home/faruk/gdrive/Segmentator/data/faruk/pt7/T2s.nii.gz')

mask = load("//home/faruk/gdrive/Segmentator/data/faruk/pt7/brain_mask.nii.gz").get_data()
mask[mask > 0] = 1.  # binarize

basename = nii1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(nii1.get_filename())

vol1 = nii1.get_data()
vol2 = nii2.get_data()
vol3 = nii3.get_data()
dims = vol1.shape + (3,)

orig = np.zeros(dims)
orig[..., 0] = vol1 * mask
orig[..., 1] = vol2 * mask
orig[..., 2] = vol3 * mask

# simplest color balance
for i in range(orig.shape[-1]):
    img = orig[..., i]
    img = truncate_range(img, percMin=1, percMax=99, discard_zeros=True)
    out = Nifti1Image(img, affine=nii1.affine)
    save(out, os.path.join(dirname, 'input'+str(i+1)+'_simplest_cbal.nii.gz'))
