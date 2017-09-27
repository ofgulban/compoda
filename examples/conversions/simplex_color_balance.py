"""Create isometric logratio transformed coordinates for MRI data."""

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import os
import numpy as np
import tetrahydra.core as tet
from tetrahydra.utils import truncate_range, scale_range
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
orig = orig.reshape(dims[0]*dims[1]*dims[2], dims[3])

# Impute
orig[orig <= 0] = 1
comp = np.copy(orig)

# Luminance
lumi = (np.max(comp, axis=1) + np.min(comp, axis=1)) / 2.

# Closure
comp = tet.closure(comp)

# Centering
center = tet.sample_center(comp)
temp = np.ones(comp.shape) * center
comp = tet.perturb(comp, temp**-1.)
# Standardize
totvar = tet.sample_total_variance(comp, center)
comp = tet.power(comp, np.power(totvar, -1./2.))

# Use Aitchison norm and powerinf for truncation of extreme compositions
anorm_thr = 3
anorm = tet.aitchison_norm(comp)
idx_trunc = anorm > anorm_thr
truncation_power = anorm[idx_trunc] / anorm_thr
correction = np.ones(anorm.shape)
correction[idx_trunc] = truncation_power
comp_bal = tet.power(comp, correction[:, None])

# go to hexcone lattice for exports
hexc = comp_bal * lumi[:, None]

hexc = hexc.reshape(dims[0], dims[1], dims[2], dims[3])
for i in range(hexc.shape[3]):
    img = Nifti1Image(hexc[..., i], affine=nii1.affine)
    save(img, os.path.join(dirname, 'input'+str(i+1)+'_simplex_cbal.nii.gz'))
