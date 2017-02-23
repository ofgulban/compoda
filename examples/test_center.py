"""(WIP) Test sample center and sample total variance."""

import numpy as np
from tetrahydra.core import closure, sample_center, sample_total_variance, aitchison_dist
from nibabel import load

# Load data
nii1 = load('/some/data/echo_1.nii.gz')
nii2 = load('/some/data/echo_2.nii.gz')
nii3 = load('/some/data/echo_3.nii.gz')

# Preprocess
vol1 = nii1.get_data()[..., 50]
vol2 = nii2.get_data()[..., 50]
vol3 = nii3.get_data()[..., 50]
shape = vol1.shape + (3,)

comp = np.zeros(shape)
comp[:, :, :, 0] = vol1
comp[:, :, :, 1] = vol2
comp[:, :, :, 2] = vol3
comp = comp.reshape(shape[0]*shape[1]*shape[2], shape[3])

# Impute
comp[comp == 0] = 1

# Closure
comp.shape
comp = np.asarray(comp, dtype='float64')
comp = closure(comp)
center = sample_center(comp[80000:80200, :])
sample_total_variance(comp[80000:80200, :], center)
