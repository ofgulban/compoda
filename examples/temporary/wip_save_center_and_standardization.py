"""Example usages of ternary plotting library."""

import os
import numpy as np
from nibabel import load
from tetrahydra.core import (closure, perturbation,
                             sample_center, sample_total_variance)
from tempfile import TemporaryFile

# Load data
nii1 = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti/echo_1.nii.gz')
nii2 = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti/echo_2.nii.gz')
nii3 = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti/echo_3.nii.gz')

nii4 = load('/home/faruk/Data/benedikt/multi_echo_epi/trajectory_masks/2voxels_gm.nii.gz')
msk = nii4.get_data().flatten()
msk = msk > 0

basename = nii1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(nii1.get_filename())

nr_measurements = nii1.shape[3]

t = 0  # timepoint to be used in centering and standardization
vol1 = nii1.get_data()[..., t]
vol2 = nii2.get_data()[..., t]
vol3 = nii3.get_data()[..., t]
shape = vol1.shape + (3,)

comp = np.zeros(shape)
comp[:, :, :, 0] = vol1
comp[:, :, :, 1] = vol2
comp[:, :, :, 2] = vol3
comp = comp.reshape(shape[0]*shape[1]*shape[2], shape[3])
comp = comp[msk, :]  # apply mask

# Impute
comp[comp == 0] = 1
# Closure
comp = closure(comp)
# Centering
center = sample_center(comp)
print "Sample center: " + str(center)
temp = np.ones(comp.shape) * center
comp = perturbation(comp, temp**-1)
# Total variance is used in standardization in the loop
totvar = sample_total_variance(comp, center)
print "Total variance: " + str(totvar)

# saving
np.save(os.path.join(dirname, 'gm_time_descriptives'),
        {'Center': center, 'Total variance': totvar})
print 'Done.'
