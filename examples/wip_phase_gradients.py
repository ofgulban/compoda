"""Gradients in time domain for phase data."""

import os
import numpy as np
from nibabel import load, Nifti1Image, save

# Load data
nii = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti/echo_1_phase_temp_unwrap.nii.gz')
basename = nii.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(nii.get_filename())

ima = nii.get_data()
gra = np.gradient(ima)

img = vol1.reshape(dims[0], dims[1], dims[2], dims[3])
out = Nifti1Image(img, affine=nii1.affine)
save(out, basename + '_temp_unwrap.nii.gz')

print 'Finished.'
