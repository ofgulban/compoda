"""Scale phase data."""

import os
import numpy as np
from tetrahydra.utils import truncate_and_scale
from nibabel import load, Nifti1Image, save
import time
start = time.time()

# Load data
nii1 = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti/phase/prelude/echo_1_phase.nii.gz')
basename = nii1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(nii1.get_filename())
vol1 = nii1.get_data()

vol1 = truncate_and_scale(vol1, percMin=0, percMax=100, zeroTo=np.pi*2)

out = Nifti1Image(vol1, affine=nii1.affine)
save(out, basename + '_2pi.nii.gz')
