"""Scale and unwrap phase images.

Unwrapping is done in time domain. Ares with low SNR is expected to result in
faulty unwraps.

"""

import os
import numpy as np
from tetrahydra.utils import progress_output, truncate_and_scale
from nibabel import load, Nifti1Image, save
from skimage.restoration import unwrap_phase
from scipy.ndimage.filters import gaussian_filter1d
from unwrap import unwrap
import time
start = time.time()

# Load data
nii1 = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti/echo_3_phase.nii.gz')
basename = nii1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(nii1.get_filename())

vol1 = nii1.get_data()
dims = nii1.shape

mb_factor = 3  # multi-band factor
nr_slab_slc = dims[2] / mb_factor  # number of slab slices
nr_time = dims[3]
nr_vox = dims[0]*dims[1]*dims[2]

# scale phase images
vol1 = truncate_and_scale(vol1, percMin=0, percMax=100, zeroTo=2*np.pi)

#
vol1 = vol1.reshape(nr_vox, nr_time)
for v in range(nr_vox):
    temp = vol1[v, :]
    temp = unwrap_phase(temp)  # scikit image
    vol1[v, :] = temp

img = vol1.reshape(dims[0], dims[1], dims[2], dims[3])
out = Nifti1Image(img, affine=nii1.affine)
save(out, basename + '_temp_unwrap.nii.gz')

print 'Finished.'
