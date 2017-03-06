"""Example usages of ternary plotting library."""

import os
import numpy as np
import tetrahydra.core as tet
from nibabel import load, Nifti1Image, save
from tetrahydra.utils import progress_output
import time
start = time.time()

# Load data
nii1 = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti_small/echo_1_roi.nii.gz')
nii2 = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti_small/echo_2_roi.nii.gz')
nii3 = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti_small/echo_3_roi.nii.gz')

basename = nii1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(nii1.get_filename())

nr_measurements = nii1.shape[3]
nr_vox = nii1.shape[0]*nii1.shape[1]*nii1.shape[2]

t = 0  # timepoint to be used in centering and standardization
vol1 = nii1.get_data()
vol2 = nii2.get_data()
vol3 = nii3.get_data()
dims = vol1.shape + (3,)

comp = np.zeros(dims)
comp[..., 0] = vol1
comp[..., 1] = vol2
comp[..., 2] = vol3

# reshape for tet usage
comp = comp.reshape(dims[0]*dims[1]*dims[2], dims[3], dims[4])

# loop stuff
img_anorm = np.zeros(nr_vox)
img_disp = np.zeros(nr_vox)
img_center = np.zeros((nr_vox, dims[4]))
temp = np.zeros(nr_vox)
for v in range(nr_vox):
    temp = comp[v, :, :]
    # Impute
    temp[temp == 0] = 1
    # Closure
    temp = tet.closure(temp)

    # Find center center (in time)
    temp_center = tet.sample_center(temp)
    img_center[v, :] = temp_center
    img_anorm[v] = tet.aitchison_norm(temp_center)
    # Dispersion (in time)
    img_disp[v] = tet.sample_sstd(temp)
    progress_output(v+1, nr_vox, text='voxels')

img = img_disp.reshape(dims[0:3])
out = Nifti1Image(img, affine=nii1.affine)
save(out, os.path.join(dirname, 'test_sstd.nii.gz'))

img_center_ilr = tet.ilr_transformation(img_center)
img = img_center_ilr.reshape(dims[0:3] + (img_center_ilr.shape[-1],))
out = Nifti1Image(img, affine=nii1.affine)
save(out, os.path.join(dirname, 'test_center_ilr.nii.gz'))

img = img_anorm.reshape(dims[0:3])
out = Nifti1Image(img, affine=nii1.affine)
save(out, os.path.join(dirname, 'test_anorm.nii.gz'))

end = time.time()
print 'Finished in:', (end - start), 'seconds'
