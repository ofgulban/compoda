"""Example usages of ternary plotting library."""

import os
import numpy as np
from nibabel import load, Nifti1Image, save
from tetrahydra.core import (closure, perturbation, simplicial_std,
                             sample_center, sample_total_variance,
                             aitchison_norm)
from tetrahydra.utils import progress_output
from scipy.ndimage import zoom, gaussian_filter
import time
start = time.time()

# Load data
nii1 = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti/echo_1.nii.gz')
nii2 = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti/echo_2.nii.gz')
nii3 = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti/echo_3.nii.gz')

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
comp = comp.reshape(dims[0]*dims[1]*dims[2], dims[3], dims[4])

# interpolate data (optional, for testing)
comp = zoom(comp, 2, order=0)
comp = comp[..., 0::2]  # remove redundant channels after interpolation
comp = comp[..., 0::2, :]
print 'Interpolation is done.'
for i in range(dims[3]):
    for j in range(dims[4]):
        comp[:, i, j] = gaussian_filter(comp[:, i, j], 1, mode="constant", cval=0.0)

# (optional) load descriptives for centering and standardization
descriptives = np.load('/home/faruk/Data/benedikt/multi_echo_epi/masks/gm_descriptives.npy')
descriptives = descriptives.item()
center = descriptives["Center"]
print "Sample center: " + str(center)

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
    temp = closure(temp)

    # (optional) Centering
    temp_c = np.ones(temp.shape) * center
    temp = perturbation(temp, temp_c**-1)

    # Find center center (in time)
    temp_center = sample_center(temp)
    img_center[v, :] = temp_center
    img_anorm[v] = aitchison_norm(temp_center)
    # Dispersion (in time)
    img_disp[v] = simplicial_std(temp)
    progress_output(v+1, nr_vox, text='voxels')


img = img_disp.reshape(dims[0:3])
out = Nifti1Image(img, affine=nii1.affine)
save(out, os.path.join(dirname, 'test_sstd.nii.gz'))
end = time.time()

img = img_center.reshape(dims[0:3] + (dims[4],))
out = Nifti1Image(img, affine=nii1.affine)
save(out, os.path.join(dirname, 'test_center.nii.gz'))
end = time.time()

img = img_anorm.reshape(dims[0:3])
out = Nifti1Image(img, affine=nii1.affine)
save(out, os.path.join(dirname, 'test_anorm.nii.gz'))
end = time.time()

print 'Finished in:', (end - start), 'seconds'
