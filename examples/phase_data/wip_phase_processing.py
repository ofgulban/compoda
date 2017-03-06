"""Testing stuff with phase data, highly experimental."""

import os
import numpy as np
import tetrahydra.core as tet
from tetrahydra.utils import progress_output, truncate_and_scale
from nibabel import load, Nifti1Image, save
from skimage.restoration import unwrap_phase
import time
start = time.time()

# Load data
nii1 = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti/echo_1_phase.nii.gz')
nii2 = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti/echo_2_phase.nii.gz')
nii3 = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti/echo_3_phase.nii.gz')

basename = nii1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(nii1.get_filename())

vol1 = nii1.get_data()
vol2 = nii2.get_data()
vol3 = nii3.get_data()
dims = nii1.shape

comp = np.zeros(dims + (3,))
comp[..., 0] = vol1
comp[..., 1] = vol2
comp[..., 2] = vol3
dims = comp.shape
comp = comp.reshape(dims[0]*dims[1]*dims[2], dims[3], dims[4])
nr_vox = dims[0]*dims[1]*dims[2]
nr_time = dims[-2]

# scale phase images
i = 0
for i in range(dims[-1]):
    temp = comp[..., i]
    temp = truncate_and_scale(temp, percMin=0, percMax=100, zeroTo=2*np.pi)
    temp = temp - np.pi  # scikit image phase unwrap wants range [-pi, pi)]

    temp = temp.reshape(dims[0], dims[1], dims[2], dims[3])
    for j in range(1):
        unwrap_0 = temp[..., j]
        unwrap_1 = unwrap_phase(unwrap_0, wrap_around=False)
        unwrap_2 = np.copy(unwrap_0)  # initialize with original wrapped data
        for k in range(3):
            unwrap_2 = unwrap_2 + 2*np.pi * np.round((unwrap_1 - unwrap_2)
                                                     / (2 * np.pi))

        temp[..., j] = unwrap_2

    # impute
    temp[temp == 0] = 0.0000001
    comp[..., i] = temp.reshape(dims[0]*dims[1]*dims[2], dims[3])

# comp = np.rad2deg(comp)

# save processed phase images
for i in range(dims[-1]):
    img = comp[..., i]
    img = img.reshape(dims[0], dims[1], dims[2], dims[3])
    out = Nifti1Image(img, affine=nii1.affine)
    save(out, os.path.join(dirname, 'phaseUnwrap_' + str(i+1) + '.nii.gz'))
print 'unwrapping done'

# # find aitchison norm of phase compositions
# anorm = np.zeros((dims[0]*dims[1]*dims[2], dims[3]))
# for v in range(nr_vox):  # loop through voxels
#     comp[v, ...] = tet.closure(comp[v, ...], k=100)
#     anorm[v, ...] = tet.aitchison_norm(comp[v, ...])
#     # progress_output(v, nr_vox)
# print 'Anorm image computed.'
#
# img = anorm.reshape(dims[0], dims[1], dims[2], dims[3])
# out = Nifti1Image(img, affine=nii1.affine)
# save(out, os.path.join(dirname, 'phase_anorm.nii.gz'))
#
# # -----------------------------------------------------------------------------
#
# # save phase temporal gradient
# gra = np.zeros((nr_vox, nr_time-1, 3))
# for t in range(nr_time-1):
#     # gra[:, t] = tet.aitchison_dist(comp[:, t, :], comp[:, t+1, :])
#     gra[:, t, :] = comp[:, t, :] - comp[:, t+1, :]
#
# for i in range(dims[-1]):
#     img = gra[..., i]
#     img = img.reshape(dims[0], dims[1], dims[2], nr_time-1)
#     out = Nifti1Image(img, affine=nii1.affine)
#     save(out, os.path.join(dirname, 'phase'+str(i+1)+'_temp_gra.nii.gz'))
#
# # save gradient magnitude
# tgramag = np.sum(np.abs(gra), axis=1)
# img = tgramag.reshape(dims[0], dims[1], dims[2], dims[4])
# out = Nifti1Image(img, affine=nii1.affine)
# save(out, os.path.join(dirname, 'phase_temp_gramag.nii.gz'))

# -----------------------------------------------------------------------------

# # manipulate magnitude images
# nii4 = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti/echo_1.nii.gz')
# nii5 = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti/echo_2.nii.gz')
# nii6 = load('/home/faruk/Data/benedikt/multi_echo_epi/nifti/echo_3.nii.gz')
#
# vol4 = nii4.get_data()
# vol5 = nii5.get_data()
# vol6 = nii6.get_data()
# mag = np.zeros(dims)
# mag[..., 0] = vol4
# mag[..., 1] = vol5
# mag[..., 2] = vol6
# mag = mag.reshape(dims[0]*dims[1]*dims[2], dims[3], dims[4])
#
# for i in range(dims[-1]):
#     mag[..., i] = tet.closure(mag[..., i], k=100)
#
# for t in range(nr_time):
#     mag[:, t, :] = tet.perturb(mag[:, t, :], comp[:, t, ::-1])
#
# print 'Magnitude manipulation is done.'
#
# for i in range(dims[-1]):
#     img = mag[..., i]
#     img = img.reshape(dims[0], dims[1], dims[2], dims[3])
#     out = Nifti1Image(img, affine=nii1.affine)
#     save(out, os.path.join(dirname, 'magcor_' + str(i+1) + '.nii.gz'))
#
# print 'Finished.'
