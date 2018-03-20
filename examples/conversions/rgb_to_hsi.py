"""Red Green Blue (RGB) to Hue Saturation Intensity (HSI) transformation."""

import os
import numpy as np
from nibabel import load, save, Nifti1Image

# Load data
nii1 = load('/home/faruk/Data/Faruk/rgb_hsi/T1.nii.gz')
nii2 = load('/home/faruk/Data/Faruk/rgb_hsi/PD.nii.gz')
nii3 = load('/home/faruk/Data/Faruk/rgb_hsi/T2s.nii.gz')

basename = nii1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(nii1.get_filename())

vol1 = nii1.get_data()
vol2 = nii2.get_data()
vol3 = nii3.get_data()
dims = vol1.shape + (3,)

rgb = np.zeros(dims)
rgb[..., 0] = vol1
rgb[..., 1] = vol2
rgb[..., 2] = vol3
rgb = rgb.reshape(dims[0]*dims[1]*dims[2], dims[3])

hsi_matrix = np.array([
    [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
    [1./np.sqrt(6), 1./np.sqrt(6), -2./np.sqrt(6)],
    [1./np.sqrt(2), -1./np.sqrt(3), 0.]])

temp = np.zeros(rgb.shape)
for v in range(rgb.shape[0]):
    temp[v, :] = np.dot(hsi_matrix, rgb[v, :].T)
temp = temp.reshape(dims)
intensity, v1, v2 = temp[..., 0], temp[..., 1], temp[..., 2]

# Export intensity
out = Nifti1Image(intensity, affine=nii1.affine)
save(out, os.path.join(dirname, 'intensity.nii.gz'))

# Export hue
hue = np.arctan(v2/v1) * (180/np.pi)
out = Nifti1Image(hue, affine=nii1.affine)
save(out, os.path.join(dirname, 'hue.nii.gz'))

# Export saturation
sat = np.sqrt(v1**2. + v2**2.)
out = Nifti1Image(sat, affine=nii1.affine)
save(out, os.path.join(dirname, 'saturation.nii.gz'))

print('Finished.')
