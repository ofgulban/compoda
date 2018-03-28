"""Export metrics related to barycentric coordinates of MRI Data."""

import os
import numpy as np
from nibabel import load, save, Nifti1Image
import compoda.core as coda

# Load data
nii1 = load('/path/to/file1.nii.gz')
nii2 = load('/path/to/file2.nii.gz')
nii3 = load('/path/to/file3.nii.gz')

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

# impute
rgb[rgb == 0] = 1.

# barycentric decomposition
intensity = np.sum(rgb, axis=-1) / rgb.shape[-1]
bary = coda.closure(rgb)
anorm = coda.aitchison_norm(bary)

# prepare reference vector for anglular difference
# TODO: export for all three main references
ref = np.ones(bary.shape)
ref[:, 0] = ref[:, 0] * 0.05
ref[:, 1] = ref[:, 1] * 0.9
ref[:, 2] = ref[:, 2] * 0.05

# compute aitchion angular difference
ainp = coda.aitchison_inner_product(bary, ref)

# Export inner product
img = ainp.reshape(dims[:-1])
out = Nifti1Image(img, affine=nii1.affine)
save(out, os.path.join(dirname, 'a_innerp.nii.gz'))

ref_norm = coda.aitchison_norm(ref)
idx = bary[:, 0] > bary[:, 2]  # NOTE: Selected in relation to the reference
ang_dif = np.zeros(anorm.shape)
ang_dif[idx] = np.arccos(ainp[idx]/(anorm[idx] * ref_norm[idx]))
ang_dif[~idx] = 2*np.pi - np.arccos(ainp[~idx]/(anorm[~idx] * ref_norm[~idx]))

# Export intensity
img = intensity.reshape(dims[:-1])
out = Nifti1Image(img, affine=nii1.affine)
save(out, os.path.join(dirname, 'a_intensity.nii.gz'))

# Export aitchison norm (saturation)
img = anorm.reshape(dims[:-1])
out = Nifti1Image(img, affine=nii1.affine)
save(out, os.path.join(dirname, 'a_norm.nii.gz'))

# Export aitchison angular difference (hue)
img = ang_dif.reshape(dims[:-1])
out = Nifti1Image(img, affine=nii1.affine)
save(out, os.path.join(dirname, 'a_angdif.nii.gz'))

print('Finished.')
