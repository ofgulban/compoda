"""Create isometric logratio transformed coordinates for MRI data."""

import os
import numpy as np
import tetrahydra.core as tet
from tetrahydra.utils import truncate_and_scale
from nibabel import load, save, Nifti1Image

"""Load Data"""
#
nii1 = load('/home/faruk/Data/Faruk/bias_corr/T1_restore.nii.gz')
nii2 = load('/home/faruk/Data/Faruk/bias_corr/PD_restore.nii.gz')
nii3 = load('/home/faruk/Data/Faruk/bias_corr/T2s_restore.nii.gz')

basename = nii1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(nii1.get_filename())

vol1 = nii1.get_data()
vol2 = nii2.get_data()
vol3 = nii3.get_data()
dims = vol1.shape + (3,)

comp = np.zeros(dims)
comp[..., 0] = vol1
comp[..., 1] = vol2
comp[..., 2] = vol3

comp = comp.reshape(dims[0]*dims[1]*dims[2], dims[3])

# Impute
comp[comp == 0] = 1.

# Closure
comp = tet.closure(comp)

# Isometric logratio transformed
ilr = tet.ilr_transformation(comp)

ilr = ilr.reshape(dims[0], dims[1], dims[2], dims[3]-1)

# save the new coordinates
for i in range(ilr.shape[-1]):
    img = ilr[..., i]
    # scale is done for FSL-FAST otherwise it cannot find clusters 
    img = truncate_and_scale(img, percMin=0, percMax=100, zeroTo=2000)
    out = Nifti1Image(img, affine=nii1.affine)
    save(out, os.path.join(dirname, 'ilr_coord_'+str(i+1)+'.nii.gz'))
