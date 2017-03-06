"""Simplex based tool for combining channels pairwise."""

import os
import numpy as np
from nibabel import load, save, Nifti1Image
from tetrahydra.core import closure, aitchison_norm
from tetrahydra.utils import truncate_and_scale
from retinex_for_mri.core import multi_scale_retinex_3d
from retinex_for_mri.filters import anisodiff3

# load nifti
file_path_1 = '/media/Data_Drive/Benchmark_Data/compositional_data/Pebre/comb/SE_kT_comb.nii.gz'
file_path_2 = '/media/Data_Drive/Benchmark_Data/compositional_data/Pebre/comb/STE_kT_comb.nii.gz'
dir_name = os.path.dirname(file_path_1)

nii_1 = load(file_path_1)
nii_2 = load(file_path_2)

SE = nii_1.get_data()
STE = nii_2.get_data()


SE = multi_scale_retinex_3d(SE, scales=[1, 2, 3])
STE = multi_scale_retinex_3d(STE, scales=[1, 2, 3])

SE = truncate_and_scale (SE, percMin=0, percMax=100)
STE = truncate_and_scale (STE, percMin=0, percMax=100)

out = Nifti1Image(SE, affine=np.eye(4))
save(out, os.path.join(dir_name, 'SE_MSR.nii.gz'))
out = Nifti1Image(STE, affine=np.eye(4))
save(out, os.path.join(dir_name, 'STE_MSR.nii.gz'))

comp = 1-SE/STE  # comp stands for component

out = Nifti1Image(comp, affine=np.eye(4))
save(out, os.path.join(dir_name, 'res_comp.nii.gz'))

rel_cont = (STE - comp) / SE
out = Nifti1Image(rel_cont, affine=np.eye(4))
save(out, os.path.join(dir_name, 'rel_cont.nii.gz'))
