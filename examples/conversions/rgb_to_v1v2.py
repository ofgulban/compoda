"""Create isometric logratio transformed coordinates for MRI data."""

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import os
import numpy as np
import tetrahydra.core as tet
from tetrahydra.utils import truncate_and_scale
from nibabel import load, save, Nifti1Image

"""Load Data"""
#
nii1 = load('/home/faruk/Data/brainweb/source/t1_icbm_normal_1mm_pn3_rf0.nii.gz')
nii2 = load('/home/faruk/Data/brainweb/source/pd_icbm_normal_1mm_pn3_rf0.nii.gz')
nii3 = load('/home/faruk/Data/brainweb/source/t2_icbm_normal_1mm_pn3_rf0.nii.gz')

mask = load("/home/faruk/Data/brainweb/derived/phantom_brain.nii.gz").get_data()
# mask = load("/home/faruk/Data/Faruk/brain_mask.nii.gz").get_data()
mask[mask > 0] = 1.  # binarize

basename = nii1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(nii1.get_filename())

vol1 = nii1.get_data()
vol2 = nii2.get_data()
vol3 = nii3.get_data()
dims = vol1.shape + (3,)

comp = np.zeros(dims)
comp[..., 0] = vol1 * mask
comp[..., 1] = vol2 * mask
comp[..., 2] = vol3 * mask

comp = comp.reshape(dims[0]*dims[1]*dims[2], dims[3])

# Impute
comp[comp == 0] = 1.

# Closure
comp = tet.closure(comp)

# Plot related operations
p_mask = mask.reshape(dims[0]*dims[1]*dims[2])
p_comp = comp[p_mask > 0]

# Centering
# center = tet.sample_center(p_comp)
center = np.array([[ 0.06521165,  0.66942364,  0.26536471]])  # 0 noise center
print "Sample center: " + str(center)
c_temp = np.ones(p_comp.shape) * center
p_comp = tet.perturb(p_comp, c_temp**-1)

# Isometric logratio transformation for plotting
ilr = tet.ilr_transformation(p_comp)

# Plot 2D histogram of ilr transformed data
plt.hist2d(ilr[:, 0], ilr[:, 1], bins=500, norm=LogNorm(),  # vmax=100,
           cmap='inferno')
plt.xlabel('$v_1$')
plt.ylabel('$v_2$')
plt.axes().set_aspect('equal')
plt.axes().set_xlim([-3, 3])
plt.axes().set_ylim([-3, 3])
plt.colorbar()

# plot axes of primary colors on top
pri_temp = np.arange(1, 101)
pri = np.ones([300, 3])
pri[0:100, 0] = pri_temp  # primary color 1
pri[100:200, 1] = pri_temp  # primary color 2
pri[200:300, 2] = pri_temp  # primary color 3
pri = tet.closure(pri)
# center the primary guides the same way
c_temp = np.ones(pri.shape) * center
pri = tet.perturb(pri, c_temp**-1)
pri = tet.ilr_transformation(tet.closure(pri))
plt.scatter(pri[0:100, 0], pri[0:100, 1], color='red', s=3)
plt.scatter(pri[100:200, 0], pri[100:200, 1], color='green', s=3)
plt.scatter(pri[200:300, 0], pri[200:300, 1], color='blue', s=3)
plt.show()


# Isometric logratio transformation for nifti output
ilr = tet.ilr_transformation(comp)

# save the new coordinates
ilr = ilr.reshape(dims[0], dims[1], dims[2], dims[3]-1)
for i in range(ilr.shape[-1]):
    img = ilr[..., i]
    # scale is done for FSL-FAST otherwise it cannot find clusters
    img = truncate_and_scale(img, percMin=0, percMax=100, zeroTo=2000)
    img = img * mask  # swap masked and imputed regions with zeros
    out = Nifti1Image(img, affine=nii1.affine)
    save(out, os.path.join(dirname, 'ilr_coord_'+str(i+1)+'.nii.gz'))

print('Finished.')
