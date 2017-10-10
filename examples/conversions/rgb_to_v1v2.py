"""Create isometric logratio transformed coordinates for MRI data."""

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import tetrahydra.core as tet
from tetrahydra.utils import truncate_range, scale_range
from nibabel import load, save, Nifti1Image

# Load data
nii1 = load('/home/faruk/gdrive/Segmentator/data/faruk/pt7/T1.nii.gz')
nii2 = load('/home/faruk/gdrive/Segmentator/data/faruk/pt7/PD.nii.gz')
nii3 = load('/home/faruk/gdrive/Segmentator/data/faruk/pt7/T2s.nii.gz')

mask = load("//home/faruk/gdrive/Segmentator/data/faruk/pt7/brain_mask.nii.gz").get_data()
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

# Isometric logratio transformation before nay centering
ilr_orig = tet.ilr_transformation(np.copy(p_comp))

# Centering
center = tet.sample_center(p_comp)
print "Sample center: " + str(center)
c_temp = np.ones(p_comp.shape) * center
p_comp = tet.perturb(p_comp, c_temp**-1)
# Standardize
totvar = tet.sample_total_variance(p_comp, center)
p_comp = tet.power(p_comp, np.power(totvar, -1./2.))

# Isometric logratio transformation for plotting
ilr = tet.ilr_transformation(p_comp)

# Plots
fig = plt.figure()
limits = [-2.5, 2.5]
ax_1 = plt.subplot(121)
# Plot 2D histogram of ilr transformed data
_, _, _, h_1 = ax_1.hist2d(ilr_orig[:, 0], ilr_orig[:, 1], bins=2000,
                           cmap='gray_r')
h_1.set_norm(LogNorm(vmax=np.power(10, 3)))
plt.colorbar(h_1)
ax_1.set_title('Before Centering')
ax_1.set_xlabel('$v_1$')
ax_1.set_ylabel('$v_2$')
ax_1.set_aspect('equal')
ax_1.set_xlim(limits)
ax_1.set_ylim(limits)

ax_2 = plt.subplot(122)
# Plot 2D histogram of ilr transformed data
_, _, _, h_2 = ax_2.hist2d(ilr[:, 0], ilr[:, 1], bins=2000, cmap='gray_r')
h_2.set_norm(LogNorm(vmax=np.power(10, 3)))
plt.colorbar(h_2)
ax_2.set_title('After Centering')
ax_2.set_xlabel('$v_1$')
ax_2.set_ylabel('$v_2$')
ax_2.set_aspect('equal')
ax_2.set_xlim(limits)
ax_2.set_ylim(limits)

# plot axes of primary colors on top
c_axes = np.array([[15., 1., 1.], [1., 15., 1.], [1., 1., 15.]])
c_axes = tet.closure(c_axes)
# (optional) center the primary guides the same way
# c_temp = np.ones(pri.shape) * center
# pri = tet.perturb(pri, c_temp**-1)
c_axes = tet.ilr_transformation(c_axes)
caxw = 0.025  # width
ax_1.add_patch(patches.FancyArrow(0, 0, c_axes[0, 0], c_axes[0, 1], width=caxw,
                                  facecolor='r', edgecolor='none'))
ax_1.add_patch(patches.FancyArrow(0, 0, c_axes[1, 0], c_axes[1, 1], width=caxw,
                                  facecolor='g', edgecolor='none'))
ax_1.add_patch(patches.FancyArrow(0, 0, c_axes[2, 0], c_axes[2, 1], width=caxw,
                                  facecolor='b', edgecolor='none'))
ax_2.add_patch(patches.FancyArrow(0, 0, c_axes[0, 0], c_axes[0, 1], width=caxw,
                                  facecolor='r', edgecolor='none'))
ax_2.add_patch(patches.FancyArrow(0, 0, c_axes[1, 0], c_axes[1, 1], width=caxw,
                                  facecolor='g', edgecolor='none'))
ax_2.add_patch(patches.FancyArrow(0, 0, c_axes[2, 0], c_axes[2, 1], width=caxw,
                                  facecolor='b', edgecolor='none'))

plt.show()

print('Exporting ilr coordinates...')
# ilr transformation for nifti output (also considering unplotted data)
# Centering
c_temp = np.ones(comp.shape) * center
comp = tet.perturb(comp, c_temp**-1)
# Standardize
comp = tet.power(comp, np.power(totvar, -1./2.))
ilr = tet.ilr_transformation(comp)

# save the new coordinates
ilr = ilr.reshape(dims[0], dims[1], dims[2], dims[3]-1)
for i in range(ilr.shape[-1]):
    img = ilr[..., i]
    # scale is done for FSL-FAST otherwise it cannot find clusters
    img = truncate_range(img, percMin=0, percMax=100)
    img = scale_range(img, scale_factor=2000)
    # img[mask == 0] = 0  # swap masked and imputed regions with zeros
    out = Nifti1Image(img, affine=nii1.affine)
    save(out, os.path.join(dirname, 'ilr_coord_'+str(i+1)+'.nii.gz'))

print('Finished.')
