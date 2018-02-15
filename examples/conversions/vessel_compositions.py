"""Plor compositions of large images."""

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import compoda.core as coda
from compoda.utils import truncate_range, scale_range
from nibabel import load, save, Nifti1Image

# Load data
nii1 = load('/media/Data_Drive/ISILON/006_SEGMENTATION/VESSELS/S01/coda/S01_SES1_T1.nii.gz')
nii2 = load('/media/Data_Drive/ISILON/006_SEGMENTATION/VESSELS/S01/coda/S01_SES1_PD.nii.gz')
nii3 = load('/media/Data_Drive/ISILON/006_SEGMENTATION/VESSELS/S01/coda/S01_T2s.nii.gz')

mask = load('/media/Data_Drive/ISILON/006_SEGMENTATION/VESSELS/S01/coda/S01_vessels_suptemp_NAT_v04_veins.nii.gz').get_data()
mask[mask > 0] = 1.  # binarize
idx_mask = mask > 0

basename = nii1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(nii1.get_filename())

vol1 = nii1.get_data()
vol2 = nii2.get_data()
vol3 = nii3.get_data()
dims = (int(np.sum(mask)), 3)

# only work on voxels within mask to not blow up the memory
comp = np.zeros(dims)
comp[..., 0] = vol1[idx_mask]
comp[..., 1] = vol2[idx_mask]
comp[..., 2] = vol3[idx_mask]

# (optional) truncate and rescale
# for i in range(comp.shape[1]):
#     temp = comp[:, i]
#     temp = truncate_range(temp)
#     temp = scale_range(temp, scale_factor=1000)
#     comp[:, i] = temp

# Impute
comp[comp == 0] = 1.

# Closure
comp = coda.closure(comp)

# Isometric logratio transformation before any centering
ilr_orig = coda.ilr_transformation(np.copy(comp))

# Centering
center = coda.sample_center(comp)
print("Sample center: " + str(center))
c_temp = np.ones(comp.shape) * center
p_comp = coda.perturb(comp, c_temp**-1)
# Standardize
totvar = coda.sample_total_variance(comp, center)
comp = coda.power(comp, np.power(totvar, -1./2.))

# Isometric logratio transformation for plotting
ilr = coda.ilr_transformation(comp)

# Plots
fig = plt.figure()
limits = [-2.5, 2.5]
ax_1 = plt.subplot(121)
# Plot 2D histogram of ilr transformed data
_, _, _, h_1 = ax_1.hist2d(ilr_orig[:, 0], ilr_orig[:, 1], bins=1000,
                           cmap='gray_r')
h_1.set_norm(LogNorm(vmax=np.power(10, 1)))
plt.colorbar(h_1, fraction=0.046, pad=0.04)
ax_1.set_title('Before Centering')
ax_1.set_xlabel('$v_1$')
ax_1.set_ylabel('$v_2$')
ax_1.set_aspect('equal')
ax_1.set_xlim(limits)
ax_1.set_ylim(limits)

ax_2 = plt.subplot(122)
# Plot 2D histogram of ilr transformed data
_, _, _, h_2 = ax_2.hist2d(ilr[:, 0], ilr[:, 1], bins=1000, cmap='gray_r')
h_2.set_norm(LogNorm(vmax=np.power(10, 1)))
plt.colorbar(h_2, fraction=0.046, pad=0.04)
ax_2.set_title('After Centering')
ax_2.set_xlabel('$v_1$')
ax_2.set_ylabel('$v_2$')
ax_2.set_aspect('equal')
ax_2.set_xlim(limits)
ax_2.set_ylim(limits)

# plot axes of primary colors on top
nr_nodes, max_node = 2, 15
caxw = 1  # width
for a in range(3):  # loop through the primary axes
    # create a set of compositions along a primary axis
    nodes = np.linspace(1, max_node, nr_nodes)
    c_axis = np.ones([nr_nodes, 3])
    c_axis[:, a] = nodes
    c_axis = coda.closure(c_axis)
    c_axis = coda.ilr_transformation(c_axis)
    ax_1.add_patch(patches.Polygon(c_axis, closed=False, linewidth=caxw,
                                   facecolor='k', edgecolor='k'))

for a in range(3):
    # create a set of compositions along a primary axis
    nodes = np.linspace(1, max_node, nr_nodes)
    c_axis = np.ones([nr_nodes, 3])
    c_axis[:, a] = nodes
    c_axis = coda.closure(c_axis)

    # (optional) center the primary guides the same way
    c_temp = np.ones(c_axis.shape) * center
    c_axis = coda.perturb(c_axis, c_temp**-1.)
    c_axis = coda.power(c_axis, np.power(totvar, -1./2.))

    c_axis = coda.ilr_transformation(c_axis)
    ax_2.add_patch(patches.Polygon(c_axis, closed=False, linewidth=caxw,
                                   facecolor='k', edgecolor='k'))

plt.show()

# # export write ilr coords
# out = np.zeros(vol1.shape)
# out[idx_mask] = ilr[:, 0]
# pim = Nifti1Image(out, affine=nii1.affine)
# save(pim, os.path.join(dirname, 'ilr_coord_1.nii.gz'))
#
# out[idx_mask] = ilr[:, 1]
# pim = Nifti1Image(out, affine=nii1.affine)
# save(pim, os.path.join(dirname, 'ilr_coord_2.nii.gz'))
