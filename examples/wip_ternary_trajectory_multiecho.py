"""Example usages of ternary plotting library."""

import os
import ternary
import numpy as np
import matplotlib.pyplot as plt
from nibabel import load
import tetrahydra.core as tet

np.seterr(divide='ignore', invalid='ignore')

# Load data
nii1 = load('/home/faruk/Data/benedikt/multi_echo_epi/band_pass/echo_1_lpass.nii.gz')
nii2 = load('/home/faruk/Data/benedikt/multi_echo_epi/band_pass/echo_2_lpass.nii.gz')
nii3 = load('/home/faruk/Data/benedikt/multi_echo_epi/band_pass/echo_3_lpass.nii.gz')

nii4 = load('/home/faruk/Data/benedikt/multi_echo_epi/trajectory_masks/6voxels_gm.nii.gz')
print nii4.get_filename()
msk = nii4.get_data().flatten()
msk = msk > 0

basename = nii1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(nii1.get_filename())

nr_measurements = nii1.shape[3]

# load descriptives for centering and standardization
descriptives = np.load('/home/faruk/Data/benedikt/multi_echo_epi/masks/wm_descriptives.npy')
descriptives = descriptives.item()
center = descriptives["Center"]
print "Sample center: " + str(center)
totvar = descriptives['Total variance']
print "Total variance: " + str(totvar)

vol1 = nii1.get_data()
vol2 = nii2.get_data()
vol3 = nii3.get_data()
shape = vol1.shape + (3,)

comp = np.zeros(shape)
comp[..., 0] = vol1
comp[..., 1] = vol2
comp[..., 2] = vol3
comp = comp.reshape(shape[0]*shape[1]*shape[2], shape[3], shape[4])
comp = comp[msk, :, :]  # apply mask
traj = comp[3, :, :]  # select only one voxel for trajectory

# Impute
traj[traj == 0] = 1
# Closure
traj = tet.closure(traj)
# Center
temp = np.ones(traj.shape) * center
traj = tet.perturb(traj, temp**-1)
# Standardize
traj = tet.power(traj, np.power(totvar, -1./2.))

# Scale data if needed
scale = 1000
traj = traj * scale

# Ternary scatter Plot
fontsize = 12
figure, tax = ternary.figure(scale=scale)
tax.set_title("Trajectory Plot", fontsize=15)
# tax.plot(traj, linewidth=0.1, alpha=1)
tax.plot_colored_trajectory(traj, cmap="hsv", linewidth=2, alpha=1)
tax.left_axis_label("Echo 1", fontsize=fontsize)
tax.right_axis_label("Echo 2", fontsize=fontsize)
tax.bottom_axis_label("Echo 2", fontsize=fontsize)
# tax.boundary(linewidth=0)
tax.gridlines(multiple=scale/4., color="Black")
tax.show()
