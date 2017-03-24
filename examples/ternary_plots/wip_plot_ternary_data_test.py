"""Example usages of ternary plotting library."""

import os
import ternary
import numpy as np
from nibabel import load
from tetrahydra.core import (closure, perturb,
                             sample_center, sample_total_variance)
from tetrahydra.utils import truncate_and_scale
np.seterr(divide='ignore', invalid='ignore')

# Load data
nii1 = load('/home/faruk/Data/brainweb/no_noise/t1_icbm_normal_1mm_pn0_rf0.nii.gz')
nii2 = load('/home/faruk/Data/brainweb/no_noise/pd_icbm_normal_1mm_pn0_rf0.nii.gz')
nii3 = load('/home/faruk/Data/brainweb/no_noise/t2_icbm_normal_1mm_pn0_rf0.nii.gz')

basename = nii1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(nii1.get_filename())

v = 0
vol1 = nii1.get_data()[..., :]
vol2 = nii2.get_data()[..., :]
vol3 = nii3.get_data()[..., :]
shape = vol1.shape + (3,)

# Preprocess
# vol1 = truncate_and_scale(vol1, percMin=0, percMax=100)
# vol2 = truncate_and_scale(vol2, percMin=0, percMax=100)
# vol3 = truncate_and_scale(vol3, percMin=0, percMax=100)

comp = np.zeros(shape)
comp[:, :, :, 0] = vol1
comp[:, :, :, 1] = vol2
comp[:, :, :, 2] = vol3
comp = comp.reshape(shape[0]*shape[1]*shape[2], shape[3])

# Impute
comp[comp == 0] = 1

# Closure
comp = closure(comp)

# Centering
center = sample_center(comp)
test = np.ones(comp.shape) * center
comp = perturb(comp, center**-1)

# Scale data if needed
scale = 50  # increase this for more resolution in ternary plot
comp = comp * scale

# Heatmap related
start = 0
a = []
for i in range(start, scale + (1 - start)):
    for j in range(start, scale + (1 - start) - i):
        k = scale - i - j
        a.append((i, j, k))

# Ternary binning
counts = []
for i in range(len(a)):  # for every vertex
    # for j in range(len(a[i])):  # for every index in a vertex
    idx_range_min = ((comp[:, 0] > a[i][0]-0.5) &
                     (comp[:, 1] > a[i][1]-0.5) &
                     (comp[:, 2] > a[i][2]-0.5))

    idx_range_max = ((comp[:, 0] < a[i][0]+0.5) &
                     (comp[:, 1] < a[i][1]+0.5) &
                     (comp[:, 2] < a[i][2]+0.5))
    idx_range = (idx_range_min & idx_range_max)
    counts.append(np.log((idx_range).sum()))

b = tuple(a)
c = dict()

for i in range(len(b)):
    c[b[i]] = counts[i]

# Ternary scatter Plot
fontsize = 12
figure, tax = ternary.figure(scale=scale)
tax.set_title("Ternay heat map", fontsize=fontsize)
tax.left_axis_label("_1_", fontsize=fontsize)
tax.right_axis_label("_2_", fontsize=fontsize)
tax.bottom_axis_label("_3_", fontsize=fontsize)
tax.boundary(linewidth=0.1)
tax.gridlines(multiple=20, color="blue")
tax.heatmap(c, scale, style="hexagonal", vmin=0, vmax=max(counts))
tax.show()
