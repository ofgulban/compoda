import os
import math
import ternary
import numpy as np
from nibabel import load, save, Nifti1Image
from AutoScale import AutoScale
np.seterr(divide='ignore', invalid='ignore')

"""Load Data"""
#
vol1 = load('/home/faruk/Data/Michelle/roi/T2s_roi.nii.gz')
vol2 = load('/home/faruk/Data/Michelle/roi/PD_roi.nii.gz')
vol3 = load('/home/faruk/Data/Michelle/roi/T1_roi.nii.gz')

basename = vol1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(vol1.get_filename())
niiHeader, niiAffine = vol1.header, vol1.affine
shape = vol1.shape + (3,)

# Preprocess
vol1 = AutoScale(vol1.get_data(), percMin=0, percMax=100)
vol2 = AutoScale(vol2.get_data(), percMin=0, percMax=100)
vol3 = AutoScale(vol3.get_data(), percMin=0, percMax=100)

rgb = np.zeros(shape)
rgb[:, :, :, 0] = vol1
rgb[:, :, :, 1] = vol2
rgb[:, :, :, 2] = vol3
rgb = rgb.reshape(shape[0]*shape[1]*shape[2], shape[3])

# Closure
sum_rgb = np.sum(rgb, axis=1)
for i in range(shape[3]):
    rgb[:, i] = rgb[:, i]/sum_rgb

# scale data if needed
scale = 50
rgb = rgb * scale

# heatmap related
start = 0
a = []
for i in range(start, scale + (1 - start)):
    for j in range(start, scale + (1 - start) - i):
        k = scale - i - j
        a.append((i, j, k))

# ternary binning
counts = []
for i in range(len(a)):  # for every vertex
    # for j in range(len(a[i])):  # for every index in a vertex
    idx_range_min = ((rgb[:, 0] > a[i][0]-0.5) &
                     (rgb[:, 1] > a[i][1]-0.5) &
                     (rgb[:, 2] > a[i][2]-0.5))

    idx_range_max = ((rgb[:, 0] < a[i][0]+0.5) &
                     (rgb[:, 1] < a[i][1]+0.5) &
                     (rgb[:, 2] < a[i][2]+0.5))
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
tax.left_axis_label("T2*", fontsize=fontsize)
tax.right_axis_label("PD", fontsize=fontsize)
tax.bottom_axis_label("T1w", fontsize=fontsize)
tax.boundary(linewidth=0.1)
tax.gridlines(multiple=20, color="blue")
tax.heatmap(c, scale, style="hexagonal", vmin=0, vmax=max(counts))
tax.show()
