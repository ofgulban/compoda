"""Example usages of ternary plotting library."""

import os
import ternary
import numpy as np
import matplotlib.pyplot as plt
from nibabel import load
from tetrahydra.utils import truncate_and_scale
np.seterr(divide='ignore', invalid='ignore')

# Load data
nii1 = load('/media/Data_Drive/Benchmark_Data/compositional_data/Benedikt/nifti/echo_1.nii.gz')
nii2 = load('/media/Data_Drive/Benchmark_Data/compositional_data/Benedikt/nifti/echo_2.nii.gz')
nii3 = load('/media/Data_Drive/Benchmark_Data/compositional_data/Benedikt/nifti/echo_3.nii.gz')

nii4 = load('/media/Data_Drive/Benchmark_Data/compositional_data/Benedikt/nifti/bet_mask_ero10.nii.gz')
msk = nii4.get_data().flatten()
msk = msk > 0

basename = nii1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(nii1.get_filename())

nr_measurements = nii1.shape[3]
v=0
for v in range(nr_measurements):
    vol1 = nii1.get_data()[..., v]
    vol2 = nii2.get_data()[..., v]
    vol3 = nii3.get_data()[..., v]
    shape = vol1.shape + (3,)

    # # normalize
    # vol1 = truncate_and_scale(vol1, percMin=0, percMax=100)
    # vol2 = truncate_and_scale(vol2, percMin=0, percMax=100)
    # vol3 = truncate_and_scale(vol3, percMin=0, percMax=100)

    rgb = np.zeros(shape)
    rgb[:, :, :, 0] = vol1
    rgb[:, :, :, 1] = vol2
    rgb[:, :, :, 2] = vol3
    rgb = rgb.reshape(shape[0]*shape[1]*shape[2], shape[3])
    rgb = rgb[msk, :]  # apply mask

    # Closure
    sum_rgb = np.sum(rgb, axis=1)
    for i in range(shape[3]):
        rgb[:, i] = rgb[:, i]/sum_rgb

    # Scale data if needed
    scale = 100
    rgb = rgb * scale

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
    tax.set_title("Ternary heat map", fontsize=fontsize)
    tax.left_axis_label("Echo 1", fontsize=fontsize)
    tax.right_axis_label("Echo 2", fontsize=fontsize)
    tax.bottom_axis_label("Echo 2", fontsize=fontsize)
    tax.boundary(linewidth=0)
    tax.gridlines(multiple=scale/4., color="white")
    tax.heatmap(c, scale, style="hexagonal", vmin=0, vmax=5)
    # tax.show()
    tax.savefig(os.path.join(dirname, 'plots_mostlywmcsf/test'+str(v).zfill(3)+'.png'))
    plt.close(figure)  # clear figure
    print str(v+1)+'/'+str(nr_measurements)
