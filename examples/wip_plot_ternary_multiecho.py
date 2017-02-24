"""Example usages of ternary plotting library."""

import os
import ternary
import numpy as np
import matplotlib.pyplot as plt
from nibabel import load
from tetrahydra.core import closure, perturbation, powering
np.seterr(divide='ignore', invalid='ignore')

# Load data
nii1 = load('/home/faruk/Data/benedikt/echo_1.nii.gz')
nii2 = load('/home/faruk/Data/benedikt/echo_2.nii.gz')
nii3 = load('/home/faruk/Data/benedikt/echo_3.nii.gz')

nii4 = load('/home/faruk/Data/benedikt/mostly_gm.nii.gz')
msk = nii4.get_data().flatten()
msk = msk > 0

basename = nii1.get_filename().split(os.extsep, 1)[0]
dirname = os.path.dirname(nii1.get_filename())

nr_measurements = nii1.shape[3]

# load descriptives for centering and standardization
descriptives = np.load('/home/faruk/Data/benedikt/gm_descriptives.npy')
descriptives = descriptives.item()
center = descriptives["Center"]
print "Sample center: " + str(center)
totvar = descriptives['Total variance']
print "Total variance: " + str(totvar)

for v in range(nr_measurements):
    vol1 = nii1.get_data()[..., v]
    vol2 = nii2.get_data()[..., v]
    vol3 = nii3.get_data()[..., v]
    shape = vol1.shape + (3,)

    comp = np.zeros(shape)
    comp[:, :, :, 0] = vol1
    comp[:, :, :, 1] = vol2
    comp[:, :, :, 2] = vol3
    comp = comp.reshape(shape[0]*shape[1]*shape[2], shape[3])
    comp = comp[msk, :]  # apply mask

    # Impute
    comp[comp == 0] = 1
    # Closure
    comp = closure(comp)
    # Center
    temp = np.ones(comp.shape) * center
    comp = perturbation(comp, temp**-1)
    # Standardize
    comp = powering(comp, np.power(totvar, -1./2.))

    # Scale data if needed
    scale = 80
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
    tax.set_title("Ternary heat map", fontsize=fontsize)
    tax.left_axis_label("Echo 1", fontsize=fontsize)
    tax.right_axis_label("Echo 2", fontsize=fontsize)
    tax.bottom_axis_label("Echo 2", fontsize=fontsize)
    tax.boundary(linewidth=0)
    tax.gridlines(multiple=scale/4., color="white")
    tax.heatmap(c, scale, style="hexagonal", vmin=0, vmax=5)
    # tax.show()
    tax.savefig(os.path.join(dirname, 'mostlygm/'+str(v).zfill(3)+'.png'))
    plt.close(figure)  # clear figure
    print str(v+1)+'/'+str(nr_measurements)
