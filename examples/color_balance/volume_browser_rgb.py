"""Visualize RGB volume slices."""

from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from nibabel import load
from compoda.utils import scale_range
np.seterr(divide='ignore', invalid='ignore')

"""Load Data"""
#
vol1 = load('/home/faruk/gdrive/Segmentator/data/faruk/pt7/input1_simplex_cbal.nii.gz')
vol2 = load('/home/faruk/gdrive/Segmentator/data/faruk/pt7/input2_simplex_cbal.nii.gz')
vol3 = load('/home/faruk/gdrive/Segmentator/data/faruk/pt7/input3_simplex_cbal.nii.gz')

basename = vol1.get_filename().split(os.extsep, 1)[0]

niiName = vol1.get_filename()
niiHeader = vol1.header
niiAffine = vol1.affine
shape = vol1.shape
shape = (shape[0], shape[1], shape[2], 3)


# matplotlib wants RGB range between 0 and 1
vol1 = scale_range(vol1.get_data(), scale_factor=1)
vol2 = scale_range(vol2.get_data(), scale_factor=1)
vol3 = scale_range(vol3.get_data(), scale_factor=1)

rgb = np.zeros(shape)
rgb[:, :, :, 0] = vol1
rgb[:, :, :, 1] = vol2
rgb[:, :, :, 2] = vol3
del vol1, vol2, vol3

# plot 3D ima by default
fig = plt.figure()
ax = fig.add_subplot(111)
slc = ax.imshow(rgb[:, :, int(rgb.shape[2]/2)], vmin=0, vmax=1,
                interpolation='none')
plt.axis('off')
plt.subplots_adjust(bottom=0.2)


def UpdateDataBrowser(val):
    """Browse imshow data."""
    global rgb
    # Scale slider value [0,1) to dimension index to allow variation in shape
    sliceNr = int(sSliceNr.val*rgb.shape[2])
    slc.set_data(rgb[:, :, sliceNr])
    slc.set_extent((0, rgb.shape[1]-1, rgb.shape[0]-1, 0))
    fig.canvas.draw_idle()


# ima browser slider
axSliceNr = plt.axes([0.1, 0.1, 0.25, 0.025], facecolor='0.875')
sSliceNr = Slider(axSliceNr, 'Slice', 0, 0.999, valinit=0.5, valfmt='%0.3f')
sSliceNr.on_changed(UpdateDataBrowser)


# cycle button
def CycleView(event):
    """Cycle through viewing dimensions."""
    global rgb, cycleCount
    cycleCount = (cycleCount+1) % 3
    rgb = np.transpose(rgb, (2, 0, 1, 3))


cycleax = plt.axes([0.6, 0.10, 0.075, 0.075])
bCycle = Button(cycleax, 'Cycle\nView', color='0.875', hovercolor='0.975')
cycleCount = 0
bCycle.on_clicked(CycleView)
bCycle.on_clicked(UpdateDataBrowser)

plt.show()
