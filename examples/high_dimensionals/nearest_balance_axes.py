"""Compute nearest balance axes and saturation for high dimensional images."""

import mat73
import cv2
import compoda.core as coda
import nibabel as nb
import numpy as np


# Load MATLAB file that has a stack of scalar images
data_dict = mat73.loadmat("/home/faruk/gdrive/thingsonthings/post-compositonal_visuals/data_from_kendrick/rawimg.mat")
data = data_dict["rawimg"]
dims = data.shape

# Remove negative by translation
data[np.isnan(data)] = 0
data = data - (data.min() - 1)
data = data.reshape([dims[0] * dims[1], dims[2]])

# =============================================================================
# Compute Aitchison norm (aka saturation)
bary = coda.closure(data)
anorm = coda.aitchison_norm(bary)

# Cast to uint8 and reshape
img = anorm - anorm.min()
img = img / (anorm.max() - anorm.min()) * 255
img = np.asarray(img, dtype=("uint8"))
img = np.reshape(img, dims[0:2])

cv2.imwrite("/home/faruk/gdrive/thingsonthings/post-compositonal_visuals/test_anorm.png", img)

# =============================================================================
# Compute nearest balance axes
bary2 = np.sort(bary, axis=1)  # Sort components to simplify the problem
