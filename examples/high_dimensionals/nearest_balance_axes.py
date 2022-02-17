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
idx_nan = np.isnan(data)
data[idx_nan] = 0

# data = np.abs(data) + 0.1
# data = data - (data.min() - 1)
idx_clip = data < 0
data[data < 0] = 0
data += 0.001
data = data.reshape([dims[0] * dims[1], dims[2]])

# =============================================================================
# L2 Norm
norm_L2 = np.sqrt(np.sum(np.power(data, 2), axis=1))
img = norm_L2 - norm_L2.min()
img = img / (norm_L2.max() - norm_L2.min()) * 255
img = np.asarray(img, dtype=("uint8"))
img = np.reshape(img, dims[0:2])
cv2.imwrite("/home/faruk/gdrive/thingsonthings/post-compositonal_visuals/test_L2norm.png", img)

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

# Reference points in simplex space
ref = np.asarray(
    [[1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
     [1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
     [1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
     [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
     [1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1],
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1],
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1],
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1],
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1]])
ref = coda.closure(ref)

# Scaling: make ref points lying on a sphere
ref_anorm = coda.aitchison_norm(ref)
ref_scaled = ref_anorm[0, None] / ref_anorm
ref_scaled = coda.power(ref, ref_scaled[..., None])

# -----------------------------------------------------------------------------
# Compute distances
dist = np.zeros((bary2.shape[0], ref.shape[0]))
for i in range(ref.shape[0]):
    dist[:, i] = coda.aitchison_dist(bary2, ref_scaled[None, i, :])

# Classification
idx_min = np.argmin(dist, axis=-1)

# Save output as grayscale
img = idx_min / (ref.shape[0]-1) * 255
img = np.asarray(img, dtype=("uint8"))
img = np.reshape(img, dims[0:2])

img[idx_nan[:, :, 0]] = 0
img[np.min(idx_clip, axis=-1)] = 0

cv2.imwrite("/home/faruk/gdrive/thingsonthings/post-compositonal_visuals/test_class.png", img)
print("Finished.")
