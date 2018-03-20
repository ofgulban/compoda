"""Color transforamtions on 2D image."""

import matplotlib
from skimage import data, color
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import numpy as np
import compoda.core as coda
from compoda.utils import truncate_range
from matplotlib import rcParams
rcParams['font.family'] = "serif"
# rcParams['mathtext.fontset'] = 'dejavuserif'

# load image
orig = data.rocket()
img = np.copy(orig) * 1.0
dims = img.shape

# impute zeros
idx1 = (img[..., 0] <= 0) + (img[..., 1] <= 0) + (img[..., 2] <= 0)
gauss = gaussian(img, sigma=0.5, multichannel=True)
img[idx1, :] = gauss[idx1, :]

# rgb to hsv for control
hsv = color.rgb2hsv(img)
hsv[..., 0] = -np.abs(hsv[..., 0] - 0.5) + 0.5

# coda stuff
angdif_norm_int = np.zeros(img.shape)
temp = img.reshape(dims[0] * dims[1], dims[2]) * 1.0
bary = coda.closure(temp, 100)
anorm = coda.aitchison_norm(bary)

# prepare reference vector for angular difference
ref = np.ones(bary.shape)
ref[:, 0] = ref[:, 0] * 0.8
ref[:, 1] = ref[:, 1] * 0.1
ref[:, 2] = ref[:, 2] * 0.1
# compute aitchion angular difference
ainp = coda.aitchison_inner_product(bary, ref)
ref_norm = coda.aitchison_norm(ref)
ang_dif = np.zeros(anorm.shape)
# deal with zero norms
idx = anorm != 0
# (choose one) wrapped angle range
ang_dif[idx] = np.arccos(ainp[idx]/(anorm[idx] * ref_norm[idx]))
ang_dif[np.isnan(ang_dif)] = 0  # fix nans assigned to bright
# (choose one) full angle range
idx_s = bary[:, 1] > bary[:, 2]
# ang_dif[idx_s] = np.arccos(ainp[idx_s]/(anorm[idx_s] * ref_norm[idx_s]))
# ang_dif[~idx_s] = 2*np.pi - np.arccos(ainp[~idx_s]/(anorm[~idx_s] * ref_norm[~idx_s]))
# truncate anorm
anorm = truncate_range(anorm, percMin=0, percMax=99)

# reassign
angdif_norm_int[..., 0] = ang_dif.reshape(dims[:-1])
angdif_norm_int[..., 1] = anorm.reshape(dims[:-1])
angdif_norm_int[..., 2] = np.sum(img, axis=-1) / 3.

# Plots
plt.subplot(3, 4, 1)
plt.imshow(orig)
plt.title('Original')
plt.axis('off')

# RGB seperated
scalar_color = 'Greys_r'
plt.subplot(3, 4, 2)
plt.imshow(img[..., 0], cmap=scalar_color)
plt.title('Red')
plt.axis('off')
plt.subplot(3, 4, 3)
plt.imshow(img[..., 1], cmap=scalar_color)
plt.title('Green')
plt.axis('off')
plt.subplot(3, 4, 4)
plt.imshow(img[..., 2], cmap=scalar_color)
plt.title('Blue')
plt.axis('off')

# HSV seperated
plt.subplot(3, 4, 6)
plt.imshow(hsv[..., 0], cmap=scalar_color)
plt.title('Hue')
plt.axis('off')
plt.subplot(3, 4, 7)
plt.imshow(hsv[..., 1], cmap=scalar_color)
plt.title('Saturation')
plt.axis('off')
plt.subplot(3, 4, 8)
plt.imshow(hsv[..., 2], cmap=scalar_color)
plt.title('Value')
plt.axis('off')

# coda seperated
plt.subplot(3, 4, 10)
plt.imshow(angdif_norm_int[..., 0], cmap=scalar_color)
plt.title('Angular diff. in $\mathrm{\mathbb{S}}^3$')
plt.axis('off')
plt.subplot(3, 4, 11)
plt.imshow(angdif_norm_int[..., 1], cmap=scalar_color)
plt.title('Norm in $\mathrm{\mathbb{S}}^3$')
plt.axis('off')
plt.subplot(3, 4, 12)
plt.imshow(angdif_norm_int[..., 2], cmap=scalar_color)
plt.title('Intensity')
plt.axis('off')

plt.show()
