"""Exercise different logratio trasnformations."""

import numpy as np
import matplotlib.pyplot as plt
from utils import (closure, cart_to_quasipolar, rad_to_deg, geometric_mean,
                   alr_transforation, clr_transforation, ilr_transformation,
                   aitchison_inner_product)

data = np.ones([7, 3])
data[1, 0] *= 100
data[2, 0:2] *= 100
data[3, 1] *= 100
data[4, 1:] *= 100
data[5, 2] *= 100
data[6, 0::2] *= 100

data = closure(data)

alr = alr_transforation(data)
plt.scatter(alr[:, 0], alr[:, 1], alpha=0.5)
plt.show()

clr = clr_transforation(data)
plt.scatter(clr[:, 0], clr[:, 1], alpha=0.5)
plt.show()

ilr = ilr_transformation(data)
plt.scatter(ilr[:, 0], ilr[:, 1], alpha=0.5)
plt.show()
