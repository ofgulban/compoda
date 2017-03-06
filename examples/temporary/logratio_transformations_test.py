"""Exercise different logratio trasnformations.

TODO: Turn this script into trasnformation tests.

"""

import numpy as np
import seaborn
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tetrahydra.core as tet

data = np.ones([7, 3])
data[1, 0] *= 100
data[2, 0:2] *= 100
data[3, 1] *= 100
data[4, 1:] *= 100
data[5, 2] *= 100
data[6, 0::2] *= 100
data = tet.closure(data)  # R^D

# alr
alr = tet.alr_transformation(data)  # R^D to S^(D-1)
ialr = tet.inverse_alr_transformation(alr)  # S^(D-1) to R^(D-1)
np.testing.assert_almost_equal(ialr, data, decimal=7, verbose=True)

# clr
clr = tet.clr_transformation(data)  # R^D to S^D
iclr = tet.inverse_clr_transformation(clr)  # S^D to R^D

# ilr
ilr = tet.ilr_transformation(data)  # R^D to S^(D-1)
iilr = tet.inverse_ilr_transformation(ilr)  # S^(D-1) to R^(D-1)
np.testing.assert_almost_equal(iilr, data, decimal=7, verbose=True)

# plots
fig = plt.figure()
ax_1 = plt.subplot(131)
ax_2 = plt.subplot(132,  projection='3d')
ax_3 = plt.subplot(133)
ax_1.set_title('Additive log-ratio transformation\n(alr)')
ax_2.set_title('Centered log-ratio transformation\ntransformation(clr)')
ax_3.set_title('Isometric log-ratio transformation\ntransformation(ilr)')
ax_1.set_aspect('equal', 'datalim')
ax_2.set_aspect('equal')
ax_3.set_aspect('equal', 'datalim')
ax_1.scatter(alr[:, 0], alr[:, 1], color='red')
ax_2.scatter(clr[:, 0], clr[:, 1], clr[:, 2], color='green')
ax_3.scatter(ilr[:, 0], ilr[:, 1], color='blue')

plt.show()
