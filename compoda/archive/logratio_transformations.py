"""Create a figure to compare ilr, clr and alr transformations."""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import compoda.core as tet

temp = np.arange(1, 101)

data = np.ones([700, 3])

# primary color 1
data[0:100, 0] = temp
# primary color 1 & 2 mixture
data[100:200, 0] = temp
data[100:200, 1] = temp
# primary color 2
data[200:300, 1] = temp
# primary color 2 & 3 mixture
data[300:400, 1] = temp
data[300:400, 2] = temp
# primary color 3
data[400:500, 2] = temp
# primary color 3 & 1 mixture
data[500:600, 2] = temp
data[500:600, 0] = temp

# constant primary color 1, increasing primary color 2 mixture and vice versa
data[600:610, 0] = 100
data[600:610, 1] = temp[5::10]
data[610:620, 1] = 100
data[610:620, 0] = temp[5::10]
# constant primary color 2, increasing primary color 3 mixture and vice versa
data[620:630, 1] = 100
data[620:630, 2] = temp[5::10]
data[630:640, 2] = 100
data[630:640, 1] = temp[5::10]
# constant primary color 3, increasing primary color 1 mixture and vice versa
data[640:650, 2] = 100
data[640:650, 0] = temp[5::10]
data[650:660, 0] = 100
data[650:660, 2] = temp[5::10]

# closure
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
ax_1.scatter(alr[:, 0], alr[:, 1], color='red', s=2)
ax_2.scatter(clr[:, 0], clr[:, 1], clr[:, 2], color='green', s=2)
ax_3.scatter(ilr[:, 0], ilr[:, 1], color='blue', s=2)

plt.show()
