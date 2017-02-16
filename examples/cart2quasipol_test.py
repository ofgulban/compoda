"""Convert n dimensional carthesian coordinates to quasipolar coordinates."""

import numpy as np
import time
from tetrahydra.utils import rad_to_deg, deg_to_rad, cart_to_quasipolar

# Testing angle conversions
coord = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
rad_to_deg(cart_to_quasipolar(coord)[:, 1:])

x, y = 1., 1.
r = np.sqrt(x**2 + y**2)
theta_1 = np.arccos(y/r)
np.degrees(theta_1)

# Test zone
angle_1 = deg_to_rad(45)
angle_2 = deg_to_rad(225)

np.cos(angle_1)
np.cos(angle_2)

np.arccos(np.cos(angle_1/2))
np.arccos(np.cos(angle_2/2))

rad_to_deg(np.arccos(np.cos(angle_1/2))*2)
rad_to_deg(np.arccos(np.cos(angle_2/2))*2)

# Speed tests
start = time.time()
print('computing n dimensional quasi-polar coordinates...')

faruk = np.ones([10000, 30])
results = cart_to_quasipolar(faruk)

end = time.time()
print 'coordinates are computed in:', (end - start), 'seconds'
