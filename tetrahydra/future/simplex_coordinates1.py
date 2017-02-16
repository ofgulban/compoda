#! /usr/bin/env python
"""Simplex coordinates 1.

Author:
John Burkardt

License:
This code is distributed under the GNU LGPL license.

Modified:
28 June 2015
27 December 2016 (PEP8, PEP257, scipy style docstring changes)

"""


def simplex_coordinates1(m):
    """Compute Cartesian coordinates of simplex vertices.

    Parameters
    ----------
    m : int
        Spatial dimension of simplex.

    Returns
    -------
    x : 1d-array, shape [m, m+1]
        Vertex coordinates of m dimensional simplex in real space.

    Discussion
    ----------
    1) The simplex will have its centroid at 0.
    2) The sum of the vertices will be zero.
    3) The distance of each vertex from the origin will be 1.
    4) The length of each edge will be constant.

    The dot product of the vectors defining any two vertices will be:
        -1/M
    This also means the angle subtended by the vectors from the origin
    to any two distinct vertices will be:
        arccos(-1/M)

    """
    import numpy as np

    x = np.zeros([m, m + 1])

    for k in range(0, m):
        # Set X(K,K) so that sum ( X(1:K,K)^2 ) = 1.
        s = 0.0
        for i in range(0, k):
            s = s + x[i, k] ** 2

    x[k, k] = np.sqrt(1.0 - s)

    # Set X(K,J) for J = K+1 to M+1 by using the fact XK dot XJ = - 1 / M.
    for j in range(k + 1, m + 1):
        s = 0.0
        for i in range(0, k):
            s = s + x[i, k] * x[i, j]
        x[k, j] = (-1.0 / float(m) - s) / x[k, k]

    return x


def simplex_coordinates1_test(m):
    """Test SIMPLEX_COORDINATES1.

    Parameters
    ----------
    Input, integer M, the spatial dimension.
    """
    import numpy as np
    import platform
    from r8_factorial import r8_factorial
    from r8mat_transpose_print import r8mat_transpose_print
    from simplex_volume import simplex_volume

    print ('')
    print ('SIMPLEX_COORDINATES1_TEST')
    print ('  Python version: %s' % (platform.python_version()))
    print ('  Test SIMPLEX_COORDINATES1')

    x = simplex_coordinates1(m)
    r8mat_transpose_print(m, m+1, x, '  Simplex vertex coordinates:')

    s = 0.0
    for i in range(0, m):
        s = s + (x[i, 0] - x[i, 1]) ** 2

    side = np.sqrt(s)
    volume = simplex_volume(m, x)
    volume2 = np.sqrt(m+1) / r8_factorial(m) \
        / np.sqrt(2.0 ** m) * side ** m

    print ('')
    print ('  Side length =     %g' % (side))
    print ('  Volume =          %g' % (volume))
    print ('  Expected volume = %g' % (volume2))

    xtx = np.dot(np.transpose(x), x)
    r8mat_transpose_print(m+1, m+1, xtx, '  Dot product matrix:')

    # Terminate.
    print ('')
    print ('SIMPLEX_COORDINATES1_TEST')
    print ('  Normal end of execution.')

    return


if (__name__ == '__main__'):
    from timestamp import timestamp
    timestamp()
    simplex_coordinates1_test(3)
    timestamp()
