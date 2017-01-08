#! /usr/bin/env python
"""Simplex coordinates 2.

Author:
John Burkardt

License:
This code is distributed under the GNU LGPL license.

Modified:
28 June 2015
27 December 2016 (PEP8, PEP257, scipy style docstring changes)

"""


def simplex_coordinates2(m):
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
    -----------
    This routine uses a simple approach to determining the coordinates
    of the vertices of a regular simplex in n dimensions.

    We want the vertices of simplex to satisfy the following
    conditions:

    1) The centroid, or average of the vertices, is 0.
    2) The distance of each vertex from the centroid is 1.
       By 1), this is equivalent to requiring that the sum of the
       squares of the coordinates of any vertex be 1.
    3) The distance between any pair of vertices is equal and is not 0.
    4) The dot product of any two coordinate vectors for distinct
       vertices is -1/M; equivalently, the angle subtended by two
       distinct vertices from the centroid is arccos ( -1/M).

    Note that if we choose the first M vertices to be the columns of
    the MxM identity matrix, we are almost there.  By symmetry, the
    last column must have all entries equal to some value A. Because
    the square of the distance between the last column and any other
    column must be 2 (because that's the distance between any pair of
    columns), we deduce that:
        (A-1)^2 + (M-1)*A^2 = 2, hence A = (1-sqrt(1+M))/M.
    Now compute the centroid C of the vertices, and subtract that, to
    center the simplex around the origin. Finally, compute the norm of
    one column, and rescale the matrix of coordinates so each vertex
    has unit distance from the origin.

    (This approach devised by John Burkardt, 19 September 2010)

    """
    import numpy as np

    x = np.zeros([m, m + 1])

    for j in range(0, m):
        x[j, j] = 1.0

    a = (1.0 - np.sqrt(float(1 + m))) / float(m)

    for i in range(0, m):
        x[i, m] = a

    # Adjust coordinates so the centroid is at zero.
    c = np.zeros(m)
    for i in range(0, m):
        s = 0.0
        for j in range(0, m + 1):
            s = s + x[i, j]
        c[i] = s / float(m + 1)

    for j in range(0, m + 1):
        for i in range(0, m):
            x[i, j] = x[i, j] - c[i]

    # Scale so each column has norm 1.
    s = 0.0
    for i in range(0, m):
        s = s + x[i, 0] ** 2
    s = np.sqrt(s)

    for j in range(0, m + 1):
        for i in range(0, m):
            x[i, j] = x[i, j] / s

    return x


def simplex_coordinates2_test(m):
    """Test SIMPLEX_COORDINATES2.

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
    print ('SIMPLEX_COORDINATES2_TEST')
    print ('  Python version: %s' % (platform.python_version()))
    print ('  Test SIMPLEX_COORDINATES2')

    x = simplex_coordinates2(m)

    r8mat_transpose_print(m, m + 1, x, '  Simplex vertex coordinates:')

    s = 0.0
    for i in range(0, m):
        s = s + (x[i, 0] - x[i, 1]) ** 2

    side = np.sqrt(s)

    volume = simplex_volume(m, x)

    volume2 = np.sqrt(m + 1) / r8_factorial(m) \
        / np.sqrt(2.0 ** m) * side ** m

    print ('')
    print ('  Side length =     %g' % (side))
    print ('  Volume =          %g' % (volume))
    print ('  Expected volume = %g' % (volume2))

    xtx = np.dot(np.transpose(x), x)

    r8mat_transpose_print(m + 1, m + 1, xtx, '  Dot product matrix:')

    # Terminate.
    print ('')
    print ('SIMPLEX_COORDINATES2_TEST')
    print ('  Normal end of execution.')
    return


if (__name__ == '__main__'):
    from timestamp import timestamp
    timestamp()
    simplex_coordinates2_test(3)
    timestamp()
