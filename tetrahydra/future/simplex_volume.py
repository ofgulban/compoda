#! /usr/bin/env python
#
def simplex_volume ( m, x ):

#*****************************************************************************80
#
## SIMPLEX_VOLUME computes the volume of a simplex.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    28 June 2015
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, integer M, the spatial dimension.
#
#    Input, real X(M,M+1), the coordinates of the vertices
#    of a simplex in M dimensions.
#
#    Output, real VOLUME, the volume of the simplex.
#
  import numpy as np
  from simplex01_volume import simplex01_volume

  a = np.zeros ( [ m, m ] )
  for j in range ( 0, m ):
    for i in range ( 0, m ):
      a[i,j] = x[i,j] - x[i,m]

  volume = abs ( np.linalg.det ( a ) )

  volume01 = simplex01_volume ( m )

  volume = volume * volume01

  return volume

def simplex_volume_test ( ) :

#*****************************************************************************80
#
## SIMPLEX_VOLUME_TEST tests SIMPLEX_VOLUME.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    28 June 2015
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform
  from r8mat_transpose_print import r8mat_transpose_print

  print ( '' )
  print ( 'SIMPLEX_VOLUME_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  SIMPLEX_VOLUME returns the volume of a simplex' )
  print ( '  in M dimensions.' )

  m = 2
  x2 = np.array ( [ \
    [ 0.0, 7.0, 4.0 ], \
    [ 0.0, 2.0, 4.0 ] ] )
  r8mat_transpose_print ( m, m + 1, x2, '  Triangle:' )
  value = simplex_volume ( m, x2 )

  print ( '' )
  print ( '  Volume = %g' % ( value ) )

  m = 3
  x3 = np.array ( [ \
    [ 0.0, 7.0, 4.0, 0.0 ], \
    [ 0.0, 2.0, 4.0, 0.0 ], \
    [ 0.0, 0.0, 0.0, 6.0 ] ] )
  r8mat_transpose_print ( m, m + 1, x3, '  Tetrahedron:' )
  value = simplex_volume ( m, x3 )

  print ( '' )
  print ( '  Volume = %g' % ( value ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'SIMPLEX_VOLUME_TEST' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  from timestamp import timestamp
  timestamp ( )
  simplex_volume_test ( )
  timestamp ( )

