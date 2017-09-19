"""tetrahydra setup.

To install, using the commandline do:
    pip install -e /path/to/tetrahydra

"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='tetrahydra',
      version='0.1.0',
      description='Compositional data methods for MRI data.',
      url='https://github.com/ofgulban/tetrahydra',
      download_url='',
      author='Omer Faruk Gulban',
      author_email='faruk.gulban@maastrichtuniversity.nl',
      license='GNU Geneal Public License Version 3',
      packages=['tetrahydra'],
      install_requires=['numpy', 'scipy'],
      zip_safe=False)
