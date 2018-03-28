"""Compoda setup.

To install, using the command line do:
    pip install -e /path/to/compoda

Notes for PyPI:
python setup.py sdist upload -r pypitest
python setup.py sdist upload -r pypi

"""

from setuptools import setup

VERSION = '0.3.2'

setup(name='compoda',
      version=VERSION,
      description='Compositional data analysis tools implemented in Python.',
      url='https://github.com/ofgulban/compoda',
      download_url=('https://github.com/ofgulban/compoda/archive/'
                    + VERSION + '.tar.gz'),
      author='Omer Faruk Gulban',
      author_email='faruk.gulban@maastrichtuniversity.nl',
      license='GNU Geneal Public License Version 3',
      packages=['compoda'],
      install_requires=['numpy', 'scipy'],
      zip_safe=False)
