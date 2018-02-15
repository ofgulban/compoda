"""Compoda setup.

To install, using the commandline do:
    pip install -e /path/to/compoda

"""

from setuptools import setup

setup(name='compoda',
      version='0.3.0',
      description='Compositional data analysis tools implemented in Python.',
      url='https://github.com/ofgulban/compoda',
      download_url='',
      author='Omer Faruk Gulban',
      author_email='faruk.gulban@maastrichtuniversity.nl',
      license='GNU Geneal Public License Version 3',
      packages=['compoda'],
      install_requires=['numpy', 'scipy'],
      zip_safe=False)
