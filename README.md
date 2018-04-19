[![DOI](https://zenodo.org/badge/78312374.svg)](https://zenodo.org/badge/latestdoi/78312374) [![Build Status](https://travis-ci.org/ofgulban/compoda.svg?branch=master)](https://travis-ci.org/ofgulban/compoda) [![Build status](https://ci.appveyor.com/api/projects/status/plrtbnjvf09h38xn?svg=true)](https://ci.appveyor.com/project/ofgulban/compoda)
 [![codecov](https://codecov.io/gh/ofgulban/compoda/branch/master/graph/badge.svg)](https://codecov.io/gh/ofgulban/compoda) [![Code Health](https://landscape.io/github/ofgulban/compoda/master/landscape.svg?style=flat)](https://landscape.io/github/ofgulban/compoda/master)



<img src="/visuals/logo.png" width=200 align="right" />

# Compoda

Compositional data analysis tools implemented in python.

Currently, this library is primarily being developed for (but not limited to) magnetic resonance images with multiple contrasts. For further details, check my [preprint on arXiv](https://arxiv.org/abs/1705.03457) which is presented in [CoDaWork 2017](http://www.compositionaldata.com/codawork2017/).

## Dependencies

**[Python 3.6](https://www.python.org/downloads/release/python-363/)** or **[Python 2.7](https://www.python.org/download/releases/2.7/)** (compatible with both).

| Package                                                 | Tested version |
|---------------------------------------------------------|----------------|
| [NumPy](http://www.numpy.org/)                          | 1.14.2         |
| [Scipy](https://www.scipy.org/)                         | 1.0.0          |

#### Additionally required for example scripts:

| Package                                                 | Tested version |
|---------------------------------------------------------|----------------|
| [matplotlib](http://matplotlib.org/)                    | 2.2.2          |
| [NiBabel](http://nipy.org/nibabel/)                     | 2.2.1          |


## Installation

Then run this command in your command line (If you are using [**Python 2.7**](https://www.python.org/download/releases/2.7/) make sure to have [**pip**](https://en.wikipedia.org/wiki/Pip_(package_manager)) installed):

```bash
pip install compoda
```

or as an alternative
- clone this repository and change directory to:
```bash
cd /path/to/compoda
```
- Install the requirements by running the following command:
```bash
pip install -r requirements.txt
```
- Install compoda:
```bash
python setup.py install
```

## Support

Please use [GitHub issues](https://github.com/ofgulban/compoda/issues) for questions, bug reports or feature requests.

## License

The project is licensed under [GNU General Public License Version 3](http://www.gnu.org/licenses/gpl.html).

## References

* [Compositional data analysis in a nutshell.](http://www.sediment.uni-goettingen.de/staff/tolosana/extra/CoDaNutshell.pdf)

* Aitchison, J. (1982). The Statistical Analysis of Compositional Data. Journal of the Royal Statistical Society, 44(2), 139–177.

* Aitchison, J. (2002). A Concise Guide to Compositional Data Analysis. CDA Workshop Girona, 24, 73–81.

* Pawlowsky-Glahn, V., Egozcue, J. J., & Tolosana-Delgado, R. (2015). Modelling and Analysis of Compositional Data. Chichester, UK: John Wiley & Sons, Ltd. http://doi.org/10.1002/9781119003144
