[![DOI](https://zenodo.org/badge/78312374.svg)](https://zenodo.org/badge/latestdoi/78312374) [![Build Status](https://travis-ci.org/ofgulban/tetrahydra.svg?branch=master)](https://travis-ci.org/ofgulban/tetrahydra)  [![codecov](https://codecov.io/gh/ofgulban/tetrahydra/branch/master/graph/badge.svg)](https://codecov.io/gh/ofgulban/tetrahydra) [![Code Health](https://landscape.io/github/ofgulban/tetrahydra/master/landscape.svg?style=flat)](https://landscape.io/github/ofgulban/tetrahydra/master)



<img src="/misc/logo.png" width=300 align="right" />

# Tetrahydra

Compositional data analysis tools for multi-modal images.

Currently, this library is primarily being developed for (but not limited to) magnetic resonance images with multiple contrasts. For further details, check my [working short-paper](https://arxiv.org/abs/1705.03457) which is presented in [CoDaWork 2017](http://www.compositionaldata.com/codawork2017/).

## Dependencies

[**Python 2.7**](https://www.python.org/download/releases/2.7/)

| Package                                                 | Tested version |
|---------------------------------------------------------|----------------|
| [NumPy](http://www.numpy.org/)                          | 1.13.1         |
| [Scipy](https://www.scipy.org/)                         | 0.19.1         |

#### Additionally required for example scripts:

| Package                                                 | Tested version |
|---------------------------------------------------------|----------------|
| [matplotlib](http://matplotlib.org/)                    | 1.5.3          |
| [NiBabel](http://nipy.org/nibabel/)                     | 2.1.0          |


## Installation & Quick Start

```bash
cd /path/to/tetrahydra/
pip install -r requirements.txt
python setup.py install
```

## Support

Please use [GitHub issues](https://github.com/ofgulban/tetrahydra/issues) for questions, bug reports or feature requests.

## License

The project is licensed under [GNU General Public License Version 3](http://www.gnu.org/licenses/gpl.html).

## References

* Aitchison, J. (1982). The Statistical Analysis of Compositional Data. Journal of the Royal Statistical Society, 44(2), 139–177.

* Aitchison, J. (2002). A Concise Guide to Compositional Data Analysis. CDA Workshop Girona, 24, 73–81.

* Pawlowsky-Glahn, V., Egozcue, J. J., & Tolosana-Delgado, R. (2015). Modelling and Analysis of Compositional Data. Chichester, UK: John Wiley & Sons, Ltd. http://doi.org/10.1002/9781119003144

* [Compositional data analysis in a nutshell.](http://www.sediment.uni-goettingen.de/staff/tolosana/extra/CoDaNutshell.pdf)
