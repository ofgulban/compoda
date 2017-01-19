# -*- coding: utf-8 -*-
"""
Function of the depth sampling library.

The purpose of this script is to demonstrate normalisation of depth profiles on
simulated data. This is useful to understand which effect normalisation (e.g.
by division or subtraction) has on the depth profiles.
"""

# Part of py_depthsampling library
# Copyright (C) 2016  Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
from ds_pltAcrDpth import funcPltAcrDpth
from utils import aitchison_norm, closure

# -----------------------------------------------------------------------------
# *** Define parameters:

# Number of depths:
varNumDpth = 100

# Scaling factor for amplitude of the mid-GM 'bump' (higher value --> higher
# amplitude):
varBumpAmp = 1.0

# Scaling factor for the width of the mid-GM 'bump' (higher value --> sharper
# bump):
varBumpWidth = 8.0

# Figure scaling factor:
varDpi = 75.0

# Output folder:
strPthOt = '/media/Data_Drive/People/Ingo/parcon_demo/'
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Create profile templates

# Linear datapoints:
aryLin = np.linspace(0.0, 1.0, varNumDpth, endpoint=True)

# Sinusoidal datapoints:
arySin = np.sin(np.linspace(0.0, np.pi, varNumDpth, endpoint=True))

# Make the sinusoid more sharp:
arySin = np.power(arySin, varBumpWidth)

# Scale the sinusoid:
arySin = np.multiply(arySin, varBumpAmp)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Create condition profiles

# Linear & sinusiodal term additive:
aryTmp01 = np.add(np.add(aryLin, 0.0),
                  np.add(arySin, 0.0))
aryTmp02 = np.add(np.add(aryLin, 1.0),
                  np.add(arySin, 1.0))
aryTmp03 = np.add(np.add(aryLin, 2.0),
                  np.add(arySin, 2.0))
aryTmp04 = np.add(np.add(aryLin, 3.0),
                  np.add(arySin, 3.0))
aryDemoAdd = np.vstack((aryTmp01,
                        aryTmp02,
                        aryTmp03,
                        aryTmp04))
# aryDemoAdd = aryDemoAdd.T

# Linear & sinusiodal term multiplicative:
aryTmp01 = np.add(np.multiply(aryLin, 1.0),
                  np.multiply(arySin, 1.0))
aryTmp02 = np.add(np.multiply(aryLin, 2.0),
                  np.multiply(arySin, 2.0))
aryTmp03 = np.add(np.multiply(aryLin, 3.0),
                  np.multiply(arySin, 3.0))
aryTmp04 = np.add(np.multiply(aryLin, 4.0),
                  np.multiply(arySin, 4.0))
aryDemoMul = np.vstack((aryTmp01,
                        aryTmp02,
                        aryTmp03,
                        aryTmp04))
# aryDemoMul = aryDemoMul.T

# Linear is additive, sinusoidal is multiplicative:
aryTmp01 = np.add(np.add(aryLin, 0.0),
                  np.multiply(arySin, 1.0))
aryTmp02 = np.add(np.add(aryLin, 1.0),
                  np.multiply(arySin, 2.0))
aryTmp03 = np.add(np.add(aryLin, 2.0),
                  np.multiply(arySin, 3.0))
aryTmp04 = np.add(np.add(aryLin, 3.0),
                  np.multiply(arySin, 4.0))
aryDemoMix01 = np.vstack((aryTmp01,
                        aryTmp02,
                        aryTmp03,
                        aryTmp04))
# aryDemoMix01 = aryDemoMix01.T

# Linear is multiplicative, sinusoidal is additive:
aryTmp01 = np.add(np.multiply(aryLin, 1.0),
                  np.add(arySin, 0.0))
aryTmp02 = np.add(np.multiply(aryLin, 2.0),
                  np.add(arySin, 1.0))
aryTmp03 = np.add(np.multiply(aryLin, 3.0),
                  np.add(arySin, 2.0))
aryTmp04 = np.add(np.multiply(aryLin, 4.0),
                  np.add(arySin, 3.0))
aryDemoMix02 = np.vstack((aryTmp01,
                        aryTmp02,
                        aryTmp03,
                        aryTmp04))
# aryDemoMix02 = aryDemoMix01.T

# Scale all profiles, so that they range from 0.5 to 3.5 (just for
# visualisation):
aryDemoMul = np.divide(aryDemoMul,
                       np.multiply(np.max(aryDemoMul),
                                   0.34))
aryDemoAdd = np.divide(aryDemoAdd,
                       np.multiply(np.max(aryDemoAdd),
                                   0.34))
aryDemoMix01 = np.divide(aryDemoMix01,
                       np.multiply(np.max(aryDemoMix01),
                                   0.34))

aryDemoMix02 = np.divide(aryDemoMix02,
                       np.multiply(np.max(aryDemoMix02),
                                   0.34))

aryDemoMul = np.add(aryDemoMul, 0.5)
aryDemoAdd = np.add(aryDemoAdd, 0.5)
aryDemoMix01 = np.add(aryDemoMix01, 0.5)
aryDemoMix02 = np.add(aryDemoMix02, 0.5)
# -----------------------------------------------------------------------------

# Compute aitchison norm

aryDemoMul.shape

aryDemoMul = closure(aryDemoMul.T).T
aryDemoAdd = closure(aryDemoAdd.T).T
aryDemoMix01 = closure(aryDemoMix01.T).T
aryDemoMix02 = closure(aryDemoMix02.T).T

# -----------------------------------------------------------------------------

# # Compute aitchison norm
#
# aryDemoMul.shape
#
# aryDemoMul = aitchison_norm(aryDemoMul.T).T
# aryDemoAdd = aitchison_norm(aryDemoAdd.T).T
# aryDemoMix01 = aitchison_norm(aryDemoMix01.T).T
# aryDemoMix02 = aitchison_norm(aryDemoMix02.T).T
#
# aryDemoMul = aryDemoMul[np.newaxis, :]
# aryDemoAdd = aryDemoAdd[np.newaxis, :]
# aryDemoMix01 = aryDemoMix01[np.newaxis, :]
# aryDemoMix02 = aryDemoMix02[np.newaxis, :]

# -----------------------------------------------------------------------------
# *** Plot profile templates

# Array with values for error bars:
aryError = np.zeros(aryDemoAdd.shape)
aryError = np.add(aryError, 0.01)

# Label on x axis
strXlabel = 'Cortical depth'

# Label on y axis
strYlabel = 'Signal change'

# Condition labels:
lstConLbl = ['72.0%', '16.3%', '6.1%', '2.5%']

# Scale linear term and add 2nd dimension:
aryTmp = np.array([np.multiply(aryLin, 1.0)], ndmin=2)

# -----------------------------------------------------------------------------
# *** Create plots before normalisation

# Array with values for error bars:
aryError = np.zeros(aryDemoAdd.shape)
aryError = np.add(aryError, 0.001)

# Plot profiles:
funcPltAcrDpth(aryDemoAdd,
               aryError,
               varNumDpth,
               4,
               varDpi,
               0.0,
               1.0,
               False,
               lstConLbl,
               strXlabel,
               strYlabel,
               'Linear & sinusiodal term additive',
               False,
               (strPthOt + 'plt_03_closure.svg'))

funcPltAcrDpth(aryDemoMul,
               aryError,
               varNumDpth,
               4,
               varDpi,
               0.0,
               1.0,
               False,
               lstConLbl,
               strXlabel,
               strYlabel,
               'Linear & sinusiodal term multiplicative',
               False,
               (strPthOt + 'plt_04_closure.svg'))

funcPltAcrDpth(aryDemoMix01,
               aryError,
               varNumDpth,
               4,
               varDpi,
               0.0,
               1.0,
               False,
               lstConLbl,
               strXlabel,
               strYlabel,
               'Linear = add., sinusoidal = mult.',
               False,
               (strPthOt + 'plt_05_closure.svg'))

funcPltAcrDpth(aryDemoMix02,
               aryError,
               varNumDpth,
               4,
               varDpi,
               0.0,
               1.0,
               False,
               lstConLbl,
               strXlabel,
               strYlabel,
               'Linear = mult., sinusoidal = add.',
               False,
               (strPthOt + 'plt_06_closure.svg'))
# -----------------------------------------------------------------------------
