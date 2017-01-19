# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

@author: Ingo Marquardt, 06.11.2016
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

import numpy as np  # noqa
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def funcPltAcrDpth(aryData,     # Data to be plotted: aryData[Condition, Depth]
                   aryError,    # Error shading: aryError[Condition, Depth]
                   varNumDpth,  # Number of depth levels (on the x-axis)
                   varNumCon,   # Number of conditions (separate lines)
                   varDpi,      # Resolution of the output figure
                   varYmin,     # Minimum of Y axis
                   varYmax,     # Maximum of Y axis
                   lgcCnvPrct,  # Boolean: whether to convert y axis to %
                   lstConLbl,   # Labels for conditions (separate lines)
                   strXlabel,   # Label on x axis
                   strYlabel,   # Label on y axis
                   strTitle,    # Figure title
                   lgcLgnd,     # Boolean: whether to plot a legend
                   strPath):    # Output path for the figure
    """
    Plot values across depth level, separately for conditions.

    The purpose of this function is to plot data & error bars across cortical
    depth levels (x-axis), separately for conditions (separate lines).
    """
    # Create figure:
    fgr01 = plt.figure(figsize=(1200.0/varDpi, 800.0/varDpi),
                       dpi=varDpi)

    # Create axis:
    axs01 = fgr01.add_subplot(111)

    # Vector for x-data:
    vecX = range(0, varNumDpth)

    # Prepare colour map:
    objClrNorm = colors.Normalize(vmin=0, vmax=(varNumCon - 1))
    objCmap = plt.cm.winter

    # Loop through conditions:
    for idxCon in range(0, varNumCon):

        # Adjust the colour of current line:
        vecClrTmp = objCmap(objClrNorm(idxCon))

        # Plot depth profile for current input file:
        plt01 = axs01.plot(vecX,  #noqa
                           aryData[idxCon, :],
                           color=vecClrTmp,
                           alpha=0.9,
                           label=(lstConLbl[idxCon]),
                           linewidth=8.0,
                           antialiased=True)

        # Plot error shading:
        plot02 = axs01.fill_between(vecX,  #noqa
                                    np.subtract(aryData[idxCon, :],
                                                aryError[idxCon, :]),
                                    np.add(aryData[idxCon, :],
                                           aryError[idxCon, :]),
                                    alpha=0.4,
                                    edgecolor=vecClrTmp,
                                    facecolor=vecClrTmp,
                                    linewidth=0,
                                    # linestyle='dashdot',
                                    antialiased=True)

    # Reduce framing box:
    axs01.spines['top'].set_visible(False)
    axs01.spines['right'].set_visible(False)
    axs01.spines['bottom'].set_visible(True)
    axs01.spines['left'].set_visible(True)

    # Set x-axis range:
    axs01.set_xlim([-1, varNumDpth])
    # Set y-axis range:
    axs01.set_ylim([varYmin, varYmax])

    # Which x values to label with ticks (WM & CSF boundary):
    # axs01.set_xticks([-0.5, (varNumDpth - 0.5)])
    axs01.set_xticks([4.0, (varNumDpth - 4.0)])

    # Set tick labels for x ticks:
    axs01.set_xticklabels(['WM', 'CSF'])

    # Which y values to label with ticks:
    vecYlbl = np.linspace(varYmin, varYmax, num=5, endpoint=True)
    # vecYlbl = np.arange(varYmin, varYmax, 0.02)
    # Round:
    # vecYlbl = np.around(vecYlbl, decimals=2)
    # Set ticks:
    axs01.set_yticks(vecYlbl)
    # Convert labels to percent?
    if lgcCnvPrct:
        # Multiply by 100 to convert to percent:
        vecYlbl = np.multiply(vecYlbl, 100.0)
        # Convert labels from float to a list of strings, with well-defined
        # number of decimals (including trailing zeros):
        lstYlbl = [None] * vecYlbl.shape[0]
        for idxLbl in range(vecYlbl.shape[0]):
            lstYlbl[idxLbl] = '{:0.1f}'.format(vecYlbl[idxLbl])
    else:
        # Convert labels from float to a list of strings, with well-defined
        # number of decimals (including trailing zeros):
        lstYlbl = [None] * vecYlbl.shape[0]
        for idxLbl in range(vecYlbl.shape[0]):
            lstYlbl[idxLbl] = '{:0.2f}'.format(vecYlbl[idxLbl])
    # Set tick labels for y ticks:
    axs01.set_yticklabels(lstYlbl)

    # Set x & y tick font size:
    axs01.tick_params(labelsize=36,
                      top='off',
                      right='off')

    # Adjust labels:
    axs01.set_xlabel(strXlabel,
                     fontsize=36)
    axs01.set_ylabel(strYlabel,
                     fontsize=36)

    # Adjust title:
    axs01.set_title(strTitle, fontsize=36, fontweight="bold")

    # Legend for axis 1:
    if lgcLgnd:
        axs01.legend(loc=0,
                     frameon=False,
                     prop={'size': 36})

    # Save figure:
    fgr01.savefig(strPath,
                  facecolor='w',
                  edgecolor='w',
                  orientation='landscape',
                  transparent=False,
                  frameon=None)

    # Close figure:
    plt.close(fgr01)
