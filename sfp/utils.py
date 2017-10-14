#!/usr/bin/python
"""various utils
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


class MidpointNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_mean(x, y, **kwargs):
    """plot line through center points, for use with seaborn's map_dataframe
    """
    data = kwargs.pop('data')
    plot_data = data.groupby(x)[y].mean()
    plt.plot(plot_data.index, plot_data.values, **kwargs)


def scatter_ci_col(x, y, ci, **kwargs):
    """plot center points and specified CIs, for use with seaborn's map_dataframe

    based on seaborn.linearmodels.scatterplot. CIs are taken from a column in this function.
    """
    data = kwargs.pop('data')
    plot_data = data.groupby(x)[y].mean()
    plot_cis = data.groupby(x)[ci].mean()
    plt.scatter(plot_data.index, plot_data.values, **kwargs)
    for (x, ci), y in zip(plot_cis.iteritems(), plot_data.values):
        plt.plot([x, x], [y+ci, y-ci], **kwargs)


def scatter_ci_dist(x, y, ci_vals=[2.5, 97.5], **kwargs):
    """plot center points and specified CIs, for use with seaborn's map_dataframe

    based on seaborn.linearmodels.scatterplot. CIs are taken from a distribution in this
    function. Therefore, it's assumed that the values being passed to it are values from a
    bootstrap distribution.

    by default, this draws the 95% confidence interval. to change this, change the ci_vals
    argument.
    """
    data = kwargs.pop('data')
    plot_data = data.groupby(x)[y].mean()
    plot_cis = data.groupby(x)[y].apply(np.percentile, ci_vals)
    plt.scatter(plot_data.index, plot_data.values, **kwargs)
    for x, (ci_low, ci_high) in plot_cis.iteritems():
        plt.plot([x, x], [ci_low, ci_high], **kwargs)


def scatter_heat(x, y, c, **kwargs):
    plt.scatter(x, y, c=c, cmap='RdBu_r', s=50, norm=MidpointNormalize(midpoint=0),
                vmin=kwargs['vmin'], vmax=kwargs['vmax'])
