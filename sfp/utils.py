#!/usr/bin/python
"""various utils
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


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
    argument. for instance, if you only want to draw the mean point, pass ci_vals=[50, 50] (this is
    eqiuvalent to just calling plt.scatter)
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


def im_plot(im, **kwargs):
    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
        ax.imshow(im, cmap='gray', **kwargs)
    else:
        ax = plt.imshow(im, cmap='gray', **kwargs)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


def add_img_to_xaxis(fig, ax, img, rel_position, size=.1, **kwargs):
    """add image to x-axis

    after calling this, you probably want to make your x axis invisible:
    `ax.xaxis.set_visible(False)` or things will look confusing

    rel_position: float between 0 and 1, specifies where on the x axis you want to place the
    image. You'll need to play arond with this, I don't have a good way of doing this automatically
    (for instance, lining up with existing tick marks). This interacts with size. if you want the
    left edge to line up with the beginning of the x-axis, this should be 0, but if you want the
    right edge to line up with the end of the x-axis, this should be around 1-size

    size: float between 0 and 1, size of image, relative to overall plot size. it appears that
    around .1 or .2 is a good size.
    """
    xl, yl, xh, yh = np.array(ax.get_position()).ravel()
    w = xh - xl

    ax1 = fig.add_axes([xl + w*rel_position, yl-size, size, size])
    ax1.axison = False
    im_plot(img, ax=ax1, **kwargs)


def log_norm_pdf(x, a, mu, sigma):
    """the pdf of the log normal distribution, with a scale factor
    """
    # the normalizing term isn't necessary, but we keep it here for propriety's sake
    return a * (1/(x*sigma*np.sqrt(2*np.pi))) * np.exp(-(np.log(x)-mu)**2/(2*sigma**2))


def fit_log_norm(x, y, **kwargs):
    """fit log norm to data and plot the result

    to be used with seaborn.FacetGrid.map_dataframe
    """
    data = kwargs.pop('data')
    plot_data = data.groupby(x)[y].mean()

    popt, pcov = sp.optimize.curve_fit(log_norm_pdf, plot_data.index, plot_data.values)
    plt.plot(plot_data.index, log_norm_pdf(plot_data.index, *popt), **kwargs)

    # to plot ci band, use ax.fill_between(x, low, high, facecolor=color, alpha=.2)
