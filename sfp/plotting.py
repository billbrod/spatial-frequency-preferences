#!/usr/bin/python
"""high-level functions to make relevant plots
"""
import matplotlib as mpl
# we do this because sometimes we run this without an X-server, and this backend doesn't need one
mpl.use('svg')
import argparse
from . import utils
import warnings
import os
from . import tuning_curves
from . import stimuli as sfp_stimuli
from . import first_level_analysis
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from sklearn import linear_model

LOGPOLAR_SUPERCLASS_ORDER = ['radial', 'forward spiral', 'mixtures', 'angular', 'reverse spiral']
CONSTANT_SUPERCLASS_ORDER = ['vertical', 'forward diagonal', 'off-diagonal', 'horizontal',
                             'reverse diagonal']


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def myLogFormat(y, pos):
    """formatter that only shows the required number of decimal points

    this is for use with log-scaled axes, and so assumes that everything greater than 1 is an
    integer and so has no decimal points

    to use (or equivalently, with `axis.xaxis`):
    ```
    from matplotlib import ticker
    axis.yaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))
    ```

    modified from https://stackoverflow.com/a/33213196/4659293
    """
    # Find the number of decimal places required
    if y < 1:
        # because the string representation of a float always starts "0."
        decimalplaces = len(str(y)) - 2
    else:
        decimalplaces = 0
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)


def _jitter_data(data, jitter):
    """optionally jitter data some amount

    jitter can be None / False (in which case no jittering is done), a number (in which case we add
    uniform noise with a min of -jitter, max of jitter), or True (in which case we do the uniform
    thing with min/max of -.1/.1)

    based on seaborn.linearmodels._RegressionPlotter.scatter_data
    """
    if jitter is None or jitter is False:
        return data
    else:
        if jitter is True:
            jitter = .1
        return data + np.random.uniform(-jitter, jitter, len(data))


def im_plot(im, **kwargs):
    try:
        cmap = kwargs.pop('cmap')
    except KeyError:
        cmap = 'gray'
    try:
        ax = kwargs.pop('ax')
        ax.imshow(im, cmap=cmap, **kwargs)
    except KeyError:
        ax = plt.imshow(im, cmap=cmap, **kwargs)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


def plot_median(x, y, plot_func=plt.plot, **kwargs):
    """plot the median points, for use with seaborn's map_dataframe

    plot_func specifies what plotting function to call on the median points (e.g., plt.plot,
    plt.scatter)
    """
    data = kwargs.pop('data')
    plot_data = data.groupby(x)[y].median()
    plot_func(plot_data.index, plot_data.values, **kwargs)


def plot_ci(x, y, ci_vals=[16, 84], **kwargs):
    """fill between the specified CIs, for use with seaborn's map_dataframe
    """
    data = kwargs.pop('data')
    alpha = kwargs.pop('alpha', .2)
    plot_data = data.groupby(x)[y].apply(np.percentile, ci_vals)
    plot_values = np.vstack(plot_data.values)
    plt.fill_between(plot_data.index, plot_values[:, 0], plot_values[:, 1], alpha=alpha, **kwargs)


def scatter_ci_col(x, y, ci, **kwargs):
    """plot center points and specified CIs, for use with seaborn's map_dataframe

    based on seaborn.linearmodels.scatterplot. CIs are taken from a column in this function.
    """
    data = kwargs.pop('data')
    plot_data = data.groupby(x)[y].median()
    plot_cis = data.groupby(x)[ci].median()
    plt.scatter(plot_data.index, plot_data.values, **kwargs)
    for (x, ci), y in zip(plot_cis.iteritems(), plot_data.values):
        plt.plot([x, x], [y+ci, y-ci], **kwargs)


def scatter_ci_dist(x, y, ci_vals=[16, 84], x_jitter=None, **kwargs):
    """plot center points and specified CIs, for use with seaborn's map_dataframe

    based on seaborn.linearmodels.scatterplot. CIs are taken from a distribution in this
    function. Therefore, it's assumed that the values being passed to it are values from a
    bootstrap distribution.

    by default, this draws the 68% confidence interval. to change this, change the ci_vals
    argument. for instance, if you only want to draw the median point, pass ci_vals=[50, 50] (this
    is equivalent to just calling plt.scatter)
    """
    data = kwargs.pop('data')
    plot_data = data.groupby(x)[y].median()
    plot_cis = data.groupby(x)[y].apply(np.percentile, ci_vals)
    x_data = _jitter_data(plot_data.index, x_jitter)
    plt.scatter(x_data, plot_data.values, **kwargs)
    for x, (_, (ci_low, ci_high)) in zip(x_data, plot_cis.iteritems()):
        plt.plot([x, x], [ci_low, ci_high], **kwargs)


def plot_median_fit(x, y, model=linear_model.LinearRegression(), x_vals=None, **kwargs):
    """plot a model fit to the median points, for use with seaborn's map_dataframe

    we first find the median values of y when grouped by x and then fit model to them and plot
    *only* the model's predictions (thus you'll need to call another function if you want to see
    the actual data). we don't cross-validate or do anything fancy.

    model should be an (already-initialized) class that has a fit (which accepts X and y) and a
    predict method (which accepts X; for instance, anything from sklearn). It should expect X to be
    2d; we reshape our 1d data to be 2d (since this is what the sklearn models expect)
    """
    data = kwargs.pop('data')
    if 'exclude' in kwargs['label']:
        # we don't plot anything with exclude in the label
        return
    plot_data = data.groupby(x)[y].median()
    model.fit(plot_data.index.values.reshape(-1, 1), plot_data.values)
    if x_vals is None:
        x_vals = plot_data.index
    plt.plot(x_vals, model.predict(np.array(x_vals).reshape(-1, 1)), **kwargs)


def stimuli_properties(df, save_path=None):
    """plot some summaries of the stimuli properties

    this plots three pieces stimulus properties as a function of their position in frequency space
    (either w_x / w_y or w_r / w_a): superclass (radial, angular, etc / horizontal, vertical,
    etc), distance from the origin in frequency space, and angle in the frequency space. these
    various properties of the stimuli will be used to summarize results later on and so are
    important to have a handle on.

    df: pandas DataFrame containing stimulus information. should be either the stimulus description
    dataframe or the first level results dataframe.
    """
    if 'voxel' in df.columns:
        df = df[df.voxel == df.voxel.unique()[0]]
    else:
        df = df.dropna()
        df.class_idx = df.class_idx.astype(int)
        df = df.drop_duplicates('class_idx').set_index('class_idx')
        df = df.rename(columns={'index': 'stimulus_index'})
        df = first_level_analysis._add_freq_metainfo(df)
    if 'baseline' in df.stimulus_superclass.unique():
        df = df[df.stimulus_superclass != 'baseline']
    figsize = (19, 5)
    cmaps = [sns.color_palette(n_colors=5), sns.cubehelix_palette(as_cmap=True),
             sns.diverging_palette(10, 220, as_cmap=True)]
    try:
        df['w_a']
        freq_names = ['w_r', 'w_a']
        if 181 in df['w_a'].unique():
            # then this is the pilot data, which goes out further in frequency space
            if df['freq_space_angle'].max() > 2:
                # then this is the first pilot, which has a bit different angles
                ylim, xlim = [-75, 200], [-150, 250]
                figsize = (20, 6)
            else:
                ylim, xlim = [-176, 212], [-28, 311]
        else:
            ylim, xlim = [-125, 150], [-20, 220]
        cmaps[0] = dict((i, j) for i, j in zip(LOGPOLAR_SUPERCLASS_ORDER, cmaps[0]))
    except KeyError:
        freq_names = ['w_x', 'w_y']
        ylim, xlim = [-.098, .118], [-.0157, .173]
        cmaps[0] = dict((i, j) for i, j in zip(CONSTANT_SUPERCLASS_ORDER, cmaps[0]))
    norms = [None, None, MidpointNormalize(df.freq_space_angle.min(),
                                           df.freq_space_angle.max(), midpoint=0)]
    titles = ['Frequency superclass', 'Frequency distance', "Frequency angle"]
    color_prop = ['stimulus_superclass', 'freq_space_distance', 'freq_space_angle']
    with sns.axes_style('white'):
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for i, ax in enumerate(axes.flatten()):
            # zorder ensures that the lines are plotted before the points in the scatterplot
            ax.plot(xlim, [0, 0], 'k--', alpha=.5, zorder=1)
            ax.plot([0, 0], ylim, 'k--', alpha=.5, zorder=2)
            if i == 0:
                handles = []
                labels = []
                for lab, g in df.groupby(color_prop[i]):
                    pts = ax.scatter(g[freq_names[0]].values, g[freq_names[1]].values,
                                     c=cmaps[i][lab], edgecolors='k', zorder=3)
                    handles.append(pts)
                    labels.append(lab)
                ax.legend(handles, labels)
            elif 0 < i < 3:
                pts = ax.scatter(df[freq_names[0]].values, df[freq_names[1]].values,
                                 c=df[color_prop[i]].values, cmap=cmaps[i], edgecolors='k',
                                 norm=norms[i], zorder=3)
                fig.colorbar(pts, ax=ax, fraction=.046, pad=.1)
            else:
                ax.set_visible(False)
                continue
            ax.set_aspect('equal')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_title(titles[i], fontsize=15)
            ax.set_xlabel("$\omega_%s$" % freq_names[0][-1], fontsize=20)
            ax.set_ylabel("$\omega_%s$" % freq_names[1][-1], fontsize=20)
    sns.despine()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def local_spatial_frequency(df, save_path=None, **kwargs):
    """plot the local spatial frequency for all stimuli

    df: first_level_analysis results dataframe.

    all kwargs get passed to plt.plot
    """
    if 'rounded_freq_space_distance' in df.columns:
        hue_label = 'rounded_freq_space_distance'
        col_order = LOGPOLAR_SUPERCLASS_ORDER
    else:
        hue_label = 'freq_space_distance'
        col_order = CONSTANT_SUPERCLASS_ORDER

    def mini_plot(x, y, **kwargs):
        # this function converts the stringified floats of eccentricity to floats and correctly
        # orders them for plotting.
        x = [np.mean([float(j) for j in i.split('-')]) for i in x.values]
        plot_vals = sorted(zip(x, y.values), key=lambda pair: pair[0])
        plt.plot([i for i, _ in plot_vals], [j for _, j in plot_vals], **kwargs)

    with sns.axes_style('white'):
        g = sns.FacetGrid(df, hue=hue_label, col='stimulus_superclass', palette='Reds', col_wrap=3,
                          col_order=col_order)
        g.map(mini_plot, 'eccen', 'local_sf_magnitude', **kwargs)
        g.add_legend()
        plt.subplots_adjust(top=.9)
        g.fig.suptitle("Local spatial frequencies across eccentricities for all stimuli",
                       fontsize=15)
    if save_path is not None:
        g.fig.savefig(save_path, bbox_inches='tight')
    return g


def plot_data(df, x_col='freq_space_distance', median_only=False, ci_vals=[16, 84],
              save_path=None, **kwargs):
    """plot the raw amplitude estimates, either with or without confidence intervals

    if df is the summary dataframe, we'll use the amplitude_estimate_std_error column as the
    confidence intervals (in this case, ci_vals is ignored). otherwise, we'll estimate them
    directly from the bootstrapped data using np.percentile; in this case, ci_vals determines what
    percentile to plot (by default, the 68% confidence interval)

    x_col determines what to have on the x-axis 'freq_space_distance' or
    'rounded_freq_space_distance' will probably work best. you can try 'freq_space_angle' or 'Local
    spatial frequency (cpd)', but there's no guarantee those will plot well.

    if median_only is True, will not plot confidence intervals.

    kwargs will get passed to plt.scatter and plt.plot, via scatter_ci_dist / scatter_ci_col
    """
    if 'rounded_freq_space_distance' in df.columns:
        col_order = [i for i in LOGPOLAR_SUPERCLASS_ORDER if i in df.stimulus_superclass.unique()]
    else:
        col_order = [i for i in CONSTANT_SUPERCLASS_ORDER if i in df.stimulus_superclass.unique()]

    ylim = kwargs.pop('ylim', None)
    xlim = kwargs.pop('xlim', None)
    aspect = kwargs.pop('aspect', 1)
    g = sns.FacetGrid(df, hue='eccen', palette='Reds', size=5, row='varea', col_order=col_order,
                      hue_order=sorted(df.eccen.unique()), col='stimulus_superclass', ylim=ylim,
                      xlim=xlim, aspect=aspect)
    if 'amplitude_estimate_std_error' in df.columns:
        g.map_dataframe(plot_median, x_col, 'amplitude_estimate_median')
        if not median_only:
            g.map_dataframe(scatter_ci_col, x_col, 'amplitude_estimate_median',
                            'amplitude_estimate_std_error', **kwargs)
    else:
        g.map_dataframe(plot_median, x_col, 'amplitude_estimate')
        if not median_only:
            g.map_dataframe(scatter_ci_dist, x_col, 'amplitude_estimate', ci_vals=ci_vals,
                            **kwargs)
    g.map_dataframe(plot_median, x_col, 'baseline', linestyle='--')
    for ax in g.axes.flatten():
        ax.set_xscale('log', basex=2)
    g.add_legend()
    g.fig.suptitle("Amplitude estimates as a function of frequency")
    plt.subplots_adjust(top=.9)
    if save_path is not None:
        g.fig.savefig(save_path, bbox_inches='tight')
    return g


def _plot_grating_approx_and_save(grating, grating_type, save_path, **kwargs):
    figsize = kwargs.pop('figsize', (5, 5))
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    im_plot(grating, ax=axes, **kwargs)
    if grating_type == 'grating':
        axes.set_title("Windowed view of actual grating")
    elif grating_type == 'approx':
        axes.set_title("Windowed view of linear approximation")
    if save_path is not None:
        try:
            fig.savefig(save_path % grating_type, bbox_inches='tight')
        except TypeError:
            save_path = os.path.splitext(save_path)[0] + "_" + grating_type + os.path.splitext(save_path)[1]
            fig.savefig(save_path, bbox_inches='tight')


def plot_grating_approximation(grating, dx, dy, num_windows=10, phase=0, w_r=None, w_a=None,
                               origin=None, stim_type='logpolar', save_path=None, **kwargs):
    """plot the "windowed approximation" of a grating

    note that this will not create the grating or its gradients (dx/dy), it only plots them. For
    this to work, dx and dy must be in cycles / pixel

    this will work for either regular 2d gratings or the log polar gratings we use as stimuli,
    though it will look slightly different depending on which you do. In the regular case, the
    space between windows will be mid-gray, while for the log polar gratings, it will be
    black. This allows for a creation of a neat illusion for some regular gratings (use
    grating=sfp.utils.create_sin_cpp(1080, .005, .005) to see an example)!

    if `grating` is one of our log polar gratings, then w_r and w_a also need to be set. if it's a
    regular 2d grating, then they should both be None.

    num_windows: int, the number of windows in each direction that we'll use. as this gets larger,
    the approximation will look better and better (and this will take a longer time to run)

    save_path: str, optional. If set, will save plots. in order to make comparison easier, will save
    two separate plots (one of the windowed grating, one of the linear approximation). if save_path
    does not include %s, will append _grating and _approx (respectively) to filename

    kwargs will be past to im_plot.
    """
    size = grating.shape[0]
    # we need to window the gradients dx and dy so they only have values where the grating does
    # (since they're derived analytically, they'll have values everywhere)
    mid_val = {'pilot': 127}.get(stim_type, 128)
    dx = utils.mask_array_like_grating(grating, dx, mid_val)
    dy = utils.mask_array_like_grating(grating, dy, mid_val)
    mask_spacing = np.round(size / num_windows)
    # for this to work, the gratings must be non-overlapping
    mask_size = np.round(mask_spacing / 2) - 1
    masked_grating = np.zeros((size, size))
    masked_approx = np.zeros((size, size))
    masks = np.zeros((size, size))
    for i in range(mask_size, size, mask_spacing):
        for j in range(mask_size, size, mask_spacing):
            loc_x, loc_y = i, j
            mask = utils.create_circle_mask(loc_x, loc_y, mask_size, size)
            masks += mask
            masked_grating += mask * grating
            masked_approx += mask * utils.local_grad_sin(dx, dy, loc_x, loc_y, w_r, w_a, phase,
                                                         origin, stim_type)
    # in order to make the space between the masks black, that area should have the minimum
    # value, -1. but for the above to all work, that area needs to be 0, so this corrects that.
    masked_approx[~masks.astype(bool)] -= 1
    _plot_grating_approx_and_save(masked_grating, 'grating', save_path, **kwargs)
    _plot_grating_approx_and_save(masked_approx, 'approx', save_path, **kwargs)
    return masked_grating, masked_approx


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


def stimuli_linear_approximation(stim, stim_df, stim_type, num_windows=11, stim_idx=None, phi=None,
                                 freq_space_distance=None, freq_space_angle=None,
                                 stimulus_superclass=None, save_path=None, **kwargs):
    """plot the linear approximation of specific stimulus
    """
    if stim_idx is None:
        stim_idx = utils.find_stim_idx(stim_df, stimulus_superclass=stimulus_superclass, phi=phi,
                                       freq_space_distance=freq_space_distance,
                                       freq_space_angle=freq_space_angle)
    props = stim_df.loc[stim_idx]
    freqs = {}
    for f in ['w_r', 'w_a', 'w_x', 'w_y']:
        try:
            freqs[f] = props[f]
        except KeyError:
            freqs[f] = None
    stim = stim[stim_idx]
    dx, dy, _, _ = sfp_stimuli.create_sf_maps_cpp(props.res, stim_type=stim_type, **freqs)
    return plot_grating_approximation(stim, dx, dy, num_windows, props.phi, w_r=freqs['w_r'],
                                      w_a=freqs['w_a'], stim_type=stim_type,
                                      save_path=save_path, **kwargs)


def stimuli(stim, stim_df, save_path=None, **kwargs):
    """plot a bunch of stimuli with specific properties, pulled out of stim_df

    possible keys for kwargs: {'w_r'/'w_x', 'w_a'/'w_y', 'phi', 'res', 'alpha', 'stimulus_index',
    'class_idx', 'stimulus_superclass', 'freq_space_angle', 'freq_space_distance'}. The values
    should be either a list or a single value. if a single value, will assume that all stimuli
    share that property. all lists should be the same length. if a property isn't set, then we
    assume it's not important and so will grab the lowest stimuli with the lowest index that
    matches all specified properties.
    """
    stim_props = {}
    stim_num = None
    figsize = kwargs.pop('figsize', None)
    for k, v in kwargs.iteritems():
        if hasattr(v, "__iter__") and not isinstance(v, str):
            if stim_num is None:
                stim_num = len(v)
            else:
                if stim_num != len(v) and len(v) != 1:
                    raise Exception("All stimulus properties must have the same length!")
            stim_props[k] = v
        else:
            stim_props[k] = [v]
    if stim_num is None:
        stim_num = 1
    for k, v in stim_props.iteritems():
        if len(v) == 1:
            stim_props[k] = stim_num * v
    stim_idx = []
    for i in range(stim_num):
        stim_idx.append(utils.find_stim_idx(stim_df,
                                            **dict((k, v[i]) for k, v in stim_props.iteritems())))
    if figsize is None:
        figsize = (5 * min(stim_num, 4), 5 * np.ceil(stim_num / 4.))
    fig = plt.figure(figsize=figsize)
    # ADD DESCRIPTIVE TITLES
    for i, idx in enumerate(stim_idx):
        ax = fig.add_subplot(np.ceil(stim_num / 4.).astype(int), min(stim_num, 4), i+1)
        im_plot(stim[idx, :, :], ax=ax)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)


def plot_tuning_curve(ci_vals=[16, 84], norm=False, **kwargs):
    data = kwargs.pop('data')
    color = kwargs.pop('color')
    if 'bootstrap_num' in data.columns:
        xs, ys = [], []
        for n, g in data.groupby('bootstrap_num'):
            x, y = tuning_curves.get_tuning_curve_xy_from_df(g, norm=norm)
            xs.append(x)
            ys.append(y)
        xs = np.array(xs)
        if (xs != xs[0]).any():
            raise Exception("Somehow we got different xs for the tuning curves of some "
                            "bootstraps!")
        ys = np.array(ys)
        y_median = np.median(ys, 0)
        y_cis = np.percentile(ys, ci_vals, 0)
        plt.fill_between(xs[0], y_cis[0], y_cis[1], alpha=.2, facecolor=color)
        plt.semilogx(xs[0], y_median, basex=2, color=color, **kwargs)
    else:
        x, y = tuning_curves.get_tuning_curve_xy_from_df(data, norm=norm)
        plt.semilogx(x, y, basex=2, color=color, **kwargs)


def _restrict_df(df, **kwargs):
    for k, v in kwargs.iteritems():
        try:
            df = df[df[k].isin(v)]
        except TypeError:
            df = df[df[k] == v]
    return df


def check_tuning_curves(tuning_df, save_path_template, **kwargs):
    """create all the tuning curve plots

    this takes the dataframe containing the tuning curves and creates plots of all of them, so they
    can be visibly checked. note that this will take a while and create *many* plots, especially
    when run on the full dataframes. It is thus *not* meant to be run from a notebook and it will
    close the plots as it creates and saves them.

    kwargs can contain columns in the tuning_df and values to limit them to.
    """
    tuning_df = _restrict_df(tuning_df, **kwargs)
    gb_cols = ['varea']
    title_template = 'varea={}'
    if 'bootstrap_num' in tuning_df.columns:
        gb_cols += ['bootstrap_num']
        title_template += ', bootstrap={:02d}'
    mode_bounds = (tuning_df.mode_bound_lower.unique()[0], tuning_df.mode_bound_upper.unique()[0])
    for n, g in tuning_df.groupby(gb_cols):
        f = sns.FacetGrid(g, row='eccen', col='stimulus_superclass', hue='frequency_type',
                          xlim=mode_bounds, aspect=.7, size=5)
        f.map(plt.scatter, 'frequency_value', 'amplitude_estimate')
        f.map_dataframe(plot_tuning_curve)
        f.map_dataframe(plot_median, 'frequency_value', 'baseline', linestyle='--')
        f.add_legend()
        f.set_titles("eccen={row_name} | {col_name}")
        if len(gb_cols) == 1:
            # then there's only one value in n (and thus, in gb_cols)
            suptitle = title_template.format(n)
        else:
            suptitle = title_template.format(*n)
        f.fig.suptitle(suptitle)
        plt.subplots_adjust(top=.95)
        f.savefig(save_path_template % (suptitle.replace(', ', '_')))
        plt.close(f.fig)


def check_hypotheses(tuning_df, save_path_template=None, norm=False, ci_vals=[16, 84],
                     plot_data=True, **kwargs):
    tuning_df = _restrict_df(tuning_df, **kwargs)
    gb_cols = ['varea']
    title_template = 'varea={}'
    col_order = [i for i in LOGPOLAR_SUPERCLASS_ORDER+CONSTANT_SUPERCLASS_ORDER
                 if i in tuning_df.stimulus_superclass.unique()]
    for n, g in tuning_df.groupby(gb_cols):
        f = sns.FacetGrid(g, hue='eccen', palette='Reds', size=5, row='frequency_type',
                          col='stimulus_superclass', col_order=col_order)
        if plot_data:
            f.map_dataframe(plot_median, 'frequency_value', 'amplitude_estimate',
                            plot_func=plt.scatter)
        f.map_dataframe(plot_tuning_curve, norm=norm, ci_vals=ci_vals)
        for ax in f.axes.flatten():
            ax.set_xscale('log', basex=2)
            ax.set_xlim((2**-5, 2**10))
            if norm:
                ax.set_ylim((0, 1.2))
        f.add_legend()
        suptitle = title_template.format(n)
        f.fig.suptitle("Median amplitude estimates with tuning curves, %s" % suptitle)
        sns.plt.subplots_adjust(top=.93)
        f.set_titles("{row_name} | {col_name}")
        if save_path_template is not None:
            f.fig.savefig(save_path_template % suptitle, bbox_inches='tight')


def check_hypotheses_with_data(tuning_df, save_path_template=None, ci_vals=[16, 84], **kwargs):
    check_hypotheses(tuning_df, save_path_template, False, ci_vals, True, **kwargs)


def check_hypotheses_normalized(tuning_df, save_path_template=None, ci_vals=[16, 84], **kwargs):
    check_hypotheses(tuning_df, save_path_template, True, ci_vals, False, **kwargs)


def tuning_params(tuning_df, save_path=None, **kwargs):
    tuning_df = _restrict_df(tuning_df, **kwargs)
    tuning_df = tuning_df[['frequency_type', 'tuning_curve_amplitude', 'tuning_curve_sigma',
                           'tuning_curve_mu', 'tuning_curve_peak', 'tuning_curve_bandwidth']]
    tuning_df['tuning_curve_peak'] = np.log2(tuning_df.tuning_curve_peak)
    g = sns.PairGrid(tuning_df, hue='frequency_type', aspect=1)
    g.map_offdiag(plt.scatter)
    g.map_diag(sns.distplot)
    g.add_legend()
    if save_path is not None:
        g.fig.savefig(save_path)


def period_summary_plot(df, pRF_size_slope=.25394, pRF_size_offset=.100698,
                        model=linear_model.LinearRegression(), n_windows=4, size=1080,
                        max_visual_angle=24, plot_view='full', center_spot_rad=2,
                        stimulus_superclass=None, save_path=None):
    """make plot that shows the optimal number of periods per receptive field

    df: dataframe containing the summarized preferred periods. should contain three columns:
    stimulus_superclass, eccen, and preferred_period. will fit model separately to each
    stimulus_superclass (by default, linear regression, fitting the intercept) and then, at each
    eccentricity, show the appropriate period (from the fitted line). because we allow the
    intercept to be non-zero

    this works by using the coefficients of the linear fits (with zero intercept) relating
    eccentricity to the period of the optimal grating stimulus (from measurements for both radial
    and angular stimuli) and to V1 receptive field size. We show the views of the radial stimuli
    along the vertical axis and the views of the angular along the horizontal axis.

    pRF_size_slope should be for the diameter of what you want to display. for example, the default
    is 4 * stddev, or the diameter of two standard deviations.

    n_windows: int, the number of windows on each side of the origin to show

    plot_view: {'full', 'quarter', 'aligned'}. whether to plot a full, quarter, or aligned view. if
    aligned, can only plot one stimulus superclass and will do always do so along the horizontal
    meridian (for combining by hand afterwards).

    stimulus_superclass: which stimulus superclasses to plot. if None, will plot all that are in
    the dataframe. note that we don't re-adjust spacing to make it look good for fewer than 4
    superclasses, so that's on you.
    """
    def get_logpolar_freq(slope, intercept, ecc):
        coeff = (slope*ecc + intercept) / ecc
        return np.round((2 * np.pi) / coeff)

    def fit_model(data, model=linear_model.LinearRegression()):
        x = data.eccen.values
        y = data.preferred_period.values
        model.fit(x.reshape(-1, 1), y)
        return pd.Series({'coeff': model.coef_[0], 'intercept': model.intercept_})

    def logpolar_solve_for_global_phase(x, y, w_r, w_a, local_phase=0):
        origin = ((size+1) / 2., (size+1) / 2.)
        x_orig, y_orig = np.meshgrid(np.array(range(1, size+1))-origin[0],
                                     np.array(range(1, size+1))-origin[1])
        local_x = x_orig[y, x]
        local_y = y_orig[y, x]
        return np.mod(local_phase - (((w_r*np.log(2))/2.)*np.log2(local_x**2+local_y**2) +
                                     w_a*np.arctan2(local_y, local_x)), 2*np.pi)

    if stimulus_superclass is None:
        stimulus_superclass = df.stimulus_superclass.unique()
    if plot_view == 'aligned' and len(stimulus_superclass) != 1:
        raise Exception("Can only plot aligned view if plotting 1 stimulus_superclass")
    df = df.groupby('stimulus_superclass').apply(fit_model)
    windowed_plot = np.zeros((size, size))
    R = sfp_stimuli.mkR(size) * (float(max_visual_angle) / size)
    ecc_to_pix = pRF_size_slope * (float(size) / max_visual_angle) + pRF_size_offset
    masks = np.zeros((size, size))
    meridian = int(size / 2)
    # we put this into masks so it doesn't get adjusted at the end and we set it to -1 so it
    # appears black.
    masks[meridian-center_spot_rad:meridian+center_spot_rad,
          meridian-center_spot_rad:meridian+center_spot_rad] = 1
    windowed_plot[masks.astype(bool)] = -1
    view_range = range(meridian, size, int(meridian / (n_windows+1)))[1:]
    for loc in view_range:
        # we do x and y separately because there's a chance a rounding issue will mean they differ
        # slightly
        if plot_view == 'full':
            diag_value_1 = (loc - meridian) * np.sin(np.pi / 4)
            diag_value_2 = (loc - meridian) * -np.sin(np.pi / 4)
            window_locs = [(loc, meridian, 'radial'), (meridian, loc, 'angular'),
                           (size-loc, meridian, 'radial'), (meridian, size-loc, 'angular'),
                           (meridian+diag_value_1, meridian+diag_value_1, 'forward spiral'),
                           (meridian+diag_value_1, meridian+diag_value_2, 'reverse spiral'),
                           (meridian+diag_value_2, meridian+diag_value_2, 'forward spiral'),
                           (meridian+diag_value_2, meridian+diag_value_1, 'reverse spiral')]
        elif plot_view == 'quarter':
            diag_value_1 = (loc - meridian) * np.sin(np.pi / 6)
            diag_value_2 = (loc - meridian) * np.sin(np.pi / 3)
            window_locs = [(size-loc, meridian, 'radial'), (meridian, loc, 'angular'),
                           (meridian-diag_value_1, meridian+diag_value_2, 'forward spiral'),
                           (meridian-diag_value_2, meridian+diag_value_1, 'reverse spiral')]
        elif plot_view == 'aligned':
            window_locs = [(meridian, loc, stimulus_superclass[0])]
        for loc_y, loc_x, stim_class in window_locs:
            if stim_class not in stimulus_superclass:
                continue
            loc_x, loc_y = int(loc_x), int(loc_y)
            ecc = R[loc_y, loc_x]
            # ecc * ecc_to_pix gives you the diameter, but we want the radius
            mask = utils.create_circle_mask(loc_x, loc_y, (ecc * ecc_to_pix) / 2, size)
            masks += mask
            opt_w = get_logpolar_freq(df.loc[stim_class].coeff, df.loc[stim_class].intercept, ecc)
            if stim_class == 'angular':
                phase = logpolar_solve_for_global_phase(loc_x, loc_y, 0, opt_w)
                windowed_plot += mask * sfp_stimuli.log_polar_grating(size, w_a=opt_w, phi=phase)
            elif stim_class == 'radial':
                phase = logpolar_solve_for_global_phase(loc_x, loc_y, opt_w, 0)
                windowed_plot += mask * sfp_stimuli.log_polar_grating(size, w_r=opt_w, phi=phase)
            elif stim_class == 'forward spiral':
                opt_w = np.round(opt_w / np.sqrt(2))
                phase = logpolar_solve_for_global_phase(loc_x, loc_y, opt_w, opt_w)
                windowed_plot += mask * sfp_stimuli.log_polar_grating(size, w_r=opt_w, w_a=opt_w,
                                                                      phi=phase)
            elif stim_class == 'reverse spiral':
                opt_w = np.round(opt_w / np.sqrt(2))
                phase = logpolar_solve_for_global_phase(loc_x, loc_y, opt_w, -opt_w)
                windowed_plot += mask * sfp_stimuli.log_polar_grating(size, w_r=opt_w, w_a=-opt_w,
                                                                      phi=phase)
    windowed_plot[~masks.astype(bool)] += 1
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im_plot(windowed_plot, ax=ax)
    if save_path is not None:
        fig.savefig(save_path)
    return windowed_plot


def _parse_save_path_for_kwargs(save_path):
    kwargs = dict(i.split('=') for i in save_path.split('_'))
    # we know all are ints
    return dict(({'bootstrap': 'bootstrap_num'}.get(k, k), int(v)) for k, v in kwargs.iteritems())


if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=("Creates the descriptive plots for one first level results dataframe")
        )
    parser.add_argument("dataframe_path",
                        help=("path to first level results or tuning curves dataframe. we'll "
                              "attempt to find the other dataframe as well."))
    parser.add_argument("stim_dir", help="path to directory containing stimuli")
    parser.add_argument("--plot_to_make", default=None, nargs='*',
                        help=("Which plots to create. If none, will create all. Possible options: "
                              "localsf (plotting.local_spatial_frequency), stim_prop (plotting."
                              "stimuli_properties), data (plotting.plot_data), "
                              "tuning_curves_check_varea={v}[_bootstrap={b:02d}] (plotting."
                              "check_tuning_curves; requires tuning curve dataframe), "
                              "hypotheses_data_varea={v} (plotting.check_hypotheses_with_data; "
                              "requires tuning curve dataframe), or tuning_params "
                              "(plotting.tuning_params; requires tuning curve dataframe)"))
    args = vars(parser.parse_args())
    d = utils.create_data_dict(args['dataframe_path'], args['stim_dir'])
    first_level_save_stem = d['df_filename'].replace('.csv', '')
    if 'tuning_df' in d.keys():
        tuning_save_stem = d['tuning_df_filename'].replace('.csv', '')
        tuning_df_present = True
    else:
        tuning_df_present = False
    if args['plot_to_make'] is None:
        local_spatial_frequency(d['df'], first_level_save_stem+"_localsf.svg")
        stimuli_properties(d['df'], first_level_save_stem+"_stim_prop.svg")
        plot_data(d['df'], save_path=first_level_save_stem+'_data.svg')
        if tuning_df_present:
            check_tuning_curves(d['tuning_df'], tuning_save_stem+"_tuning_curves_check_%s.svg")
            check_hypotheses_with_data(d['tuning_df'], tuning_save_stem+"_hypotheses_data_%s.svg")
            tuning_params(d['tuning_df'], tuning_save_stem+"_tuning_params.svg")
        else:
            warnings.warn("Unable to create tuning curves, hypotheses check, or tuning param plots"
                          " because tuning curve df hasn't been created!")
    else:
        for p in args['plot_to_make']:
            if 'localsf' == p:
                local_spatial_frequency(d['df'], first_level_save_stem+"_localsf.svg")
            elif 'stim_prop' == p:
                stimuli_properties(d['df'], first_level_save_stem+"_stim_prop.svg")
            elif 'tuning_curves_check' in p:
                if tuning_df_present:
                    p_kwargs = _parse_save_path_for_kwargs(p.replace('tuning_curves_check_', ''))
                    check_tuning_curves(d['tuning_df'], tuning_save_stem+"_tuning_curves_check_%s.svg",
                                        **p_kwargs)
                else:
                    raise Exception("Unable to create tuning curves plot because tuning curve df "
                                    "hasn't been created!")
            elif 'data' == p:
                plot_data(d['df'], save_path=first_level_save_stem+'_data.svg')
            elif 'hypotheses_data' in p:
                if tuning_df_present:
                    p_kwargs = _parse_save_path_for_kwargs(p.replace('hypotheses_data_', ''))
                    check_hypotheses_with_data(d['tuning_df'], tuning_save_stem+"_hypotheses_data_%s.svg",
                                               **p_kwargs)
                else:
                    raise Exception("Unable to create hypotheses check with data plot because "
                                    "tuning curve df hasn't been created!")
            elif 'tuning_params' == p:
                if tuning_df_present:
                    tuning_params(d['tuning_df'], tuning_save_stem+"_tuning_params.svg")
                else:
                    raise Exception("Unable to create tuning params plot because "
                                    "tuning curve df hasn't been created!")
            else:
                raise Exception("Don't know how to make plot %s!" % p)
