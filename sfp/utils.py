#!/usr/bin/python
"""various utils
"""

import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pyPyrTools as ppt
from bids.grabbids import BIDSLayout
import pandas as pd
import first_level_analysis
import tuning_curves
import warnings


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


def create_sin_cpp(size, w_x, w_y, phase=0, origin=None):
    """create a full 2d sine wave, with frequency in cycles / pixel
    """
    if origin is None:
        origin = [(size+1) / 2., (size+1) / 2.]
    x = np.array(range(1, size+1))
    x, y = np.meshgrid(x - origin[0], x - origin[1])
    return np.cos(2*np.pi*x*w_x + 2*np.pi*y*w_y + phase)


def create_sin_cpd(size, w_x_cpd, w_y_cpd, phase=0, stim_rad_deg=12):
    """create a full 2d sine wave, with frequency in cycles / degree

    this converts the desired cycles / degree into the frequency shown in an image by using the
    stim_rad_deg, the radius of the image in degrees of visual angle.
    """
    w_x_pix = w_x_cpd / (size / (2*float(stim_rad_deg)))
    w_y_pix = w_y_cpd / (size / (2*float(stim_rad_deg)))
    return create_sin_cpp(size, w_x_pix, w_y_pix, phase)


def create_circle_mask(x, y, rad, size):
    """create a circular mask

    this returns a circular mask centered at pixel (x, y) with radius rad in a size by size
    image. This can then be multiplied by an image of the same size to mask out everything else.
    """
    x_grid = np.array(range(size))
    x_grid, y_grid = np.meshgrid(x_grid, x_grid)
    mask = np.zeros((size, size))
    mask[(x_grid - x)**2 + (y_grid - y)**2 <= rad**2] = 1
    return mask


def mask_array_like_grating(masked, array_to_mask, mid_val=128, val_to_set=0):
    """mask array_to_mask the way that masked has been masked

    this takes two arrays of the same size, grating and array_to_mask. masked should already be
    masked into an annulus, while array_to_mask should be unmasked. This then finds the inner and
    outer radii of that annulus and applies the same mask to array_to_mask. the value in the masked
    part of masked should be mid_val (by default, 128, as you'd get when the grating runs from 0 to
    255; mid_val=0, with the grating going from -1 to 1 is also likely) and the value that you want
    to set array_to_mask to is val_to_set (same reasonable values as mid_val)
    """
    R = ppt.mkR(masked.shape)
    x, y = np.where(masked != mid_val)
    Rmin = R[x, y].min()
    try:
        array_to_mask[R < Rmin] = val_to_set
    except IndexError:
        # then there is no R<Rmin
        pass
    Rmax = R[x, y].max()
    try:
        array_to_mask[R > Rmax] = val_to_set
    except IndexError:
        # then there is no R>Rmax
        pass
    return array_to_mask


def flat_hyperbola(x, a):
    """hyperbola which is flat until 1 degree
    """
    b = 1.
    period = x*a
    period[x < b] = a*b
    return 1./period


def fit_log_norm(x, y, **kwargs):
    """fit log norm to data and plot the result

    to be used with seaborn.FacetGrid.map_dataframe

    x: string, column in data which contains the x values for this plot.

    y: string, column in data which contains the y values for this plot.

    kwargs must contain `data`, the DataFrame with data to plot.
    """
    data = kwargs.pop('data')
    plot_data = data.groupby(x)[y].mean()

    try:
        popt, pcov = sp.optimize.curve_fit(tuning_curves.log_norm_pdf, plot_data.index, plot_data.values)
    except RuntimeError:
        # since data is a Series, this is the best way to do this.
        idx = [i for i in data.iloc[0].index if i not in [x, y]]
        warnings.warn("The following was not well fit by a log Gaussian and so is"
                      " skipped:\n%s" % data.iloc[0][idx])
    else:
        plt.plot(plot_data.index, tuning_curves.log_norm_pdf(plot_data.index, *popt), **kwargs)


def fit_log_norm_ci(x, y, ci_vals=[2.5, 97.5], **kwargs):
    """fit log norm to different bootstraps and plot the resulting mean and confidence interval.

    to be used with seaborn.FacetGrid.map_dataframe.

    because this goes through all the bootstraps and calculates their log normal tuning curves
    separately, it's takes much more time than fit_log_norm

    the data passed here must contain a column named `bootstrap_num`, which specifies which number
    bootstrap the observation corresponds to. Each value of bootstrap_num will be fit
    separately. It's recommended (i.e., this function was written assuming), therefore, that your
    data only contains one y value per value of bootstrap_num and value of x.

    x: string, column in data which contains the x values for this plot.

    y: string, column in data which contains the x values for this plot.

    ci_vals: 2-tuple or list of length 2 of floats, optional. the min and max percentile you wish
    to plot as a shaded region. For example, if you wish to plot the 95% confidence interval, then
    ci_vals=[2.5, 97.5] (the default); if you wish to plot the 68%, then ci_vals=[16, 84].

    kwargs must contain `data`, the DataFrame with data to plot.
    """
    data = kwargs.pop('data')
    if 'color' in kwargs:
        color = kwargs.pop('color')
    lines = []
    for boot in data.bootstrap_num.unique():
        plot_data = data.groupby(x)[[y, 'bootstrap_num']].apply(lambda x, j: x[x.bootstrap_num==j], boot)
        plot_idx = plot_data.index.get_level_values(x)
        plot_vals = plot_data[y].values
        try:
            popt, _ = sp.optimize.curve_fit(tuning_curves.log_norm_pdf, plot_idx, plot_vals)
        except RuntimeError:
            # since data is a Series, this is the best way to do this.
            idx = [i for i in data.iloc[0].index if i not in [x, y]]
            warnings.warn("The following bootstrap was not well fit by a log Gaussian and so is"
                          " skipped:\n%s" % data.iloc[0][idx])
        else:
            lines.append(tuning_curves.log_norm_pdf(plot_idx, *popt))
    lines = np.array(lines)
    lines_mean = lines.mean(0)
    cis = np.percentile(lines, ci_vals, 0)
    plt.fill_between(plot_idx, cis[0], cis[1], alpha=.2, facecolor=color, **kwargs)
    plt.plot(plot_idx, lines_mean, color=color, **kwargs)
    return lines


def local_grad_sin(dx, dy, loc_x, loc_y, w_r=None, w_a=None, phase=0, origin=None,
                   stim_type='logpolar'):
    """create a local 2d sin grating based on the gradients dx and dy

    this uses the gradients at location loc_x, loc_y to create a small grating to approximate a
    larger one at that location. This can be done either for the log polar gratings we use as
    stimuli (in which case w_r and w_a should be set to get the phase correct) or a regular 2d sin
    grating (like the one created by create_sin_cpp), in which case they can be left at 0

    dx and dy should be in cycles / pixel

    stim_type: {'logpolar', 'pilot', 'constant'}. what type of stimuli we're creating an
    approximation of (see stimuli.create_sf_maps_cpp for an explanation). used to determine how to
    calculate the local phase.
    """
    size = dx.shape[0]
    x, y = np.meshgrid(np.array(range(1, size+1)) - loc_x,
                       np.array(range(1, size+1)) - loc_y)
    if origin is None:
        origin = ((size+1) / 2., (size+1) / 2.)
    x_orig, y_orig = np.meshgrid(np.array(range(1, size+1))-origin[0],
                                 np.array(range(1, size+1))-origin[1])
    local_x = x_orig[loc_y, loc_x]
    local_y = y_orig[loc_y, loc_x]

    w_x = 2 * np.pi * dx[loc_y, loc_x]
    w_y = 2 * np.pi * dy[loc_y, loc_x]

    # the local phase is just the value of the actual grating at that point (see the explanation in
    # sfp.stimuli.create_sf_maps_cpp about why this works).
    if stim_type == 'constant':
        if w_r is not None or w_a is not None:
            raise Exception("If stim_type is constant, w_r / w_a must be None!")
        local_phase = np.mod(w_x * local_x + w_y * local_y + phase, 2*np.pi)
    elif stim_type == 'logpolar':
        local_phase = np.mod(((w_r * np.log(2))/2.) * np.log2(local_x**2 + local_y**2) +
                             w_a * np.arctan2(local_y, local_x) + phase, 2*np.pi)
    elif stim_type == 'pilot':
        alpha = 50
        local_phase = np.mod((w_r / np.pi) * np.log2(local_x**2 + local_y**2 + alpha**2) +
                             w_a * np.arctan2(local_y, local_x) + phase, 2*np.pi)
    if w_x == 0 and w_y == 0:
        return 0
    else:
        return np.cos(w_x*x + w_y*y + local_phase)


def _plot_and_save(grating, grating_type, save_path, **kwargs):
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
    dx = mask_array_like_grating(grating, dx, mid_val)
    dy = mask_array_like_grating(grating, dy, mid_val)
    mask_spacing = np.round(size / num_windows)
    # for this to work, the gratings must be non-overlapping
    mask_size = np.round(mask_spacing / 2) - 1
    masked_grating = np.zeros((size, size))
    masked_approx = np.zeros((size, size))
    masks = np.zeros((size, size))
    for i in range(mask_size, size, mask_spacing):
        for j in range(mask_size, size, mask_spacing):
            loc_x, loc_y = i, j
            mask = create_circle_mask(loc_x, loc_y, mask_size, size)
            masks += mask
            masked_grating += mask * grating
            masked_approx += mask * local_grad_sin(dx, dy, loc_x, loc_y, w_r, w_a, phase, origin,
                                                   stim_type)
    # in order to make the space between the masks black, that area should have the minimum
    # value, -1. but for the above to all work, that area needs to be 0, so this corrects that.
    masked_approx[~masks.astype(bool)] -= 1
    _plot_and_save(masked_grating, 'grating', save_path, **kwargs)
    _plot_and_save(masked_approx, 'approx', save_path, **kwargs)
    return masked_grating, masked_approx


def find_stim_idx(stim_df, **kwargs):
    stim_df = stim_df.dropna()
    stim_df.class_idx = stim_df.class_idx.astype(int)
    stim_df = stim_df.rename(columns={'index': 'stimulus_index'})
    stim_df = first_level_analysis._add_freq_metainfo(stim_df)
    props = stim_df.copy()
    key_order = ['stimulus_superclass', 'phi', 'freq_space_angle', 'freq_space_distance']
    key_order += [k for k in kwargs.iterkeys() if k not in key_order]
    for k in key_order:
        v = kwargs.get(k, None)
        if v is not None:
            if isinstance(v, basestring):
                val = v
            else:
                val = props.iloc[np.abs(props[k].values - v).argsort()[0]][k]
            props = props[props[k] == val]
    return props.index[0]


def find_stim_for_first_level(filename, stim_dir):
    if 'pilot00' in filename:
        stim_type = 'pilot'
        stim = np.load(os.path.join(stim_dir, 'pilot00_unshuffled.npy'))
        stim_df = pd.read_csv(os.path.join(stim_dir, 'pilot00_unshuffled_stim_description.csv'))
    elif 'pilot01' in filename:
        stim_type = 'pilot'
        stim = np.load(os.path.join(stim_dir, 'pilot01_unshuffled.npy'))
        stim_df = pd.read_csv(os.path.join(stim_dir, 'pilot01_unshuffled_stim_description.csv'))
    else:
        if 'task-sfpconstant' in filename:
            stim_type = 'constant'
            stim = np.load(os.path.join(stim_dir, 'constant_unshuffled.npy'))
            stim_df = pd.read_csv(os.path.join(stim_dir, 'constant_unshuffled_stim_description.csv'))
        else:
            stim_type = 'logpolar'
            stim = np.load(os.path.join(stim_dir, 'unshuffled.npy'))
            stim_df = pd.read_csv(os.path.join(stim_dir, 'unshuffled_stim_description.csv'))
    return {'stim': stim, 'stim_df': stim_df, 'stim_type': stim_type}


def load_data(subject, session=None, task=None, df_mode='full',
              bids_dir='~/Data/spatial_frequency_preferences', **kwargs):
    """load in the first level results, stimuli, and stimuli dataframe; return as a dict

    this loads in first level results and assocaited stimuli and stimuli description dataframe
    for one session.

    you must specify the subject and either the session (in which case we'll load in that one) or
    the task (in which case we ignore all pilot sessions and find the session that matches that
    task).

    df_mode determines whether we load in the full or summary dataframe.

    kwargs should specify other first_level_analysis keywords that are specified on the save path:
    vareas (list of ints), eccen (2-tuple of ints, the range of eccentricity considered), eccen_bin
    (boolean), hemi_bin (boolean), atlas_type ({"prior", "posterior"}), and mat_type (string
    specifying the type of design matrix).

    bids_dir should be the path to the bids directory

    if more than one file is found satisfying all constraints, then we'll throw an exception.
    """
    bids_dir = os.path.expanduser(bids_dir)
    if 'derivatives' not in bids_dir:
        bids_dir = os.path.join(bids_dir, 'derivatives')
    if 'first_level_analysis' not in bids_dir:
        bids_dir = os.path.join(bids_dir, 'first_level_analysis')
    stim_dir = os.path.abspath(os.path.join(bids_dir, '..', '..', 'stimuli'))
    layout = BIDSLayout(bids_dir)
    if "sub-" in subject:
        subject = subject.replace('sub-', '')
    layout = BIDSLayout(bids_dir)
    if session is not None:
        if "ses-" in str(session):
            session = session.replace("ses-", "")
        files = layout.get("file", subject=subject, session=session, type=df_mode)
    else:
        if "task-" in task:
            task = task.replace('task-', '')
        # the bit in session is regex to return a string that doesn't start with pilot
        files = layout.get("file", subject=subject, type=df_mode, session=r'^((?!pilot).+)')
    for k, v in kwargs.iteritems():
        if k == 'vareas':
            if not isinstance(v, basestring):
                val = '-'.join([str(i) for i in v])
            else:
                val = v
        elif k == 'eccen':
            if not isinstance(v, basestring):
                val = '-'.join([str(i) for i in v])
            else:
                val = v
        elif k == 'eccen_bin':
            if v:
                val = "eccen_bin"
            else:
                val = ""
        elif k == 'hemi_bin':
            if v:
                val = "hemi_bin"
            else:
                val = ""
        elif k in ['atlas_type', 'mat_type']:
            val = v
        else:
            raise Exception("Don't know how to handle kwargs %s!" % k)
        files = [f for f in files if val in f]
    if len(files) != 1:
        raise Exception("Cannot find unique first level results csv that satisfies all "
                        "specifications! Matching files: %s" % files)
    return_dict = {'df': pd.read_csv(files[0]), 'df_filename': files[0]}
    return_dict.update(find_stim_for_first_level(files[0], stim_dir))
    return return_dict
