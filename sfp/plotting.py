#!/usr/bin/python
"""high-level functions to make relevant plots
"""
import matplotlib as mpl
# we do this because sometimes we run this without an X-server, and this backend doesn't need one
mpl.use('svg')
import argparse
import utils
import warnings
import os
import tuning_curves
import stimuli as sfp_stimuli
import first_level_analysis
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp

LOGPOLAR_SUPERCLASS_ORDER = ['circular', 'forward spiral', 'mixtures', 'radial', 'reverse spiral']
CONSTANT_SUPERCLASS_ORDER = ['vertical', 'forward diagonal', 'off-diagonal', 'horizontal',
                             'reverse diagonal']


def stimuli_properties(df, save_path=None):
    """plot some summaries of the stimuli properties

    this plots three pieces stimulus properties as a function of their position in frequency space
    (either w_x / w_y or w_r / w_a): superclass (circular, radial, etc / horizontal, vertical,
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
    norms = [None, None, utils.MidpointNormalize(df.freq_space_angle.min(),
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
        g.map(mini_plot, 'eccen', 'Local spatial frequency (cpd)', **kwargs)
        g.add_legend()
        plt.subplots_adjust(top=.9)
        g.fig.suptitle("Local spatial frequencies across eccentricities for all stimuli",
                       fontsize=15)
    if save_path is not None:
        g.fig.savefig(save_path, bbox_inches='tight')


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
        if hasattr(v, "__iter__") and not isinstance(v, basestring):
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


def compare_hypotheses(df, size=4, aspect=2, save_path=None):
    """make plots to compare hypotheses
    """
    if 'rounded_freq_space_distance' in df.columns:
        col_order = [i for i in LOGPOLAR_SUPERCLASS_ORDER if i in df.stimulus_superclass.unique()]
    else:
        col_order = [i for i in CONSTANT_SUPERCLASS_ORDER if i in df.stimulus_superclass.unique()]

    if 'bootstrap_num' in df.columns:
        df = df[['eccen', 'amplitude_estimate', 'freq_space_distance', 'Local spatial frequency (cpd)',
                 'bootstrap_num', 'stimulus_superclass']]
        df = pd.melt(df, ['eccen', 'amplitude_estimate', 'bootstrap_num', 'stimulus_superclass'],
                     var_name='Frequency')
        log_norm_func = utils.fit_log_norm_ci
        plot_kwargs = {'ci_vals': [16, 84]}
    else:
        df = df[['eccen', 'amplitude_estimate_median', 'freq_space_distance',
                 'Local spatial frequency (cpd)', 'stimulus_superclass']]
        df = pd.melt(df, ['eccen', 'amplitude_estimate_median', 'stimulus_superclass'],
                     var_name='Frequency')
        df = df.rename(columns={'amplitude_estimate_median': 'amplitude_estimate'})
        log_norm_func = utils.fit_log_norm
        plot_kwargs = {}

    with sns.axes_style('white'):
        g = sns.FacetGrid(df, hue='eccen', row='Frequency', palette='Reds', sharex=False,
                          col='stimulus_superclass', aspect=aspect, col_order=col_order, size=size,
                          row_order=['Local spatial frequency (cpd)', 'freq_space_distance'])
        g.map_dataframe(log_norm_func, 'value', 'amplitude_estimate', **plot_kwargs)
        # we use this so we can just plot the mean value (it plays nicely with FacetGrid)
        g.map(sns.regplot, 'value', 'amplitude_estimate', x_estimator=np.mean, fit_reg=False, ci=0)
        xmin = 2**np.floor(np.log2(df.groupby('Frequency').value.min()))
        xmax = 2**np.ceil(np.log2(df.groupby('Frequency').value.max()))
        for i, (ax, (n, _)) in enumerate(zip(g.axes.flatten(), df.groupby(['Frequency', 'stimulus_superclass']))):
            if i < df.stimulus_superclass.nunique():
                ax.set_title('Response as function of local spatial frequency (cycles / degree) | '
                             'stimulus_superclass = %s' % n[1])
            else:
                ax.set_title('Response as function of stimulus | stimulus_superclass = %s' % n[1])
                
            ax.set_xlim([xmin[n[0]], xmax[n[0]]])
            ax.set_xscale('log', basex=2)
            ax.set_ylabel("Response amplitude estimate")
        legend_data = {}
        legend_order = []
        for k in sorted(g._legend_data.keys()):
            new_name = "%i-%i" % (int(k.split('-')[0]), int(k.split('-')[1]))
            legend_data[new_name] = g._legend_data[k]
            legend_order.append(new_name)
        g.add_legend(legend_data, "Eccentricity (degrees)", legend_order)
        g.set_xlabels('Frequency')
    if save_path is not None:
        g.fig.savefig(save_path, bbox_inches='tight')
    return g


# NEED TO GET FIGURE TITLE SIZES BETTER AND FIGURE OUT RELIABLE WAY TO PLACE IMAGES WITH DIFFERENT
# NUMBERS OF COLUMNS
def compare_hypotheses_talk(df, axis_imgs, axis_img_locs=[(.025, .05), .6], save_path=None,
                            **kwargs):
    """make talk version of compare hypotheses plot

    axis_imgs: the specified images to plot on the x-axis. The images will be masked to place on
    the x-axis of the local frequency plot and will be used unmodified on the stimulus space plot.

    axis_img_locs: list made up of 2-tuples and floats. the relative positions of the images on the
    x axis. should have one entry for each image in axis_imgs. if a 2-tuple, the first is the
    position in the local frequency plot, the second in the stimulus space plot. if a float, same
    position will be used for both. will be passed to utils.add_img_to_xaxis as the rel_position
    arg.

    returns the FacetGrid containing the plot
    """
    # this arbitrary bit of code makes a well-sized mask
    mask = utils.create_circle_mask(750, 350, 10*1080/(2*2*12), 1080)
    windowed_axis_imgs = [mask * s + ~mask.astype(bool)*127 for s in axis_imgs]
    with sns.plotting_context('poster'):
        g = compare_hypotheses(df, 5)
        plt.subplots_adjust(hspace=.6)
        for i, ax in enumerate(g.axes.flatten()):
            img_idx = {True: 0, False: 1}.get(i < df.stimulus_superclass.nunique())
            for pos, img in zip(axis_img_locs, [windowed_axis_imgs, axis_imgs][img_idx]):
                try:
                    add_img_to_xaxis(g.fig, ax, img, pos[img_idx], vmin=0, vmax=255, size=.15)
                except TypeError:
                    add_img_to_xaxis(g.fig, ax, img, pos, vmin=0, vmax=255, size=.15)
            ax.xaxis.set_visible(False)
    if save_path is not None:
        g.fig.savefig(save_path, bbox_inches='tight')
    return g


def get_tuning_curve_properties(df, id_vars=['stimulus_superclass', 'eccen', 'bootstrap_num']):
    """fits log normal tuning curves to data, returns peak and bandwidth
    """
    new_df = []
    for i, (n, g) in enumerate(df.groupby(id_vars)):
        # there are two possibilities: either g only contains one amplitude estimate for each
        # spatial frequency or it contains several (which means that id_vars didn't restrict it
        # down too far). the following guarantees that we have one value per local spatial
        # frequency, leaving it unchanged if that's already the case
        data = g.groupby('Local spatial frequency (cpd)')['amplitude_estimate'].mean()
        popt, _ = sp.optimize.curve_fit(utils.log_norm_pdf, data.index, data.values)
        # popt contains a, mu, and sigma, in that order
        mode, var = utils.log_norm_describe(popt[1], popt[2])
        df_data = dict((k, v) for k, v in zip(id_vars, n))
        df_data.update({'peak': mode, 'bandwidth': var})
        # we want to change eccentricity to a number
        if 'eccen' in df_data:
            df_data['eccen'] = np.mean([float(i) for i in df_data['eccen'].split('-')])
        new_df.append(pd.DataFrame(df_data, index=[i]))
    return pd.concat(new_df)


def peak_spatial_frequency(df, id_vars=['stimulus_superclass']):
    """create peak spatial frequency plot
    """
    df = df[['stimulus_superclass', 'eccen', 'Local spatial frequency (cpd)',
             'amplitude_estimate', 'bootstrap_num']]
    df = get_tuning_curve_properties(df, id_vars + ['eccen'])
    peak_df = []
    for i, (n, g) in enumerate(df.groupby(id_vars)):
        peak_a, _ = sp.optimize.curve_fit(utils.flat_hyperbola, g.eccen.values, g.peak.values)
        peak_df.append(g.assign(peak_hyperbola_param=peak_a[0]))
    peak_df = pd.concat(peak_df)
    return peak_df


def plot_tuning_curve(**kwargs):
    data = kwargs.pop('data')
    x, y = tuning_curves.get_tuning_curve_xy_from_df(data)
    plt.semilogx(x, y, basex=2, **kwargs)


def check_tuning_curves(tuning_df, save_path_template, **kwargs):
    """create all the tuning curve plots

    this takes the dataframe containing the tuning curves and creates plots of all of them, so they
    can be visibly checked. note that this will take a while and create *many* plots, especially
    when run on the full dataframes. It is thus *not* meant to be run from a notebook and it will
    close the plots as it creates and saves them.

    kwargs can contain columns in the tuning_df and values to limit them to.
    """
    for k, v in kwargs.iteritems():
        try:
            tuning_df = tuning_df[tuning_df[k].isin(v)]
        except TypeError:
            tuning_df = tuning_df[tuning_df[k] == v]
    gb_cols = ['varea']
    title_template = 'varea={}'
    if 'bootstrap_num' in tuning_df.columns:
        gb_cols += ['bootstrap_num']
        title_template += ', bootstrap={:02d}'
    for n, g in tuning_df.groupby(gb_cols):
        f = sns.FacetGrid(g, row='eccen', col='stimulus_superclass', hue='frequency_type')
        f.map(plt.scatter, 'frequency_value', 'amplitude_estimate')
        f.map_dataframe(plot_tuning_curve)
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
                              "stimuli_properties) or tuning_curves_check_varea={v}[_bootstrap"
                              "={b:02d}] (plotting.check_tuning_curves)"))
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
        if tuning_df_present:
            check_tuning_curves(d['tuning_df'], tuning_save_stem+"_tuning_curves_check_%s.svg")
        else:
            warnings.warn("Unable to create tuning curves plot because tuning curve df hasn't "
                          "been created!")
    else:
        for p in args['plot_to_make']:
            if 'localsf' == p:
                local_spatial_frequency(d['df'], first_level_save_stem+"_localsf.svg")
            elif 'stim_prop' == p:
                stimuli_properties(d['df'], first_level_save_stem+"_stim_prop.svg")
            elif 'tuning_curves_check' in p:
                if tuning_df_present:
                    p_kwargs = p.replace('tuning_curves_check_', '')
                    p_kwargs = dict(i.split('=') for i in p_kwargs.split('_'))
                    # we know both are ints
                    p_kwargs = dict(({'bootstrap': 'bootstrap_num'}.get(k, k), int(v)) for k, v in p_kwargs.iteritems())
                    check_tuning_curves(d['tuning_df'], tuning_save_stem+"_tuning_curves_check_%s.svg",
                                        **p_kwargs)
                else:
                    raise Exception("Unable to create tuning curves plot because tuning curve df "
                                    "hasn't been created!")
            else:
                raise Exception("Don't know how to make plot %s!" % p)
    # if 'circular' in d['df'].stimulus_superclass.values:
    #     superclass_order = LOGPOLAR_SUPERCLASS_ORDER
    # else:
    #     superclass_order = CONSTANT_SUPERCLASS_ORDER
    # for n in superclass_order:
    #     df = d['df']
    #     df = df[(df.stimulus_superclass == n) & (df.varea == 1)]
    #     compare_hypotheses(df, save_path=save_stem+"_tuning_curves_%s.svg" % n)
