#!/usr/bin/python
"""high-level functions to make relevant plots
"""
import matplotlib as mpl
# we do this because sometimes we run this without an X-server, and this backend doesn't need one
mpl.use('svg')
import argparse
import utils
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
        fig, axes = sns.plt.subplots(1, 3, figsize=figsize)
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
    dx, dy, _, _ = sfp_stimuli.create_sf_maps_cpp(props.res, w_r=freqs['w_r'], w_a=freqs['w_a'],
                                              w_x=freqs['w_x'], w_y=freqs['w_y'],
                                              stim_type=stim_type)
    return utils.plot_grating_approximation(stim, dx, dy, num_windows, props.phi, w_r=freqs['w_r'],
                                            w_a=freqs['w_a'], stim_type=stim_type,
                                            save_path=save_path, **kwargs)


def stimuli(stim, stim_df, **kwargs):
    """plot a bunch of stimuli with specific properties, pulled out of stim_df

    kwargs can be any of the columns in stim_df and the values should be either a list or a single
    value. if a single value, will assume that all stimuli share that property. all lists should be
    the same length. if a property isn't set, then we assume it's not important and so will grab
    the lowest stimuli with the lowest index that matches all specified properties.
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
        utils.im_plot(stim[idx, :, :], ax=ax)
    plt.tight_layout()


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
                    utils.add_img_to_xaxis(g.fig, ax, img, pos[img_idx], vmin=0, vmax=255, size=.15)
                except TypeError:
                    utils.add_img_to_xaxis(g.fig, ax, img, pos, vmin=0, vmax=255, size=.15)
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


def check_tuning_curves(tuning_df, save_path_template):
    """create all the tuning curve plots

    this takes the dataframe containing the tuning curves and creates plots of all of them, so they
    can be visibly checked. note that this will take a while and create *many* plots, especially
    when run on the full dataframes. It is thus *not* meant to be run from a notebook and it will
    close the plots as it creates and saves them.
    """
    gb_cols = ['varea']
    if 'bootstrap_num' in tuning_df.columns:
        gb_cols += ['bootstrap_num']
    for n, g in tuning_df.groupby(gb_cols):
        f = sns.FacetGrid(g, row='eccen', col='stimulus_superclass', hue='frequency_type')
        f.map(plt.scatter, 'frequency_value', 'amplitude_estimate')
        f.map_dataframe(plot_tuning_curve)
        f.add_legend()
        f.set_titles("eccen={row_name} | {col_name}")
        if isinstance(n, basestring) or not hasattr(n, '__iter__'):
            # then there's only one value in n (and thus, in gb_cols)
            suptitle = "%s=%s" % (gb_cols[0], n)
        else:
            suptitle = ", ".join("%s=%s" % (i, j) for i, j in zip(gb_cols, n))
        f.fig.suptitle(suptitle)
        plt.subplots_adjust(top=.95)
        f.savefig(save_path_template % (suptitle.replace(', ', '_')))
        plt.close(f.fig)


# NEED TO HANDLE MULTIPLE VISUAL AREAS
if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=("Creates the descriptive plots for one first level results dataframe")
        )
    parser.add_argument("first_level_results_path", help="path to first level results dataframe")
    parser.add_argument("stim_dir", help="path to directory containing stimuli")
    args = vars(parser.parse_args())
    d = utils.create_data_dict(**args)
    first_level_save_stem = d['df_filename'].replace('.csv', '')
    tuning_save_stem = d['tuning_df_filename'].replace('.csv', '')
    local_spatial_frequency(d['df'], first_level_save_stem+"_localsf.svg")
    stimuli_properties(d['df'], first_level_save_stem+"_stim_prop.svg")
    check_tuning_curves(d['tuning_df'], tuning_save_stem+"_tuning_curves_check_%s.svg")
    # if 'circular' in d['df'].stimulus_superclass.values:
    #     superclass_order = LOGPOLAR_SUPERCLASS_ORDER
    # else:
    #     superclass_order = CONSTANT_SUPERCLASS_ORDER
    # for n in superclass_order:
    #     df = d['df']
    #     df = df[(df.stimulus_superclass == n) & (df.varea == 1)]
    #     compare_hypotheses(df, save_path=save_stem+"_tuning_curves_%s.svg" % n)
