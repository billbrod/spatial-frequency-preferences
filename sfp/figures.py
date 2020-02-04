#!/usr/bin/python
"""functions to create the figures for publication
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from . import summary_plots
from . import analyze_model
from . import plotting
from . import model
from . import utils
from . import first_level_analysis


def existing_studies_df():
    """create df summarizing earlier studies

    there have been a handful of studies looking into this, so we want
    to summarize them for ease of reference. Each study is measuring
    preferred spatial frequency at multiple eccentricities in V1 using
    fMRI (though how exactly they determine the preferred SF and the
    stimuli they use vary)

    This dataframe contains the following columns:
    - Paper: the reference for this line
    - Eccentricity: the eccentricity (in degrees) that they measured
      preferred spatial frequency at
    - Preferred spatial frequency (cpd): the preferred spatial frequency
      measured at this eccentricity (in cycles per degree)
    - Preferred period (dpc): the preferred period measured at this
      eccentricity (in degrees per cycle); this is just the inverse of
      the preferred spatial frequency

    The eccentricity / preferred spatial frequency were often not
    reported in a manner that allowed for easy extraction of the data,
    so the values should all be taken as approximate, as they involve me
    attempting to read values off of figures / colormaps.

    Papers included (and their reference in the df):
    - Sasaki (2001): Sasaki, Y., Hadjikhani, N., Fischl, B., Liu, A. K.,
      Marret, S., Dale, A. M., & Tootell, R. B. (2001). Local and global
      attention are mapped retinotopically in human occipital
      cortex. Proceedings of the National Academy of Sciences, 98(4),
      2077–2082.
    - Henriksson (2008): Henriksson, L., Nurminen, L., Hyv\"arinen,
      Aapo, & Vanni, S. (2008). Spatial frequency tuning in human
      retinotopic visual areas. Journal of Vision, 8(10),
      5. http://dx.doi.org/10.1167/8.10.5
    - Kay (2011): Kay, K. N. (2011). Understanding Visual Representation
      By Developing Receptive-Field Models. Visual Population Codes:
      Towards a Common Multivariate Framework for Cell Recording and
      Functional Imaging, (), 133–162.
    - Hess (dominant eye, 2009): Hess, R. F., Li, X., Mansouri, B.,
      Thompson, B., & Hansen, B. C. (2009). Selectivity as well as
      sensitivity loss characterizes the cortical spatial frequency
      deficit in amblyopia. Human Brain Mapping, 30(12),
      4054–4069. http://dx.doi.org/10.1002/hbm.20829 (this paper reports
      spatial frequency separately for dominant and non-dominant eyes in
      amblyopes, only the dominant eye is reported here)
    - D'Souza (2016): D'Souza, D. V., Auer, T., Frahm, J., Strasburger,
      H., & Lee, B. B. (2016). Dependence of chromatic responses in v1
      on visual field eccentricity and spatial frequency: an fmri
      study. JOSA A, 33(3), 53–64.
    - Farivar (2017): Farivar, R., Clavagnier, S., Hansen, B. C.,
      Thompson, B., & Hess, R. F. (2017). Non-uniform phase sensitivity
      in spatial frequency maps of the human visual cortex. The Journal
      of Physiology, 595(4),
      1351–1363. http://dx.doi.org/10.1113/jp273206
    - Olsson (pilot, model fit): line comes from a model created by Noah
      Benson in the Winawer lab, fit to pilot data collected by
      Catherine Olsson (so note that this is not data). Never ended up
      in a paper, but did show in a presentation at VSS 2017: Benson NC,
      Broderick WF, Müller H, Winawer J (2017) An anatomically-defined
      template of BOLD response in
      V1-V3. J. Vis. 17(10):585. DOI:10.1167/17.10.585

    Returns
    -------
    df : pd.DataFrame
        Dataframe containing the optimum spatial frequency at multiple
        eccentricities from the different papers

    """
    data_dict = {
        'Paper': ['Sasaki (2001)',]*7,
        'Preferred spatial frequency (cpd)': [1.25, .9, .75, .7, .6, .5, .4],
        'Eccentricity': [0, 1, 2, 3, 4, 5, 12]
    }
    data_dict['Paper'].extend(['Henriksson (2008)', ]*5)
    data_dict['Preferred spatial frequency (cpd)'].extend([1.2, .68, .46, .40, .18])
    data_dict['Eccentricity'].extend([1.7, 4.7, 6.3, 9, 19])

    # This is only a single point, so we don't plot it
    # data_dict['Paper'].extend(['Kay (2008)'])
    # data_dict['Preferred spatial frequency (cpd)'].extend([4.5])
    # data_dict['Eccentricity'].extend([ 2.9])

    data_dict['Paper'].extend(['Kay (2011)']*5)
    data_dict['Preferred spatial frequency (cpd)'].extend([4, 3, 10, 10, 2])
    data_dict['Eccentricity'].extend([2.5, 4, .5, 1.5, 7])

    data_dict['Paper'].extend(["Hess (dominant eye, 2009)"]*3)
    data_dict['Preferred spatial frequency (cpd)'].extend([2.25, 1.9, 1.75])
    data_dict['Eccentricity'].extend([2.5, 5, 10])

    data_dict['Paper'].extend([ "D'Souza (2016)",]*3)
    data_dict['Preferred spatial frequency (cpd)'].extend([2, .95, .4])
    data_dict['Eccentricity'].extend([1.4, 4.6, 9.8])

    data_dict['Paper'].extend(['Farivar (2017)']*2)
    data_dict['Preferred spatial frequency (cpd)'].extend([3, 1.5,])
    data_dict['Eccentricity'].extend([.5, 3])

    data_dict['Paper'].extend([ 'Olsson (pilot, model fit)']*10)
    data_dict['Preferred spatial frequency (cpd)'].extend([2.11, 1.76, 1.47, 2.75, 1.24, 1.06, .88, .77, .66, .60])
    data_dict['Eccentricity'].extend([2, 3, 4, 1, 5, 6, 7, 8, 9, 10])

    # Predictions of the scaling hypothesis -- currently unused
    # ecc = np.linspace(.01, 20, 50)
    # fovea_cutoff = 0
    # # two possibilities here
    # V1_RF_size = np.concatenate([np.ones(len(ecc[ecc<fovea_cutoff])),
    #                              np.linspace(1, 2.5, len(ecc[ecc>=fovea_cutoff]))])
    # V1_RF_size = .2 * ecc

    df = pd.DataFrame(data_dict)
    df = df.sort_values(['Paper','Eccentricity',])
    df["Preferred period (dpc)"] = 1. / df['Preferred spatial frequency (cpd)']

    return df


def _demean_df(df, gb_cols=['subject'], y='cv_loss'):
    """demean a column of the dataframe

    Calculate the mean of `y` across the values in some other column(s)
    `gb_cols`, then demean `y` and return df with several new columns:
    - `demeaned_{y}`: each y with the gb_cols-wise average of y
      subtracted off
    - `{y}_mean`: the gb_cols-wise average of y
    - `{y}_mean_overall`: the average of `{y}_mean`
    - `remeaned_{y}`: the `demeaned_{y}` with `{y}_mean_overall` added
      back to it

    If you use this with the defaults, the overall goal of this is to
    enable us to look at how the cv_loss varies across models, because
    the biggest effect is the difference in cv_loss across
    subjects. Demeaning the cv_loss on a subject-by-subject basis
    enables us to put all the subjects together so we can look for
    patterns across models. For example, we can then compute error bars
    that only capture the variation across models, but not across
    subjects. Both remeaned or demeaned will capture this, the question
    is what values to have on the y-axis. If you use demeaned, you'll
    have negative loss, which might be confusing. If you use remeaned,
    the y-axis values will be the average across subjects, which might
    be easier to interpret.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to demean
    gb_cols : list, optional
        columns to calculate the mean across (name comes from pandas
        groupby operation)
    y : str, optional
        the column to demean

    Returns
    -------
    df : pd.DataFrame
        dataframe with new, demeaned column

    """
    df = df.set_index(gb_cols)
    df[f'{y}_mean'] = df.groupby(gb_cols)[y].mean()
    # here we take the average over the averages. we do this so that we
    # weight all of the groups the same. For example, if
    # gb_cols=['subject'] (as default) and one subject had twice as many
    # rows (because it had two sessions in df, for example), then this
    # ensures that subject isn't twice as important when computing the
    # mean (which would be the case if we used df[f'{y}_mean'].mean()
    # instead)
    df[f'{y}_mean_overall'] = df.groupby(gb_cols)[y].mean().mean()
    df[f'demeaned_{y}'] = df[y] - df[f'{y}_mean']
    df[f'remeaned_{y}'] = df[f'demeaned_{y}'] + df[f'{y}_mean_overall']
    return df.reset_index()


def prep_df(df, task):
    """prepare the dataframe by restricting to the appropriate subset

    The dataframe created by earlier analysis steps contains all
    scanning sessions and potentially multiple visual areas. for our
    figures, we just want to grab the relevant scanning sessions and
    visual areas (V1), so this function helps do that. If df has the
    'frequency_type' column (i.e., it's summarizing the 1d tuning
    curves), we also restrict to the "local_sf_magnitude" rows (rather
    than "frequency_space")

    Parameters
    ----------
    df : pd.DataFrame
        dataframe that will be used for plotting figures. contains some
        summary of (either 1d or 2d) model information across sessions.
    task : {'task-sfrescaled', 'task-sfpconstant'}
        this determines which task we'll grab: task-sfprescaled or
        task-sfpconstant. task-sfp is also exists, but we consider that
        a pilot task and so do not allow it for the creation of figures
        (the stimuli were not contrast-rescaled).

    Returns
    -------
    df : pd.DataFrame
        The restricted dataframe.

    """
    if task not in ['task-sfprescaled', 'task-sfpconstant']:
        raise Exception("Only task-sfprescaled and task-sfpconstant are allowed!")
    df = df.query("task==@task")
    if 'frequency_type' in df.columns:
        df = df.query("frequency_type=='local_sf_magnitude'")
    if 'varea' in df.columns:
        df = df.query("varea==1")
    return df


def prep_model_df(df):
    """prepare models df for plotting

    For plotting purposes, we want to rename the model parameters from
    their original values (e.g., sf_ecc_slope, abs_mode_cardinals) to
    those we use in the equation (e.g., a, p_1). We do that by simply
    remapping the names from those given at plotting.ORIG_PARAM_ORDER to
    those in plotting.PLOT_PARAM_ORDER. we additionally add a new
    column, param_category, which we use to separate out the three types
    of parameters: sigma, the effect of eccentricity, and the effect of
    orientation / retinal angle.

    Parameters
    ----------
    df : pd.DataFrame
        models dataframe, that is, the dataframe that summarizes the
        parameter values for a variety of models

    Returns
    -------
    df : pd.DataFrame
        The remapped dataframe.

    """
    rename_params = dict((k, v) for k, v in zip(plotting.ORIG_PARAM_ORDER,
                                                plotting.PLOT_PARAM_ORDER))
    df = df.set_index('model_parameter')
    df.loc['sigma', 'param_category'] = 'sigma'
    df.loc[['sf_ecc_slope', 'sf_ecc_intercept'], 'param_category'] = 'eccen'
    df.loc[['abs_mode_cardinals', 'abs_mode_obliques', 'rel_mode_cardinals', 'rel_mode_obliques',
            'abs_amplitude_cardinals', 'abs_amplitude_obliques', 'rel_amplitude_cardinals',
            'rel_amplitude_obliques'], 'param_category'] = 'orientation'
    df = df.reset_index()
    df['model_parameter'] = df.model_parameter.map(rename_params)
    return df


def append_precision_col(df, col='preferred_period',
                         gb_cols=['subject', 'session', 'varea', 'stimulus_superclass', 'eccen']):
    """append column giving precision of another column and collapse

    this function gives the precision of the value found in a single
    column (across the columns that are NOT grouped-by) and collapses
    across those columns. The intended use case is to determine the
    precision of a parameter estimate across bootstraps for each
    (subject, session) (for the 2d model) or for each (subject, session,
    stimulus_superclass, eccen) (for the 1d model).

    precision is the inverse of the variance, so let :math:`c` be the
    68% confidence interval of the column value, then precision is
    :math:`\frac{1}{(c/2)^2}`

    finally, we collapse across gb_cols, returning the median and
    precision of col for each combination of values from those columns.

    Parameters
    ----------
    df : pd.DataFrame
        the df that contains the values we want the precision for
    col : str, optional
        the name of the column that contains the values we want the
        precision for
    gb_cols : list, optional
        list of strs containing the columns we want to groupby. we will
        compute the precision separately for each combination of values
        here.

    Returns
    -------
    df : pd.DataFrame
        the modified df, containing the median and precision of col
        (also contains the medians of the other values in the original
        df, but not their precision)

    """
    gb = df.groupby(gb_cols)
    df = df.set_index(gb_cols)
    df[f'{col}_precision'] = gb[col].apply(first_level_analysis._precision_dist)
    df = df.reset_index()
    return df.groupby(gb_cols).median().reset_index()


def precision_weighted_bootstrap(df, n_bootstraps=100, col='preferred_period',
                                 gb_cols=['varea', 'stimulus_superclass', 'eccen']):
    """calculate the precision-weighted bootstrap of a column

    to combine across subjects, we want to use a precision-weighted
    average, rather than a regular average, because we are trying to
    summarize the true value across the population and our uncertainty
    in it. Therefore, we down-weight subjects whose estimate is
    noisier. Similar to append_precision_col(), we groupby over some of
    the columns to combine info across them (gb_cols here should be a
    subset of those used for append_precision_col())

    You should plot the values here with scatter_ci_dist() or something
    similar to draw the 68% CI of the distribution here (not sample it
    to draw the CI)

    Parameters
    ----------
    df : pd.DataFrame
        the df that we want to bootstrap (must already have precision
        column, i.e., this should be the df returned by
        append_precision_col())
    n_bootstraps : int, optional
        the number of independent bootstraps to draw
    col : str, optional
        the name of the column that contains the values we want to draw
        bootstraps for
    gb_cols : list, optional
        list of strs containing the columns we want to groupby. we will
        compute the bootstraps for each combination of values here.

    Returns
    -------
    df : pd.DataFrame
        the df containing the bootstraps of precision-weighted
        mean. this will only contain the following columns: col,
        *gb_cols, and bootstrap_num

    """
    if type(gb_cols) != list:
        raise Exception("gb_cols must be a list!")
    bootstraps = []
    for n, g in df.groupby(gb_cols):
        # n needs to be a list of the same length as gb_cols for the
        # dict(zip()) call to work, but if len(gb_cols) == 1, then it
        # will be a single str (or int or float or whatever), so we
        # convert it to a list real quick
        if len(gb_cols) == 1:
            n = [n]
        tmp = dict(zip(gb_cols, n))
        for j in range(n_bootstraps):
            t = g.sample(len(g), replace=True)
            tmp[col] = np.average(t[col], weights=t[f'{col}_precision'])
            tmp['bootstrap_num'] = j
            bootstraps.append(pd.DataFrame(tmp, [0]))
    bootstraps = pd.concat(bootstraps).reset_index(drop=True)
    if 'subject' in df.columns and 'subject' not in gb_cols:
        bootstraps['subject'] = 'all'
    return bootstraps


def _summarize_1d(df, reference_frame, y, row, col, height, **kwargs):
    """helper function for pref_period_1d and bandwidth_1d

    since they're very similar functions.

    "eccen" is always plotted on the x-axis, and hue is always
    "stimulus_type" (unless overwritten with kwargs)

    Parameters
    ----------
    df : pd.DataFrame
        pandas DataFrame summarizing all the 1d tuning curves, as
        created by the summarize_tuning_curves.py script. If you want
        confidence intervals, this should be the "full" version of that
        df (i.e., including the fits to each bootstrap).
    y : str
        which column of the df to plot on the y-axis
    reference_frame : {'relative', 'absolute'}
        whether the data contained here is in the relative or absolute
        reference frame. this will determine both the palette used and
        the hue_order
    row : str
        which column of the df to facet the plot's rows on
    col : str
        which column of the df to facet the plot's column on
    height : float
        height of each plot facet
    kwargs :
        all passed to summary_plots.main() (most of these then get
        passed to sns.FacetGrid, see the docstring of summary_plots.main
        for more info)

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    pal = plotting.stimulus_type_palette(reference_frame)
    hue_order = plotting.get_order('stimulus_type', reference_frame)
    col_order, row_order = None, None
    if col is not None:
        col_order = plotting.get_order(col, col_unique=df[col].unique())
    if row is not None:
        row_order = plotting.get_order(row, col_unique=df[row].unique())
    g = summary_plots.main(df, row=row, col=col, y=y, eccen_range=(0, 11), hue_order=hue_order,
                           linewidth=2, xlim=(0, 12), x_jitter=[None, .2],height=height,
                           plot_func=[plotting.plot_median_fit, plotting.scatter_ci_dist],
                           palette=pal, col_order=col_order, row_order=row_order, **kwargs)
    g.set_xlabels('Eccentricity (deg)')
    g._legend.set_title("Stimulus class")
    g.fig.subplots_adjust(top=.85)
    return g


def pref_period_1d(df, reference_frame='relative', row='session', col='subject', height=4,
                   **kwargs):
    """plot the preferred period of the 1d model fits

    Note that we do not restrict the input dataframe in any way, so we
    will plot all data contained within it. If this is not what you want
    (e.g., you only want to plot some of the tasks), you'll need to do
    the restrictions yourself before passing df to this function

    The only difference between this and the bandwidth_1d function is
    what we plot on the y-axis, and how we label it.

    Parameters
    ----------
    df : pd.DataFrame
        pandas DataFrame summarizing all the 1d tuning curves, as
        created by the summarize_tuning_curves.py script. If you want
        confidence intervals, this should be the "full" version of that
        df (i.e., including the fits to each bootstrap).
    reference_frame : {'relative', 'absolute'}, optional
        whether the data contained here is in the relative or absolute
        reference frame. this will determine both the palette used and
        the hue_order
    row : str, optional
        which column of the df to facet the plot's rows on
    col : str, optional
        which column of the df to facet the plot's column on
    height : float, optional
        height of each plot facet
    kwargs :
        passed to sfp.figures._summarize_1d

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    kwargs.setdefault('ylim', (0, 4))
    g = _summarize_1d(df, reference_frame, 'preferred_period', row, col, height, **kwargs)
    g.set_ylabels('Preferred period (dpc)')
    yticks = [i for i in range(4) if i <= kwargs['ylim'][1]]
    g.set(yticks=yticks)
    g.fig.suptitle("Preferred period of 1d tuning curves in each eccentricity band")
    return g


def bandwidth_1d(df, reference_frame='relative', row='session', col='subject', height=4, **kwargs):
    """plot the bandwidth of the 1d model fits

    Note that we do not restrict the input dataframe in any way, so we
    will plot all data contained within it. If this is not what you want
    (e.g., you only want to plot some of the tasks), you'll need to do
    the restrictions yourself before passing df to this function

    The only difference between this and the pref_period_1d function is
    what we plot on the y-axis, and how we label it.

    Parameters
    ----------
    df : pd.DataFrame
        pandas DataFrame summarizing all the 1d tuning curves, as
        created by the summarize_tuning_curves.py script. If you want
        confidence intervals, this should be the "full" version of that
        df (i.e., including the fits to each bootstrap).
    reference_frame : {'relative', 'absolute'}, optional
        whether the data contained here is in the relative or absolute
        reference frame. this will determine both the palette used and
        the hue_order
    row : str, optional
        which column of the df to facet the plot's rows on
    col : str, optional
        which column of the df to facet the plot's column on
    height : float, optional
        height of each plot facet
    kwargs :
        passed to sfp.figures._summarize_1d

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    g = _summarize_1d(df, reference_frame, 'tuning_curve_bandwidth', row, col, height, **kwargs)
    g.set_ylabels('Tuning curve FWHM (octaves)')
    g.fig.suptitle("Full-Width Half-Max of 1d tuning curves in each eccentricity band")
    return g


def existing_studies_figure(df, y="Preferred period (dpc)"):
    """Plot the results from existing studies

    See the docstring for figures.existing_studies_df() for more
    details on the information displayed in this figure.

    Parameters
    ----------
    df : pd.DataFrame
        The existing studies df, as returned by the function
        figures.existing_studies_df().
    y : {'Preferred period (dpc)', 'Preferred spatial frequency (cpd)'}
        Whether to plot the preferred period or preferred spatial
        frequency on the y-axis. If preferred period, the y-axis is
        linear; if preferred SF, the y-axis is log-scaled (base 2). The
        ylims will also differ between these two

    Returns
    -------
    g : sns.FacetGrid
        The FacetGrid containing the plot

    """
    pal = sns.color_palette('Set3', df.Paper.nunique())
    g = sns.FacetGrid(df, hue='Paper', size=4, aspect=1.2, palette=pal)
    if y == "Preferred period (dpc)":
        g.map(plt.plot, 'Eccentricity', y, marker='o', linewidth=2)
        g.ax.set_ylim((0, 6))
    elif y == "Preferred spatial frequency (cpd)":
        g.map(plt.semilogy, 'Eccentricity', y, marker='o', linewidth=2, basey=2)
        g.ax.set_ylim((0, 11))
        g.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(plotting.myLogFormat))
    g.ax.set_xlim((0, 20))
    g.add_legend()
    g.ax.set_title("Summary of human V1 fMRI results")
    g.ax.set_xlabel('Eccentricity of receptive field center (deg)')
    return g


def input_schematic(prf_loc=(250, 250), prf_radius=100, stim_freq=(.01, .03)):
    """schematic to explain 2d model inputs

    This schematic explains the various inputs of our 2d model:
    eccentricity, retinotopic angle, spatial frequency, and
    orientation. It does this with a little diagram of a pRF with a
    local stimulus, with arrows and labels.

    The location and size of the pRF, as well as the frequency of the
    stimulus, are all modifiable, and the labels and arrows will update
    themselves. The arrows should behave appropriately, but it's hard to
    guarantee that the labels will always look good (their positioning
    is relative, so it will at least be close). You are restricted to
    placing the pRF inside the first quadrant, which helps make the
    possibilities more reasonable.

    Parameters
    ----------
    prf_loc : tuple, optional
        2-tuple of floats, location of the prf. Both numbers must lie
        between 0 and 500 (i.e., we require this to be in the first
        quadrant). Max value on both x and y axes is 500.
    prf_radius : float, optional
        radius of the prf, in pixels. the local stimulus will have half
        this radius
    stim_freq : tuple, optional
        2-tuple of floats, the (x_freq, y_freq) of the stimulus, in
        cycles per pixel

    Returns
    -------
    fig : plt.Figure
        Figure containing the schematic

    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    def get_xy(distance, angle, origin=(500, 500)):
        return [o + distance * func(angle) for o, func in zip(origin, [np.cos, np.sin])]
    def draw_arrow(xy, xytext, text="", arrowprops={}, **kwargs):
        arrowprops['linewidth'] = 2
        ax.annotate(text, xy=xy, xytext=xytext, xycoords='data', textcoords='data',
                    arrowprops=arrowprops, **kwargs)
    pal = sns.color_palette('deep', 2)
    if (np.array(prf_loc) > 500).any() or (np.array(prf_loc) < 0).any():
        raise Exception("the coordinates of prf_loc must be between 0 and 500, but got "
                        f"value {prf_loc}!")
    # prf_loc is in coordinates relative to the center, so we convert that here
    abs_prf_loc = [500 + i for i in prf_loc]
    mask = utils.create_circle_mask(*abs_prf_loc, prf_radius/2, 1001)
    mask[mask==0] = np.nan
    stim = mask * utils.create_sin_cpp(1001, *stim_freq)
    plotting.im_plot(stim, ax=ax, origin='lower')
    ax.axhline(500, c='.5')
    ax.axvline(500, c='.5')
    ax.set(xlim=(450, 1001), ylim=(450, 1001))
    for s in ax.spines.keys():
        ax.spines[s].set_visible(False)
    prf = mpl.patches.Circle(abs_prf_loc, prf_radius, fc='none', ec='k', linewidth=4,
                             linestyle='--', zorder=10)
    ax.add_artist(prf)
    prf_ecc = np.sqrt(np.square(prf_loc).sum())
    prf_angle = np.arctan2(*prf_loc[::-1])
    phi_loc = get_xy(100, prf_angle)
    e_loc = get_xy(prf_ecc/2, prf_angle + np.pi/13)
    draw_arrow((500, 500), abs_prf_loc, arrowprops={'connectionstyle': 'arc3', 'arrowstyle': '<-',
                                                    'color': pal[1]})
    ax.text(*e_loc, r'$e$', {'size': 15})
    draw_arrow(phi_loc, (600, 500), arrowprops={'connectionstyle': 'angle3', 'arrowstyle': '-',
                                                'color': pal[1]})
    ax.text(600, 500 + 50*np.sin(prf_angle/2), r'$\phi$', {'size': 15})
    # so that this is the normal vector, the 7000 is just an arbitrary
    # scale factor to make the vector a reasonable length
    normal_len = 7000 * np.sqrt(np.square(stim_freq).sum())
    normal_angle = np.arctan2(*stim_freq[::-1])
    omega_loc = get_xy(normal_len, normal_angle, abs_prf_loc)
    draw_arrow(abs_prf_loc, omega_loc, r'$\omega$', {'connectionstyle': 'arc3', 'arrowstyle': '<-',
                                                     'color': pal[0]}, size=15)
    arc_loc = get_xy(1.2*normal_len/2, normal_angle, abs_prf_loc)
    draw_arrow(arc_loc, (abs_prf_loc[0] + 1.2*normal_len/2, abs_prf_loc[1]),
               arrowprops={'connectionstyle': 'angle3', 'arrowstyle': '-', 'color': pal[0]})
    draw_arrow((abs_prf_loc[0] + normal_len, abs_prf_loc[1]), abs_prf_loc,
               arrowprops={'connectionstyle': 'angle3', 'arrowstyle': '-', 'color': '.5',
                           'linestyle': ':'})
    theta_loc = get_xy(1.3*normal_len/2, normal_angle/4, abs_prf_loc)
    ax.text(*theta_loc, r'$\theta$', {'size': 15})
    return fig


def model_schematic():
    """Create model schematic.

    In order to better explain the model, its predictions, and the
    effects of its parameters, we create a model schematic that shows
    the effects of the different p parameters (those that control the
    effect of stimulus orientation and retinotopic angle on preferred
    period).

    Returns
    -------
    fig : plt.Figure
        Figure containing the schematic

    """
    abs_model = model.LogGaussianDonut('full', sf_ecc_slope=.2, sf_ecc_intercept=.2,
                                       abs_mode_cardinals=.4, abs_mode_obliques=.1)
    rel_model = model.LogGaussianDonut('full', sf_ecc_slope=.2, sf_ecc_intercept=.2,
                                       rel_mode_cardinals=.4, rel_mode_obliques=.1)
    full_model = model.LogGaussianDonut('full', sf_ecc_slope=.2, sf_ecc_intercept=.2,
                                        abs_mode_cardinals=.4, abs_mode_obliques=.1,
                                        rel_mode_cardinals=.4, rel_mode_obliques=.1)
    # we can't use the plotting.feature_df_plot / feature_df_polar_plot
    # functions because they use FacetGrids, each of which creates a
    # separate figure and we want all of this to be on one figure.
    fig = plt.figure(figsize=(15, 15))
    gs = mpl.gridspec.GridSpec(figure=fig, ncols=3, nrows=3)
    projs = ['rectilinear', 'polar']
    labels = [r'$p_1>p_2>0$', r'$p_3>p_4>0$', r'$p_1=p_3>p_2=p_4>0$']

    axes = []
    for i, m in enumerate([abs_model, rel_model, full_model]):
        model_axes = [fig.add_subplot(gs[i, j], projection=projs[j]) for j in range(2)]
        if i==0:
            title = True
        else:
            title = False
        model_axes = plotting.model_schematic(m, model_axes[:2], [(-.1, 4.2), (-.1, 3)], title)
        if i != 2:
            [ax.set(xlabel='') for ax in model_axes]
        model_axes[0].text(-.25, .5, labels[i], rotation=90, transform=model_axes[0].transAxes,
                           va='center', fontsize=15)
        axes.append(model_axes)
        

    # this needs to be created after the model plots so we can grab
    # their axes
    legend_axis = fig.add_subplot(gs[1, -1])
    legend_axis.legend(*axes[1][1].get_legend_handles_labels(), loc='center left')
    legend_axis.axis('off')

    return fig


def _catplot(df, x='subject', y='cv_loss', hue='fit_model_type', height=8, aspect=.75,
             ci=95, plot_kind='strip', x_rotate=True, **kwargs):
    """wrapper around seaborn.catplot

    several figures call seaborn.catplot and are pretty similar, so this
    function bundles a bunch of the stuff we do:
    1. determine the proper order for hue and x
    2. determine the proper palette for hue
    3. always use np.median as estimator and 'full' legend
    4. optionally rotate x-axis labels (and add extra room if so)
    5. add a horizontal line at the x-axis if we have both negative and
       positive values

    Parameters
    ----------
    df : pd.DataFrame
        pandas DataFrame
    x : str, optional
        which column of the df to plot on the x-axis
    y : str, optional
        which column of the df to plot on the y-axis
    hue : str, optional
        which column of the df to facet as the hue
    height : float, optional
        height of each plot facet
    aspect : float, optional
        aspect ratio of each facet
    ci : int, optional
        size of the confidence intervals (ignored if plot_kind=='strip')
    plot_kind : {'point', 'bar', 'strip', 'swarm', 'box', 'violin', or 'boxen'}, optional
        type of plot to make, i.e., sns.catplot's kind argument. see
        that functions docstring for more details. only 'point' and
        'strip' are expected, might do strange things otherwise
    x_rotate : bool or int, optional
        whether to rotate the x-axis labels or not. if True, we rotate
        by 25 degrees. if an int, we rotate by that many degrees. if
        False, we don't rotate. If labels are rotated, we'll also shift
        the bottom of the plot up to avoid cutting off the bottom.
    kwargs :
        passed to sns.catplot

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    df.fit_model_type = df.fit_model_type.map(dict(zip(plotting.MODEL_ORDER,
                                                       plotting.MODEL_PLOT_ORDER)))
    hue_order = plotting.get_order(hue, col_unique=df[hue].unique())
    order = plotting.get_order(x, col_unique=df[x].unique())
    pal = plotting.get_palette(hue, col_unique=df[hue].unique())
    if plot_kind == 'strip':
        # want the different hues to be in a consistent order on the
        # x-axis, which requires this
        kwargs.update({'jitter': False, 'dodge': True})
    g = sns.catplot(x, y, hue, data=df, hue_order=hue_order, legend='full', height=height,
                    kind=plot_kind, aspect=aspect, order=order, palette=pal, ci=ci,
                    estimator=np.median, **kwargs)
    for ax in g.axes.flatten():
        if x_rotate:
            if x_rotate is True:
                x_rotate = 25
            labels = ax.get_xticklabels()
            if labels:
                ax.set_xticklabels(labels, rotation=x_rotate)
        if (df[y] < 0).any() and (df[y] > 0).any():
            ax.axhline(color='grey', linestyle='dashed')
    if x_rotate:
        if x == 'subject':
            g.fig.subplots_adjust(bottom=.15)
        else:
            g.fig.subplots_adjust(bottom=.2)
    return g


def cross_validation_raw(df):
    """plot raw cross-validation loss

    This does no pre-processing of the df and plots subjects on the
    x-axis, model type as hue. (NOTE: this means if there are multiple
    scanning sessions for each subject, the plot will combine them,
    which is probably NOT what you want)

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing the output of the cross-validation
        analyses, combined across sessions (i.e., theo utput of
        combine_model_cv_summaries snakemake rule)

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    g = _catplot(df)
    g.fig.suptitle("Cross-validated loss across subjects")
    g.set(ylabel="Cross-validated loss", xlabel="Subject")
    g._legend.set_title("Model type")
    return g


def cross_validation_demeaned(df, remeaned=False):
    """plot demeaned cross-validation loss

    This function demeans the cross-validation loss on a
    subject-by-subject basis, then plots subjects on the x-axis, model
    type as hue. (NOTE: this means if there are multiple scanning
    sessions for each subject, the plot will combine them, which is
    probably NOT what you want)

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing the output of the cross-validation
        analyses, combined across sessions (i.e., theo utput of
        combine_model_cv_summaries snakemake rule)
    remeaned : bool, optional
        whether to use the demeaned cross-validation loss or the
        remeaned one. Remeaned has the mean across subjects added back
        to it, so that there won't be any negative y-values. This will
        only affect the values on the y-axis; the relative placements of
        the points will all be the same.

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    df = _demean_df(df)
    if remeaned:
        name = 'remeaned'
    else:
        name = 'demeaned'
    g = _catplot(df, y=f'{name}_cv_loss')
    g.fig.suptitle(f"{name.capitalize()} cross-validated loss across subjects")
    g.set(ylabel=f"Cross-validated loss ({name} by subject)", xlabel="Subject")
    g._legend.set_title("Model type")
    return g


def cross_validation_model(df, plot_kind='strip', remeaned=False):
    """plot demeaned cross-validation loss, as function of model type

    This function demeans the cross-validation loss on a
    subject-by-subject basis, then plots model type on the x-axis,
    subject as hue. (NOTE: this means if there are multiple scanning
    sessions for each subject, the plot will combine them, which is
    probably NOT what you want)

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing the output of the cross-validation
        analyses, combined across sessions (i.e., the output of
        combine_model_cv_summaries snakemake rule)
    plot_kind : {'strip', 'point'}, optional
        whether to create a strip plot (each subject as a separate
        point) or a point plot (combine across subjects, plotting the
        median and bootstrapped 95% CI)
    remeaned : bool, optional
        whether to use the demeaned cross-validation loss or the
        remeaned one. Remeaned has the mean across subjects added back
        to it, so that there won't be any negative y-values. This will
        only affect the values on the y-axis; the relative placements of
        the points (and the size of the error bars if
        `plot_kind='point'`) will all be the same.

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    df = _demean_df(df)
    if plot_kind == 'strip':
        hue = 'subject'
        legend_title = "Subject"
    elif plot_kind == 'point':
        hue = 'fit_model_type'
    if remeaned:
        name = 'remeaned'
    else:
        name = 'demeaned'
    g = _catplot(df, x='fit_model_type', y=f'{name}_cv_loss', hue=hue, plot_kind=plot_kind)
    g.fig.suptitle(f"{name.capitalize()} cross-validated loss across model types")
    g.set(ylabel=f"Cross-validated loss ({name} by subject)", xlabel="Model type")
    # if plot_kind=='point', then there is no legend, so the following
    # would cause an error
    if plot_kind == 'strip':
        g._legend.set_title(legend_title)
    return g


def model_types():
    """Create plot showing which model fits which parameters

    We have 11 different parameters, which might seem like a lot, so we
    do cross-validation to determine whether they're all necessary. This
    plot shows which parameters are fit by each model, in a little
    table.

    Returns
    -------
    fig : plt.Figure
        The figure with the plot on it

    """
    model_names = plotting.MODEL_PLOT_ORDER
    parameters = [r'$\sigma$', r'$a$', r'$b$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$', r'$A_1$',
                  r'$A_2$', r'$A_3$', r'$A_4$']
    model_variants = np.zeros((len(model_names), len(parameters))).astype(bool)
    # everyone fits sigma
    model_variants[:, 0] = True
    model_variants[1:, 1] = True
    model_variants[0, 2] = True
    model_variants[2:, 2] = True
    model_variants[3, [3, 4]] = True
    model_variants[4, [5, 6]] = True
    model_variants[5, [3, 4, 5, 6]] = True
    model_variants[6, [3, 4, 7, 8]] = True
    model_variants[7, [5, 6, 9, 10]] = True
    model_variants[8, 3:] = True
    model_variants = pd.DataFrame(model_variants, model_names, parameters)
    green, red = sns.color_palette('deep', 4)[2:]
    pal = sns.blend_palette([red, green])
    fig = plt.figure(figsize=(6, 5))
    ax = sns.heatmap(model_variants, cmap=pal, cbar=False)
    ax.set_yticklabels(model_names, rotation=0)
    return fig


def model_parameters(df, plot_kind='point', visual_field='all', fig=None, add_legend=True,
                     **kwargs):
    """plot model parameter values, across subjects

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing all the model parameter values, across
        subjects. note that this should first have gone through
        prep_model_df, which renames the values of the model_parameter
        columns so they're more pleasant to look at on the plot and adds
        a column, param_category, which enables us to break up the
        figure into three subplots
    plot_kind : {'point', 'strip', 'dist'}, optional
        What type of plot to make. If 'point' or 'strip', it's assumed
        that df contains only the fits to the median data across
        bootstraps (thus, one value per subject per parameter); if
        'dist', it's assumed that df contains the fits to all bootstraps
        (thus, 100 values per subject per parameter). this function
        should run if those are not true, but it will look weird:
        - 'point': point plot, so show 95% CI across subjects
        - 'strip': strip plot, so show each subject as a separate point
        - 'dist': distribution, show each each subject as a separate
          point with their own 68% CI across bootstraps
    visual_field : str, optional
        in addition to fitting the model across the whole visual field,
        we also fit the model to some portions of it (the left half,
        right half, etc). this arg allows us to easily modify the title
        of the plot to make it clear which portion of the visual field
        we're plotting. If 'all' (the default), we don't modify the
        title at all, otherwise we append "in {visual_field} visual
        field" to it.
    fig : plt.Figure or None, optional
        the figure to plot on. If None, we create a new figure. Intended
        use case for this is to plot the data from multiple sessions on
        the same axes (with different display kwargs), in order to
        directly compare how parameter values change.
    add_legend : bool, optional
        whether to add a legend or not. If True, will add just outside
        the right-most axis
    kwargs :
        Passed directly to the plotting function, which depends on the
        value of plot_kind

    Returns
    -------
    fig : plt.Figure
        Figure containin the plot

    """
    # in order to make the distance between the hues appear roughly
    # equivalent, need to set the ax_xlims in a particular way
    ax_xlims = [[-1, 1], [-1, 2], [-.5, 7.5]]
    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 10),
                                 gridspec_kw={'width_ratios': [.15, .3, .6]})
    else:
        axes = fig.axes
    order = plotting.get_order('model_parameter', col_unique=df.model_parameter.unique())
    if plot_kind == 'point':
        pal = plotting.get_palette('model_parameter', col_unique=df.model_parameter.unique(),
                                   as_dict=True)
    elif plot_kind == 'strip':
        pal = plotting.get_palette('subject', col_unique=df.subject.unique(), as_dict=True)
        hue_order = plotting.get_order('subject', col_unique=df.subject.unique())
    elif plot_kind == 'dist':
        # then we're showing this across subjects
        if 'subject' in df.columns and df.subject.nunique() > 1:
            pal = plotting.get_palette('subject', col_unique=df.subject.unique(), as_dict=True)
            hue_order = plotting.get_order('subject', col_unique=df.subject.unique())
            gb_col = 'subject'
            # copied from how seaborn's stripplot handles this, by looking
            # at lines 368 and 1190 in categorical.py (version 0.9.0)
            dodge = np.linspace(0, .8 - (.8 / df.subject.nunique()), df.subject.nunique())
            dodge -= dodge.mean()
        # else we've combined across all subjects
        else:
            pal = plotting.get_palette('model_parameter', col_unique=df.model_parameter.unique(),
                                       as_dict=True)
            gb_col = 'model_parameter'
            dodge = np.zeros(df.model_parameter.nunique())
    for i, ax in enumerate(axes):
        cat = ['sigma', 'eccen', 'orientation'][i]
        tmp = df.query("param_category==@cat")
        ax_order = [i for i in order if i in tmp.model_parameter.unique()]
        if plot_kind == 'point':
            sns.pointplot('model_parameter', 'fit_value', 'model_parameter', data=tmp,
                          estimator=np.median, ax=ax, order=ax_order, palette=pal, ci=95, **kwargs)
        elif plot_kind == 'strip':
            # want to make sure that the different hues end up in the
            # same order everytime, which requires doing this with
            # jitter and dodge
            sns.stripplot('model_parameter', 'fit_value', 'subject', data=tmp, ax=ax,
                          order=ax_order, palette=pal, hue_order=hue_order, jitter=False,
                          dodge=True, **kwargs)
        elif plot_kind == 'dist':
            handles, labels = [], []
            for j, (n, g) in enumerate(tmp.groupby(gb_col)):
                dots, _, _ = plotting.scatter_ci_dist('model_parameter', 'fit_value', data=g,
                                                      label=n, ax=ax, color=pal[n],
                                                      x_dodge=dodge[j], x_order=ax_order, **kwargs)
                handles.append(dots)
                labels.append(n)
        ax.set(xlim=ax_xlims[i])
        if ax.legend_:
            ax.legend_.remove()
        if i==2 and add_legend:
            if plot_kind == 'dist':
                legend = ax.legend(handles, labels, loc=(1.01, .3), borderaxespad=0, frameon=False)
            else:
                legend = ax.legend(loc=(1.01, .3), borderaxespad=0, frameon=False)
            # explicitly adding the legend artist allows us to add a
            # second legend if we want
            ax.add_artist(legend)
        ax.axhline(color='grey', linestyle='dashed')
        ax.set(ylabel='Fit value', xlabel='Parameter')
    suptitle = "Model parameters"
    if visual_field != 'all':
        suptitle += f' in {visual_field} visual field'
    fig.suptitle(suptitle)
    fig.subplots_adjust(top=.85)
    return fig


def model_parameters_pairplot(df, drop_outlier=False):
    """plot pairwise distribution of model parameters

    There's one very obvious outlier (sub-wlsubj007, ses-04, bootstrap
    41), where the $a$ parameter (sf_ecc_slope) is less than 0 (other
    parameters are also weird). If you want to drop that, set
    drop_outlier=True

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing all the model parameter values, across
        subjects. note that this should first have gone through
        prep_model_df, which renames the values of the model_parameter
        columns so they're more pleasant to look at on the plot
    drop_outlier : bool, optional
        whether to drop the outlier or not (see above)

    Returns
    -------
    g : sns.PairGrid
        the PairGrid containing the plot

    """
    pal = plotting.get_palette('subject', col_unique=df.subject.unique())
    pal = dict(zip(df.subject.unique(), pal))

    df = pd.pivot_table(df, index=['subject', 'bootstrap_num'], columns='model_parameter',
                        values='fit_value').reset_index()

    # this is a real outlier: one subject, one bootstrap (see docstring)
    if drop_outlier:
        df = df[df.get('$a$') > 0]

    g = sns.pairplot(df, hue='subject', vars=plotting.PLOT_PARAM_ORDER, palette=pal)
    for ax in g.axes.flatten():
        ax.axhline(color='grey', linestyle='dashed')
        ax.axvline(color='grey', linestyle='dashed')
    return g


def model_parameters_compare_plot(df, bootstrap_df):
    """plot comparison of model parameters from bootstrap vs median fits

    we have two different ways of fitting the data: to all of the
    bootstraps or just to the median across bootstraps. if we compare
    the resulting parameter values, they shouldn't be that different,
    which is what we do here.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing all the model parameter values, across
        subjects. note that this should first have gone through
        prep_model_df, which renames the values of the model_parameter
        columns so they're more pleasant to look at on the plot
    bootstrap_df : pd.DataFrame
        dataframe containing all the model parameter values, across
        subjects and bootstraps. note that this should first have gone
        through prep_model_df, which renames the values of the
        model_parameter columns so they're more pleasant to look at on
        the plot

    Returns
    -------
    g : sns.FacetGrid
        the FacetGrid containing the plot

    """
    pal = plotting.get_palette('subject', col_unique=df.subject.unique(), as_dict=True)
    order = plotting.get_order('subject', col_unique=df.subject.unique())
    compare_cols = ['model_parameter', 'subject', 'session', 'task']
    compare_df = df[compare_cols + ['fit_value']]
    tmp = bootstrap_df[compare_cols + ['fit_value']].rename(columns={'fit_value': 'fit_value_bs'})
    compare_df = pd.merge(tmp, compare_df, on=compare_cols)
    compare_df = compare_df.sort_values(compare_cols)
    g = sns.FacetGrid(compare_df, col='model_parameter', hue='subject', col_wrap=4, sharey=False,
                      aspect=2.5, height=3, col_order=plotting.PLOT_PARAM_ORDER, hue_order=order,
                      palette=pal)
    g.map_dataframe(plotting.scatter_ci_dist, 'subject', 'fit_value_bs')
    g.map_dataframe(plt.scatter, 'subject', 'fit_value')
    for ax in g.axes.flatten():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25)
    return g


def feature_df_plot(df, avg_across_retinal_angle=False, reference_frame='relative',
                    feature_type='pref-period', visual_field='all'):
    """plot model predictions based on parameter values

    This function is used to create plots showing the preferred period
    as a function of eccentricity, as given by the model. Right now, it
    always plots each subject separately, and will plot confidence
    intervals based on bootstraps if possible (i.e., if df contains the
    column 'bootstrap_num'). You can optionally average over the
    retinotopic angles or keep them separate, and you can plot the
    predictions for stimuli in the relative or absolute reference frame.

    This function converts the model paramter value df into the
    feature_df by calling analyze_model.create_feature_df. 

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing all the model parameter values, across
        subjects.
    avg_across_retinal_angle : bool, optional
        whether to average across the different retinotopic angles
        (True) or plot each of them on separate subplots (False). only
        relevant if feature_type=='pref-period' (others all plot
        something as function of retinotopic angle on polar plots)
    reference_frame : {'relative', 'absolute'}, optional
        whether the you want to plot the predictions for stimuli in the
        relative or absolute reference frame (i.e., annuli and pinwheels
        or constant gratings).
    feature_type : {'pref-period', 'pref-period-contour', 'iso-pref-period', 'max-amp'}
        what type of feature to create the plot for:
        - pref-period: plot preferred period as a function of
          eccentricity (on a Cartesian plot)
        - pref-period-contour: plot preferred period as a function of
          retinotopic angle at several different eccentricities (on a
          polar plot)
        - iso-pref-period: plot iso-preferred period lines as a function
          of retinotopic angle, for several different preferred periods
          (on a polar plot)
        - max-amp: plot max amplitude as a function of retinotopic angle
          (on a polar plot)
    visual_field : str, optional
        in addition to fitting the model across the whole visual field,
        we also fit the model to some portions of it (the left half,
        right half, etc). this arg allows us to easily modify the title
        of the plot to make it clear which portion of the visual field
        we're plotting. If 'all' (the default), we don't modify the
        title at all, otherwise we append "in {visual_field} visual
        field" to it.

    Returns
    -------
    g : sns.FacetGrid
        the FacetGrid containing the plot

    """
    kwargs = {'top': .9}
    if df.bootstrap_num.nunique() > 1:
        # then we have each subject's bootstraps, so we use
        # scatter_ci_dist to plot across them
        plot_func = plotting.scatter_ci_dist
        kwargs.update({'draw_ctr_pts': False, 'ci_mode': 'fill', 'join': True})
    else:
        plot_func = sns.lineplot
    if feature_type == 'pref-period':
        if avg_across_retinal_angle:
            pre_boot_gb_func = 'mean'
            row = None
        else:
            pre_boot_gb_func = None
            row = 'Retinotopic angle (rad)'
        df = analyze_model.create_feature_df(df, reference_frame=reference_frame)
        g = plotting.feature_df_plot(df, col='subject', row=row, pre_boot_gb_func=pre_boot_gb_func,
                                     plot_func=plot_func, **kwargs)
    else:
        kwargs.update({'hspace': .3, 'all_tick_labels': ['r']})
        if feature_type == 'pref-period-contour':
            df = analyze_model.create_feature_df(df, reference_frame=reference_frame,
                                                 eccentricity=[5],
                                                 retinotopic_angle=np.linspace(0, 2*np.pi, 49))
            g = plotting.feature_df_polar_plot(df, col='subject', row='Eccentricity (deg)',
                                               r='Preferred period (dpc)', plot_func=plot_func, **kwargs)
        elif feature_type == 'iso-pref-period':
            df = analyze_model.create_feature_df(df, 'preferred_period_contour', period_target=[1],
                                                 reference_frame=reference_frame)
            g = plotting.feature_df_polar_plot(df, col='subject', row='Preferred period (dpc)',
                                               plot_func=plot_func,
                                               title='Iso-preferred period contours', **kwargs)
        elif feature_type == 'max-amp':
            # this will have only one row, in which case we should use
            # the default value
            kwargs.update({'top': .76})
            df = analyze_model.create_feature_df(df, 'max_amplitude',
                                                 reference_frame=reference_frame)
            g = plotting.feature_df_polar_plot(df, col='subject', r='Max amplitude',
                                               plot_func=plot_func, title='Max amplitude', **kwargs)
        else:
            raise Exception(f"Don't know what to do with feature_type {feature_type}!")
    if visual_field != 'all':
        g.fig._suptitle.set_text(g.fig._suptitle.get_text() + f' in {visual_field} visual field')
    return g
