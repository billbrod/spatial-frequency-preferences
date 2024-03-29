#!/usr/bin/python
"""functions to create the figures for publication
"""
import seaborn as sns
import math
import pyrtools as pt
import neuropythy as ny
import os.path as op
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pandas as pd
import re
import itertools
from sklearn import linear_model
from . import summary_plots
from . import analyze_model
from . import plotting
from . import model
from . import utils
from . import first_level_analysis
from . import style


def create_precision_df(paths, summary_func=np.mean,
                        df_filter_string='drop_voxels_with_mean_negative_amplitudes,drop_voxels_near_border'):
    """Create dataframe summarizing subjects' precision

    When combining parameter estimates into an 'overall' value, we want
    to use the precision of each subject's data. To do that, we take the
    first level summary dfs (using regex to extract the subject,
    session, and task from the path) and call `summary_func` on the
    `precision` column. This gives us a single number, which we'll use
    when performing the precision-weighted mean

    df_filter_string can be used to filter the voxels we examine, so
    that we look only at those voxels that the model was fit to

    Parameters
    ----------
    paths : list
        list of strings giving the paths to the first level summary
        dfs.
    summary_func : callable, optional
        function we use to summarize the precision. Must take an array
        as its first input, not require any other inputs, and return a
        single value
    df_filter_string : str or None, optional
        a str specifying how to filter the voxels in the dataset. see
        the docstrings for sfp.model.FirstLevelDataset and
        sfp.model.construct_df_filter for more details. If None, we
        won't filter. Should probably use the default, which is what all
        models are trained using.

    Returns
    -------
    df : pd.DataFrame
        dataframe containing one row per (subject, session) pair, giving
        the precision for that scanning session. used to weight
        bootstraps

    """
    regex_names = ['subject', 'session', 'task']
    regexes = [r'(sub-[a-z0-9]+)', r'(ses-[a-z0-9]+)', r'(task-[a-z0-9]+)']
    df = []
    for p in paths:
        tmp = pd.read_csv(p)
        if df_filter_string is not None:
            df_filter = model.construct_df_filter(df_filter_string)
            tmp = df_filter(tmp).reset_index()
        val = summary_func(tmp.precision.values)
        if hasattr(val, '__len__') and len(val) > 1:
            raise Exception(f"summary_func {summary_func} returned more than one value!")
        data = {'precision': val}
        for n, regex in zip(regex_names, regexes):
            res = re.findall(regex, p)
            if len(set(res)) != 1:
                raise Exception(f"Unable to infer {n} from path {p}!")
            data[n] = res[0]
        df.append(pd.DataFrame(data, [0]))
    return pd.concat(df).reset_index(drop=True)


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
    - Preferred period (deg): the preferred period measured at this
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

    data_dict['Paper'].extend(["D'Souza (2016)"]*3)
    data_dict['Preferred spatial frequency (cpd)'].extend([2, .95, .4])
    data_dict['Eccentricity'].extend([1.4, 4.6, 9.8])

    data_dict['Paper'].extend(['Farivar (2017)']*2)
    data_dict['Preferred spatial frequency (cpd)'].extend([3, 1.5,])
    data_dict['Eccentricity'].extend([.5, 3])

    # model fit and never published, so don't include.
    # data_dict['Paper'].extend(['Olsson (pilot, model fit)']*10)
    # data_dict['Preferred spatial frequency (cpd)'].extend([2.11, 1.76, 1.47, 2.75, 1.24, 1.06, .88, .77, .66, .60])
    # data_dict['Eccentricity'].extend([2, 3, 4, 1, 5, 6, 7, 8, 9, 10])

    # these values gotten using web plot digitizer and then rounded to 2
    # decimal points
    data_dict["Paper"].extend(['Aghajari (2020)']*9)
    data_dict['Preferred spatial frequency (cpd)'].extend([2.24, 1.62, 1.26,
                                                           1.09, 0.88, 0.75,
                                                           0.78, 0.75, 0.70])
    data_dict['Eccentricity'].extend([0.68, 1.78, 2.84, 3.90, 5.00, 6.06, 7.16,
                                      8.22, 9.28])

    # Predictions of the scaling hypothesis -- currently unused
    # ecc = np.linspace(.01, 20, 50)
    # fovea_cutoff = 0
    # # two possibilities here
    # V1_RF_size = np.concatenate([np.ones(len(ecc[ecc<fovea_cutoff])),
    #                              np.linspace(1, 2.5, len(ecc[ecc>=fovea_cutoff]))])
    # V1_RF_size = .2 * ecc

    df = pd.DataFrame(data_dict)
    df = df.sort_values(['Paper', 'Eccentricity'])
    df["Preferred period (deg)"] = 1. / df['Preferred spatial frequency (cpd)']

    return df


def _demean_df(df, y='cv_loss', extra_cols=[]):
    """demean a column of the dataframe

    Calculate the mean of `y` across the values in the 'subject' and
    'loss_func' columns, then demean `y` and return df with several new
    columns:
    - `demeaned_{y}`: each y with `{y}_mean` subtracted off
    - `{y}_mean`: the average of y per subject per loss_func
    - `{y}_mean_overall`: the average of `{y}_mean` per loss_func
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
    y : str, optional
        the column to demean
    extra_cols : list, optionla
        list of columns to de/remean using the mean from `y`. for
        example, you might want to de/remean the noise_ceiling using the
        mean from the cross-validation loss

    Returns
    -------
    df : pd.DataFrame
        dataframe with new, demeaned column

    """
    gb_cols = ['subject', 'loss_func']
    df = df.set_index(gb_cols)
    y_mean = df.groupby(gb_cols)[y].mean()
    df[f'{y}_mean'] = y_mean
    # here we take the average over the averages. we do this so that we weight
    # all of the groups the same. For example, if gb_cols=['subject'] and one
    # subject had twice as many rows (because it had two sessions in df, for
    # example), then this ensures that subject isn't twice as important when
    # computing the mean (which would be the case if we used
    # df[f'{y}_mean'].mean() instead). We do, however, want to do this
    # separately for each loss function, since they'll probably have different
    # means
    df = df.reset_index()
    df = df.set_index('loss_func')
    df[f'{y}_mean_overall'] = y_mean.reset_index().groupby('loss_func')[y].mean()
    df[f'demeaned_{y}'] = df[y] - df[f'{y}_mean']
    df[f'remeaned_{y}'] = df[f'demeaned_{y}'] + df[f'{y}_mean_overall']
    for col in extra_cols:
        df[f'demeaned_{col}'] = df[col] - df[f'{y}_mean']
        df[f'remeaned_{col}'] = df[f'demeaned_{col}'] + df[f'{y}_mean_overall']
    return df.reset_index()


def prep_df(df, task, groupaverage=False):
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
    groupaverage : bool, optional
        whether to grab only the groupaverage subjects (if True) or
        every other subject (if False). Note that we'll grab/drop both
        i-linear and i-nearest if they're both present

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
    if 'fit_model_type' in df.columns:
        df.fit_model_type = df.fit_model_type.map(dict(zip(plotting.MODEL_ORDER,
                                                           plotting.MODEL_PLOT_ORDER)))
    if 'subject' in df.columns:
        df.subject = df.subject.map(dict(zip(plotting.SUBJECT_ORDER,
                                             plotting.SUBJECT_PLOT_ORDER)))
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


def precision_weighted_bootstrap(df, seed, n_bootstraps=100, col='preferred_period',
                                 gb_cols=['varea', 'stimulus_superclass', 'eccen'],
                                 precision_col='preferred_period_precision'):
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
    seed : int
        seed for numpy's RNG
    n_bootstraps : int, optional
        the number of independent bootstraps to draw
    col : str, optional
        the name of the column that contains the values we want to draw
        bootstraps for
    gb_cols : list, optional
        list of strs containing the columns we want to groupby. we will
        compute the bootstraps for each combination of values here.
    precision_col : str, optional
        name of the column that contains the precision, used in the
        precision-weighted mean

    Returns
    -------
    df : pd.DataFrame
        the df containing the bootstraps of precision-weighted
        mean. this will only contain the following columns: col,
        *gb_cols, and bootstrap_num

    """
    np.random.seed(seed)
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
            tmp[col] = np.average(t[col], weights=t[precision_col])
            tmp['bootstrap_num'] = j
            bootstraps.append(pd.DataFrame(tmp, [0]))
    bootstraps = pd.concat(bootstraps).reset_index(drop=True)
    if 'subject' in df.columns and 'subject' not in gb_cols:
        bootstraps['subject'] = 'all'
    return bootstraps


def _summarize_1d(df, reference_frame, y, row, col, height, facetgrid_legend,
                  **kwargs):
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
    kwargs.setdefault('xlim', (0, 12))
    g = summary_plots.main(df, row=row, col=col, y=y, eccen_range=(0, 11),
                           hue_order=hue_order, height=height,
                           plot_func=[plotting.plot_median_fit, plotting.plot_median_fit,
                                      plotting.scatter_ci_dist],
                           # these three end up being kwargs passed to the
                           # functions above, in order
                           x_jitter=[None, None, .2],
                           x_vals=[(0, 10.5), None, None],
                           linestyle=['--', None, None],
                           palette=pal, col_order=col_order,
                           row_order=row_order,
                           facetgrid_legend=facetgrid_legend, **kwargs)
    g.set_xlabels('Eccentricity (deg)')
    if facetgrid_legend:
        g._legend.set_title("Stimulus class")
    return g


def pref_period_1d(df, context='paper', reference_frame='relative',
                   row='session', col='subject', col_wrap=None, **kwargs):
    """Plot the preferred period of the 1d model fits.

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
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up
    reference_frame : {'relative', 'absolute'}, optional
        whether the data contained here is in the relative or absolute
        reference frame. this will determine both the palette used and
        the hue_order
    row : str, optional
        which column of the df to facet the plot's rows on
    col : str, optional
        which column of the df to facet the plot's column on
    kwargs :
        passed to sfp.figures._summarize_1d

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    # if we're wrapping columns, then we need this to take up the full width in
    # order for it to be readable
    if col_wrap is not None:
        fig_width = 'full'
    else:
        fig_width = 'half'
    params, fig_width = style.plotting_style(context, figsize=fig_width)
    if col_wrap is not None:
        fig_width /= col_wrap
        # there is, as of seaborn 0.11.0, a bug that interacts with our xtick
        # label size and height (see
        # https://github.com/mwaskom/seaborn/issues/2293), which causes an
        # issue if col_wrap == 3. this manual setting is about the same size
        # and fixes it
        if col_wrap == 3:
            fig_width = 2.23
    elif col is not None:
        fig_width /= df[col].nunique()
    plt.style.use(params)
    if context == 'paper':
        facetgrid_legend = False
        kwargs.setdefault('xlim', (0, 11.55))
        kwargs.setdefault('ylim', (0, 2.1))
    else:
        kwargs.setdefault('ylim', (0, 4))
        facetgrid_legend = True
    g = _summarize_1d(df, reference_frame, 'preferred_period', row, col,
                      fig_width, facetgrid_legend, col_wrap=col_wrap, **kwargs)
    g.set_ylabels('Preferred period (deg)')
    yticks = [i for i in range(4) if i <= kwargs['ylim'][1]]
    g.set(yticks=yticks)
    if context != 'paper':
        g.fig.suptitle("Preferred period of 1d tuning curves in each eccentricity band")
        g.fig.subplots_adjust(top=.85)
    else:
        if len(g.axes) == 1:
            # remove title if there's only one plot (otherwise it tells us which
            # subject is which)
            g.axes.flatten()[0].set_title('')
        for ax in g.axes.flatten():
            ax.axhline(color='gray', linestyle='--')
            ax.axvline(color='gray', linestyle='--')
            ax.set(xticks=[0, 2, 4, 6, 8, 10])
        g.fig.subplots_adjust(wspace=.05, hspace=.15)
    return g


def bandwidth_1d(df, context='paper', reference_frame='relative',
                 row='session', col='subject', units='octaves', **kwargs):
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
    units : {'octaves', 'degrees}, optional
        Whether to plot this data in octaves (in which case we expect it to be
        flat with eccentricity) or degrees (in which case we expect it to scale
        with eccentricity)
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up
    reference_frame : {'relative', 'absolute'}, optional
        whether the data contained here is in the relative or absolute
        reference frame. this will determine both the palette used and
        the hue_order
    row : str, optional
        which column of the df to facet the plot's rows on
    col : str, optional
        which column of the df to facet the plot's column on
    kwargs :
        passed to sfp.figures._summarize_1d

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    params, fig_width = style.plotting_style(context, figsize='half')
    plt.style.use(params)
    if context == 'paper':
        facetgrid_legend = False
        kwargs.setdefault('xlim', (0, 11.55))
    else:
        facetgrid_legend = True
    if units == 'degrees':
        if 'tuning_curve_bandwidth_degrees' not in df.columns:
            df['tuning_curve_bandwidth_degrees'] = df.apply(utils._octave_to_degrees, 1)
        y = 'tuning_curve_bandwidth_degrees'
    elif units == 'octaves':
        y = 'tuning_curve_bandwidth'
        kwargs.setdefault('ylim', (0, 8))
    g = _summarize_1d(df, reference_frame, y, row, col,
                      fig_width, facetgrid_legend, **kwargs)
    g.set_ylabels(f'Tuning curve FWHM ({units})')
    if context != 'paper':
        g.fig.suptitle("Full-Width Half-Max of 1d tuning curves in each eccentricity band")
        g.fig.subplots_adjust(top=.85)
    elif len(g.axes) == 1:
        # remove title if there's only one plot (otherwise it tells us which
        # subject is which)
        g.axes.flatten()[0].set_title('')
    return g


def existing_studies_figure(df, y="Preferred period (deg)", legend=True, context='paper'):
    """Plot the results from existing studies

    See the docstring for figures.existing_studies_df() for more
    details on the information displayed in this figure.

    Parameters
    ----------
    df : pd.DataFrame
        The existing studies df, as returned by the function
        figures.existing_studies_df().
    y : {'Preferred period (deg)', 'Preferred spatial frequency (cpd)'}
        Whether to plot the preferred period or preferred spatial
        frequency on the y-axis. If preferred period, the y-axis is
        linear; if preferred SF, the y-axis is log-scaled (base 2). The
        ylims will also differ between these two
    legend : bool, optional
        Whether to add a legend or not
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up

    Returns
    -------
    g : sns.FacetGrid
        The FacetGrid containing the plot

    """
    params, fig_width = style.plotting_style(context, figsize='half')
    plt.style.use(params)
    fig_height = fig_width / 1.2
    pal = sns.color_palette('Set2', df.Paper.nunique())
    pal = dict(zip(df.Paper.unique(), pal))
    if 'Current study' in df.Paper.unique():
        pal['Current study'] = (0, 0, 0)
    g = sns.FacetGrid(df, hue='Paper', height=fig_height, aspect=1.2, palette=pal)
    if y == "Preferred period (deg)":
        g.map(plt.plot, 'Eccentricity', y, marker='o')
        g.ax.set_ylim((0, 6))
    elif y == "Preferred spatial frequency (cpd)":
        g.map(plt.semilogy, 'Eccentricity', y, marker='o', basey=2)
        g.ax.set_ylim((0, 11))
        g.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(plotting.myLogFormat))
    g.ax.set_xlim((0, 20))
    if context == 'poster':
        g.ax.set(xticks=[0, 5, 10, 15, 20])
        g.ax.set_title("Summary of human V1 fMRI results")
    if legend:
        g.add_legend()
        # facetgrid doesn't let us set the title fontsize directly, so need to do
        # this hacky work-around
        g.fig.legends[0].get_title().set_size(mpl.rcParams['legend.title_fontsize'])
    g.ax.set_xlabel('Eccentricity of receptive field center (deg)')
    return g


def input_schematic(context='paper', prf_loc=(250, 250), prf_radius=100,
                    stim_freq=(.01, .03)):
    """Schematic to explain 2d model inputs.

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
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up
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
    params, fig_width = style.plotting_style(context, figsize='half')
    plt.style.use(params)
    figsize = (fig_width, fig_width)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    def get_xy(distance, angle, origin=(500, 500)):
        return [o + distance * func(angle) for o, func in
                zip(origin, [np.cos, np.sin])]

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
    prf = mpl.patches.Circle(abs_prf_loc, prf_radius, fc='none', ec='k', linewidth=2,
                             linestyle='--', zorder=10)
    ax.add_artist(prf)
    prf_ecc = np.sqrt(np.square(prf_loc).sum())
    prf_angle = np.arctan2(*prf_loc[::-1])
    e_loc = get_xy(prf_ecc/2, prf_angle + np.pi/13)
    plotting.draw_arrow(ax, (500, 500), abs_prf_loc, arrowprops={'connectionstyle': 'arc3',
                                                                 'arrowstyle': '<-',
                                                                 'color': pal[1]})
    ax.text(*e_loc, r'$r_v$')
    ax.text(600, 500 + 100*np.sin(prf_angle/2), r'$\theta_v$')
    angle = mpl.patches.Arc((500, 500), 200, 200, 0, 0, np.rad2deg(prf_angle),
                            fc='none', ec=pal[1], linestyle='-')
    ax.add_artist(angle)
    # so that this is the normal vector, the 7000 is just an arbitrary
    # scale factor to make the vector a reasonable length
    normal_len = 7000 * np.sqrt(np.square(stim_freq).sum())
    normal_angle = np.arctan2(*stim_freq[::-1])
    omega_loc = get_xy(normal_len, normal_angle, abs_prf_loc)
    plotting.draw_arrow(ax, abs_prf_loc, omega_loc, r'$\omega_l$', {'connectionstyle': 'arc3',
                                                                  'arrowstyle': '<-',
                                                                  'color': pal[0]})
    angle = mpl.patches.Arc(abs_prf_loc, 1.2*normal_len, 1.2*normal_len, 0, 0,
                            # small adjustment appears to be necessary for some
                            # reason -- but really only for some spatial
                            # frequencies.
                            np.rad2deg(normal_angle)-3,
                            fc='none', ec=pal[0], linestyle='-')
    ax.add_artist(angle)
    plotting.draw_arrow(ax, (abs_prf_loc[0] + normal_len, abs_prf_loc[1]), abs_prf_loc,
                        arrowprops={'connectionstyle': 'angle3', 'arrowstyle': '-', 'color': '.5',
                                    'linestyle': ':'})
    theta_loc = get_xy(1.3*normal_len/2, normal_angle/2, abs_prf_loc)
    ax.text(*theta_loc, r'$\theta_l$')
    return fig


def model_schematic(context='paper'):
    """Create model schematic.

    In order to better explain the model, its predictions, and the
    effects of its parameters, we create a model schematic that shows
    the effects of the different p parameters (those that control the
    effect of stimulus orientation and retinotopic angle on preferred
    period).

    This creates only the polar plots (showing the preferred period contours),
    and doesn't have a legend; it's intended that you call
    compose_figures.add_legend to add the graphical one (and a space has been
    left for it)

    Parameters
    ----------
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up

    Returns
    -------
    fig : plt.Figure
        Figure containing the schematic

    """
    params, fig_width = style.plotting_style(context, figsize='half')
    plt.style.use(params)
    figsize = (fig_width, fig_width/3)
    if context == 'paper':
        orientation = np.linspace(0, np.pi, 4, endpoint=False)
    elif context == 'poster':
        orientation = np.linspace(0, np.pi, 2, endpoint=False)
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
    fig, axes = plt.subplots(1, 3, figsize=figsize,
                             subplot_kw={'projection': 'polar'})
    labels = [r'$p_1>p_2>0$', r'$p_3>p_4>0$',
              # can't have a newline in a raw string, so have to combine them
              # in the last label here
              r'$p_1=p_3>$'+'\n'+r'$p_2=p_4>0$']

    for i, (m, ax) in enumerate(zip([abs_model, rel_model, full_model], axes)):
        plotting.model_schematic(m, [ax], [(-.1, 3)], False,
                                 orientation=orientation)
        if i != 0:
            ax.set(ylabel='')
        ax.set(xlabel='')
        ax.set_title(labels[i])
        ax.set(xticklabels=[], yticklabels=[])

    fig.subplots_adjust(wspace=.075)

    return fig


def model_schematic_large(context='paper'):
    """Create larger version of model schematic.

    In order to better explain the model, its predictions, and the
    effects of its parameters, we create a model schematic that shows
    the effects of the different p parameters (those that control the
    effect of stimulus orientation and retinotopic angle on preferred
    period).

    Note that this includes both linear and polar plots, and will probably be
    way too large

    Parameters
    ----------
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up

    Returns
    -------
    fig : plt.Figure
        Figure containing the schematic

    """
    if context == 'paper':
        orientation = np.linspace(0, np.pi, 4, endpoint=False)
        size_scale = 1
    elif context == 'poster':
        size_scale = 1.5
        orientation = np.linspace(0, np.pi, 2, endpoint=False)
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
    fig = plt.figure(figsize=(size_scale*15, size_scale*15))
    gs = mpl.gridspec.GridSpec(figure=fig, ncols=3, nrows=3)
    projs = ['rectilinear', 'polar']
    labels = [r'$p_1>p_2>0$', r'$p_3>p_4>0$', r'$p_1=p_3>p_2=p_4>0$']

    axes = []
    for i, m in enumerate([abs_model, rel_model, full_model]):
        model_axes = [fig.add_subplot(gs[i, j], projection=projs[j]) for j in range(2)]
        if i == 0:
            title = True
        else:
            title = False
        model_axes = plotting.model_schematic(m, model_axes[:2], [(-.1, 4.2), (-.1, 3)], title,
                                              orientation=orientation)
        if i != 2:
            [ax.set(xlabel='') for ax in model_axes]
        model_axes[0].text(size_scale*-.25, .5, labels[i], rotation=90,
                           transform=model_axes[0].transAxes, va='center',
                           fontsize=1.5*mpl.rcParams['font.size'])
        axes.append(model_axes)

    # this needs to be created after the model plots so we can grab
    # their axes
    legend_axis = fig.add_subplot(gs[1, -1])
    legend_axis.legend(*axes[1][1].get_legend_handles_labels(), loc='center left')
    legend_axis.axis('off')

    return fig


def _catplot(df, x='subject', y='cv_loss', hue='fit_model_type', height=8, aspect=.9,
             ci=68, plot_kind='strip', x_rotate=False, legend='full', orient='v', **kwargs):
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
    legend : str or bool, optional
        the legend arg to pass through to seaborn.catplot, see its
        docstrings for more details
    orient : {'h', 'v'}, optional
        orientation of plot (horizontal or vertical)
    kwargs :
        passed to sns.catplot

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    hue_order = plotting.get_order(hue, col_unique=df[hue].unique())
    if 'order' in kwargs.keys():
        order = kwargs.pop('order')
    else:
        order = plotting.get_order(x, col_unique=df[x].unique())
    pal = plotting.get_palette(hue, col_unique=df[hue].unique(),
                               doubleup='doubleup' in x)
    if plot_kind == 'strip':
        # want the different hues to be in a consistent order on the
        # x-axis, which requires this
        kwargs.update({'jitter': False, 'dodge': True})
    if orient == 'h':
        x_copy = x
        x = y
        y = x_copy
        aspect = 1/aspect
        kwargs['sharex'] = False
    else:
        kwargs['sharey'] = False
    if 'dodge' not in kwargs.keys():
        kwargs['dodge'] = 0
    # facetgrid seems to ignore the defaults for these, but we want to use them
    # so its consistent with other figures
    gridspec_kws = {k: mpl.rcParams[f'figure.subplot.{k}']
                    for k in ['top', 'bottom', 'left', 'right']}
    g = sns.catplot(x, y, hue, data=df, hue_order=hue_order, legend=legend, height=height,
                    kind=plot_kind, aspect=aspect, order=order, palette=pal, ci=ci,
                    estimator=np.median, orient=orient, facet_kws={'gridspec_kws': gridspec_kws},
                    **kwargs)
    for ax in g.axes.flatten():
        if x_rotate:
            if x_rotate is True:
                x_rotate = 25
            labels = ax.get_xticklabels()
            if labels:
                ax.set_xticklabels(labels, rotation=x_rotate, ha='right')
        if orient == 'v':
            if (df[y] < 0).any() and (df[y] > 0).any():
                ax.axhline(color='grey', linestyle='dashed')
        else:
            if (df[x] < 0).any() and (df[x] > 0).any():
                ax.axvline(color='grey', linestyle='dashed')
    if x_rotate:
        if x == 'subject':
            g.fig.subplots_adjust(bottom=.15)
        else:
            g.fig.subplots_adjust(bottom=.2)
    return g


def cross_validation_raw(df, seed, noise_ceiling_df=None, orient='v', context='paper'):
    """plot raw cross-validation loss

    This does no pre-processing of the df and plots subjects on the
    x-axis, model type as hue. (NOTE: this means if there are multiple
    scanning sessions for each subject, the plot will combine them,
    which is probably NOT what you want)

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing the output of the cross-validation
        analyses, combined across sessions (i.e., the output of
        combine_model_cv_summaries snakemake rule)
    seed : int
        seed for numpy's RNG
    noise_ceiling_df : pd.DataFrame
        dataframe containing the results of the noise ceiling analyses
        for all subjects (i.e., the output of the
        noise_ceiling_monte_carlo_overall rule)
    orient : {'h', 'v'}, optional
        orientation of plot (horizontal or vertical)
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    np.random.seed(seed)
    height = 8
    aspect = .9
    s = 5
    if context == 'poster':
        height *= 2
        aspect = 1
        s *= 2
    if noise_ceiling_df is not None:
        merge_cols = ['subject', 'mat_type', 'atlas_type', 'session', 'task', 'vareas', 'eccen']
        df = pd.merge(df, noise_ceiling_df, 'outer', on=merge_cols, suffixes=['_cv', '_noise'])
    g = _catplot(df.query('loss_func in ["weighted_normed_loss", "normed_loss", "cosine_distance_scaled"]'),
                 legend=False, height=height, s=s, x_rotate=True, orient=orient,
                 col='loss_func')
    if noise_ceiling_df is not None:
        g.map_dataframe(plotting.plot_noise_ceiling, 'subject', 'loss')
    g.fig.suptitle("Cross-validated loss across subjects")
    if orient == 'v':
        g.set(ylabel="Cross-validated loss", xlabel="Subject")
    elif orient == 'h':
        g.set(xlabel="Cross-validated loss", ylabel="Subject")
    g.add_legend()
    g._legend.set_title("Model type")
    ylims = [(0, .06), (0, .0022), (0, .0022)]
    for i, ax in enumerate(g.axes.flatten()):
        ax.set(ylim=ylims[i])
    return g


def cross_validation_demeaned(df, seed, remeaned=False, orient='v', context='paper'):
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
        analyses, combined across sessions (i.e., the output of
        combine_model_cv_summaries snakemake rule)
    seed : int
        seed for numpy's RNG
    remeaned : bool, optional
        whether to use the demeaned cross-validation loss or the
        remeaned one. Remeaned has the mean across subjects added back
        to it, so that there won't be any negative y-values. This will
        only affect the values on the y-axis; the relative placements of
        the points will all be the same.
    orient : {'h', 'v'}, optional
        orientation of plot (horizontal or vertical)
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    np.random.seed(seed)
    height = 8
    aspect = .9
    if context == 'poster':
        height *= 2
        aspect = 1
    df = _demean_df(df)
    if remeaned:
        name = 'remeaned'
    else:
        name = 'demeaned'
    g = _catplot(df, y=f'{name}_cv_loss', height=height, aspect=aspect, x_rotate=True,
                 orient=orient, col='loss_func')
    g.fig.suptitle(f"{name.capitalize()} cross-validated loss across subjects")
    if orient == 'v':
        g.set(ylabel=f"Cross-validated loss ({name} by subject)", xlabel="Subject")
    elif orient == 'h':
        g.set(xlabel=f"Cross-validated loss ({name} by subject)", ylabel="Subject")
    g._legend.set_title("Model type")
    return g


def cross_validation_model(df, seed, plot_kind='strip', remeaned=False, noise_ceiling_df=None,
                           orient='v', sort=False, doubleup=False, context='paper'):
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
    seed : int
        seed for numpy's RNG
    plot_kind : {'strip', 'point'}, optional
        whether to create a strip plot (each subject as a separate
        point) or a point plot (combine across subjects, plotting the
        median and bootstrapped 68% CI)
    remeaned : bool, optional
        whether to use the demeaned cross-validation loss or the
        remeaned one. Remeaned has the mean across subjects added back
        to it, so that there won't be any negative y-values. This will
        only affect the values on the y-axis; the relative placements of
        the points (and the size of the error bars if
        `plot_kind='point'`) will all be the same.
    noise_ceiling_df : pd.DataFrame
        dataframe containing the results of the noise ceiling analyses
        for all subjects (i.e., the output of the
        noise_ceiling_monte_carlo_overall rule)
    orient : {'h', 'v'}, optional
        orientation of plot (horizontal or vertical)
    sort : bool, optional
        whether to sort the models by the median loss of the
        weighted_normed_loss or show them in numbered order
    doubleup : bool, optional
        whether to "double-up" models so that we plot two models on the same
        row if they're identical except for fitting A3/A4. this then shows the
        version fitting A3/A4 as a fainter color of the version that doesn't.
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    kwargs = {}
    np.random.seed(seed)
    params, fig_width = style.plotting_style(context, figsize='half')
    plt.style.use(params)
    if doubleup:
        height = fig_width * .855
    else:
        height = fig_width
    aspect = 1
    if noise_ceiling_df is not None:
        merge_cols = ['subject', 'mat_type', 'atlas_type', 'session', 'task', 'vareas', 'eccen']
        noise_ceiling_df = noise_ceiling_df.groupby(merge_cols).median().reset_index()
        df = pd.merge(df, noise_ceiling_df, 'inner', on=merge_cols, suffixes=['_cv', '_noise'])
        extra_cols = ['loss']
    else:
        extra_cols = []
    df = _demean_df(df, extra_cols=extra_cols)
    if plot_kind == 'strip':
        hue = 'subject'
        legend_title = "Subject"
        legend = 'full'
    elif plot_kind == 'point':
        hue = 'fit_model_type'
        legend = False
    if remeaned:
        name = 'remeaned'
    else:
        name = 'demeaned'
    if sort:
        gb = df.query("loss_func == 'weighted_normed_loss'").groupby('fit_model_type')
        kwargs['order'] = gb[f'{name}_cv_loss'].median().sort_values(ascending=False).index
    if doubleup:
        df['fit_model_doubleup'] = df.fit_model_type.map(dict(zip(plotting.MODEL_PLOT_ORDER,
                                                                  plotting.MODEL_PLOT_ORDER_DOUBLEUP)))
        x = 'fit_model_doubleup'
        if noise_ceiling_df is not None:
            nc_map = {k: k for k in range(1, 8)}
            nc_map.update({10: 8, 12: 9})
            df['fit_model_nc'] = df.fit_model_doubleup.map(nc_map)
    else:
        x = 'fit_model_type'
        if noise_ceiling_df is not None:
            df['fit_model_nc'] = df.fit_model_type
    g = _catplot(df, x=x, y=f'{name}_cv_loss', hue=hue,
                 col='loss_func', plot_kind=plot_kind, height=height,
                 aspect=aspect, orient=orient, legend=legend, **kwargs)
    title = f"{name.capitalize()} cross-validated loss across model types"
    if noise_ceiling_df is not None:
        g.map_dataframe(plotting.plot_noise_ceiling, 'fit_model_nc', f'{name}_loss', ci=0,
                        orient=orient)
        title += "\n Median noise ceiling shown as blue line"
    if orient == 'v':
        g.set(ylabel=f"Cross-validated loss ({name} by subject)", xlabel="Model type")
    elif orient == 'h':
        g.set(xlabel=f"Cross-validated loss ({name} by subject)", ylabel="")
    # if plot_kind=='point', then there is no legend, so the following
    # would cause an error
    if plot_kind == 'strip':
        g._legend.set_title(legend_title)
    # don't want title in the paper version
    if context != 'paper':
        g.fig.suptitle(title)
    else:
        if orient == 'h':
            # also want to remove the y axis, since it's duplicating the one from
            # the other figure
            for ax in g.axes.flatten():
                ax.yaxis.set_visible(False)
                ax.spines['left'].set_visible(False)
                if plot_kind == 'point':
                    # this way, the ylims line up whether or not we plotted the
                    # noise ceiling line
                    if doubleup:
                        ax.set_ylim((8.5, -0.5))
                    else:
                        ax.set_ylim((13.5, -0.5))
    return g


def model_types(context='paper', palette_type='model', annotate=False,
                order=None, doubleup=False):
    """Create plot showing which model fits which parameters.

    We have 11 different parameters, which might seem like a lot, so we
    do cross-validation to determine whether they're all necessary. This
    plot shows which parameters are fit by each model, in a little
    table.

    Parameters
    ----------
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up
    palette_type : {'model', 'simple', 'simple_r', seaborn palette name}, optional
        palette to use for this plot. if 'model', the parameter each
        model fits is shown in its color (as used in other plots). If
        'simple' or 'simple_r', we'll use a white/black colormap with
        either black (if 'simple') or white (if 'simple_r') showing the
        parameter is fit. Else, should be a str giving a seaborn palette
        name, i.e., an arg that can be passed to seaborn.color_palette.
    annotate : bool, optional
        whether to annotate the schematic with info on the parameter
        categories (e.g., period/amplitude, eccentricity/orientation,
        etc)
    order : pandas index or None, optional
        If None, we plot the models in the default order. Else, should be an
        index object that gives the order to plot them in (from top to bottom).

    Returns
    -------
    fig : plt.Figure
        The figure with the plot on it

    """
    params, fig_width = style.plotting_style(context, figsize='half')
    # these ticks don't add anything and are confusing
    params['xtick.bottom'] = False
    params['ytick.left'] = False
    plt.style.use(params)
    figsize = (fig_width, fig_width)
    extra_space = 0
    model_names = plotting.MODEL_PLOT_ORDER
    parameters = plotting.PLOT_PARAM_ORDER
    model_variants = np.zeros((len(model_names), len(parameters)))
    if palette_type == 'model':
        pal = plotting.get_palette('fit_model_type', col_unique=model_names,
                                   doubleup=doubleup)
        try:
            pal = pal.tolist()
        except AttributeError:
            # then it's already a list
            pass
        pal = [(1, 1, 1)] + pal
        fill_vals = dict(zip(range(len(model_names)), range(1, len(model_names)+1)))
    else:
        if palette_type.startswith('simple'):
            black, white = [(0, 0, 0), (1, 1, 1)]
            if palette_type.endswith('_r'):
                pal = [black, white]
            else:
                pal = [white, black]
        else:
            pal = sns.color_palette(palette_type, 2)
        fill_vals = dict(zip(range(len(model_names)), len(model_names) * [True]))
    if not doubleup:
        model_variants[0, [0, 2]] = fill_vals[0]
        model_variants[1, [0, 1]] = fill_vals[1]
        model_variants[2, [0, 1, 2]] = fill_vals[2]
        model_variants[3, [0, 1, 2, 3, 4]] = fill_vals[3]
        model_variants[4, [0, 1, 2, 5, 6]] = fill_vals[4]
        model_variants[5, [0, 1, 2, 3, 4, 5, 6]] = fill_vals[5]
        model_variants[6, [0, 1, 2, 7, 8]] = fill_vals[6]
        model_variants[7, [0, 1, 2, 9, 10]] = fill_vals[7]
        model_variants[8, [0, 1, 2, 7, 8, 9, 10]] = fill_vals[8]
        model_variants[9, [0, 1, 2, 3, 4, 7, 8]] = fill_vals[9]
        model_variants[10, [0, 1, 2, 5, 6, 9, 10]] = fill_vals[10]
        model_variants[11, [0, 1, 2, 3, 4, 5, 6, 7, 8]] = fill_vals[11]
        model_variants[12, [0, 1, 2, 3, 4, 5, 6, 9, 10]] = fill_vals[12]
        model_variants[13, :] = fill_vals[13]
        # while in theory, we want square to be True here too, we messed with
        # all the size in such a way that it works with it set to False
        square = False
    else:
        model_variants[0, [0, 2]] = fill_vals[0]
        model_variants[1, [0, 1]] = fill_vals[1]
        model_variants[2, [0, 1, 2]] = fill_vals[2]
        model_variants[3, [0, 1, 2, 3, 4]] = fill_vals[3]
        model_variants[4, [0, 1, 2, 5, 6]] = fill_vals[4]
        model_variants[5, [0, 1, 2, 3, 4, 5, 6]] = fill_vals[5]
        model_variants[6, [0, 1, 2, 7, 8]] = fill_vals[6]
        model_variants[2, [9, 10]] = fill_vals[7]
        model_variants[6, [9, 10]] = fill_vals[8]
        model_variants[9, [0, 1, 2, 3, 4, 7, 8]] = fill_vals[9]
        model_variants[4, [9, 10]] = fill_vals[10]
        model_variants[11, [0, 1, 2, 3, 4, 5, 6, 7, 8]] = fill_vals[11]
        model_variants[5, [9, 10]] = fill_vals[12]
        model_variants[11, [9, 10]] = fill_vals[13]
        # drop the rows that are all 0s
        model_variants = model_variants[~(model_variants==0).all(1)]
        warnings.warn("when doubling-up, we just use sequential numbers for models "
                      "(the numbers therefore have a different meaning than for "
                      "non-doubled-up version)")
        model_names = np.arange(1, model_variants.shape[0]+1)
        square = True
    model_variants = pd.DataFrame(model_variants, model_names, parameters)
    if order is not None:
        model_variants = model_variants.reindex(order)
    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(model_variants, cmap=pal, cbar=False, square=square)
    ax.set_yticklabels(model_variants.index, rotation=0)
    ax.set_ylabel("Model type")
    # we want the labels on the top here, not the bottom
    ax.tick_params(labelbottom=False, labeltop=True, pad=-2)
    if annotate:
        arrowprops = {'connectionstyle': 'bar', 'arrowstyle': '-', 'color': '0'}
        text = ['Eccentricity', 'Absolute', 'Relative', 'Absolute', 'Relative']
        text = ['Ecc', 'Abs', 'Rel', 'Abs', 'Rel']
        for i, pos in enumerate(range(1, 10, 2)):
            plotting.draw_arrow(ax, ((pos+.5)/11, 1.08+extra_space),
                                ((pos+1.5)/11, 1.08+extra_space), arrowprops=arrowprops,
                                xycoords='axes fraction', textcoords='axes fraction')
            ax.text((pos+1)/11, 1.11+extra_space, text[i], transform=ax.transAxes,
                    ha='center', va='bottom')
        arrowprops['connectionstyle'] = f'bar,fraction={.3/5}'
        plotting.draw_arrow(ax, (1.5/11, 1.17+extra_space), (6.5/11, 1.17+extra_space),
                            arrowprops=arrowprops,
                            xycoords='axes fraction', textcoords='axes fraction')
        ax.text(4/11, 1.22+extra_space, 'Period', transform=ax.transAxes,
                ha='center', va='bottom')
        arrowprops['connectionstyle'] = f'bar,fraction={.3/3}'
        plotting.draw_arrow(ax, (7.5/11, 1.17+extra_space), (10.5/11, 1.17+extra_space),
                            arrowprops=arrowprops,
                            xycoords='axes fraction', textcoords='axes fraction')
        ax.text(9/11, 1.22+extra_space, 'Amplitude', transform=ax.transAxes,
                ha='center', va='bottom')
    return fig


def model_parameters(df, plot_kind='point', visual_field='all', fig=None, add_legend=True,
                     context='paper', **kwargs):
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
        - 'point': point plot, so show 68% CI across subjects
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
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up
    kwargs :
        Passed directly to the plotting function, which depends on the
        value of plot_kind

    Returns
    -------
    fig : plt.Figure
        Figure containin the plot

    """
    params, fig_width = style.plotting_style(context, figsize='full')
    plt.style.use(params)
    # in order to make the distance between the hues appear roughly
    # equivalent, need to set the ax_xlims in a particular way
    n_ori_params = df.query("param_category=='orientation'").model_parameter.nunique()
    ax_xlims = [[-.5, .5], [-.5, 1.5], [-.5, n_ori_params - .5]]
    yticks = [[0, .5, 1, 1.5, 2, 2.5], [0, .1, .2, .3, .4], [-.03, 0, .03, .06, .09]]
    axhline = [2]
    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_width/2),
                                 gridspec_kw={'width_ratios': [.12, .25, .63],
                                              'wspace': .3})
    else:
        axes = fig.axes
    order = plotting.get_order('model_parameter', col_unique=df.model_parameter.unique())
    if plot_kind == 'point':
        pal = plotting.get_palette('model_parameter', col_unique=df.model_parameter.unique(),
                                   as_dict=True)
    elif plot_kind == 'strip':
        # then we're showing this across subjects
        if 'subject' in df.columns and df.subject.nunique() > 1:
            hue = 'subject'
        # this is sub-groupaverage
        else:
            hue = 'groupaverage_seed'
        pal = plotting.get_palette(hue, col_unique=df[hue].unique(), as_dict=True)
        hue_order = plotting.get_order(hue, col_unique=df[hue].unique())
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
            yticks = [[0, .5, 1, 1.5, 2, 2.5, 3.0],
                      [-.1, 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
                      [-.2, -.1, 0, .1, .2, .3]]
            ax_xlims = [[-1, 1], [-1, 2], [-.75, n_ori_params-.5]]
            axhline += [1]
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
                          estimator=np.median, ax=ax, order=ax_order, palette=pal, ci=68, **kwargs)
        elif plot_kind == 'strip':
            # want to make sure that the different hues end up in the
            # same order everytime, which requires doing this with
            # jitter and dodge
            sns.stripplot('model_parameter', 'fit_value', hue, data=tmp, ax=ax,
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
        ax.set(xlim=ax_xlims[i], yticks=yticks[i])
        ax.tick_params(pad=0)
        if ax.legend_:
            ax.legend_.remove()
        if i == 2:
            if add_legend:
                if plot_kind == 'dist':
                    legend = ax.legend(handles, labels, loc='lower center', ncol=3,
                                       borderaxespad=0, frameon=False,
                                       bbox_to_anchor=(.49, -.3), bbox_transform=fig.transFigure)
                else:
                    legend = ax.legend(loc=(1.01, .3), borderaxespad=0, frameon=False)
                # explicitly adding the legend artist allows us to add a
                # second legend if we want
                ax.add_artist(legend)
        if i in axhline:
            ax.axhline(color='grey', linestyle='dashed')
        if i == 0:
            ax.set(ylabel='Parameter value')
    fig.text(.5, 0, "Parameter", ha='center')
    if context != 'paper':
        # don't want title in paper context
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
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    return g


def training_loss_check(df, hue='test_subset', thresh=.2):
    """check last epoch training loss

    in order to check that one of the models didn't get stuck in a local
    optimum in, e.g., one of the cross-validation folds or bootstraps,
    we here plot the loss for each subject and model, with median and
    68% CI across batches. they should hopefully look basically all the
    same

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with the last epoch loss, as created by
        `analyze_model.collect_final_loss`
    hue : str, optional
        which df column to use as the hue arg for the FacetGrid
    thresh : float, optional
        the loss threshold for getting stuck in local optima. we
        annotate the plot with any training sessions whose median
        training loss on the last epoch is above this value

    Returns
    -------
    g : sns.FacetGrid
        the FacetGrid containing the plot

    """
    # to make sure we show the full dataframe below, from
    # https://stackoverflow.com/a/42293737
    pd.set_option('display.max_columns', None)
    # from https://stackoverflow.com/a/25352191
    pd.set_option('display.max_colwidth', -1)
    df.fit_model_type = df.fit_model_type.map(dict(zip(plotting.MODEL_ORDER,
                                                       plotting.MODEL_PLOT_ORDER_FULL)))
    order = plotting.get_order('fit_model_type', col_unique=df.fit_model_type.unique())
    col_order = plotting.get_order('subject', col_unique=df.subject.unique())
    g = sns.FacetGrid(df, col='subject', hue=hue, col_wrap=4, sharey=False,
                      aspect=2.5, height=3, col_order=col_order)
    g.map_dataframe(plotting.scatter_ci_dist, 'fit_model_type', 'loss', x_jitter=True,
                    x_order=order)
    for ax in g.axes.flatten():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
        if ax.get_ylim()[1] > thresh:
            ax.hlines(thresh, 0, len(df.fit_model_type.unique())-1, 'gray', 'dashed')
    # find those training sessions with loss above the threshold
    above_thresh = df.groupby(['subject', 'fit_model_type', hue]).loss.median()
    above_thresh = above_thresh.reset_index().query('loss > @thresh')
    if len(above_thresh) > 0:
        g.fig.text(1.01, .5, ("Probable local optima (median last epoch training loss > "
                              f"{thresh}):\n" + str(above_thresh)))
    g.fig.suptitle("Last epoch median training loss (with 68% CI across batches) on each CV fold")
    g.fig.subplots_adjust(top=.92)
    return g


def feature_df_plot(df, avg_across_retinal_angle=False, reference_frame='relative',
                    feature_type='pref-period', visual_field='all', context='paper',
                    col_wrap=None, scatter_ref_pts=False, **kwargs):
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
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up
    col_wrap : int or None, optional
        col_wrap argument to pass through to seaborn FacetGrid
    scatter_ref_pts : bool, optional
        if True, we plot black points every 45 degrees on the polar plots to
        serve as a reference (only used in paper context). if False, do
        nothing.
    kwargs :
        Passed to plotting.feature_df_plot

    Returns
    -------
    g : sns.FacetGrid
        the FacetGrid containing the plot

    """
    aspect = 1
    params, fig_width = style.plotting_style(context, figsize='full')
    plt.style.use(params)
    kwargs.setdefault('top', .9)
    axes_titles = True
    title_kwargs = {}
    adjust_kwargs = {}
    if df.bootstrap_num.nunique() > 1 or 'groupaverage_seed' in df.columns:
        # then we have each subject's bootstraps or the groupaverage
        # subject (which has also already been bootstrapped), so we use
        # scatter_ci_dist to plot across them
        plot_func = plotting.scatter_ci_dist
        kwargs.update({'draw_ctr_pts': False, 'ci_mode': 'fill', 'join': True})
    else:
        plot_func = sns.lineplot
    # in this case, we have the individual fits
    if 'groupaverage_seed' not in df.columns:
        gb_cols = ['subject', 'bootstrap_num']
        col = 'subject'
        pre_boot_gb_cols = ['subject', 'reference_frame', 'Stimulus type', 'bootstrap_num',
                            'Eccentricity (deg)']
    # in this case, we have the sub-groupaverage
    else:
        gb_cols = ['groupaverage_seed']
        col = None
        pre_boot_gb_cols = ['reference_frame', 'Stimulus type', 'groupaverage_seed',
                            'Eccentricity (deg)']
    if 'col' in kwargs.keys():
        if col is None or df[col].nunique() == 1:
            col = kwargs.pop('col')
            gb_cols += [col]
            pre_boot_gb_cols += [col]
        else:
            raise Exception("Cannot set col if we're plotting individual fits!")
    # if we're faceting over something, need to separate it out when creating
    # the feature df
    if 'hue' in kwargs.keys():
        gb_cols += [kwargs['hue']]
        pre_boot_gb_cols += [kwargs['hue']]
    if col is None or df[col].nunique() == 1:
        facetgrid_legend = False
        suptitle = False
        axes_titles = False
        split_oris = True
        col = 'orientation_type'
        ori_map = {k: ['cardinals', 'obliques'][i%2] for i, k in
                   enumerate(np.linspace(0, np.pi, 4, endpoint=False))}
        pre_boot_gb_cols += [col]
        if feature_type == 'pref-period':
            kwargs.setdefault('height', (fig_width/2) / aspect)
        else:
            # the polar plots have two subplots, so they're half the height of the
            # pref-period one in order to get the same width
            kwargs.setdefault('height', (fig_width/4) / aspect)
    else:
        if context != 'paper':
            facetgrid_legend = True
            suptitle = True
        else:
            facetgrid_legend = False
            suptitle = False
        split_oris = False
        if col_wrap is not None:
            # there is, as of seaborn 0.11.0, a bug that interacts with our
            # xtick label size and height (see
            # https://github.com/mwaskom/seaborn/issues/2293), which causes an
            # issue if col_wrap == 3. this manual setting is about the same
            # size and fixes it
            if col_wrap == 3:
                kwargs.setdefault('height', 2.23)
            else:
                kwargs.setdefault('height', (fig_width / col_wrap) / aspect)
    if feature_type in ['pref-period', 'pref-sf']:
        if context == 'poster':
            aspect = 1.3
        else:
            kwargs.setdefault('ylim', (0, 2.1))
            kwargs.setdefault('xlim', (0, 11.55))
        if avg_across_retinal_angle:
            pre_boot_gb_func = 'mean'
            row = None
        else:
            pre_boot_gb_func = None
            row = 'Retinotopic angle (rad)'
        if split_oris:
            orientation = np.linspace(0, np.pi, 2, endpoint=False)
        else:
            orientation = np.linspace(0, np.pi, 4, endpoint=False)
        df = analyze_model.create_feature_df(df, feature_type.replace('pref-', 'preferred_'),
                                             reference_frame=reference_frame,
                                             gb_cols=gb_cols,
                                             orientation=orientation)
        if split_oris:
            df['orientation_type'] = df['Orientation (rad)'].map(ori_map)
        y = {'pref-period': 'Preferred period (deg)',
             'pref-sf': 'Preferred spatial frequency (cpd)'}[feature_type]
        g = plotting.feature_df_plot(df, y=y, col=col, row=row, pre_boot_gb_func=pre_boot_gb_func,
                                     plot_func=plot_func, aspect=aspect,
                                     pre_boot_gb_cols=pre_boot_gb_cols, col_wrap=col_wrap,
                                     facetgrid_legend=facetgrid_legend, **kwargs)
    else:
        kwargs.update({'all_tick_labels': ['r'], })
        if context == 'paper':
            orientation = np.linspace(0, np.pi, 4, endpoint=False)
            kwargs.update({'ylabelpad': 10, 'theta_ticklabels': [], 'wspace': .1,
                           'hspace': .1})
        elif context == 'poster':
            orientation = np.linspace(0, np.pi, 2, endpoint=False)
            kwargs.update({'top': .76, 'r_ticks': [.25, .5, .75, 1], 'wspace': .3,
                           'r_ticklabels': ['', .5, '', 1], 'ylabelpad': 60,
                           'hspace': .3})
        if feature_type == 'pref-period-contour':
            rticks = np.arange(.25, 1.5, .25)
            if context == 'paper':
                rticklabels = ['' for i in rticks]
            else:
                rticklabels = [j if j == 1 else '' for i, j in enumerate(rticks)]
            if not split_oris:
                # there's a weird interaction where if we set the rticks before
                # calling scatter (which we do when split_oris is True), it
                # competely messes up the plot. unsure why.
                kwargs.update({'r_ticks': rticks, 'r_ticklabels': rticklabels})
            df = analyze_model.create_feature_df(df, reference_frame=reference_frame,
                                                 eccentricity=[5], orientation=orientation,
                                                 retinotopic_angle=np.linspace(0, 2*np.pi, 49),
                                                 gb_cols=gb_cols)
            if split_oris:
                df['orientation_type'] = df['Orientation (rad)'].map(ori_map)
                kwargs['ylim'] = (0, 1.25)
            row = 'Eccentricity (deg)'
            if df[row].nunique() == 1:
                row = None
            r = 'Preferred period (deg)'
            g = plotting.feature_df_polar_plot(df, col=col, row=row,
                                               r=r, plot_func=plot_func, col_wrap=col_wrap,
                                               aspect=aspect,
                                               pre_boot_gb_cols=pre_boot_gb_cols,
                                               facetgrid_legend=facetgrid_legend, **kwargs)
            if context == 'paper':
                for axes in g.axes:
                    axes[0].set_ylabel('Preferred\nperiod (deg)')
        elif feature_type == 'iso-pref-period':
            if context == 'poster':
                kwargs.update({'r_ticks': list(range(1, 9)),
                               'r_ticklabels': [i if i%2==0 else '' for i in range(1, 9)]})
            if split_oris:
                df['orientation_type'] = df['Orientation (rad)'].map(ori_map)
            df = analyze_model.create_feature_df(df, 'preferred_period_contour', period_target=[1],
                                                 reference_frame=reference_frame,
                                                 orientation=orientation, gb_cols=gb_cols)
            r = 'Eccentricity (deg)'
            row = 'Preferred period (deg)'
            if df[row].nunique() == 1:
                row = None
            g = plotting.feature_df_polar_plot(df, col=col, r=r, row=row,
                                               plot_func=plot_func, aspect=aspect,
                                               title='ISO-preferred period contours',
                                               pre_boot_gb_cols=pre_boot_gb_cols,
                                               col_wrap=col_wrap,
                                               facetgrid_legend=facetgrid_legend, **kwargs)
        elif feature_type == 'max-amp':
            rticks = np.arange(.25, 1.5, .25)
            if context == 'paper':
                rticklabels = ['' for i in rticks]
            else:
                rticklabels = [j if j == 1 else '' for i, j in enumerate(rticks)]
            if not split_oris:
                # there's a weird interaction where if we set the rticks before
                # calling scatter (which we do when split_oris is True), it
                # competely messes up the plot. unsure why.
                kwargs.update({'r_ticks': rticks, 'r_ticklabels': rticklabels})
            df = analyze_model.create_feature_df(df, 'max_amplitude', orientation=orientation,
                                                 reference_frame=reference_frame, gb_cols=gb_cols)
            if split_oris:
                df['orientation_type'] = df['Orientation (rad)'].map(ori_map)
                kwargs['ylim'] = (0, 1.15)
            r = 'Max amplitude'
            g = plotting.feature_df_polar_plot(df, col=col, r=r, 
                                               aspect=aspect, plot_func=plot_func,
                                               title='Relative amplitude', col_wrap=col_wrap,
                                               pre_boot_gb_cols=pre_boot_gb_cols,
                                               facetgrid_legend=facetgrid_legend, **kwargs)
            ylabel = 'Relative amplitude'
            # doesn't look good with multiple rows
            if context == 'paper' and col_wrap is None:
                # the location argument here does nothing, since we over-ride
                # it with the bbox_to_anchor and bbox_transform arguments. the
                # size and size_vertical values here look weird because they're
                # in polar units (so size is in theta, size_vertical is in r)
                asb = AnchoredSizeBar(g.axes[0, 0].transData, 0, '1', 'center',
                                      frameon=False, size_vertical=1,
                                      bbox_to_anchor=(.52, 1),
                                      sep=5,
                                      bbox_transform=g.fig.transFigure)
                g.axes[0, 0].add_artist(asb)
                ylabel = ylabel.replace(' ', '\n')
            for axes in g.axes:
                axes[0].set_ylabel(ylabel)
        else:
            raise Exception(f"Don't know what to do with feature_type {feature_type}!")
        if split_oris:
            th = np.linspace(0, 2*np.pi, 8, endpoint=False)
            r_val = 1 # df[r].mean()
            if scatter_ref_pts:
                for ax in g.axes.flatten():
                    ax.scatter(th, len(th)*[r_val], c='k',
                               s=mpl.rcParams['lines.markersize']**2 / 2)
            # for some reason, can't call the set_rticks until after all
            # scatters have been called, or they get messed up
            for ax in g.axes.flatten():
                ax.set_yticks(rticks)
                ax.set_yticklabels(rticklabels)
        else:
            adjust_kwargs.update({'wspace': -.1, 'hspace': .15})
        if context == 'paper':
            for ax in g.axes.flatten():
                if ax.get_xlabel():
                    ax.set_xlabel(ax.get_xlabel(), labelpad=-5)
            # remove the xlabel from one of them and place the remaining one in
            # between the two subplots, because it's redundant
            g.axes[0, 0].set_xlabel('')
            # these can have their xlabel removed, since the legend will clarify
            if feature_type in ['pref-period-contour', 'max-amp']:
                g.axes[0, 1].set_xlabel('')
            else:
                g.axes[0, 1].set_xlabel(g.axes.flatten()[1].get_xlabel(), x=-.05,
                                        ha='center', labelpad=-5)
            title_kwargs['pad'] = -13
    if visual_field != 'all':
        g.fig._suptitle.set_text(g.fig._suptitle.get_text() + f' in {visual_field} visual field')
    if not suptitle:
        g.fig.suptitle('')
    if not axes_titles:
        for ax in g.axes.flatten():
            ax.set_title('')
    else:
        g.set_titles(col_template="{col_name}", **title_kwargs)
        g.tight_layout()
    g.fig.subplots_adjust(**adjust_kwargs)
    return g


def existing_studies_with_current_figure(df, seed=None, precision_df=None, y="Preferred period (deg)",
                                         context='paper'):
    """Plot results from existing studies with our results

    This is the same plot as `existing_studies_figure()`, with the
    results from our study plotted as a black line (so see that figure
    for more details).

    Note that the `df` argument here is the dataframe containing results
    from this study, NOT the results from previous studies (we call the
    `existing_studies_df()` function here)

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing all the model parameter values, across
        subjects.
    seed : int or None
        seed for numpy's RNG. can only be None if precision_df is None
    precision_df : pd.dataFrame or None, optional
        dataframe containing the precision for each scanning session in
        df. If None, we won't do any bootstrapping, and so assume this
        already has only one subject
    y : {'Preferred period (deg)', 'Preferred spatial frequency (cpd)'}
        Whether to plot the preferred period or preferred spatial
        frequency on the y-axis. If preferred period, the y-axis is
        linear; if preferred SF, the y-axis is log-scaled (base 2). The
        ylims will also differ between these two
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up

    Returns
    -------
    g : sns.FacetGrid
        The FacetGrid containing the plot

    """
    # this gets us the median parameter value for each subject and fit
    # model type
    df = df.groupby(['subject', 'model_parameter', 'fit_model_type']).median().reset_index()
    if precision_df is not None:
        df = df.merge(precision_df, on=['subject'])
        df = precision_weighted_bootstrap(df, seed, 100, 'fit_value', ['model_parameter', 'fit_model_type'],
                                          'precision')
    gb_cols = [c for c in ['subject', 'bootstrap_num'] if c in df.columns]
    df = analyze_model.create_feature_df(df, reference_frame='relative', gb_cols=gb_cols)
    df = df.groupby(['subject', 'reference_frame', 'Eccentricity (deg)', 'bootstrap_num']).agg('mean').reset_index()
    df['Preferred spatial frequency (cpd)'] = 1 / df['Preferred period (deg)']
    g = existing_studies_figure(existing_studies_df(), y, False, context)
    _, line, _ = plotting.scatter_ci_dist('Eccentricity (deg)', y, data=df,
                                          color='k', join=True, ax=g.ax,
                                          linewidth=1.5*plt.rcParams['lines.linewidth'],
                                          ci=68, estimator=np.median,
                                          draw_ctr_pts=False, ci_mode='fill');
    data = g._legend_data.copy()
    data['Current study'] = line[0]
    g.add_legend(data, label_order=g.hue_names + ['Current study'])
    # facetgrid doesn't let us set the title fontsize directly, so need to do
    # this hacky work-around
    g.fig.legends[0].get_title().set_size(mpl.rcParams['legend.title_fontsize'])
    return g


def mtf(mtf_func, df=None, context='paper'):
    """Plot the MTF as a function of spatial frequencies

    This plots the function we use to invert the display MTF when constructing
    our stimuli. We plot a semilogx plot, from 1/512 to 1/2 cycles per pixel,
    labeled as pixels per period (the reciprocal of spatial frequency), with
    y-values going from .5 to 1

    Parameters
    ----------
    mtf_func : function
        python function that takes array of spatial frequencies as its only
        argument and returns the MTF at those spatial frequencies.
    df : pd.DataFrame or None, optional
        If not None, the data used to fit this function, which we'll plot as
        points on the figure.
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up

    Returns
    -------
    fig : plt.figure
        Figure containing the MTF plot

    """
    sfs = np.linspace(0, .5)
    params, fig_width = style.plotting_style(context, figsize='half')
    plt.style.use(params)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width*.65))
    ax.semilogx(sfs, mtf_func(sfs), 'C0', basex=2)
    if df is not None:
        ax.semilogx(df.display_freq, df.corrected_contrast, 'C0o', basex=2)
    ticks = [512, 128, 32, 8, 2]
    ax.set(xticks=[1/i for i in ticks], xticklabels=ticks, xlabel='Pixels per period',
           ylabel='Michelson contrast', yticks=[.5, .75, 1])
    fig.tight_layout()
    return fig


def sigma_interpretation(df):
    """Generate string interpreting relative size of a, b, and sigma.

    This function returns a string (meant to be printed or saved to txt file)
    that describes the preferred period at 0, the standard deviation, and how
    many degrees you'd have to move in order to shift your preferred period by
    a single standard deviation.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing all the model parameter values, across
        subjects. note that this should first have gone through
        prep_model_df, which renames the values of the model_parameter
        columns.

    Returns
    -------
    result : str
        string containing the description discussed above

    """
    # get the median value of the parameters we're interested
    median_params = df.groupby('model_parameter').fit_value.median()
    a = median_params['$a$']
    b = median_params['$b$']
    sigma = median_params['$\sigma$']
    n_degrees = (b * (2**sigma - 1)) / a
    pref_period_there = b + n_degrees * a
    # as described on the wiki page for FWHM:
    # https://en.wikipedia.org/wiki/Full_width_at_half_maximum. That's for a
    # regular Gaussian, but the same calculation works here, just in octave
    # units (as equivalent to $\log_2(SF_{.5H} / SF_{.5L})$, where those SFs
    # are the spatial frequency where the curve reaches half-max above and
    # below the peak, respectively)
    fwhm = 2*np.sqrt(2*np.log(2)) * sigma
    result = (
        f"Preferred period at 0 degrees is {b:.03f}, with slope {a:.03f}.\n"
        f"Standard deviation of the log-Gaussian is {sigma:.03f} octaves (equivalent to FWHM of {fwhm:.03f} octaves).\n"
        f"Therefore, you'd need to move to {n_degrees:.03f} degrees eccentricity to move by a std dev.\n"
        f"At that eccentricity, preferred period is {pref_period_there:.03f}.\n"
        "All this is calculated using the median across bootstraps, average across polar angle and orientations."
    )
    return result


def compare_cv_models(first_level_df, targets, predictions, model_names, loss_func='normed_loss',
                      df_filter_string='drop_voxels_with_mean_negative_amplitudes,drop_voxels_near_border',
                      context='paper', voxel_n_check=9):
    """Create plots to help understand differences in model performance.

    This creates several plots to compare the predictions of different models.
    We make pairwise comparisons between each of them:

    1. Plot pairwise difference in loss as a function of eccentricity (each
       comparison on a separate row) (1 plot).

    2. Plot the `voxel_n_check` voxels that are the best for each model in each
       pairwise comparison (2 plots per pairwise comparison). We plot the voxel
       response as a function of spatial frequency, and then curves for each
       model. This means that we're collapsing across stimulus orientation
       (variation in those responses just shown as confidence intervals).

    Because we're only plotting response as a function of spatial frequency
    (and not of stimulus orientation), this is really only sufficient for
    comparing models 1 to 3, those models whose responses are isotropic.
    Modification to this would be necessary to make informative plots for the
    other models.

    Parameters
    ----------
    first_level_df : pd.DataFrame
        DataFrame containing the responses of each voxel to each stimulus. Note
        that we only use the median response, so the summary dataframe (vs
        full, which includes separate bootstraps) should be used.
    targets : torch.tensor
        tensor containing the targets for the model, i.e., the responses and
        precision of the voxels-to-fit, as saved out by
        sfp.analyze_model.calc_cv_error
    predictions : list
        list of tensors containing the predictions for each model, as saved out
        by sfp.analyze_model.calc_cv_error
    model_names : list
        list of strings containing the names (for plotting purposes) of each
        model, in same order as predictions.
    loss_func : str, optional
        The loss function to compute. One of: {'weighted_normed_loss',
        'crosscorrelation', 'normed_loss', 'explained_variance_score',
        'cosine_distance', 'cosine_distance_scaled'}.
    df_filter_string : str or None, optional
        a str specifying how to filter the voxels in the dataset. see
        the docstrings for sfp.model.FirstLevelDataset and
        sfp.model.construct_df_filter for more details. If None, we
        won't filter. Should probably use the default, which is what all
        models are trained using.
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up
    voxel_n_check : int, optional
        Number of voxels to plot in second plot type. As you get farther away
        from default value (9), more likely that plot will look weird.

    Returns
    -------
    figs : list
        List containing the created figures

    """
    params, fig_width = style.plotting_style(context, figsize='full')
    plt.style.use(params)
    if df_filter_string is not None:
        df_filter = model.construct_df_filter(df_filter_string)
        first_level_df = df_filter(first_level_df)
    voxels = first_level_df.drop_duplicates('voxel')
    voxels['voxel_new'] = np.arange(len(voxels))
    tmp = first_level_df.set_index('voxel')
    tmp['voxel_new'] = voxels['voxel_new']
    first_level_df = tmp.reset_index()
    for name, pred in zip(model_names, predictions):
        loss = analyze_model._calc_loss(pred, targets, loss_func, False)
        voxels[f'{name}_loss'] = loss
    # this is the number of combinations of the values in model names with
    # length 2. for some reason, itertools objects don't have len()
    n_combos = int(math.factorial(len(model_names)) / 2 /
                   math.factorial(len(model_names)-2))
    fig, axes = plt.subplots(n_combos, 2, squeeze=False,
                             figsize=(fig_width, n_combos/2*fig_width))
    predictions = dict(zip(model_names, predictions))
    voxel_comp_figs = []
    for i, (name_1, name_2) in enumerate(itertools.combinations(model_names, 2)):
        loss_name = f'{name_1}_loss - {name_2}_loss'
        voxels[loss_name] = voxels[f'{name_1}_loss'] - voxels[f'{name_2}_loss']
        ymax = voxels[loss_name].max() + voxels[loss_name].max() / 10
        ymin = voxels[loss_name].min() + voxels[loss_name].min() / 10
        sns.scatterplot(x='eccen', y=loss_name, data=voxels, ax=axes[i, 0])
        axes[i, 0].set_ylim(ymin, ymax)
        sns.regplot(x='eccen', y=loss_name, data=voxels, ax=axes[i, 1],
                    x_estimator=np.median, x_bins=50)
        axes[i, 1].set(ylabel='')
        axes[i, 0].hlines(0, voxels.eccen.min(), voxels.eccen.max(), linestyles='dashed')
        axes[i, 1].hlines(0, voxels.eccen.min(), voxels.eccen.max(), linestyles='dashed')

        vox_idx = voxels[loss_name].values.argsort()
        vox_idx = np.concatenate([vox_idx[-voxel_n_check:], vox_idx[:voxel_n_check]])

        tmp = first_level_df.query(f"voxel_new in @vox_idx")

        data = []
        for j, v in enumerate(vox_idx):
            d = {}
            for name in model_names:
                pred = predictions[name]
                val = pred[v]
                # need to normalize predictions for comparison
                val = val / val.norm(2, -1, True)
                d[name] = val.detach()
            d['voxel_new'] = v
            d['stimulus_class'] = np.arange(48)
            d['better_model'] = {True: name_2, False: name_1}[j < voxel_n_check]
            data.append(pd.DataFrame(d))
        t = pd.concat(data)

        tmp = tmp.merge(t, 'left', on=['voxel_new', 'stimulus_class'],
                        validate='1:1', )
        tmp = tmp.rename(columns={'amplitude_estimate_median_normed':
                                  'voxel_response'})
        tmp = pd.melt(tmp, ['voxel_new', 'local_sf_magnitude', 'stimulus_class',
                            'better_model', 'eccen'],
                      value_vars=['voxel_response'] + model_names,
                      var_name='model', value_name='response')
        for name, other_name in zip([name_1, name_2], [name_2, name_1]):
            g = sns.relplot(x='local_sf_magnitude', y='response',
                            data=tmp.query(f"better_model=='{name}'"),
                            hue='model', col='voxel_new', kind='line',
                            col_wrap=3, height=fig_width/3)
            g.fig.suptitle(f'better_model = {name} (vs {other_name})')
            if voxel_n_check > 6:
                g.fig.subplots_adjust(top=.9)
            elif voxel_n_check > 3:
                g.fig.subplots_adjust(top=.85)
            else:
                g.fig.subplots_adjust(top=.75)
            g.set(xscale='log')
            for ax in g.axes.flatten():
                vox_id = int(re.findall('\d+', ax.get_title())[0])
                ax.set_title(ax.get_title() + f",\neccen = {tmp.query('voxel_new==@vox_id').eccen.unique()[0]:.02f}")
            voxel_comp_figs.append(g.fig)
    fig.tight_layout()
    return [fig] + voxel_comp_figs


def theory_background_figure(context):
    """Create figure with some small info on background theory.

    Parameters
    ----------
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in seaborn's
        set_context function). if poster, will scale things up (but only paper
        has been tested)

    Returns
    -------
    fig : plt.figure
        Figure containing this plot

    """
    tiles_path = op.join(op.dirname(op.realpath(__file__)), '..', 'reports', 'figures',
                         'tiles.png')
    tiles = plt.imread(tiles_path)
    tiles = tiles / tiles.max()
    params, fig_width = style.plotting_style(context, figsize='full')
    params['axes.titlesize'] = '8'
    params['axes.labelsize'] = '8'
    params['legend.fontsize'] = '8'
    warnings.warn("We adjust the font size for axes titles, labels, and legends down to "
                  "8pts (so this will probably look wrong if context is not paper)!")
    plt.style.use(params)
    fig = plt.figure(figsize=(fig_width, fig_width/2))
    gs = fig.add_gridspec(4, 4, hspace=.65)
    fig.add_subplot(gs[:2, 0])
    fig.add_subplot(gs[2:, 0])
    fig.add_subplot(gs[1:3, 1])
    fig.add_subplot(gs[1:3, -2])
    fig.add_subplot(gs[:2, -1])
    fig.add_subplot(gs[2:, -1])
    axes = np.array(fig.axes).flatten()

    for ax in axes[:2]:
        ax.axis('off')
    pt.imshow((tiles+.5)/1.5, ax=axes[0], zoom=110/256, title=None,
              vrange=(0, 1))
    pt.imshow((tiles+.5)/1.5, ax=axes[1], zoom=110/256, title=None,
              vrange=(0, 1))

    axes[0].set_title(r'SF preferences $\bf{constant}$'+'\nacross visual field')
    axes[1].set_title(r'SF preferences $\bf{scale}$'+'\nwith eccentricity')

    ecc = np.linspace(.01, 20, 50)
    V1_pRF_size = 0.063485 * ecc
    constant_hyp = 2*np.ones(len(ecc))
    pal = sns.color_palette('Dark2', n_colors=2)
    for i, ax in enumerate(axes[2:4].flatten()):
        if i == 0:
            ax.semilogy(ecc, 1./V1_pRF_size, '-', label='scaling',
                        linewidth=2, basey=2, c=pal[0])
            ax.set_ylim((.25, 10))
            ax.plot(ecc, constant_hyp, c=pal[1], linewidth=2, label='constant')
            ax.set(xticks=[], yticks=[], ylabel='Preferred SF (cpd)',
                   xlabel='Eccentricity')
        elif i == 1:
            ax.plot(ecc, V1_pRF_size, linewidth=2, label='scaling', c=pal[0])
            ax.plot(ecc, 1./constant_hyp, c=pal[1], linewidth=2, label='constant')
            ax.set(xlabel='Eccentricity', xticks=[], yticks=[],
                   ylabel='Preferred period (deg)')
    axes[3].legend(frameon=False, bbox_to_anchor=(-.1, -.1), loc='upper center')
    axes[3].annotate('', xy=(.5, 1), xytext=(-.65, 1), xycoords='axes fraction',
                        arrowprops={'arrowstyle': '<->', 'color': 'k',
                                    'connectionstyle': 'arc3,rad=-.3'})
    axes[3].text(-.075, 1.2, r'$\frac{1}{f(x)}$', ha='center', va='bottom',
                 transform=axes[3].transAxes)

    # from Eero, this is about what it should be
    V1_RF_size = .2 * ecc
    V1_pRF_size_slope = 0.063485
    V1_pRF_size_offset = 0
    V1_pRF_size_error = 0.052780

    for i, ax in enumerate(axes[4:].flatten()):
        ax.fill_between(ecc, (V1_pRF_size_slope - V1_pRF_size_error/2.)*ecc + V1_pRF_size_offset,
                        (V1_pRF_size_slope + V1_pRF_size_error/2.)*ecc + V1_pRF_size_offset,
                        alpha=.1, color=pal[0])
        ax.plot(ecc, V1_pRF_size_slope*ecc+V1_pRF_size_offset, linewidth=2, label='scaling', c=pal[0])
        if i == 0:
            for e in [1,5,10,15,20]:
                ax.plot([0, 20], [V1_pRF_size_slope*e+V1_pRF_size_offset,
                                  V1_pRF_size_slope*e+V1_pRF_size_offset], '--', c='k',
                        linewidth=1)
            ax.set(title="Full-field gratings", xticks=[], yticks=[])
        if i == 1:
            for j in [-1, -.5, 0, .5, 1]:
                ax.plot(ecc, (V1_pRF_size_slope + j*V1_pRF_size_error/2.)*ecc + V1_pRF_size_offset,
                        linestyle='--', c='k', linewidth=1)
            ax.set(xlabel='Eccentricity', xticks=[], yticks=[], title='Scaled gratings')
        ax.set_ylabel("Preferred period (deg)")

    return fig


def voxel_exclusion(df, context='paper'):
    """Create plot showing how many voxels were excluded from model fitting.

    WARNING: Currently this is not context-compliant -- the figure ends up much
    wider than allowed. If we want to use this in paper, will change that.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing the voxel exclusion info, as created by the
        snakemake rule voxel_exclusion_df
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in seaborn's
        set_context function). if poster, will scale things up (but only paper
        has been tested)

    Returns
    -------
    g : sns.FacetGrid
        FacetGrid containing the plot

    """
    params, fig_width = style.plotting_style(context, figsize='full')
    plt.style.use(params)
    if 'ecc in 1-12,drop_voxels_with_any_negative_amplitudes' in df.columns:
        arg_str = 'any'
    elif 'ecc in 1-12,drop_voxels_with_mean_negative_amplitudes' in df.columns:
        arg_str = 'mean'
    neg = df['ecc in 1-12'] - df[f'ecc in 1-12,drop_voxels_with_{arg_str}_negative_amplitudes']
    border = df['ecc in 1-12'] - df['ecc in 1-12,drop_voxels_near_border']
    df[f'ecc in 1-12,drop_voxels_with_{arg_str}_negative_amplitudes,drop_voxels_near_border - independent'] = df['ecc in 1-12'] - (neg + border)
    neg_prop = dict(zip(df.subject, neg / df['ecc in 1-12']))
    neg = dict(zip(df.subject, neg))

    map_dict = {'total_voxels': 0,
                'ecc in 1-12': 1,
                f'ecc in 1-12,drop_voxels_with_{arg_str}_negative_amplitudes': 2,
                'ecc in 1-12,drop_voxels_near_border': 3,
                f'ecc in 1-12,drop_voxels_with_{arg_str}_negative_amplitudes,drop_voxels_near_border': 4,
                f'ecc in 1-12,drop_voxels_with_{arg_str}_negative_amplitudes,drop_voxels_near_border - independent': 5}
    id_vars = [c for c in df.columns if c not in map_dict.keys()]
    df = pd.melt(df, id_vars, value_name='number_of_voxels')
    df['exclusion_criteria'] = df.variable.map(map_dict)
    col_order = plotting.get_order('subject', col_unique=df.subject.unique())

    g = sns.catplot(x='exclusion_criteria', y='number_of_voxels', data=df,
                    col='subject', kind='point', col_wrap=6, aspect=.5,
                    height=(1/.5)*(2*fig_width/6), col_order=col_order)
    for ijk, data in g.facet_data():
        ax = g.axes[ijk[1]]
        ax.scatter(4, data.query('exclusion_criteria==4').number_of_voxels, c='r', zorder=100)
    txt = '\n'.join([f'{v}: {k}' for k,v in map_dict.items()])
    g.fig.text(1, .75, txt, va='center')
    txt = '\n'.join([f'{s}: {neg[s]} ({neg_prop[s]:.3f})' for s in col_order])
    txt = "Number of voxels dropped because of negative amplitude (proportion on stimuli)\n\n" + txt
    g.fig.text(1, .25, txt, va='center')

    return g


def _create_model_prediction_df(df, trained_model, voxel_label,
                                for_relative_plot=False,
                                extend_sf=False):
    """Create df containing model predictions for a single voxel

    Will contain 48 rows, with the following columns: model_predictions (normed
    predictions of trained_model to the spatial frequency seen by this voxel),
    voxel (voxel_label), stimulus_class (0 to 47, giving the stimulus label),
    peak_sf (if add_peak_sf is True, this gives the preferred spatial frequency
    of this voxel, at each observed orientation).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the responses of a single voxel to stimuli. Should
        only have one response per stimulus (thus, the summary df), and must
        have columns eccen, angle, local_sf_magnitude, and local_sf_xy_direction.
    trained_model : sfp.model.LogGaussianDonut
        Trained model whose responses we want to get.
    voxel_label : str
        The label for this voxel.
    for_relative_plot : bool, optional
        If True, will add a column giving the peak spatial frequency for this
        voxel at each observed orientation and evaluate the model at 36
        frequencies log-spaced from two decades below to two decades above the
        peak (rather than the presented frequencies), at the four main
        orientations.
    extend_sf : bool, optional
        If True, we instead generate predictions for local spatial frequencies
        from .01 to 100 cpd (logspaced, 36 samples), for the four main angles.
        Cannot be True if for_relative_plot is True.

    Returns
    -------
    data : pd.DataFrame
        DataFrame containing the above info

    """
    data = {}
    assert df.eccen.nunique() == 1 and df.angle.nunique() == 1, "_create_model_prediction_df must be called on the df with responses to a single voxel!"
    sfs = df.drop_duplicates('stimulus_class')[['local_sf_magnitude',
                                                'local_sf_xy_direction']]
    sfs = torch.tensor(sfs.values)
    prf_loc = torch.tensor(df[['eccen', 'angle']].values)
    predictions = trained_model.evaluate(sfs[:, 0], sfs[:, 1], prf_loc[:, 0], prf_loc[:, 1])
    predictions_norm = predictions.norm(2, -1, True)
    if extend_sf:
        if for_relative_plot:
            raise Exception("At most one of for_relative_plot and extend_sf can be true, but both were true!")
        # get the 4 main orientations
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles = df.query('freq_space_angle in @angles').drop_duplicates('freq_space_angle')
        angles = angles.local_sf_xy_direction.values
        n_samps = 36
        freqs = []
        for a in angles:
            freqs.extend(np.logspace(-2, 2, n_samps))
        sfs = torch.tensor([freqs, np.concatenate([n_samps*[a] for a in angles])]).transpose(0, 1)
        data['local_sf_magnitude'] = sfs[:, 0].detach().numpy()
        # we use the same norm as before, in order to make sure things line up correctly
        predictions = trained_model.evaluate(sfs[:, 0], sfs[:, 1],
                                             prf_loc[0, 0], prf_loc[0, 1])
    elif for_relative_plot:
        # get the 4 main orientations
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles = df.query('freq_space_angle in @angles').drop_duplicates('freq_space_angle')
        angles = angles.local_sf_xy_direction.values
        peak_sf = []
        freqs = []
        n_samps = 36
        for a in angles:
            peak_sf.append(trained_model.preferred_sf(a, prf_loc[0, 0], prf_loc[0, 1]).item())
            freqs.extend(np.logspace(np.log10(peak_sf[-1]/100), np.log10(peak_sf[-1]*100), n_samps))
        sfs = torch.tensor([freqs, np.concatenate([n_samps*[a] for a in angles])]).transpose(0, 1)
        peak_sf = np.concatenate([n_samps*[p] for p in peak_sf])
        data['peak_sf'] = peak_sf
        data['local_sf_magnitude'] = sfs[:, 0].detach().numpy()
        # we use the same norm as before, in order to make sure things line up correctly
        predictions = trained_model.evaluate(sfs[:, 0], sfs[:, 1],
                                             prf_loc[0, 0], prf_loc[0, 1])
    else:
        data['stimulus_class'] = np.arange(48)
    data['model_predictions'] = (predictions / predictions_norm).detach().squeeze()
    data['voxel'] = voxel_label
    return pd.DataFrame(data)


def _remap_frequencies(df, freq_mag_col='local_sf_magnitude'):
    """Create plotting_sf column in df

    for each voxel, our stimuli have several orientations. ideally, these
    orientations would all have the exact same spatial frequency, but they
    don't (the w_r/w_a parameters needed to be integers in order to avoid
    obvious artifacts at polar angle 0). for plotting purposes, this is
    confusing, so we map those values such that they are identical, and the
    binning that gets done later on then makes more sense.

    This adds a column, plotting_sf, which contains this info.

    Parameters
    ----------
    df : pd.DataFrame
        first level DataFrame containing the amplitude responses for a single
        subject and session. Must be the summary version (only has median across
        bootstraps).
    freq_mag_col : str, optional
        Name of the column with the spatial frequencies to remap.

    Returns
    -------
    df : pd.DataFrame
        the dataframe with plotting_sf column added.

    """
    canonical_freqs = [f for f in df.freq_space_distance.unique() if f == int(f)]
    canonical_freq_mapper = {f: min(canonical_freqs, key=lambda x: abs(x-f))
                             for f in df.freq_space_distance.unique()}
    freq_mapper = df.groupby(['voxel', 'freq_space_distance'])[freq_mag_col].median().to_dict()
    df['plotting_sf'] = df.apply(lambda x: freq_mapper[x.voxel,
                                                       canonical_freq_mapper[x.freq_space_distance]],
                                 axis=1)
    return df


def _merge_model_response_df(df, model_predictions):
    """Merge dfs with model predictions and voxel responses.

    Parameters
    ----------
    df : pd.DataFrame
        first level DataFrame containing the amplitude responses for a single
        subject and session. Must be the summary version (only has median across
        bootstraps).
    model_predictions : pd.DataFrame
        DataFrame containing the model predictions for each voxel in df.

    Returns
    -------
    df : pd.Dataframe
        The merged dataframe

    """
    try:
        df = df.merge(model_predictions, 'left', on=['voxel', 'stimulus_class'],
                      validate='1:1', )
        df = df.rename(columns={'amplitude_estimate_median_normed':
                                'voxel_response'})
        df = pd.melt(df, ['voxel', 'stimulus_class', 'eccen', 'freq_space_angle',
                          'local_sf_magnitude', 'plotting_sf'],
                     value_vars=['voxel_response', 'model_predictions'],
                     var_name='model', value_name='Response (a.u.)')
    except KeyError:
        # in this case, we're combining the relative ones, so model_predictions
        # doesn't have a stimulus_class column (and they're evaluated at
        # different frequencies)
        df = df[['voxel', 'local_sf_magnitude', 'amplitude_estimate_median_normed',
                 'peak_sf', 'subject']]
        df['model'] = 'voxel_response'
        df = df.rename(columns={'amplitude_estimate_median_normed': 'Response (a.u.)'})
        model_predictions = model_predictions.rename(columns={'model_predictions':
                                                              'Response (a.u.)'})
        model_predictions['model'] = 'model_predictions'
        df = pd.concat([df, model_predictions], sort=False)
    return df


def _voxel_responses_and_predictions(*args, label='', n_bins=10, plot_type='reg', **kwargs):
    """Plot voxel responses and model predictions.

    If label=voxel_response, we use sns.regplot (if plot_type=='reg', with
    n_bins bins on the x axis) or sns.histplot (if plot_type='hist', logscaling
    the x-axis). Else, we use sns.lineplot

    """
    if label == 'voxel_response':
        if plot_type == 'reg':
            # there are 22 unique frequencies (freq_space_distance in the
            # dataframe), but only 10 "real" ones, the others are just off by a
            # little bit (because w_a/w_r needed to be whole numbers). we add 2
            # to n_bins because seaborn's regplot bins in a way that excludes
            # the ends of the possible range of values. since we want those end
            # values (they're meaningful), we add 2 here, which gives us the
            # right number of bins
            return sns.regplot(*args, x_bins=n_bins+2,
                               fit_reg=False, label=label,
                               scatter_kws={'s': 10}, **kwargs)
        elif plot_type == 'hist':
            to_return = sns.histplot(*args, label=label,
                                     log_scale=(True, False),
                                     # rasterize to decrease size
                                     rasterized=True,
                                     **kwargs)
            # set xscale back to linear because apparently sns.histplot sets it
            # for all axes, and we want the next facet to have linear xscale
            # for when sns.lineplot is called
            plt.xscale('linear')
            return to_return
    else:
        return sns.lineplot(*args, label=label, **kwargs, zorder=10)


def example_voxels(df, trained_model, voxel_idx=[2310, 2957, 1651],
                   extend_sf=False, context='paper'):
    """Plot some example voxel data and their model fit.

    For some voxels and a trained model, plot some comparisons between the
    measured voxel responses and the model's predictions. Each voxel gets its
    own column. Nothing is done here to choose the voxels, so that must be done
    externally.

    Parameters
    ----------
    df : pd.DataFrame
        first level DataFrame containing the amplitude responses for a single
        subject and session. Must be the summary version (only has median across
        bootstraps).
    trained_model : sfp.model.LogGaussianDonut
        Trained model whose responses we want to show.
    voxel_idx : list, optional
        List of voxel ids (i.e., values from the 'voxel' column of df) to show.
        Should be selected somehow in order to make sure they're reasonably
        nice. The default values are for sub-wlsubj001, ses-04, and are roughly
        foveal, parafoveal, and peripheral, all reasonably well fit by the full
        model. Regardless of how many are here, we'll have 3 columns per row.
    extend_sf : bool, optional
        If True, we instead generate predictions for local spatial frequencies
        from .01 to 100 cpd (logspaced, 36 samples), for the four main angles.
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in seaborn's
        set_context function). if poster, will scale things up (but only paper
        has been tested)

    Returns
    -------
    g : sns.FacetGrid
        FacetGrid containing the plot

    """
    params, fig_width = style.plotting_style(context, figsize='full')
    plt.style.use(params)
    ax_height = (fig_width / 4) / .75
    df = df.query("voxel in @voxel_idx")
    data = []
    voxel = df.drop_duplicates('voxel')
    eccen_order = voxel.sort_values('eccen').voxel.values
    for i, v in enumerate(voxel_idx):
        data.append(_create_model_prediction_df(df.query('voxel==@v'),
                                                trained_model, v,
                                                extend_sf=extend_sf))
    data = pd.concat(data)
    canonical_freqs = [f for f in df.freq_space_distance.unique() if f == int(f)]
    df = _remap_frequencies(df)
    if extend_sf:
        df['model'] = 'voxel_response'
        df = df.rename(columns={'amplitude_estimate_median_normed':
                                'Response (a.u.)'})
        xlim = (.01, 100)
        ylim = (0, .225)
        yticks = [0, .2]
    else:
        df = _merge_model_response_df(df, data)
        xlim = (.1, 20)
        ylim = (.05, .225)
        yticks = []
    g = sns.FacetGrid(hue='model', data=df, col='voxel', col_wrap=3,
                      col_order=eccen_order, height=ax_height, aspect=.75)
    g.map_dataframe(_voxel_responses_and_predictions, x='plotting_sf',
                    y='Response (a.u.)', n_bins=len(canonical_freqs),
                    plot_type='reg')
    for i, ax in enumerate(g.axes.flatten()):
        vox_id = int(re.findall('voxel = (\d+)', ax.get_title())[0])
        ax.set_title(f"eccentricity = {df.query('voxel==@vox_id').eccen.unique()[0]:.02f}")
        # when extend_sf is True, we plot model predictions separately, because
        # merging the two dfs was too difficult
        if extend_sf:
            sns.lineplot(ax=ax, c='C1', label='model_prediction',
                         x='local_sf_magnitude', y='model_predictions',
                         data=data.query("voxel==@vox_id"), zorder=10)
            ax.legend_.remove()
        if i == 0:
            ax.set(ylabel='Response (a.u.)')
        if i != 1:
            ax.set(xlabel='')
        else:
            ax.set(xlabel='Local spatial frequency (cpd)')
    g.set(xscale='log', ylim=ylim, yticks=yticks, xlim=xlim)
    return g


def example_eccentricity_bins(df, context='paper'):
    """Plot some example eccentricity bins and their tuning curves.
   
    This plots the amplitude estimates and the tuning curves for a single
    subject, angular and radial stimuli, eccentricity bins 02-03 and 10-11. It
    is meant to show that the tuning curves fit the bins reasonably well.

    Parameters
    ----------
    df : pd.DataFrame
        pandas DataFrame containing the 1d tuning curves for a single subject.
        Must be the summary version (containing the fits to the median across
        bootstraps)
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in seaborn's
        set_context function). if poster, will scale things up (but only paper
        has been tested)

    Returns
    -------
    g : sns.FacetGrid
        FacetGrid containing the plot

    """
    params, fig_width = style.plotting_style(figsize='half')
    plt.style.use(params)
    if context == 'paper':
        # this isn't the exact same as what you'd get doing the line below,
        # because we use a relatively small wspace (space between axes), that
        # lets us make them a bit bigger
        height = 2.7
    else:
        height = (fig_width / 2) / .7

    df = df.query("frequency_type=='local_sf_magnitude' & "
                  "stimulus_superclass in ['angular', 'radial'] & "
                  "eccen in ['02-03', '09-10']")
    df.eccen = df.eccen.map(lambda x: {'02-03': '2-3',
                                       '09-10': '9-10'}.get(x, x))
    df = df.rename(columns={'eccen': "Eccentricity band"})
    pal = plotting.get_palette('stimulus_type', 'relative',
                               df.stimulus_superclass.unique(),
                               True)

    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=True, sharey=True,
                             figsize=(.6*2*height, height), 
                             gridspec_kw={'wspace': .1})
    artists = [axes[0].scatter([], [], s=0)]
    labels = ['Eccentricity band']
    for i, (col, data) in enumerate(df.groupby('stimulus_superclass')):
        ax = axes[i]
        for s, d in data.groupby('Eccentricity band'):
            hue = pal[col]
            ax.scatter(d.frequency_value, d.amplitude_estimate,
                       facecolor={'2-3': 'w'}.get(s, hue), edgecolor=hue)
            plotting.plot_tuning_curve(data=d, ax=ax, xlim='data',
                                       style='Eccentricity band', color=hue,
                                       dashes_dict={'2-3': (2, 2)})
            if i == 1:
                artists.append(ax.plot([], [], color='k',
                                       mfc={'2-3': 'w'}.get(s, 'k'),
                                       mec='k', marker='o',
                                       dashes={'2-3': (2, 2)}.get(s, ''))[0])
                labels.append(s+' deg')
        ax.set_xscale('log', basex=10)
        ax.set(xticks=[10**i for i in [-1, 0, 1]], ylim=(1, 3.5),)
        if i == 0:
            ax.set(ylabel='Response\n(% BOLD signal change)',
                   yticks=np.arange(1, 4.5, .5))
            ax.set_xlabel('Local spatial frequency (cpd)', ha='center',
                          x=1)
        
    fig.legend(artists, labels, frameon=False, bbox_to_anchor=(1, .5),
               bbox_transform=fig.transFigure, loc='center left',
               borderaxespad=0)
    return fig


def stimulus_schematic(stim, stim_df, context='paper'):
    """Create schematic with some example stimuli.

    Shows the two lowest frequencies from each of the four main stimulus types,
    with some annotations.

    This works with any of the stimuli created by this project: log-scaled or
    constant, rescaled or not.

    Parameters
    ----------
    stim : np.ndarray
        array containing the stimuli
    stim_df : pd.DataFrame
        dataframe containing the description of the stimuli.
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in seaborn's
        set_context function). if poster, will scale things up (but only paper
        has been tested)

    Returns
    --------
    fig : plt.figure
        Figure containing this plot

    """
    params, fig_width = style.plotting_style(context, figsize='half')
    params['font.size'] = '8'
    params['axes.titlesize'] = '8'
    params['axes.labelsize'] = '8'
    params['legend.fontsize'] = '8'
    warnings.warn("We adjust the font size for axes titles, labels, and legends down to "
                  "8pts (so this will probably look wrong if context is not paper)!")
    plt.style.use(params)
    fig, axes = plt.subplots(3, 4, figsize=(fig_width, .75*fig_width))
    # for pt.imshow to work, we need to find an integer that, when the size of
    # the image is divided by it, we get another integer (so that we
    # down-sample correctly)
    zoom = int(stim.shape[-1] / axes[0, 0].bbox.height)
    # this while will be false when zoom is a divisor of stim.shape[-1]
    while math.gcd(stim.shape[-1], zoom) != zoom:
        zoom += 1
    stim_df = first_level_analysis._add_freq_metainfo(stim_df.drop_duplicates('class_idx'))
    # drop baseline and the off-diagonal stimuli
    stim_df = stim_df.query("stimulus_superclass not in ['baseline', 'mixtures', 'off-diagonal']")
    if 'angular' in stim_df.stimulus_superclass.unique():
        col_order = ['radial', 'angular', 'forward spiral', 'reverse spiral']
        stim_type = 'relative'
    elif 'vertical' in stim_df.stimulus_superclass.unique():
        col_order = ['vertical', 'horizontal', 'forward diagonal', 'reverse diagonal']
        stim_type = 'absolute'
    pal = plotting.get_palette('stimulus_type', stim_type, col_order, True)
    for i, stim_type in enumerate(col_order):
        # get the lowest and second frequencies from each stimulus type (any
        # higher and it starts to alias at this resolution)
        g = stim_df.query("stimulus_superclass==@stim_type").iloc[[0, 2]]
        for ax, g_j in zip(axes[:, i], g.iterrows()):
            pt.imshow(stim[g_j[1]['index']], ax=ax, zoom=1/zoom, title=None)
            ax.set_frame_on(False)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        axes[-1, i].set_visible(False)
        # use the less-ambiguous label
        plot_label = plotting.SUPERCLASS_PLOT_LABELS.get(stim_type, stim_type)
        axes[0, i].set_title(plot_label.replace(' ', '\n'), rotation=0, va='bottom',
                             bbox={'fc': 'none', 'ec': pal[stim_type], 'pad': 2})
    fig.text(.515, 1/3.5, '...', transform=fig.transFigure, fontsize='xx-large',
             va='center', ha='center')
    axes[0, 0].text(0, .5, 'Low base\nfrequency', transform=axes[0, 0].transAxes,
                    rotation=90, ha='right', va='center', multialignment='center')
    # we do this using axes[0, 0] because axes[2, 0] is invisible, but we can
    # still use its transform
    axes[0, 0].text(0, .5, 'High base\nfrequency', transform=axes[2, 0].transAxes,
                    rotation=90, ha='right', va='center')
    axes[1, 0].annotate('', xy=(-.2, 1), xytext=(-.2, 0), textcoords='axes fraction',
                        xycoords='axes fraction',
                        arrowprops={'arrowstyle': '<-', 'color': '0',
                                    'connectionstyle': 'arc3'})
    axes[1, -1].annotate('', xy=(1.035, -.1), xytext=(-.035, -.1), textcoords='axes fraction',
                         xycoords='axes fraction',
                         arrowprops={'arrowstyle': '-', 'color': '0',
                                     'connectionstyle': 'arc3'})
    axes[1, -1].text(.5, -.2, '24\u00B0', transform=axes[1, -1].transAxes,
                     ha='center', va='top')
    fig.subplots_adjust(wspace=.05, hspace=.05)
    return fig


def peakiness_check(dfs, trained_models, col='subject', voxel_subset=False,
                    df_filter_string='drop_voxels_with_mean_negative_amplitudes,drop_voxels_near_border',
                    context='paper'):
    """Plot all voxels responses to check peakiness.

    The x value here is spatial frequency relative to peak (based on the
    model). This allows us to see whether the responses are "peakier" than the
    model, which would tell us that, instead of the exp(-x^2) in a Gaussian, we
    should be using a smaller exponent, e.g., exp(-x^(1.5))

    Parameters
    ----------
    dfs : pd.DataFrame or list
        first level DataFrame containing the amplitude responses for a single
        subject and session. Must be the summary version (only has median across
        bootstraps). If a list, a list of those (one per subject). Should
        contain a subject column
    trained_models : sfp.model.LogGaussianDonut or list
        Trained model whose responses we want to show. If a list, a list of
        those (one per subject).
    col : str or None, optional
        The column of the dataframe to facet columns on
    voxel_subset : bool or int, optional
        if True, we only do this for 10 voxels, to test it out (since this
        will take a while). If an int, we do it for that many voxels.
    df_filter_string : str or None, optional
        a str specifying how to filter the voxels in the dataset. see
        the docstrings for sfp.model.FirstLevelDataset and
        sfp.model.construct_df_filter for more details. If None, we
        won't filter. Should probably use the default, which is what all
        models are trained using.
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in seaborn's
        set_context function). if poster, will scale things up (but only paper
        has been tested)

    Returns
    -------
    g : sns.FacetGrid
        FacetGrid containing the plot

    """
    params, fig_width = style.plotting_style(context, figsize='full')
    # for some reason, we also draw the grid in this figure, even when
    # axes.grid is set to False. so, we manually set the linestyle to nothing
    # in order to avoid drawing the grid lines. (we don't want to do this in
    # the style params because we *do* want to draw the grid for polar plots)
    params['grid.linestyle'] = ''
    plt.style.use(params)
    ax_height = (fig_width / 4) / .75
    if not isinstance(dfs, list):
        dfs = [dfs]
    if not isinstance(trained_models, list):
        trained_models = [trained_models]
    df_overall = []
    for df, trained_model in zip(dfs, trained_models):
        if 'subject' not in df.columns:
            # this way it will run even if there's no subject specified
            df['subject'] = 'none'
        print(f"Starting subject {df['subject'].unique()}")
        if df_filter_string is not None:
            df_filter = model.construct_df_filter(df_filter_string)
            df = df_filter(df).reset_index()
        if voxel_subset is not False:
            if voxel_subset is True:
                voxel_subset = df.voxel.unique()[:10]
            elif isinstance(voxel_subset, int):
                voxel_subset = df.voxel.unique()[:voxel_subset]
            df = df.query("voxel in @voxel_subset")
        data = []
        peak_sfs = []
        for n, g in df.groupby('voxel'):
            data.append(_create_model_prediction_df(g, trained_model, n,
                                                    True))
            g = g[['local_sf_xy_direction', 'eccen', 'angle']].values
            peak_sf = trained_model.preferred_sf(*torch.tensor(g.T))
            peak_sf = pd.DataFrame({'voxel': n, 'stimulus_class': np.arange(48),
                                    'peak_sf': peak_sf.detach().numpy()})
            peak_sfs.append(peak_sf)
        data = pd.concat(data)
        data['subject'] = df.subject.unique()[0]
        peak_sfs = pd.concat(peak_sfs)
        df = pd.merge(df, peak_sfs, on=['voxel', 'stimulus_class'])
        df = _merge_model_response_df(df, data)
        # since we want to shift the tuning curves on a log axis, we divide by the
        # peak (to set it to 1 for everyone) rather than subtract it off (which
        # would shift it on a linear axis)
        df['Proportion of peak spatial frequency'] = df.local_sf_magnitude / df.peak_sf
        # want the values across orientation to be equal and they're off by a tiny
        # amount
        df['Proportion of peak spatial frequency'] = df['Proportion of peak spatial frequency'].round(8)
        df_overall.append(df)
    df_overall = pd.concat(df_overall)
    # we plot do it this way so that the model_predictions gets called first,
    # since it calls sns.lineplot and that gets messed up if the axis is
    # already logscale
    if col is not None:
        col_wrap = 4
    else:
        col_wrap = None
    print("Getting ready to plot")
    g = sns.FacetGrid(hue='model', data=df_overall, palette=['C1', 'C0'],
                      hue_order=['model_predictions', 'voxel_response'],
                      height=ax_height, col=col, col_wrap=col_wrap,
                      aspect=.75)
    g.map_dataframe(_voxel_responses_and_predictions,
                    x='Proportion of peak spatial frequency',
                    y='Response (a.u.)', plot_type='hist')
    # this range should highlight the curve
    g.set(ylim=(0, .225), yticks=[], xlim=(.01, 100), xscale='log')
    g.set_ylabels('Response (a.u.)')
    if col is not None:
        g.set_titles('{col_name}')
        g.set_xlabels('Proportion of peak spatial frequency')
    else:
        g.set_xlabels('Proportion of peak\nspatial frequency')
        if context == 'paper':
            g.set_ylabels('')
    g.fig.subplots_adjust(left=.1, right=.95)
    return g


def compare_sigma_and_pref_period(dfs, trained_models,
                                  df_filter_string='drop_voxels_with_mean_negative_amplitudes',
                                  context='paper'):
    """Create two figures comparing sigma to preferred period.

    We create two figures:

    1. Plot a 2d histogram showing sigma as a function of eccentricity, with a
       red dotted line on top showing the linear fit between the two, one
       sub-plot per subject.

    2. Plot the slope and intercept of that line and those of the relationship
       between preferred period and eccentricity as a scatter plot (hollow and
       filled for the two parameters, different subjects in different colors).

    Parameters
    ----------
    dfs : pd.DataFrame or list
        first level DataFrame containing the amplitude responses for a single
        subject and session. Must be the summary version (only has median across
        bootstraps). If a list, a list of those (one per subject). Should
        contain a subject column
    trained_models : sfp.model.LogGaussianDonut or list
        Trained model whose responses we want to show. If a list, a list of
        those (one per subject).
    df_filter_string : str or None, optional
        a str specifying how to filter the voxels in the dataset. see
        the docstrings for sfp.model.FirstLevelDataset and
        sfp.model.construct_df_filter for more details. If None, we
        won't filter. Will raise an exception if it drops voxels near the
        border (since that will bias the best-fit line)
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in seaborn's
        set_context function). if poster, will scale things up (but only paper
        has been tested)

    Returns
    -------
    g : sns.FacetGrid
        FacetGrid containing the first plot
    fig : plt.Figure
        Figure containing the second plot

    """
    params, fig_width = style.plotting_style(context, figsize='full')
    plt.style.use(params)
    ax_height = (fig_width / 3)
    if not isinstance(dfs, list):
        dfs = [dfs]
    if not isinstance(trained_models, list):
        trained_models = [trained_models]
    df_overall = []
    voxel_overall = []
    if df_filter_string is not None:
        if 'border' in df_filter_string:
            raise Exception("Don't want to drop voxels near border when computing relationship "
                            "between pRF size and eccentricity!")
        df_filter = model.construct_df_filter(df_filter_string)
    for i, (df, trained_model) in enumerate(zip(dfs, trained_models)):
        if 'subject' not in df.columns:
            # this way it will run even if there's no subject specified
            df['subject'] = 'none'
        if df_filter_string is not None:
            df = df_filter(df).reset_index()
        voxel = df.drop_duplicates('voxel')
        a, b = np.polyfit(df.eccen.values, df.sigma.values, 1)
        data = {'subject': df.subject.unique()[0],
                'sigma_slope': a, 'sigma_intercept': b,
                'period_slope': trained_model.sf_ecc_slope.item(),
                'period_intercept': trained_model.sf_ecc_intercept.item()}
        voxel['sigma_slope'] = a
        voxel['sigma_intercept'] = b
        df_overall.append(pd.DataFrame(data, index=[i]))
        voxel_overall.append(voxel)
    df_overall = pd.concat(df_overall)
    voxel_overall = pd.concat(voxel_overall)
    g = sns.FacetGrid(data=voxel_overall, col='subject', col_wrap=3,
                      height=ax_height,
                      col_order=plotting.get_order('subject', col_unique=voxel_overall.subject.unique()))

    def hist_and_reg(*args, **kwargs):
        to_return = sns.histplot(*args, **kwargs)
        data = kwargs.pop('data')
        x = np.linspace(data.eccen.min(), data.eccen.max())
        ax = plt.gca()
        ax.plot(x, data.sigma_slope.unique()[0]*x+data.sigma_intercept.unique()[0],
                'r--')
        return to_return

    g.map_dataframe(hist_and_reg, x='eccen', y='sigma')

    df_overall = pd.melt(df_overall, ['subject'])
    df_overall['parameter'] = df_overall.variable.map(lambda x: x.split('_')[-1])
    df_overall['phenomenon'] = df_overall.variable.map(lambda x: x.split('_')[0])
    df_overall = pd.pivot_table(df_overall, 'value', ['subject', 'parameter'],
                                'phenomenon').reset_index()
    params, fig_width = style.plotting_style(context, figsize='half')
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width))
    palette = plotting.get_palette('subject', col_unique=df_overall.subject.unique())
    for i, (n, h) in enumerate(df_overall.groupby('subject')):
        c = palette[i]
        ax.scatter('period', 'sigma', data=h, label=n, facecolors=[c, 'none'],
                   edgecolors=c)
        ax.plot('period', 'sigma', data=h, label='', c=c)
    # this is the subject legend -- maybe don't need?
    # put it outside the axis
    ax.add_artist(plt.legend(bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes))
    sc2 = ax.scatter([], [], edgecolors='k', facecolors='none')
    sc = ax.scatter([], [], edgecolors='k', facecolors='k')
    plt.legend([sc, sc2], ['intercept', 'slope'])
    ax.set(xlabel='Preferred period parameter', ylabel='pRF sigma parameter')
    return g, fig


def compare_surface_area_and_pref_period(model_parameter_df, subjects,
                                         mgz_template, target_ecc=6,
                                         eccen_range=(1, 12),
                                         context='paper'):
    """Compare V1 surface area and preferred period

    Compare the surface area of V1 and the preferred period across subjects.

    Parameters
    ----------
    model_parameter_df : pd.DataFrame
        dataframe containing all the model parameter values, across
        subjects.
    subjects : str or list
        Strings identifying the subjects to investigate. If a list, a list of
        those.
    mgz_template : str
        template string with the path to the varea and eccen mgz files. Should
        contain the format keys: subject, hemi, prop
    target_ecc : int, optional
        The eccentricity at which we compute the preferred period for each
        subject, for comparison against visual area
    eccen_range : tuple, optional
        Range of eccentricites to use for creating the 'surface area stimulus'
        (the surface area of V1 that corresponds to the portion of the visual
        field the stimulus was presented on)
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in seaborn's
        set_context function). if poster, will scale things up (but only paper
        has been tested)

    Returns
    -------
    g : sns.FacetGrid
        FacetGrid containing the first plot
    linreg : pd.DataFrame
        DataFrame giving the coefficient, intercept, and R^2 of a linear
        regression model between the surface area and preferred period at
        target eccentricity, bootstrapped 1000 times across subjects

    """
    params, fig_width = style.plotting_style(context, figsize='full')
    plt.style.use(params)
    ax_height = (fig_width / 2)
    feature_df = analyze_model.create_feature_df(model_parameter_df,
                                                 eccentricity=[target_ecc])
    # average preferred period over orientations and retinotopic angles
    feature_df = feature_df.groupby('subject').mean()
    if not isinstance(subjects, list):
        subjects = [subjects]
    df = []
    # feature_df doesn't have actual subject names, it has the plot ones,
    # so in order to grab the data, we need to map to that.
    map_sub = dict(zip(plotting.SUBJECT_ORDER, plotting.SUBJECT_PLOT_ORDER))
    for sub in subjects:
        if isinstance(sub, str):
            sub = ny.freesurfer_subject(sub.replace('sub-', ''))
        data = {'subject': 'sub-' + sub.name}
        plot_sub = map_sub[data['subject']]
        data[f'preferred_period_at_{target_ecc}_deg'] = feature_df.loc[plot_sub]['Preferred period (deg)']
        for hemi in ['lh', 'rh']:
            eccen = ny.load(mgz_template.format(hemi=hemi, subject=data['subject'],
                                                prop='eccen'))
            varea = ny.load(mgz_template.format(hemi=hemi, subject=data['subject'],
                                                prop='varea'))
            surface_area = sub.hemis[hemi].prop('midgray_surface_area')
            data['surface_area_full'] = np.sum(surface_area[varea==1])
            # need to stack logical_and, because they only operate on two
            # boolean arrays at a time
            mask = np.logical_and(varea==1,
                                  np.logical_and(eccen > eccen_range[0],
                                                 eccen < eccen_range[1]))
            data['surface_area_stimulus'] = np.sum(surface_area[mask])
            data['hemi'] = hemi
            df.append(pd.DataFrame(data, [0]))
    df = pd.concat(df).reset_index(drop=True)
    df = df.groupby('subject').agg({'surface_area_full': np.sum,
                                    'surface_area_stimulus': np.sum,
                                    f'preferred_period_at_{target_ecc}_deg': np.mean
                                    }).reset_index()
    # we're only going to use the full V1 surface area -- results look similar
    # either way
    df = df.drop(columns='surface_area_stimulus')
    df = pd.melt(df, ['subject', f'preferred_period_at_{target_ecc}_deg'],
                 var_name='surface_area_type', value_name='surface_area_value')

    hue_order = plotting.get_order('subject', col_unique=df.subject.unique())
    palette = plotting.get_palette('subject', col_unique=df.subject.unique())
    g = sns.relplot(x='surface_area_value', y=f'preferred_period_at_{target_ecc}_deg',
                    hue='subject', aspect=1, col='surface_area_type', data=df,
                    height=ax_height, hue_order=hue_order, palette=palette,
                    legend=False, zorder=100)
    xlim = g.axes[0, 0].get_xlim()
    ylim = g.axes[0, 0].get_ylim()
    # this way, we use seaborn to plot the linear regression across subjects,
    # while the above plots the points. we have to do this separately because
    # lmplot (the regression equivalent of relplot) won't let us plot
    # regression line aross values of the hue variable
    sns.regplot(x='surface_area_value', y=f'preferred_period_at_{target_ecc}_deg',
                data=df, scatter=False, ax=g.ax, ci=68)
    if context != 'paper':
        g.set_titles("{col_name}")
    else:
        g.set_titles('')
    g.set(xlabel='V1 surface area ($\mathrm{mm}^2$)',
          ylabel=f'Preferred period at {target_ecc} degrees (deg)',
          xlim=xlim, ylim=ylim)
    g.fig.subplots_adjust(wspace=.1)

    if len(df) != df.subject.nunique():
        raise Exception("The bootstrapping of models assumes that we have one"
                        "point per subject!")
    linreg = []
    true_x = df.surface_area_value.values.reshape(-1, 1)
    true_y = df[f'preferred_period_at_{target_ecc}_deg'].values
    for n in range(1000):
        model = linear_model.LinearRegression()
        # this resamples the subjects
        tmp = df.sample(len(df), replace=True)
        model.fit(tmp.surface_area_value.values.reshape(-1, 1),
                  tmp[f'preferred_period_at_{target_ecc}_deg'].values)
        data = {'coeff': model.coef_, 'intercept': model.intercept_,
                'bootstrap_num': n, 'R^2': model.score(true_x, true_y)}
        linreg.append(pd.DataFrame(data, index=[n]))
    linreg = pd.concat(linreg)
    return g, linreg


def feature_difference_plot(df, precision_df, diff_col='Visual field',
                            feature_type='preferred_period', seed=0,
                            n_bootstraps=100, feature_kwargs={},
                            feature_gb_cols=['subject', 'Visual field'],
                            context='paper', **plot_kwargs):
    """Plot difference between two features along some dimension.

    Intended use is to plot the difference between the preferred period of
    models fit to different portions of the visual field.

    We take the difference within subjects, and then bootstrap the
    precision-weighted average of that across subjects

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing all the model parameter values, across
        subjects.
    precision_df : pd.dataFrame
        dataframe containing the precision for each scanning session in
        df.
    diff_col : str, optional
        The column with the dimension to take difference across. Must only have
        two values.
    feature_type : str, optional
        what type of feature to create the plot for. See
        analyze_model.create_feature_df for possible values and explanations;
        only preferred_period has been tested.
    seed : int, optional
        seed for numpy's RNG.
    n_bootstraps : int, optional
        the number of independent bootstraps to draw
    feature_kwargs : dict, optional
        Additional arguments to pass to analyze_mode.create_feature_df
    feature_gb_cols : list, optional
        List of columns to groupby when creating feature_df. when we groupby
        these columns, each subset should give a single model.
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in seaborn's
        set_context function). if poster, will scale things up (but only paper
        has been tested)
    plot_kwargs :
        passed to plotting.feature_df_plot

    Returns
    -------
    g : sns.FacetGrid
        FacetGrid containing the plot.

    """
    params, fig_width = style.plotting_style(context, figsize='full')
    plot_kwargs.setdefault('height', fig_width / 2)
    plt.style.use(params)
    if context == 'paper':
        plot_kwargs.setdefault('xlim', (0, 11.55))
        plot_kwargs.setdefault('title', '')
    feature = analyze_model.create_feature_df(df, feature_type,
                                              gb_cols=feature_gb_cols,
                                              **feature_kwargs)
    if feature_type.startswith('preferred_period'):
        feature_type = 'Preferred period (deg)'
    elif feature_type == 'max_amplitude':
        feature_type = 'Max amplitude'
    idx_cols = [c for c in feature.columns if c not in [diff_col, feature_type]]
    diff_vals = sorted(feature[diff_col].unique())
    assert len(diff_vals) == 2, 'For now, diff_col must have two values!'
    feature = feature.pivot_table(feature_type, idx_cols, diff_col).reset_index()
    diff_name = f'Difference in {feature_type}'
    feature[diff_name] = feature[diff_vals[0]] - feature[diff_vals[1]]

    feature = feature.merge(precision_df, on=['subject'])
    idx_cols.remove('subject')
    feature = precision_weighted_bootstrap(feature, seed, n_bootstraps,
                                           diff_name, idx_cols,
                                           precision_col='precision')
    g = plotting.feature_df_plot(feature, y=diff_name, hue=None, yticks=None,
                                 col=None, aspect=1, color='k', top=.9,
                                 plot_func=plotting.scatter_ci_dist, join=True,
                                 ci_mode='fill', draw_ctr_pts=False,
                                 **plot_kwargs)
    return g


def stimulus_frequency(df, context='paper', **kwargs):
    """Show chosen w_r and w_a values.

    This takes the first sub-plot from plotting.stimuli_properties() and
    figure-ifies it, making it appropriate to insert in the paper.

    Parameters
    ----------
    df : pd.DataFrame
        Stimulus information dataframe.
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in seaborn's
        set_context function). if poster, will scale things up (but only paper
        has been tested)

    Returns
    -------
    g : sns.FacetGrid
        Facetgrid containing this plot


    """
    params, fig_width = style.plotting_style(context, figsize='half')
    params['axes.titlesize'] = '8'
    params['axes.labelsize'] = '8'
    params['legend.fontsize'] = '8'
    warnings.warn("We adjust the font size for axes titles, labels, and legends down to "
                  "8pts (so this will probably look wrong if context is not paper)!")
    plt.style.use(params)
    df = df.dropna()
    df.class_idx = df.class_idx.astype(int)
    df = df.drop_duplicates('class_idx').set_index('class_idx')
    df = df.rename(columns={'index': 'stimulus_index'})
    df = first_level_analysis._add_freq_metainfo(df)
    pal = plotting.get_palette('stimulus_type', 'relative',
                               df.stimulus_superclass.unique(),
                               True)
    # use the less-ambiguous labels for plotting
    df.stimulus_superclass = df.stimulus_superclass.apply(lambda x: plotting.SUPERCLASS_PLOT_LABELS.get(x, x))
    pal = {plotting.SUPERCLASS_PLOT_LABELS.get(k, k): v for k, v in pal.items()}

    def _focus(x):
        if x.name in [0, 10, 20, 30]:
            return 'first'
        elif x.name in [1, 11, 21, 31]:
            return 'second'
        elif x.name in [2, 12, 22, 32]:
            return 'third'
        else:
            return 'others'
    df['focus'] = df.apply(_focus, 1)
    g = sns.relplot(data=df[df.focus == 'others'], x='w_r', y='w_a', aspect=1,
                    hue='stimulus_superclass', zorder=2,
                    height=fig_width, palette=pal,
                    hue_order=['annulus', 'pinwheel',
                               'forward spiral',
                               'reverse spiral',
                               'mixtures'],)
    # This is the default edgewidth that seaborn's scatterplot uses
    default_ew = .08*plt.rcParams['lines.markersize']
    # this for loop allows us to tightly control the zorder, edgewidth, and
    # edgecolor of these points (which are the ones closest to the origin)
    for i, (lvl, ec, ew) in enumerate(zip(['third', 'second', 'first'],
                                      ['k', 'w', 'k'], [2, 1, 2])):
        g.map(sns.scatterplot, data=df[df.focus == lvl], x='w_r', y='w_a',
              hue='stimulus_superclass', zorder=3+i, palette=pal,
              hue_order=['annulus', 'pinwheel',
                         'forward spiral',
                         'reverse spiral',
                         'mixtures'],
              edgecolor=ec, linewidth=ew*default_ew)
    g.ax.set_aspect('equal')
    # we create and then remove the legend so that matplotlib can find the
    # legend info (we don't use seaborn's legend functionality so we can set
    # the location more specifically)
    g.legend.remove()
    # we grab these handles specifically because we've plotted each
    # stimulus_superclass multiple times -- we just want to grab one instance
    # of each.
    handles = [c for c in g.ax.collections[-5:]]
    plt.legend(handles, [c.get_label() for c in handles],
               title='', loc=(.85, .55))

    pal = plotting.get_palette('freq_space_distance', None,
                               df.freq_space_distance[:10])
    for r, c in zip(df.freq_space_distance[:10], pal):
        w = mpl.patches.Wedge((0, 0), r, -90, 90, fc=c, ec=c, width=1,
                              zorder=1)
        g.ax.add_patch(w)
    g.ax.axhline(linestyle='--', color='gray', zorder=0)
    g.ax.axvline(linestyle='--', color='gray', zorder=0)
    # because these are Latex, they appear a bit smaller
    g.ax.set_xlabel(r'$\omega_r$', fontsize=1.25*float(params['axes.labelsize']))
    g.ax.set_ylabel(r'$\omega_a$', fontsize=1.25*float(params['axes.labelsize']))
    return g


def behavioral_heatmap(outcomes, by_subject=False, context='paper'):
    """Heatmap showing behavioral data outcomes.

    Creates a heatmap with stimulus_superclass along the vertical axis and the
    outcome (in signal detection terms) along the horizontal axis.

    Percentage / colors are normalized so that correct reject + false alarm = 1
    and hit + miss = 1

    Parameters
    ----------
    outcomes : pd.DataFrame
        The behavioral dataframe created by the summarize_behavior rule.
    by_subject : bool, optional
        Whether to combine across subjects (False) or plot each subject as a
        separate heatmap (True)
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in seaborn's
        set_context function). if poster, will scale things up (but only paper
        has been tested)

    Returns
    -------
    fig : plt.Figure
        matplotlib figure containing the heatmap

    """
    params, fig_width = style.plotting_style(context, figsize={False: 'half', True: 'full'}[by_subject])
    plt.style.use(params)
    order = ['annulus', 'pinwheel', 'forward spiral', 'reverse spiral', 'mixtures', 'blank']
    pal = plotting.get_palette('stimulus_type', 'relative', outcomes.stimulus_superclass.unique(),
                               True)
    pal = {plotting.SUPERCLASS_PLOT_LABELS.get(k, k): v for k, v in pal.items()}
    def pivot_outcome(df):
        df = df.pivot('stimulus_superclass', 'outcome', 'percentage').fillna(0)
        # reorder
        return df.loc[order]

    outcomes.stimulus_superclass = outcomes.stimulus_superclass.apply(lambda x: plotting.SUPERCLASS_PLOT_LABELS.get(x, x))
    outcomes.outcome = outcomes.outcome.apply(lambda x: x.replace('_', ' '))
    gb_cols = ['stimulus_superclass', 'outcome_supercategory']
    if by_subject:
        gb_cols += ['subject']
    kwargs = {'vmin': 0, 'vmax': 1, 'fmt': '.02f', 'cmap': 'Blues',
              'annot': True}
    outcomes = outcomes.groupby(gb_cols + ['outcome']).n_trials.sum().reset_index()
    totals = outcomes.groupby(gb_cols)['n_trials'].sum().rename('total_trials')
    outcomes = outcomes.merge(totals, left_on=gb_cols, right_index=True)
    outcomes['percentage'] = outcomes['n_trials'] / outcomes.total_trials
    if not by_subject:
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width))
        ax = sns.heatmap(pivot_outcome(outcomes), square=True, ax=ax, **kwargs)
        fig = ax.figure
    else:
        def map_func(x, y, value, data=None, **kwargs):
            sns.heatmap(pivot_outcome(data), **kwargs)

        g = sns.FacetGrid(outcomes, col='subject', col_wrap=3, height=fig_width/3)
        cbar_ax = g.fig.add_axes([.92, .36, .02, .4])
        g.map_dataframe(map_func, 'stimulus_superclass', 'outcome', 'percentage',
                        cbar_ax=cbar_ax, **kwargs)
        # the meaning of x and y is backwards for heatmap compared to what
        # facetgrid expects, so we swap labels
        g.set_ylabels('stimulus_superclass')
        g.set_xlabels('outcome')
        # make room for the cbar axes
        g.fig.subplots_adjust(right=.9)
        fig = g.fig
    for ax in fig.axes:
        # remove axes labels for paper
        if context == 'paper':
            ax.set(xlabel='', ylabel='')
            # remove the subject tag, if present
            ax.set_title(ax.get_title().replace('subject = ', ''))
        # sometimes the rotation gets messed up, fo rno reason I can track
        # down, so this makes sure it's correct
        if ax.get_xticklabels() and ax.get_xticklabels()[0].get_visible():
            for lab in ax.get_xticklabels():
                lab.set_rotation(90)
        if ax.get_yticklabels() and ax.get_yticklabels()[0].get_visible():
            for lab in ax.get_yticklabels():
                # this skips over the colorbar axes labels
                if lab.get_text() in pal.keys():
                    lab.set_bbox({'fc': 'none', 'ec': pal[lab.get_text()], 'pad': 2})
                    lab.set_rotation(0)
        # don't need ticks for this plot
        ax.tick_params('x', bottom=False)
        ax.tick_params('y', left=False)
    return fig
