#!/usr/bin/python
"""functions to create the figures for publication
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from . import summary_plots
from . import analyze_model
from . import plotting


def _demean_df(df, gb_cols=['subject'], y='cv_loss'):
    """demean a column of the dataframe

    Calculate the mean of `y` across the values in some other column(s)
    `gb_cols`, then demean `y` and return df with a new column,
    `demeaned_{y}`.

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
    df[f'demeaned_{y}'] = df[y] - df[f'{y}_mean']
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


def pref_period_1d(df, reference_frame='relative', row='session', col='subject', height=4):
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

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    g = _summarize_1d(df, reference_frame, 'preferred_period', row, col, height, ylim=(0, 4))
    g.set_ylabels('Preferred period (dpc)')
    g.set(yticks=[0, 1, 2, 3])
    g.fig.suptitle("Preferred period of 1d tuning curves in each eccentricity band")
    return g


def bandwidth_1d(df, reference_frame='relative', row='session', col='subject', height=4):
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

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    g = _summarize_1d(df, reference_frame, 'tuning_curve_bandwidth', row, col, height)
    g.set_ylabels('Tuning curve FWHM (octaves)')
    g.fig.suptitle("Full-Width Half-Max of 1d tuning curves in each eccentricity band")
    return g


def _catplot(df, x='subject', y='cv_loss', hue='fit_model_type', height=8, aspect=.75,
             ci=95, plot_kind='strip', x_rotate=True):
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

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    hue_order = plotting.get_order(hue, col_unique=df[hue].unique())
    order = plotting.get_order(x, col_unique=df[x].unique())
    pal = plotting.get_palette(hue, col_unique=df[hue].unique())
    g = sns.catplot(x, y, hue, data=df, hue_order=hue_order, legend='full', height=height,
                    kind=plot_kind, aspect=aspect, order=order, palette=pal, ci=ci,
                    estimator=np.median)
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


def cross_validation_demeaned(df):
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

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    df = _demean_df(df)
    g = _catplot(df, y='demeaned_cv_loss')
    g.fig.suptitle("Demeaned cross-validated loss across subjects")
    g.set(ylabel="Cross-validated loss (demeaned by subject)", xlabel="Subject")
    g._legend.set_title("Model type")
    return g


def cross_validation_model(df, plot_kind='strip'):
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
    g = _catplot(df, x='fit_model_type', y='demeaned_cv_loss', hue=hue, plot_kind=plot_kind)
    g.fig.suptitle("Demeaned cross-validated loss across model types")
    g.set(ylabel="Cross-validated loss (demeaned by subject)", xlabel="Model type")
    # if plot_kind=='point', then there is no legend, so the following
    # would cause an error
    if plot_kind == 'strip':
        g._legend.set_title(legend_title)
    return g


def model_parameters(df, plot_kind='point'):
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

    Returns
    -------
    fig : plt.Figure
        Figure containin the plot

    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 10), gridspec_kw={'width_ratios': [.15, .3, .6]})
    order = plotting.get_order('model_parameter', col_unique=df.model_parameter.unique())
    if plot_kind == 'point':
        pal = plotting.get_palette('model_parameter', col_unique=df.model_parameter.unique(),
                                   as_dict=True)
    elif plot_kind == 'strip':
        pal = plotting.get_palette('subject', col_unique=df.subject.unique(), as_dict=True)
    elif plot_kind == 'dist':
        pal = plotting.get_palette('subject', col_unique=df.subject.unique(), as_dict=True)
    for i, ax in enumerate(axes):
        cat = ['sigma', 'eccen', 'orientation'][i]
        tmp = df.query("param_category==@cat")
        ax_order = [i for i in order if i in tmp.model_parameter.unique()]
        if plot_kind == 'point':
            sns.pointplot('model_parameter', 'fit_value', 'model_parameter', data=tmp,
                          estimator=np.median, ax=ax, order=ax_order, palette=pal, ci=95)
        elif plot_kind == 'strip':
            sns.stripplot('model_parameter', 'fit_value', 'subject', data=tmp, ax=ax,
                          order=ax_order, palette=pal)
        elif plot_kind == 'dist':
            handles, labels = [], []
            for n, g in tmp.groupby('subject'):
                dots, _, _ = plotting.scatter_ci_dist('model_parameter', 'fit_value', data=g,
                                                      label=n, ax=ax, x_jitter=.2, color=pal[n],
                                                      x_order=ax_order)
                handles.append(dots)
                labels.append(n)
        if ax.legend_:
            ax.legend_.remove()
        if i==2 and plot_kind in ['strip', 'dist']:
            if plot_kind == 'strip':
                ax.legend(loc=(1.01, .3), borderaxespad=0, frameon=False)
            else:
                ax.legend(handles, labels, loc=(1.01, .3), borderaxespad=0, frameon=False)
        ax.axhline(color='grey', linestyle='dashed')
        ax.set(ylabel='Fit value', xlabel='Parameter')
    fig.suptitle("Model parameters")
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
    pal = plotting.get_palette('subject', col_unique=df.subject.unique())
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
                    feature_type='pref-period'):
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
                                                 eccentricity=[2, 5, 10], 
                                                 retinotopic_angle=np.linspace(0, 2*np.pi, 49))
            g = plotting.feature_df_polar_plot(df, col='subject', row='Eccentricity (deg)',
                                               r='Preferred period (dpc)', plot_func=plot_func, **kwargs)
        elif feature_type == 'iso-pref-period':
            df = analyze_model.create_feature_df(df, 'preferred_period_contour',
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
    return g
