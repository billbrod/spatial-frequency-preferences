#!/usr/bin/python
"""functions to create the figures for publication
"""
import seaborn as sns
import numpy as np
from . import summary_plots
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


def prep_df(df, reference_frame):
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
    reference_frame : {'relative', 'absolute'}
        this determines which task we'll grab: task-sfprescaled (if
        "relative") or task-sfpconstant (if "absolute"). task-sfp is
        also in the relative reference frame, but doesn't have the
        contrast-rescaled stimuli, so we're considering it pilot data

    Returns
    -------
    df : pd.DataFrame
        The restricted dataframe.

    """
    if reference_frame == 'relative':
        df = df.query("task=='task-sfprescaled'")
    elif reference_frame == 'absolute':
        df = df.query("task=='task-sfpconstant'")
    if 'frequency_type' in df.columns:
        df = df.query("frequency_type=='local_sf_magnitude'")
    if 'varea' in df.columns:
        df = df.query("varea==1")
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
    g = _summarize_1d(df, reference_frame, y, row, col, height, ylim=(0, 4))
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


def _catplot(df, x='subject', y='cv_loss', hue='fit_model_type', height=8, aspect=1.5,
             ci=95, plot_kind='strip', x_rotate=False):
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
        g.fig.subplots_adjust(bottom=.25)
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
    g = _catplot(df, x='fit_model_type', y='demeaned_cv_loss', hue=hue, plot_kind=plot_kind,
                 aspect=.75, x_rotate=True)
    g.fig.suptitle("Demeaned cross-validated loss across model types")
    g.set(ylabel="Cross-validated loss (demeaned by subject)", xlabel="Model type")
    # if plot_kind=='point', then there is no legend, so the following
    # would cause an error
    if plot_kind == 'strip':
        g._legend.set_title(legend_title)
    return g


def pairplot():
    # put this here for now, I think it might be worth viewing but boy
    # is there a lot going on
    tmp = models_plot.query("task=='task-sfprescaled'")
    pivoted = pd.pivot_table(tmp, index=['subject', 'bootstrap_num'], columns='model_parameter', values='fit_value')
    pivoted = pivoted.reset_index()

    # this is a real outlier
    pivoted[pivoted.get('$a$') < 0]
    # throw away one real obvious outlier
    sns.pairplot(pivoted[pivoted.get('$a$') > 0], hue='subject', vars=parameters)
