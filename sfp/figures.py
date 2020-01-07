#!/usr/bin/python
"""functions to create the figures for publication
"""
import seaborn as sns
from . import summary_plots
from . import plotting


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
    height : int, optional
        height of each plot facet

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    pal = plotting.stimulus_type_palette(reference_frame)
    hue_order = plotting.stimulus_type_order(reference_frame)
    g = summary_plots.main(df, row=row, col=col, y='preferred_period', eccen_range=(0, 11),
                           hue_order=hue_order, linewidth=2, xlim=(0, 12), ylim=(0, 4),
                           plot_func=[plotting.plot_median_fit, plotting.scatter_ci_dist],
                           x_jitter=[None, .2], height=height, palette=pal)
    g.set_ylabels('Preferred period (dpc)')
    g.set_xlabels('Eccentricity (deg)')
    g.set(yticks=[0, 1, 2, 3])
    g._legend.set_title("Stimulus class")
    g.fig.subplots_adjust(top=.85)
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
    height : int, optional
        height of each plot facet

    Returns
    -------
    g : sns.FacetGrid
        seaborn FacetGrid object containing the plot

    """
    pal = plotting.stimulus_type_palette(reference_frame)
    hue_order = plotting.stimulus_type_order(reference_frame)
    g = summary_plots.main(df, row=row, col=col, y='tuning_curve_bandwidth', eccen_range=(0, 11),
                           hue_order=hue_order, linewidth=2, xlim=(0, 12), x_jitter=[None, .2],
                           plot_func=[plotting.plot_median_fit, plotting.scatter_ci_dist],
                           height=height, palette=pal)
    g.set_ylabels('Tuning curve FWHM (octaves)')
    g.set_xlabels('Eccentricity (deg)')
    g._legend.set_title("Stimulus class")
    g.fig.subplots_adjust(top=.85)
    g.fig.suptitle("Full-Width Half-Max of 1d tuning curves in each eccentricity band")
    return g
