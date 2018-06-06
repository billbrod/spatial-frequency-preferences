#!/usr/bin/python
"""plots from the tuning curve summary dataframe
"""
import matplotlib as mpl
# we do this because sometimes we run this without an X-server, and this backend doesn't need one
mpl.use('svg')
import inspect
import re
import os
import itertools
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


SAVE_TEMPLATE = ("tuning_curves_summary_plot_{mat_type}_{atlas_type}_{subject}_{session}_{task}_{s"
                 "timulus_superclass}_v{varea}_e{eccen_range}_row={row}_col={col}_hue={hue}_{plot_"
                 "func}_{y}.svg")


def main(summary_df, y='tuning_curve_peak', x='eccen', row='frequency_type', col='varea',
         hue='stimulus_superclass', save_path=None, sharey='row', sharex='all', plot_func=plt.plot,
         eccen_range=(1, 12), axes_style='whitegrid', eccen_soft_exclude=None, **kwargs):
    """make plots of tuning curve parameters

    kwargs can be any of varea, atlas_type, mat_type, subject, session, task, or
    stimulus_superclass, and will limit which data we show to only those whose values for that
    field match the specified value(s).

    plot_func: function to call with FacetGrid.map (or, if it's plot_median, plot_ci,
    scatter_ci_col, scatter_ci_dist, plot_median_fit, with FacetGrid.map_dataframe). Can be a
    single function or a list of functions.

    eccen_soft_exclude: None or 2-tuple with range of eccentricities. These eccentricites will be
    plotted but with a reduced alpha. They are handled by a separate call and so will not be
    connected to the rest of the data. The exception is if plot_func is
    sfp.plotting.plot_median_fit, in which case the data will not be used to fit the data, but we
    will plot the prediction for the relevant x values. The range should be within eccen_range
    (i.e., eccen_range=(1, 12) and eccen_soft_exclude=(11, 12), NOT eccen_range=(1, 11) and
    eccen_soft_exclude=(11, 12)) and should be on one end or the other.

    any kwarg arguments that don't correspond to columns of summary_df will be passed to
    sns.FacetGrid (if FacetGrid's constructor accepts it as an argument) or plot_func. note this
    means FacetGrid gets priority. If you have a kwarg that you want to pass to plot_func and it
    also gets accepted by FacetGrid (e.g., hue) then append '_plot' after its name (e.g.,
    hue_plot='stimulus_superclass')

    if plot_func is a list of functions, each of the kwargs discussed above can be either a single
    value or a list of values, in which case each of those values will be passed (in order) to the
    corresponding function. if one of the values is None, that kwarg will not be passed to that
    function.
    """
    for k, v in kwargs.copy().iteritems():
        if k in summary_df.columns:
            # user can specify None, to mean all
            if v is not None:
                if isinstance(v, basestring) or not hasattr(v, '__iter__'):
                    summary_df = summary_df[summary_df[k] == v]
                else:
                    summary_df = summary_df[summary_df[k].isin(v)]
            # but even if it's None, we want to get rid of it so we don't pass it to FacetGrid or
            # the plotting function
            kwargs.pop(k)
    summary_df = summary_df[(summary_df.eccen > eccen_range[0]) &
                            (summary_df.eccen < eccen_range[1])]
    legend_keys = summary_df[hue].unique()
    hue_kws = kwargs.pop('hue_kws', {})
    hue_order = list(kwargs.pop('hue_order', legend_keys))
    size = kwargs.pop('size', 5)
    palette = kwargs.pop('palette', 'deep')
    colors = sns.color_palette(palette, len(legend_keys))
    palette = dict((k, v) for k, v in zip(legend_keys, colors))
    if eccen_soft_exclude is not None:
        if eccen_soft_exclude[0] == eccen_range[0]:
            summary_df.loc[summary_df.eccen < eccen_soft_exclude[1], hue] = (
                summary_df.loc[summary_df.eccen < eccen_soft_exclude[1], hue].apply(
                    lambda x: 'exclude ' + x))
        elif eccen_soft_exclude[1] == eccen_range[1]:
            summary_df.loc[summary_df.eccen > eccen_soft_exclude[0], hue] = (
                summary_df.loc[summary_df.eccen > eccen_soft_exclude[0], hue].apply(
                    lambda x: 'exclude ' + x))
        if 'alpha' not in hue_kws:
            hue_kws['alpha'] = [1] * len(legend_keys) + [.5] * len(legend_keys)
        hue_order.extend(['exclude ' + i for i in hue_order])
        for k, v in palette.copy().iteritems():
            palette['exclude ' + k] = v
    if not hasattr(plot_func, '__iter__'):
        plot_func = [plot_func]
    joint_plot_func_kwargs = {}
    separate_plot_func_kwargs = [dict() for i in plot_func]
    additional_plot_args = []
    for k in set(kwargs.keys()) - set(inspect.getargspec(sns.FacetGrid.__init__)[0]):
        if k.endswith('_plot'):
            if k == 'hue_plot':
                # we need to handle hue slightly differently because of how FacetGrid.map handles
                # things
                additional_plot_args.append(kwargs.pop(k))
                if hue == additional_plot_args[-1]:
                    hue = None
            else:
                v = kwargs.pop(k)
                if isinstance(v, list):
                    for i, vi in enumerate(v):
                        if vi is not None:
                            separate_plot_func_kwargs[i][k.replace('_plot', '')] = vi
                else:
                    joint_plot_func_kwargs[k.replace('_plot', '')] = v
        else:
            v = kwargs.pop(k)
            if isinstance(v, list):
                for i, vi in enumerate(v):
                    if vi is not None:
                        separate_plot_func_kwargs[i][k] = vi
            else:
                joint_plot_func_kwargs[k] = v
    if 'ylim' not in kwargs.keys():
        if y in ['preferred_period', 'tuning_curve_bandwidth']:
            kwargs['ylim'] = (0, 10)
    with sns.axes_style(axes_style):
        g = sns.FacetGrid(summary_df, row, col, hue, aspect=1, size=size, sharey=sharey,
                          sharex=sharex, hue_order=hue_order, hue_kws=hue_kws, palette=palette,
                          **kwargs)
        for i, pf in enumerate(plot_func):
            tmp_kwargs = joint_plot_func_kwargs.copy()
            tmp_kwargs.update(separate_plot_func_kwargs[i])
            if pf.__name__ == 'plot_median_fit':
                tmp_kwargs['x_vals'] = summary_df.eccen.unique()
            if pf.__name__ in ['plot_median', 'plot_ci', 'scatter_ci_col', 'scatter_ci_dist',
                               'plot_median_fit']:
                # these functions require map_dataframe, since they need some extra info
                g.map_dataframe(pf, x, y, *additional_plot_args, **tmp_kwargs)
            else:
                g.map(pf, x, y, *additional_plot_args, **tmp_kwargs)
        if row is not None or col is not None:
            titles = "{row_name} | {col_name}"
            if col == 'varea':
                titles = titles.replace("{col_name}", "V{col_name}")
            elif row == 'varea':
                titles = titles.replace("{row_name}", "V{row_name}")
            if row is None:
                titles = titles.replace("{row_name} | ", "")
            elif col is None:
                titles = titles.replace(" | {col_name}", "")
            g.set_titles(titles)
        g._legend_data = dict((k, v) for k, v in g._legend_data.iteritems() if k in legend_keys)
        g.add_legend(label_order=legend_keys)
        if save_path is not None:
            g.fig.savefig(save_path)
    return g


if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        description=("Create plots summarizing tuning curves. the save_path will be constructed "
                     "based on the various arguments and placed in the same directory as the "
                     "tuning_curve_summary_path. If your specified tuning_curve_summary dataframe"
                     " contains multiple visual areas, atlas types, design matrix types, subjects,"
                     " sessions, tasks, or stimulus superclass, you should probably either "
                     "restrict to only show one of them (using the appropriate arguments) or facet"
                     " on them (using row, col, or hue). Otherwise your plots will probably be "
                     "incomprehensible"),
        formatter_class=CustomFormatter
        )
    parser.add_argument("tuning_curve_summary_path",
                        help=("Path to tuning curve summary dataframe that contains info we'll "
                              "plot. Note that this must end in tuning_curves_summary.csv or the"
                              " creation of the save path will not be correct"))
    parser.add_argument("--y", default="tuning_curve_peak",
                        help="What to plot on the y-axis")
    parser.add_argument("--x", default="eccen", help="What to plot on the x-axis")
    parser.add_argument("--row", default="frequency_type",
                        help="What variable to facet onto different rows")
    parser.add_argument("--col", default="varea",
                        help="What variable to facet onto different columns")
    parser.add_argument("--hue", default="stimulus_superclass",
                        help="What variable to facet onto different colors")
    parser.add_argument("--sharey", default="row",
                        help=("Whether to share y values across plots. Possible values are all, "
                              "none, row, or col"))
    parser.add_argument("--sharex", default="all",
                        help=("Whether to share x values across plots. Possible values are all, "
                              "none, row, or col"))
    parser.add_argument("--plot_func", default="plt.plot",
                        help=("What function to use to plot data. Must be a matplotlib (start with"
                              " plt.) or seaborn (start with sns.) function"))
    parser.add_argument("--varea", default=[1, 2, 3], nargs="+", type=int,
                        help=("Which visual area(s) to plot. By default we plot all of them."))
    parser.add_argument("--atlas_type", default=None,
                        help=("Which atlas_type to plot. By default we don't restrict, plotting "
                              "whatever is in the specified dataframe"))
    parser.add_argument("--mat_type", default=None,
                        help=("Which mat_type to plot. By default we don't restrict, plotting "
                              "whatever is in the specified dataframe"))
    parser.add_argument("--subject", default=None, nargs='+',
                        help=("Which subject(s) to plot. By default we don't restrict, plotting "
                              "whatever is in the specified dataframe"))
    parser.add_argument("--session", default=None, nargs='+',
                        help=("Which session(s) to plot. By default we don't restrict, plotting "
                              "whatever is in the specified dataframe"))
    parser.add_argument("--task", default=None, nargs='+',
                        help=("Which task(s) to plot. By default we don't restrict, plotting "
                              "whatever is in the specified dataframe"))
    parser.add_argument("--stimulus_superclass", default=None, nargs='+',
                        help=("Which stimulus_superclass(es) to plot. By default we don't restrict"
                              ", plotting whatever is in the specified dataframe"))
    parser.add_argument("--eccen_range", default=(1, 12), nargs=2, type=int,
                        help=("What range of eccentricities to consider. Will throw out all above "
                              "and below"))
    args = vars(parser.parse_args())
    tuning_curve_path = args.pop('tuning_curve_summary_path')
    summary_df = pd.read_csv(tuning_curve_path)
    save_path = SAVE_TEMPLATE
    save_kwargs = {}
    for k, v in args.iteritems():
        if k in save_path:
            if v is None:
                save_path = save_path.replace("{%s}" % k, '')
            else:
                if isinstance(v, list) or isinstance(v, tuple):
                    if k in ['varea', 'eccen_range']:
                        v = "-".join(str(i) for i in v)
                    else:
                        v = ",".join(str(i) for i in v)
                save_kwargs[k] = v.replace('_', '-').replace('plt.', '').replace('sns.', '')
    save_path = (tuning_curve_path.replace('tuning_curves_summary.csv', '') + "_" +
                 save_path.format(**save_kwargs))
    # we don't want two or more underscores in a row
    save_path = re.sub("_{2,}", "_", save_path)
    plot_func = args.pop('plot_func')
    if plot_func.split('.')[0] not in ['plt', 'sns']:
        raise Exception("plot_func must be a matplotlib or seaborn function! Can't "
                        "handle %s" % plot_func)
    args['plot_func'] = eval(plot_func)
    if args['y'] == 'preferred_period':
        # it only makes sense to look at preferred period with the local spatial frequency
        args['frequency_type'] = 'Local spatial frequency (cpd)'
    main(summary_df, save_path=save_path, **args)
