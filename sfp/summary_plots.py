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
         eccen_range=(1, 12), **kwargs):
    """make plots of tuning curve parameters

    kwargs can be any of varea, atlas_type, mat_type, subject, session, task, or
    stimulus_superclass, and will limit which data we show to only those whose values for that
    field match the specified value(s).

    any kwarg arguments that don't correspond to columns of summary_df will be passed to
    sns.FacetGrid (if FacetGrid's constructor accepts it as an argument) or plot_func. note this
    means FacetGrid gets priority. If you have a kwarg that you want to pass to plot_func and it
    also gets accepted by FacetGrid (e.g., hue) then append '_plot' after its name (e.g.,
    hue_plot='stimulus_superclass')
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
    plot_func_kwargs = {}
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
                plot_func_kwargs[k.replace('_plot', '')] = kwargs.pop(k)
        else:
            plot_func_kwargs[k] = kwargs.pop(k)
    if 'ylim' not in kwargs.keys():
        if y in ['preferred_period', 'tuning_curve_bandwidth']:
            kwargs['ylim'] = (0, 10)
    with sns.axes_style('whitegrid'):
        g = sns.FacetGrid(summary_df, row, col, hue, aspect=1, size=5, sharey=sharey,
                          sharex=sharex, **kwargs)
        g.map(plot_func, x, y, *additional_plot_args, **plot_func_kwargs)
        if col == 'varea':
            titles = "{row_name} | V{col_name}"
        elif row == 'varea':
            titles = "V{row_name} | {col_name}"
        else:
            titles = "{row_name} | {col_name}"
        g.set_titles(titles)
        g.add_legend()
        if save_path is not None:
            g.fig.savefig(save_path)


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
