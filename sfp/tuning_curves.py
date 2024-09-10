#!/usr/bin/python
"""fit tuning curves to first level results
"""
import matplotlib as mpl
# we do this because sometimes we run this without an X-server, and this backend doesn't need
# one. We set warn=False because the notebook uses a different backend and will spout out a big
# warning to that effect; that's unnecessarily alarming, so we hide it.
mpl.use('svg', warn=False)
import argparse
import warnings
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import numpy as np


def log_norm_pdf(x, a, mode, sigma):
    """the pdf of the log normal distribution, with a scale factor
    """
    # note that mode here is the actual mode, for us, the peak spatial frequency. this differs from
    # the 2d version we have, where we we have np.log2(x)+np.log2(p), so that p is the inverse of
    # the preferred period, the ivnerse of the mode / the peak spatial frequency.
    pdf = a * np.exp(-(np.log2(x)-np.log2(mode))**2/(2*sigma**2))
    return pdf


def get_tuning_curve_xy(a, mode, sigma, x=None, norm=False):
    if x is None:
        x = np.logspace(-20, 20, 20000, base=2)
    y = log_norm_pdf(x, a, mode, sigma)
    if norm:
        y /= y.max()
    return x, y


def get_tuning_curve_xy_from_df(df, x=None, norm=False):
    """given a dataframe with one associated tuning curve, return x and y of that tuning curve
    """
    params = {'x': x}
    for param, param_label in [('a', 'tuning_curve_amplitude'), ('mode', 'tuning_curve_peak'),
                               ('sigma', 'tuning_curve_sigma')]:
        if df[param_label].nunique() > 1:
            raise Exception("Only one tuning curve can be described in df \n%s!" % df)
        params[param] = df[param_label].unique()[0]
    return get_tuning_curve_xy(norm=norm, **params)


def log_norm_describe_full(a, mode, sigma):
    """calculate and return many descriptions of the log normal distribution

    returns the bandwidth (in octaves), low and high half max values of log normal
    curve, inf_warning, x and y.

    inf_warning is a boolean which indicates whether we calculate the variance to be infinite. this
    happens when the mode is really small as the result of an overflow and so you should probably
    examine this curve to make sure things are okay

    x and y are arrays of floats so you can plot the tuning curve over a reasonable range.
    """
    mu = np.log(mode) + sigma**2
    # we compute this because the std dev is always larger than the bandwidth, so we can use this
    # to make sure we grab the right patch of x values
    var = (np.exp(sigma**2) - 1) * (np.exp(2*mu + sigma**2))
    inf_warning = False
    if np.isinf(var):
        # if the peak of the curve would be at a *really* low or *really* high value, the variance
        # will be infinite (not really, but because of computational issues) and so we need to
        # handle it separately. this really shouldn't happen anymore, since I've constrained the
        # bounds of the mode
        if np.log2(mode) < 0:
            x = np.logspace(-300, 100, 100000, base=2)
        else:
            x = np.logspace(-100, 300, 100000, base=2)
        inf_warning = True
    else:
        xmin, xmax = np.floor(np.log2(mode) - 5*sigma), np.ceil(np.log2(mode) + 5*sigma)
        x = np.logspace(xmin, xmax, int(1000*(xmax - xmin)), base=2)
    x, y = get_tuning_curve_xy(a, mode, sigma, x)
    half_max_idx = abs(y - (y.max() / 2.)).argsort()
    if (not (x[half_max_idx[0]] > mode and x[half_max_idx[1]] < mode) and
       not (x[half_max_idx[0]] < mode and x[half_max_idx[1]] > mode)):
        print(a, mode, sigma)
        raise Exception("Something went wrong with bandwidth calculation! halfmax x values %s and"
                        " %s must lie on either side of max %s!" % (x[half_max_idx[0]],
                                                                    x[half_max_idx[1]], mode))
    low_half_max = np.min(x[half_max_idx[:2]])
    high_half_max = np.max(x[half_max_idx[:2]])
    bandwidth = np.log2(high_half_max) - np.log2(low_half_max)
    return bandwidth, low_half_max, high_half_max, inf_warning, x, y


def create_problems_report(fit_problems, inf_problems, save_path):
    """create html report showing problem cases
    """
    plots_save_path = os.path.join(save_path.replace('.html', '') + "_report_plots", "plot%03d.svg")
    if not os.path.isdir(os.path.dirname(plots_save_path)):
        os.makedirs(os.path.dirname(plots_save_path))
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'report_templates.json')) as f:
            report_template_strs = json.load(f)
    report_text = report_template_strs['HEAD']
    for title, problems in zip(["fitting curves", "finding bandwidths"],
                               [fit_problems, inf_problems]):
        report_text += "<h2>Those cases that had problems %s</h2>" % title
        for i, (labels, (x, y), (datax, datay)) in enumerate(problems):
            report_text += labels.style.render()
            plt.scatter(datax, datay)
            plt.semilogx(x, y, basex=2)
            plt.savefig(plots_save_path % i)
            plt.close()
            report_text += report_template_strs['FIGURE'] % (plots_save_path % i)
    report_text += report_template_strs['TAIL']
    with open(save_path, 'w') as f:
        f.write(report_text)


# add bounds to the command line?
def main(df, save_path=None, mode_bounds=(2**(-5), 2**11), ampl_bounds=(0, 10),
         sigma_bounds=(0, 10)):
    """fit tuning curve to first level results dataframe

    note that your mode_bounds are stimuli dependent. for the log-normal stimuli (for either the
    pilot or regular stimuli), the default value above works well. for the constant stimuli,
    (2**(-15), 2**8) seems to work
    """
    if 'bootstrap_num' in df.columns:
        additional_cols = ['bootstrap_num']
    else:
        additional_cols = []
        df = df.rename(columns={'amplitude_estimate_median': 'amplitude_estimate'})
    melt_cols = ['varea', 'eccen', 'amplitude_estimate', 'stimulus_superclass',
                 'freq_space_angle', 'baseline'] + additional_cols
    df = df[['freq_space_distance', 'local_sf_magnitude'] + melt_cols]
    df = pd.melt(df, melt_cols, var_name='frequency_type', value_name='frequency_value')
    gb_columns = ['varea', 'eccen', 'stimulus_superclass', 'frequency_type'] + additional_cols
    gb = df.groupby(gb_columns)
    tuning_df = []
    fit_problems, inf_problems = [], []
    for n, g in gb:
        # we want a sense of where this is, in order to figure out if it stalled out.
        str_labels = ", ".join("%s: %s" % i for i in zip(gb_columns, n))
        print("\nCreating tuning curves for: %s" % str_labels)
        fit_warning = False
        if 'mixtures' in n or 'off-diagonal' in n or 'baseline' in n:
            # then these points all have the same frequency and so we can't fit a frequency tuning
            # curve to them
            continue
        values_to_fit = zip(g.frequency_value.values, g.amplitude_estimate.values)
        # in python2, zip returned a list. in python3, it returns an iterable. we don't actually
        # want to iterate through it here, just index into it, so we convert it to a list
        values_to_fit = list(zip(*sorted(values_to_fit, key=lambda pair: pair[0])))
        fit_success = False
        maxfev = 100000
        tol = 1.5e-08
        while not fit_success:
            try:
                mode_guess = np.log(np.mean(values_to_fit[0]))
                if mode_guess < mode_bounds[0]:
                    mode_guess = 1
                popt, _ = optimize.curve_fit(log_norm_pdf, values_to_fit[0], values_to_fit[1],
                                             maxfev=maxfev, ftol=tol, xtol=tol,
                                             p0=[1, mode_guess, 1],
                                             # optimize.curve_fit needs to be able to take the
                                             # len(bounds), and zips have no length
                                             bounds=list(zip(ampl_bounds, mode_bounds, sigma_bounds)))
                fit_success = True
            except RuntimeError:
                fit_warning = True
                maxfev *= 10
                tol /= np.sqrt(10)
        # popt contains a, mode, and sigma, in that order
        bandwidth, lhm, hhm, inf_warning, x, y = log_norm_describe_full(popt[0], popt[1], popt[2])
        tuning_df.append(g.assign(tuning_curve_amplitude=popt[0], tuning_curve_peak=popt[1],
                         tuning_curve_sigma=popt[2], preferred_period=1./popt[1],
                         tuning_curve_bandwidth=bandwidth, high_half_max=hhm, low_half_max=lhm,
                         fit_warning=fit_warning, inf_warning=inf_warning, tol=tol, maxfev=maxfev,
                         mode_bound_lower=mode_bounds[0], mode_bound_upper=mode_bounds[1]))
        warning_cols = gb_columns + ['tol', 'maxfev', 'tuning_curve_amplitude',
                                     'tuning_curve_sigma', 'tuning_curve_peak',
                                     'tuning_curve_bandwidth']
        if fit_warning:
            warnings.warn('Fit not great for:\n%s' % str_labels.replace(', ', '\n'))
            fit_problems.append((pd.DataFrame(tuning_df[-1][warning_cols].iloc[0]).T, (x, y),
                                 (g.frequency_value.values, g.amplitude_estimate.values)))
        if inf_warning:
            inf_problems.append((pd.DataFrame(tuning_df[-1][warning_cols].iloc[0]).T, (x, y),
                                 (g.frequency_value.values, g.amplitude_estimate.values)))
    tuning_df = pd.concat(tuning_df).reset_index(drop=True)
    if save_path is not None:
        tuning_df.to_csv(save_path, index=False)
        report_save_path = save_path.replace('.csv', '_problems.html')
    else:
        report_save_path = "tuning_curve_problems.html"
    if len(fit_problems) > 0 or len(inf_problems) > 0:
        create_problems_report(fit_problems, inf_problems, report_save_path)
    return tuning_df


if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        description=("Load in first level results DataFrame and create the tuning curves that "
                     "summarize them, saving the resulting parameters in a new dataframe that can "
                     "be used to easily plot the results. Will also create an html report showing "
                     "any curves that had problems fitting."),
        formatter_class=CustomFormatter)
    parser.add_argument("first_level_results_path",
                        help=("Path to the first level results dataframe containing the data to "
                              "fit."))
    parser.add_argument("save_path",
                        help=("Path to save the resulting tuning dataframe at. The problems report"
                              ", if created, will be at the same path, with '.csv' replaced by "
                              "'_problems.html"))
    args = vars(parser.parse_args())
    df = pd.read_csv(args.pop('first_level_results_path'))
    # bounds should be task-dependent, task-sfp and task-sfprescaled have the same frequency
    # information
    if ('task-sfp' in args['save_path'].split('_') or
        'task-sfprescaled' in args['save_path'].split('_')):
        bounds = (2**(-5), 2**11)
    elif 'task-sfpconstant' in args['save_path'].split('_'):
        bounds = (2**(-15), 2**8)
    main(df, args['save_path'], mode_bounds=bounds)
