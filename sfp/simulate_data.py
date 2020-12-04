#!/usr/bin/python
"""create simulated data for testing 2d model fit
"""
import matplotlib as mpl
# we do this because sometimes we run this without an X-server, and this backend doesn't need
# one. We set warn=False because the notebook uses a different backend and will spout out a big
# warning to that effect; that's unnecessarily alarming, so we hide it.
mpl.use('svg', warn=False)
import argparse
import pandas as pd
import numpy as np
from . import stimuli as sfp_stimuli
from . import model as sfp_model


def quadratic_mean(x):
    """returns the quadratic mean: sqrt{(x_1^2 + ... + x_n^2) / n}
    """
    return np.sqrt(np.mean(x**2))


def calculate_error_distribution(first_level_df):
    """given a first_level_df, return the distribution of errors across voxels

    this requries the first_level_df to contain the column amplitude_estimate_std_error_normed
    (i.e., it's the summary df, not the full df, which contains each bootstrap).

    we take the quadratic mean of each voxel's 48 normed standard errors (that's the appropriate
    way to combine these errors) and then return an array containing one error per voxel. this
    should be sampled to determine the noise level for individual simulated voxels
    """
    errors = first_level_df.groupby(['varea', 'voxel']).amplitude_estimate_std_error_normed
    return errors.apply(quadratic_mean).values


def simulate_voxel(true_model, freqs, noise_level=0, ecc_range=(1, 12),
                   angle_range=(0, 2*np.pi), pix_diam=1080, deg_diam=24):
    """simulate a single voxel

    noise_level should be a float. to add noise, we normalize our predictions (to have an L2 of 1),
    add noise from a normal distribution with a mean of 0 and a standard deviation of
    `noise_level`, and un-normalize our predictions (by multiplying by its original L2 norm).
    """
    vox_ecc = np.random.uniform(*ecc_range)
    vox_angle = np.random.uniform(*angle_range)
    mags, direcs = [], []
    for w_r, w_a in freqs:
        _, _, m, d = sfp_stimuli.sf_cpd(pix_diam, deg_diam, vox_ecc, vox_angle, w_r=w_r, w_a=w_a)
        mags.append(m)
        direcs.append(d)
    resps = true_model.evaluate(mags, direcs, vox_ecc, vox_angle)
    resps = resps.detach().numpy()
    resps_norm = np.linalg.norm(resps, 2)
    normed_resps = resps / resps_norm
    # this means that the noise_level argument controls the size of the error
    # in the normed responses (not the un-normed ones)
    normed_resps += np.random.normal(scale=noise_level, size=len(resps))
    # since the noise_level becomes the standard deviation of a normal distribution, the precision
    # is the reciprocal of its square
    if noise_level != 0:
        precision = 1. / ((noise_level * resps_norm)**2)
    else:
        # in this case, just set the precision to 1, so it's the same for all of them. only the
        # relative precision matters anyway; if they're all identical it doesn't matter what the
        # value is.
        precision = 1.
    return pd.DataFrame({'eccen': vox_ecc, 'angle': vox_angle, 'local_sf_magnitude': mags,
                         'local_sf_xy_direction': direcs,
                         'amplitude_estimate_median': normed_resps * resps_norm,
                         'amplitude_estimate_std_error': noise_level * resps_norm,
                         'true_amplitude_estimate_median': resps,
                         'amplitude_estimate_median_normed': normed_resps,
                         'amplitude_estimate_std_error_normed': noise_level,
                         'amplitude_estimate_norm': resps_norm,
                         'precision': precision,
                         'stimulus_class': range(len(freqs))})


def simulate_data(true_model, num_voxels=100, noise_level=0, num_bootstraps=10,
                  noise_source_path=None):
    """simulate a bunch of voxels

    if noise_source_path is None, then all voxels have the same noise, which is drawn from a
    Gaussian with mean 0 and standard deviation `noise_level` (after normalization to having an L2
    norm of 1). if first_level_df is not None, then we grab the error distribution of voxels found
    in it (see `calculate_error_distribution`), multiply those values by noise_level, and sample
    once per voxel
    """
    freqs = sfp_stimuli._gen_freqs([2**i for i in np.arange(2.5, 7.5, .5)], True)
    if noise_source_path is not None:
        noise_source_df = pd.read_csv(noise_source_path)
        noise_distribution = noise_level * calculate_error_distribution(noise_source_df)
    else:
        noise_distribution = [noise_level]
    df = []
    for i in range(num_voxels):
        for j in range(num_bootstraps):
            tmp = simulate_voxel(true_model, freqs, noise_level=np.random.choice(noise_distribution))
            tmp['bootstrap_num'] = j
            tmp['voxel'] = i
            df.append(tmp)
    df = pd.concat(df)
    df['varea'] = 1
    # we want the generating model and its parameters stored here
    df['true_model_type'] = true_model.model_type
    for name, val in true_model.named_parameters():
        df['true_model_%s' % name] = val.detach().numpy()
    df['noise_level'] = noise_level
    df['noise_source_df'] = noise_source_path
    return df


def main(model_period_orientation_type='iso', model_eccentricity_type='full',
         model_amplitude_orientation_type='iso', sigma=.4, sf_ecc_slope=1, sf_ecc_intercept=0,
         abs_mode_cardinals=0, abs_mode_obliques=0, rel_mode_cardinals=0, rel_mode_obliques=0,
         abs_amplitude_cardinals=0, abs_amplitude_obliques=0, rel_amplitude_cardinals=0,
         rel_amplitude_obliques=0, num_voxels=100, noise_level=0, save_path=None,
         noise_source_path=None, num_bootstraps=100):
    """Simulate first level data to be fit with 2d tuning model.

    Note that when calling the function, you can set every parameter
    individually, but, depending on the values of the
    model_period_orientation_type, model_eccentricity_type, and
    model_amplitude_orientation_type, some of them have specific values
    (often 0), they will be set to. If this happens, a warning will be
    raised.

    if save_path is not None, should be a list with one or two strs, first
    giving the path to save the summary dataframe (median across bootstraps),
    second the path to save the full dataframe (all bootstraps). If only one
    str, we only save the summary version.

    """
    model = sfp_model.LogGaussianDonut(model_period_orientation_type, model_eccentricity_type,
                                       model_amplitude_orientation_type, sigma, sf_ecc_slope,
                                       sf_ecc_intercept, abs_mode_cardinals, abs_mode_obliques,
                                       rel_mode_cardinals,rel_mode_obliques,
                                       abs_amplitude_cardinals, abs_amplitude_obliques,
                                       rel_amplitude_cardinals, rel_amplitude_obliques)
    model.eval()
    df = simulate_data(model, num_voxels, noise_level, num_bootstraps, noise_source_path)
    df['period_orientation_type'] = model_period_orientation_type
    df['eccentricity_type'] = model_eccentricity_type
    df['amplitude_orientation_type'] = model_amplitude_orientation_type
    if save_path is not None:
        # summary dataframe
        df.groupby(['voxel', 'stimulus_class']).median().to_csv(save_path[0],
                                                                index=False)
        # full dataframe
        col_renamer = {c: c.replace('_median', '') for c in df.columns
                       if 'median' in c}
        df.rename(columns=col_renamer).to_csv(save_path[1], index=False)
    return df


if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        descriptio
                     "calling the function, you can set every parameter individually, but, "
                     "depending on the values of the model_orientation_type, model_eccentricity_"
                     "type, and model_vary_amplitude, some of them have specific values (often 0),"
                     "they will be set to. If this happens, a warning will be raised."),
        formatter_class=CustomFormatter)
    parser.add_argument("save_path", nargs='+',
                        help=("Path (should end in .csv) where we'll save the simulated data. If "
                              "one str, we only save the summary version, if two we save both "
                              "summary and full."))
    parser.add_argument("--model_period_orientation_type", '-p', default='iso',
                        help=("{iso, absolute, relative, full}\nEffect of orientation on "
                              "preferred period\n- iso: model is isotropic, "
                              "predictions identical for all orientations.\n- absolute: model can"
                              " fit differences in absolute orientation, that is, in Cartesian "
                              "coordinates, such that sf_angle=0 correponds to 'to the right'\n- "
                              "relative: model can fit differences in relative orientation, that "
                              "is, in retinal polar coordinates, such that sf_angle=0 corresponds"
                              " to 'away from the fovea'\n- full: model can fit differences in "
                              "both absolute and relative orientations"))
    parser.add_argument("--model_eccentricity_type", '-e', default='full',
                        help=("{scaling, constant, full}\n- scaling: model's relationship between"
                              " preferred period and eccentricity is exactly scaling, that is, the"
                              " preferred period is equal to the eccentricity.\n- constant: model'"
                              "s relationship between preferred period and eccentricity is exactly"
                              " constant, that is, it does not change with eccentricity but is "
                              "flat.\n- full: model discovers the relationship between "
                              "eccentricity and preferred period, though it is constrained to be"
                              " linear (i.e., model solves for a and b in period = a * "
                              "eccentricity + b)"))
    parser.add_argument("--model_amplitude_orientation_type", '-o', default='iso',
                        help=("{iso, absolute, relative, full}\nEffect of orientation on "
                              "max_amplitude\n- iso: model is isotropic, "
                              "predictions identical for all orientations.\n- absolute: model can"
                              " fit differences in absolute orientation, that is, in Cartesian "
                              "coordinates, such that sf_angle=0 correponds to 'to the right'\n- "
                              "relative: model can fit differences in relative orientation, that "
                              "is, in retinal polar coordinates, such that sf_angle=0 corresponds"
                              " to 'away from the fovea'\n- full: model can fit differences in "
                              "both absolute and relative orientations"))
    parser.add_argument("--num_voxels", '-n', default=100, help="Number of voxels to simulate",
                        type=int)
    parser.add_argument("--num_bootstraps", default=100, help="Number of bootstraps per voxel",
                        type=int)
    parser.add_argument("--sigma", '-s', default=.4, type=float, help="Sigma of log-Normal donut")
    parser.add_argument("--sf_ecc_slope", '-a', default=1, type=float,
                        help=("Slope of relationship between tuning and eccentricity for log-"
                              "Normal donut"))
    parser.add_argument("--sf_ecc_intercept", '-b', default=0, type=float,
                        help=("Intercept of relationship between tuning and eccentricity for "
                              "log-Normal donut"))
    parser.add_argument("--rel_mode_cardinals", "-rmc", default=0, type=float,
                        help=("The strength of the cardinal-effect of the relative orientation (so"
                              " angle=0 corresponds to away from the fovea) on the mode. That is, "
                              "the coefficient of cos(2*relative_orientation)"))
    parser.add_argument("--rel_mode_obliques", "-rmo", default=0, type=float,
                        help=("The strength of the oblique-effect of the relative orientation (so"
                              " angle=0 corresponds to away from the fovea) on the mode. That is, "
                              "the coefficient of cos(4*relative_orientation)"))
    parser.add_argument("--rel_amplitude_cardinals", "-rac", default=0, type=float,
                        help=("The strength of the cardinal-effect of the relative orientation (so"
                              " angle=0 corresponds to away from the fovea) on the amplitude. That"
                              " is, the coefficient of cos(2*relative_orientation)"))
    parser.add_argument("--rel_amplitude_obliques", "-rao", default=0, type=float,
                        help=("The strength of the oblique-effect of the relative orientation (so"
                              " angle=0 corresponds to away from the fovea) on the amplitude. That"
                              " is, the coefficient of cos(4*relative_orientation)"))
    parser.add_argument("--abs_mode_cardinals", "-amc", default=0, type=float,
                        help=("The strength of the cardinal-effect of the absolute orientation (so"
                              " angle=0 corresponds to the right) on the mode. That is, "
                              "the coefficient of cos(2*absolute_orientation)"))
    parser.add_argument("--abs_mode_obliques", "-amo", default=0, type=float,
                        help=("The strength of the oblique-effect of the absolute orientation (so"
                              " angle=0 corresponds to the right) on the mode. That is, "
                              "the coefficient of cos(4*absolute_orientation)"))
    parser.add_argument("--abs_amplitude_cardinals", "-aac", default=0, type=float,
                        help=("The strength of the cardinal-effect of the absolute orientation (so"
                              " angle=0 corresponds to the right) on the amplitude. That"
                              " is, the coefficient of cos(2*absolute_orientation)"))
    parser.add_argument("--abs_amplitude_obliques", "-aao", default=0, type=float,
                        help=("The strength of the oblique-effect of the absolute orientation (so"
                              " angle=0 corresponds to the right) on the amplitude. That"
                              " is, the coefficient of cos(4*absolute_orientation)"))
    parser.add_argument('--noise_source_path', default=None,
                        help=("None or path to a first level summary dataframe. If None, then all "
                              "simulated voxels have the same noise, determined by `noise_level` "
                              "argment. If a path, then we find calculate the error distribution"
                              " based on that dataframe (see `calculate_error_distribution` "
                              "function for details) and each voxel's noise level is sampled "
                              "independently from that distribution."))
    parser.add_argument("--noise_level", '-l', default=0, type=float,
                        help=("Noise level. If noise_source_path is None, this is the std dev of a"
                              " normal distribution with mean 0, which will be added to the "
                              "simulated data. If "
                              "noise_source_path is not None, then we multiply the noise "
                              "distribution obtained from that dataframe by this number (see that"
                              " variable's help for more details). In both cases, the simulated "
                              "responses are normalized to have an L2 norm of 1 before noise is"
                              " added, so this should be interpreted as relative to a unit vector"
                              ". In both cases, a value of 0 means no noise."))
    args = vars(parser.parse_args())
    main(**args)
