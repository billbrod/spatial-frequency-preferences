#!/usr/bin/python
"""create simulated data for testing 2d model fit
"""
import matplotlib as mpl
# we do this because sometimes we run this without an X-server, and this backend doesn't need one
mpl.use('svg')
import argparse
import pandas as pd
import numpy as np
import stimuli as sfp_stimuli
import first_level_analysis
import model as sfp_model


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
    resps = true_model(mags, direcs, vox_ecc, vox_angle)
    resps = resps.detach().numpy()
    resps_norm = np.linalg.norm(resps, 2)
    normed_resps = resps / resps_norm
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
                         'amplitude_estimate_std_error': noise_level,
                         'true_amplitude_estimate_median': resps,
                         'amplitude_estimate_median_normed': normed_resps,
                         'amplitude_estimate_norm': resps_norm,
                         'precision': precision,
                         'stimulus_class': range(len(freqs))})


def simulate_data(true_model, num_voxels=100, noise_level=0, noise_source_path=None):
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
        tmp = simulate_voxel(true_model, freqs, noise_level=np.random.choice(noise_distribution))
        tmp['voxel'] = i
        df.append(tmp)
    df = pd.concat(df)
    df['varea'] = 1
    # we want the generating model and its parameters stored here
    df['true_model_type'] = true_model.model_type
    for name, val in true_model.named_parameters():
        df['true_model_%s'%name] = val.detach().numpy()
    df['noise_level'] = noise_level
    df['noise_source_df'] = noise_source_path
    return df


def main(sigma=.4, sf_ecc_slope=1, sf_ecc_intercept=0, mode_cardinals=0, mode_obliques=0,
         amplitude_cardinals=0, amplitude_obliques=0, orientation_type='absolute', num_voxels=100,
         noise_level=0, save_path=None, noise_source_path=None):
    model = sfp_model.LogGaussianDonut(sigma, sf_ecc_slope, sf_ecc_intercept, mode_cardinals,
                                       mode_obliques, amplitude_cardinals, amplitude_obliques,
                                       orientation_type)
    model.eval()
    df = simulate_data(model, num_voxels, noise_level, noise_source_path)
    df['orientation_type'] = orientation_type
    if df is not None:
        df.to_csv(save_path, index=False)
    return df


if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        description=("Simulate first level data to be fit with 2d tuning model."),
        formatter_class=CustomFormatter)
    parser.add_argument("model_type",
                        help=("{'full_relative', 'full_absolute', 'iso', 'scaling', 'constant'}."
                              " Which type of model underlies the simulated data."))
    parser.add_argument("save_path",
                        help=("Path (should end in .csv) where we'll save the simulated data"))
    parser.add_argument("--num_voxels", '-n', default=100, help="Number of voxels to simulate",
                        type=int)
    parser.add_argument("--sigma", '-s', default=.4, type=float, help="Sigma of log-Normal donut")
    parser.add_argument("--sf_ecc_slope", '-e', default=1, type=float,
                        help=("Slope of relationship between tuning and eccentricity for log-Normal"
                              " donut"))
    parser.add_argument("--sf_ecc_intercept", '-i', default=0, type=float,
                        help=("Intercept of relationship between tuning and eccentricity for "
                              "log-Normal donut"))
    parser.add_argument("--mode_cardinals", "-mc", default=0, type=float,
                        help=("The strength of the cardinal-effect of orientation on the mode. "
                              "That is, the coefficient of cos(2*orientation)"))
    parser.add_argument("--mode_obliques", "-mo", default=0, type=float,
                        help=("The strength of the oblique-effect of orientation on the mode. "
                              "That is, the coefficient of cos(4*orientation)"))
    parser.add_argument("--amplitude_cardinals", "-ac", default=0, type=float,
                        help=("The strength of the cardinal-effect of orientation on the amplitude. "
                              "That is, the coefficient of cos(2*orientation)"))
    parser.add_argument("--amplitude_obliques", "-ao", default=0, type=float,
                        help=("The strength of the oblique-effect of orientation on the amplitude. "
                              "That is, the coefficient of cos(4*orientation)"))
    parser.add_argument('--noise_source_path', default=None,
                        help=("None or path to a first level summary dataframe. If None, then all "
                              "simulated voxels have the same noise, determined by `noise_level` "
                              "argment. If a path, then we find calculate the error distribution"
                              " based on that dataframe (see `calculate_error_distribution` "
                              "function for details) and each voxel's noise level is sampled "
                              "independently from that distribution."))
    parser.add_argument("--noise_level", '-l', default=0, type=float,
                        help=("Noise level. If noise_source_path is None, this is the std dev of a "
                              "normal distribution with mean 0, which will be added to the "
                              "simulated data. If "
                              "noise_source_path is not None, then we multiply the noise "
                              "distribution obtained from that dataframe by this number (see that"
                              " variable's help for more details). In both cases, the simulated "
                              "responses are normalized to have an L2 norm of 1 before noise is"
                              " added, so this should be interpreted as relative to a unit vector"
                              ". In both cases, a value of 0 means no noise."))
    parser.add_argument("--orienation_type", '-o', default='absolute',
                        help=("{'absolute', 'relative'}. Whether orientation should be absolute "
                              "(so that 0 is to the right) or relative (so that 0 is away from the"
                              " fovea)"))
    args = vars(parser.parse_args())
    main(**args)
