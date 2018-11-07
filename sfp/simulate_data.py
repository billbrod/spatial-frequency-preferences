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


def simulate_voxel(true_model, freqs, direction_type='absolute', noise_level=0, ecc_range=(0, 12),
                   angle_range=(0, 2*np.pi), pix_diam=1080, deg_diam=24):
    vox_ecc = np.random.uniform(*ecc_range)
    vox_angle = np.random.uniform(*angle_range)
    mags, direcs, rel_direcs = [], [], []
    for w_r, w_a in freqs:
        _, _, m, d = sfp_stimuli.sf_cpd(pix_diam, deg_diam, vox_ecc, vox_angle, w_r=w_r, w_a=w_a)
        _, _, rd = sfp_stimuli.sf_origin_polar_cpd(pix_diam, deg_diam, vox_ecc, vox_angle,
                                                   w_r=w_r, w_a=w_a)
        mags.append(m)
        direcs.append(d)
        rel_direcs.append(rd)
    if direction_type == 'absolute':
        resps = true_model(mags, direcs, vox_ecc, vox_angle)
    elif direction_type == 'relative':
        resps = true_model(mags, rel_direcs, vox_ecc, vox_angle)
    resps = resps.detach().numpy()
    resps += np.random.normal(scale=noise_level, size=len(resps))
    return pd.DataFrame({'eccen': vox_ecc, 'angle': vox_angle, 'local_sf_magnitude': mags,
                         'local_sf_xy_direction': direcs, 'local_sf_ra_direction': rel_direcs,
                         'amplitude_estimate_median': resps, 'amplitude_estimate_std_error': noise_level,
                         'stimulus_class': range(len(freqs))})


def simulate_data(true_model, direction_type='absolute', num_voxels=100, noise_level=0):
    freqs = sfp_stimuli._gen_freqs([2**i for i in np.arange(2.5, 7.5, .5)], True)
    df = []
    for i in range(num_voxels):
        tmp = simulate_voxel(true_model, freqs, direction_type, noise_level=noise_level)
        tmp['voxel'] = i
        df.append(tmp)
    df = pd.concat(df)
    df['varea'] = 1
    # we want the generating model and its parameters stored here
    df['true_model_type'] = true_model.model_type
    for name, val in true_model.named_parameters():
        df['true_model_%s'%name] = val.detach().numpy()
    df['noise_level'] = noise_level
    df['direction_type'] = direction_type
    # since the noise_level becomes the standard deviation of a normal distribution, the precision
    # is the reciprocal of its square
    try:
        df['precision'] = 1. / (noise_level**2)
    except ZeroDivisionError:
        # in this case, just set the precision to 1, so it's the same for all of them. only the
        # relative precision matters anyway; if they're all identical it doesn't matter what the
        # value is.
        df['precision'] = 1.
    return first_level_analysis._normalize_amplitude_estimate(df)


def main(model_type, amplitude=1, mode=2, sigma=.4, sf_ecc_slope=1, sf_ecc_intercept=0,
         direction_type='absolute', num_voxels=100, noise_level=0, save_path=None):
    if model_type == 'full':
        model = sfp_model.LogGaussianDonut(amplitude, mode, sigma, sf_ecc_slope, sf_ecc_intercept)
    elif model_type == 'constant':
        model = sfp_model.ConstantLogGaussianDonut(amplitude, mode, sigma)
    elif model_type == 'scaling':
        model = sfp_model.ScalingLogGaussianDonut(amplitude, mode, sigma)
    else:
        raise Exception("Don't know how to handle model_type %s!" % model_type)
    model.eval()
    df = simulate_data(model, direction_type, num_voxels, noise_level)
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
                        help=("{'full', 'scaling', 'constant'}. Which type of model underlies the"
                              " simualted data."))
    parser.add_argument("save_path",
                        help=("Path (should end in .csv) where we'll save the simulated data"))
    parser.add_argument("--num_voxels", '-n', default=100, help="Number of voxels to simulate",
                        type=int)
    parser.add_argument("--amplitude", '-a', default=1, type=float,
                        help="Amplitude of log-Normal donut")
    parser.add_argument("--mode", '-m', default=2, type=float, help="Mode of log-Normal donut")
    parser.add_argument("--sigma", '-s', default=.4, type=float, help="Sigma of log-Normal donut")
    parser.add_argument("--sf_ecc_slope", '-e', default=1, type=float,
                        help=("Slope of relationship between tuning and eccentricity for log-Normal"
                              " donut"))
    parser.add_argument("--sf_ecc_intercept", '-i', default=0, type=float,
                        help=("Intercept of relationship between tuning and eccentricity for "
                              "log-Normal donut"))
    parser.add_argument("--noise_level", '-l', default=0, type=float,
                        help=("Noise level (this is the std dev of normal distribution with mean "
                              "0)"))
    parser.add_argument("--direction_type", '-d', default='absolute',
                        help=("{'absolute', 'relative'}. Whether orientation should be absolute "
                              "(so that 0 is to the right) or relative (so that 0 is away from the"
                              " fovea)"))
    args = vars(parser.parse_args())
    main(**args)
