#!/usr/bin/python
"""run MCMC to fit the parameters of our model
"""
import argparse
import logging
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import theano.tensor as tt
from .model import construct_df_filter


def get_data_dict_from_df(df):
    """get a dict of arrays from the first level results dataframe
    """
    data = {}
    data_labels = ['sf_mag', 'sf_angle', 'vox_ecc', 'vox_angle', 'targets', 'std_error']
    df_labels = ['local_sf_magnitude', 'local_sf_xy_direction', 'eccen', 'angle',
                 'amplitude_estimate_median_normed', 'amplitude_estimate_std_error_normed']
    for data_name, df_name in zip(data_labels, df_labels):
        arr = df.groupby(['voxel', 'stimulus_class'])[[df_name]].mean().unstack().to_numpy()
        data[data_name] = arr.astype(np.float32)
    return data


def _parse_distrib_dict(distrib_dict):
    """returns a lambda function that accepts name to finalize
    """
    distrib = distrib_dict.pop("distribution")
    if distrib not in pm.distributions.__all__:
        raise Exception("Can't find distribution %s" % distrib)
    distrib = eval("pm.distributions.%s" % distrib)
    return lambda x: distrib(x, **distrib_dict)


def pymc_log_gauss_donut(sf_mag, sf_angle, vox_ecc, vox_angle, targets, std_error, voxel_norm=None,
                         sigma=None, sf_ecc_slope=None, sf_ecc_intercept=None):
    """this is just the PyMC3 implementation of our PyTorch model of the log-normal 2d tuning curve

    the only difference is we have one additional parameter, voxel_norm, for rescaling the overall
    response magnitude to be the same as in the data. we expect all targets to have already been
    rescaled to have an L2-norm of 1, and for the std_error to have been rescaled using that same
    per-voxel normalizing constant. Our final step is to assume there's Gaussian noise around our
    computed mean (that is, this 2d log-normal tuning curve gives the mean response), with standard
    deviation equal to the std_error passed in. SO MAKE SURE THIS std_error IS CORRECT! Otherwise,
    the MCMC sampler will have a lot of trouble.

    for now, this only supports the iso versions of the model (without the amplitudes varying). the
    code to support the other versions is here, but commented out

    the parameters (voxel_norm, sigma, sf_ecc_slope, and sf_ecc_intercept) are provided as
    arguments so you can set the prior distributions easily. if None, we use the defaults:
    Normal(mu=1, sd=.25) for voxel_norm, Wald(mu=1, lam=3) for sigma, sf_ecc_slope,
    sf_ecc_intercept. If not None, must be a dictionary containing the key 'distribution',
    specifying the name of the distribution, and the other keys must specify the parameters for
    that distribution.

    """
    model = pm.Model()
    with model:
        if voxel_norm is None:
            voxel_norm = pm.Normal('voxel_norm', mu=1, sd=.25, )
        else:
            voxel_norm = _parse_distrib_dict(voxel_norm)('voxel_norm')
        if sigma is None:
            sigma = pm.Wald('sigma', mu=1, lam=3)
        else:
            sigma = _parse_distrib_dict(sigma)('sigma')
        if sf_ecc_slope is None:
            sf_ecc_slope = pm.Wald('sf_ecc_slope', mu=1, lam=3)
        else:
            sf_ecc_slope = _parse_distrib_dict(sf_ecc_slope)('sf_ecc_slope')
        if sf_ecc_intercept is None:
            sf_ecc_intercept = pm.Wald('sf_ecc_intercept', mu=1, lam=3)
        else:
            sf_ecc_intercept = _parse_distrib_dict(sf_ecc_intercept)('sf_ecc_intercept')
        # can use the sfp.model._check_log_gaussian_parmas to check whether you want to set the
        # different parameters, based on orientation_type, etc.        
        # rel_sf_angle = sf_angle - vox_angle
        # orientation_effect = (1 + abs_mode_cardinals * tt.cos(2 * sf_angle) +
        #                       abs_mode_obliques * tt.cos(4 * sf_angle) +
        #                       rel_mode_cardinals * tt.cos(2 * rel_sf_angle) +
        #                       rel_mode_obliques * tt.cos(4 * rel_sf_angle))
        # if you set your priors intelligently, you probably don't need the clip call, but just in
        # case.
        # preferred_period = pm.math.clip(eccentricity_effect * orientation_effect, 1e-6, 1e6)
        eccentricity_effect = sf_ecc_slope * vox_ecc + sf_ecc_intercept
        preferred_period = eccentricity_effect
        
        # if you set your priors intelligently, you probably don't need the clip call, but just in
        # case.
        # max_amplitude = pm.math.clip(1 + abs_amplitude_cardinals * tt.cos(2*sf_angle) +
        #                              abs_amplitude_obliques * tt.cos(4*sf_angle) +
        #                              rel_amplitude_cardinals * tt.cos(2*rel_sf_angle) +
        #                              rel_amplitude_obliques * tt.cos(4*rel_sf_angle), 1e-6, 1e6)
        pdf = tt.exp(-((tt.log2(sf_mag) + tt.log2(preferred_period))**2) / (2*sigma**2))
        predicted_response = voxel_norm * pdf #* max_amplitude
        noisy_response = pm.Normal('noisy_response', mu=predicted_response, sd=std_error,
                                   observed=targets)
    return model


def setup_model(df, voxel_norm=None, sigma=None, sf_ecc_intercept=None, sf_ecc_slope=None,
                df_filter_string=None):
    """setup and return the model

    the parameters (voxel_norm, sigma, sf_ecc_slope, and sf_ecc_intercept) are provided as
    arguments so you can set the prior distributions easily. if None, we use the defaults:
    Normal(mu=1, sd=.25) for voxel_norm, Wald(mu=1, lam=3) for sigma, sf_ecc_slope,
    sf_ecc_intercept. If not None, must be a dictionary containing the key 'distribution',
    specifying the name of the distribution, and the other keys must specify the parameters for
    that distribution.

    df_filter_string: str or None. if not None, the string should be a single string containing at
    least one of the following, separated by commas: 'drop_voxels_with_negative_amplitudes',
    'drop_voxels_near_border', 'reduce_num_voxels:n' (where n is an integer), 'None'. This will
    then construct the function that will chain them together in the order specified (if None is
    one of the entries, we will simply return None)

    """
    logger = logging.getLogger("pymc3")
    pre_voxels = df.voxel.nunique()
    if df_filter_string is not None:
        df_filter = construct_df_filter(df_filter_string)
        if df_filter is not None:
            df = df_filter(df).reset_index()
    post_voxels = df.voxel.nunique()
    logger.info("Started with %d voxels, after filtering with df_filter_string %s have %d voxels" %
                (pre_voxels, df_filter_string, post_voxels))
    data = get_data_dict_from_df(df)
    model = pymc_log_gauss_donut(sigma=sigma, voxel_norm=voxel_norm, sf_ecc_slope=sf_ecc_slope,
                                 sf_ecc_intercept=sf_ecc_intercept, **data)
    return model


def main(first_level_results_path, voxel_norm=None, sigma=None, sf_ecc_intercept=None,
         sf_ecc_slope=None, n_samples=1000, n_chains=4, n_cores=None, save_path=None,
         random_seed=None, df_filter_string=None, **nuts_kwargs):
    """run MCMC sampling to fit 2d log-normal tuning curve model

    first_level_results_path: str. Path to the first level results dataframe containing the data to
    fit.

    the parameters (voxel_norm, sigma, sf_ecc_slope, and sf_ecc_intercept) are provided as
    arguments so you can set the prior distributions easily. if None, we use the defaults:
    Normal(mu=1, sd=.25) for voxel_norm, Wald(mu=1, lam=3) for sigma, sf_ecc_slope,
    sf_ecc_intercept. If not None, must be a dictionary containing the key 'distribution',
    specifying the name of the distribution, and the other keys must specify the parameters for
    that distribution.

    n_samples: int. The number of MCMC samples to draw

    n_chains: int. The number of MCMC chains to run.

    n_cores: int or None. The number of cores to use when sampling. If None, we use the same number as
    `n_chains`.

    save_path: str. The path to save the samples at. Must have a ".nc" extension.

    random_seed: int or None. we initialize the model with random parameters in order to try and
    avoid local optima. we set the seed before generating all those random numbers.

    df_filter_string: str or None. If not None, the string should be a single string containing at least
    one of the following, separated by commas: 'drop_voxels_with_negative_amplitudes',
    'drop_voxels_near_border', 'reduce_num_voxels:n' (where n is an integer), 'None'. This will
    then construct the function that will chain them together in the order specified (if None is
    one of the entries, we will simply return None)

    nuts_kwargs: additional arguments to pass to the NUTS sampler. Some examples: target_accept,
    step_scale, max_treedepth. See pymc3.NUTS for a list of all arguments and their accepted
    values.

    """
    df = pd.read_csv(first_level_results_path)
    model = setup_model(df, voxel_norm, sigma, sf_ecc_intercept, sf_ecc_slope, df_filter_string)
    if n_cores is None:
        n_cores = n_chains
    # n_cores cannot be larger than n_chains, or things get weird
    n_cores = min(n_cores, n_chains)
    with model:
        trace = pm.sample(n_samples, chains=n_chains, cores=n_cores, random_seed=random_seed,
                          nuts_kwargs=nuts_kwargs)
        prior = pm.sample_prior_predictive(n_samples)
        post = pm.sample_posterior_predictive(trace, n_samples)
    inference_data = az.from_pymc3(trace, prior=prior, posterior_predictive=post)
    metadata = {}
    for k in ['first_level_results_path', 'n_cores', 'random_seed', 'df_filter_string']:
        if eval(k) is None:
            # we can't serialize None into a netCDF file
            metadata[k] = "None"
        else:
            metadata[k] = eval(k)
    # we don't need to save the priors of the variables, because we're saving samples from the
    # prior
    inference_data.posterior = inference_data.posterior.assign_attrs(**metadata)
    if save_path is not None:
        inference_data.to_netcdf(save_path)
    return inference_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=("Load in the first level results Dataframe and use MCMC to train a 2d tuning "
                     "model on it. Will save the arviz inference data."))
    parser.add_argument("first_level_results_path",
                        help=("Path to the first level results dataframe containing the data to "
                              "fit."))
    parser.add_argument("save_path",
                        help=("Path stem (with '.nc' extension) where we'll save the results"))
    parser.add_argument("--random_seed", default=None, nargs='+', type=int,
                        help=("Random seed for the MCMC sampler. If None, we don't set it"))
    parser.add_argument("--n_samples", '-s', type=int, default=1000,
                        help=("The number of MCMC samples to draw"))
    parser.add_argument("--n_chains", '-c', type=int, default=4,
                        help=("The number of MCMC chains to run"))
    parser.add_argument("--n_cores", '-n', type=int, default=4,
                        help=("The number of cores to use when sampling."))
    parser.add_argument("--df_filter_string", '-d', default='drop_voxels_with_negative_amplitudes',
                        help=("{'drop_voxels_near_border', 'drop_voxels_with_negative_amplitudes',"
                              " 'reduce_num_voxels:n', 'None'}."
                              " How to filter the first level dataframe. Can be multiple of these,"
                              " separated by a comma, in which case they will be chained in the "
                              "order provided (so the first one will be applied to the dataframe "
                              "first). If 'drop_voxels_near_border', will drop all voxels whose "
                              "pRF center is one sigma away from the stimulus borders. If "
                              "'drop_voxels_with_negative_amplitudes', drop any voxel that has a "
                              "negative response amplitude. If 'reduce_num_voxels:n', will drop "
                              "all but the first n voxels. If 'None', fit on all data (obviously,"
                              " this cannot be chained with any of the others)."))
    parser.add_argument("--nuts_kwargs", nargs='+',
                        help=("Additional arguments to pass to the NUTS sampler. Should be "
                              "key=value, with multiple arguments separated by a space"))
    args = vars(parser.parse_args())
    nuts_kwargs_tmp = args.pop('nuts_kwargs')
    nuts_kwargs = {}
    if nuts_kwargs_tmp is not None:
        for val in nuts_kwargs_tmp:
            k, v = val.split('=')
            try:
                v = float(v)
            except ValueError:
                try:
                    v = bool(v)
                except ValueError:
                    pass
            nuts_kwargs[k] = v
    main(**args, **nuts_kwargs)
