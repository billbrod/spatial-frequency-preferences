#!/usr/bin/python
"""run MCMC to fit the parameters of our model
"""
import argparse
import os
import re
import logging
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import theano.tensor as tt
from .model import construct_df_filter
from sklearn import preprocessing


def get_data_dict_from_df(df):
    """get a dict of arrays from the first level results dataframe
    """
    data = {}
    data_labels = ['sf_mag', 'sf_angle', 'vox_ecc', 'vox_angle', 'targets', 'std_error']
    df_labels = ['local_sf_magnitude', 'local_sf_xy_direction', 'eccen', 'angle',
                 'amplitude_estimate_median_normed', 'amplitude_estimate_std_error_normed']
    for data_name, df_name in zip(data_labels, df_labels):
        data[data_name] = df[df_name].values.astype(np.float32)
    if 'indicator' in df:
        le = preprocessing.LabelEncoder()
        data['scan_session_label'] = le.fit_transform(df.indicator.values)
    else:
        data['scan_session_label'] = np.zeros_like(data[data_name], dtype=np.int)
    return data


def _parse_distrib_dict(distrib_dict):
    """returns a lambda function that accepts name to finalize
    """
    distrib_dict = distrib_dict.copy()
    distrib = distrib_dict.pop("distribution")
    if distrib not in pm.distributions.__all__:
        raise Exception("Can't find distribution %s" % distrib)
    distrib = eval("pm.distributions.%s" % distrib)
    return lambda x, shape: distrib(x, shape=shape, **distrib_dict)


def hyperparam(param_name, mu_mu, mu_sd, sd_sd, shape, mode='centered'):
    param_mu = pm.Gamma("%s_mu" % param_name, mu=mu_mu, sd=mu_sd)
    param_sd = pm.HalfNormal("%s_sd" % param_name, sd=sd_sd)
    if mode == 'centered':
        param = pm.Bound(pm.Normal, lower=0)(param_name, mu=param_mu, sd=param_sd, shape=shape)
    elif mode == 'non-centered':
        param_offset = pm.Normal('%s_offset' % param_name, mu=0, sd=1, shape=shape)
        param = param_mu + param_offset * param_sd
        param = pm.Deterministic(param_name, tt.switch(tt.lt(param, 1e-12), 1e-12, param))
    return param


def pymc_log_gauss_donut(sf_mag, sf_angle, vox_ecc, vox_angle, targets, std_error,
                         scan_session_label, hierarchy_type='unpooled',
                         voxel_norm={'distribution': 'Normal', 'mu': 1, 'sd': .25},
                         sigma={'distribution': 'Gamma', 'mu': 2, 'sd': 1},
                         sf_ecc_slope={'distribution': 'Gamma', 'mu': .5, 'sd': .5},
                         sf_ecc_intercept={'distribution': 'Gamma', 'mu': .5, 'sd': .5}):
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

    scan_session_label: must be a 1d array with as many elements as voxels, which labels the
    different scan sessions (which will be fit separately) (or an array of 0s, if only one scan
    session)

    """
    logger = logging.getLogger("pymc3")
    model = pm.Model()
    num_scan_sessions = len(set(scan_session_label))
    with model:
        if hierarchy_type == 'unpooled':
            logger.info("Will fit unpooled model, separate parameters for each of %d sessions" %
                        num_scan_sessions)
        elif hierarchy_type == 'pooled':
            logger.info("Will fit completely pooled model, one parameter for all %d sessions" %
                        num_scan_sessions)
            num_scan_sessions = 1
            scan_session_label = np.zeros_like(vox_ecc, dtype=np.int)
        elif hierarchy_type == 'partially pooled':
            logger.info("Will fit partially pooled model, parameters for all %d sessions share "
                        "hyperparams!" % num_scan_sessions)
            sigma = hyperparam('sigma', 2, .5, 1, num_scan_sessions, 'non-centered')
            sf_ecc_slope = hyperparam('sf_ecc_slope', .5, .3, 1, num_scan_sessions, 'non-centered')
            sf_ecc_intercept = hyperparam('sf_ecc_intercept', .5, .3, 1, num_scan_sessions,
                                          'non-centered')
        else:
            raise Exception("Don't know how to handle hierarchy_type %s!" % hierarchy_type)
        if 'voxel_norm' not in model.named_vars.keys():
            voxel_norm = _parse_distrib_dict(voxel_norm)('voxel_norm', num_scan_sessions)
        if 'sigma' not in model.named_vars.keys():
            sigma = _parse_distrib_dict(sigma)('sigma', num_scan_sessions)
        if 'sf_ecc_slope' not in model.named_vars.keys():
            sf_ecc_slope = _parse_distrib_dict(sf_ecc_slope)('sf_ecc_slope', num_scan_sessions)
        if 'sf_ecc_intercept' not in model.named_vars.keys():
            sf_ecc_intercept = _parse_distrib_dict(sf_ecc_intercept)('sf_ecc_intercept',
                                                                     num_scan_sessions)
        # I think just set the params in the hierarchy type if statements, because if they're
        # unpooled/pooled, we want to use Gamma or similar distribution to avoid negative
        # values. If they're partially pooled, do we want to just say they're normally distributed
        # around a mean which is pulled from a Gamma or something similar? but we still need it to
        # be positive
        # abs_mode_cardinals = pm.Normal('abs_mode_cardinals', mu=0, sd=.1)
        # abs_mode_obliques = pm.Normal('abs_mode_obliques', mu=0, sd=.1)
        # rel_mode_cardinals = pm.Normal('rel_mode_cardinals', mu=0, sd=.1)
        # rel_mode_obliques = pm.Normal('rel_mode_obliques', mu=0, sd=.1)
        # abs_amplitude_cardinals = pm.Normal('abs_amplitude_cardinals', mu=0, sd=.1)
        # abs_amplitude_obliques = pm.Normal('abs_amplitude_obliques', mu=0, sd=.1)
        # rel_amplitude_cardinals = pm.Normal('rel_amplitude_cardinals', mu=0, sd=.1)
        # rel_amplitude_obliques = pm.Normal('rel_amplitude_obliques', mu=0, sd=.1)
        # # can use the sfp.model._check_log_gaussian_parmas to check whether you want to set the
        # # different parameters, based on orientation_type, etc.        
        # rel_sf_angle = sf_angle - vox_angle
        # orientation_effect = (1 + abs_mode_cardinals * tt.cos(2 * sf_angle) +
        #                       abs_mode_obliques * tt.cos(4 * sf_angle) +
        #                       rel_mode_cardinals * tt.cos(2 * rel_sf_angle) +
        #                       rel_mode_obliques * tt.cos(4 * rel_sf_angle))
        # if you set your priors intelligently, you probably don't need the clip call, but just in
        # case.
        eccentricity_effect = (sf_ecc_slope[scan_session_label] * vox_ecc +
                               sf_ecc_intercept[scan_session_label])
        # preferred_period = pm.math.clip(eccentricity_effect * orientation_effect, 1e-6, 1e6)
        preferred_period = eccentricity_effect
        
        # if you set your priors intelligently, you probably don't need the clip call, but just in
        # case.
        # max_amplitude = pm.math.clip(1 + abs_amplitude_cardinals * tt.cos(2*sf_angle) +
        #                              abs_amplitude_obliques * tt.cos(4*sf_angle) +
        #                              rel_amplitude_cardinals * tt.cos(2*rel_sf_angle) +
        #                              rel_amplitude_obliques * tt.cos(4*rel_sf_angle), 1e-6, 1e6)
        pdf = tt.exp(-((tt.log2(sf_mag) + tt.log2(preferred_period))**2) / (2*sigma[scan_session_label]**2))
        predicted_response = voxel_norm[scan_session_label] * pdf #* max_amplitude
        noisy_response = pm.Normal('noisy_response', mu=predicted_response, sd=std_error,
                                   observed=targets)
    return model


def setup_model(df, df_filter_string=None, hierarchy_type='unpooled',
                voxel_norm={'distribution': 'Normal', 'mu': 1, 'sd': .25},
                sigma={'distribution': 'Gamma', 'mu': 2, 'sd': 1},
                sf_ecc_slope={'distribution': 'Gamma', 'mu': .5, 'sd': .5},
                sf_ecc_intercept={'distribution': 'Gamma', 'mu': .5, 'sd': .5}):
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
    data = get_data_dict_from_df(df)
    model = pymc_log_gauss_donut(sigma=sigma, voxel_norm=voxel_norm, sf_ecc_slope=sf_ecc_slope,
                                 sf_ecc_intercept=sf_ecc_intercept, hierarchy_type=hierarchy_type,
                                 **data)
    return model


def main(first_level_results_path, hierarchy_type='unpooled',
         voxel_norm={'distribution': 'Normal', 'mu': 1, 'sd': .25},
         sigma={'distribution': 'Gamma', 'mu': 2, 'sd': 1},
         sf_ecc_slope={'distribution': 'Gamma', 'mu': .5, 'sd': .5},
         sf_ecc_intercept={'distribution': 'Gamma', 'mu': .5, 'sd': .5},
         n_samples=1000, n_chains=4, n_cores=None, save_path=None,
         random_seed=None, df_filter_string=None, init='auto', sampler='NUTS', **nuts_kwargs):
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

    init: str. How to initialize the NUTS sampler.

    sampler: {'NUTS', 'Metroplis'}. whether to use the NUTS or metropolis sampler.

    nuts_kwargs: additional arguments to pass to the NUTS sampler. Some examples: target_accept,
    step_scale, max_treedepth. See pymc3.NUTS for a list of all arguments and their accepted
    values.

    """
    hierarchy_type = hierarchy_type.replace('-', ' ')
    if not isinstance(first_level_results_path, list):
        first_level_results_path = [first_level_results_path]
    logger = logging.getLogger("pymc3")
    df = []
    df_filter = construct_df_filter(df_filter_string)
    for path in first_level_results_path:
        tmp = pd.read_csv(path)
        if 'first_level_analysis' in path:
            tmp['session'] = path.split(os.sep)[-2]
            tmp['subject'] = path.split(os.sep)[-3]
            tmp['task'] = re.search('_(task-[a-z0-9]+)_', path).groups()[0]
            tmp['indicator'] = tmp.apply(lambda x: str((x.subject, x.session, x.task)), 1)
        if df_filter is not None:
            tmp = df_filter(tmp).reset_index()
        df.append(tmp)
    df = pd.concat(df)
    if 'indicator' in df.columns:
        n_voxels = len(df.groupby(['indicator', 'voxel']).size())
    else:
        n_voxels = df.voxel.nunique()
    logger.info("Will fit %d voxels" % (n_voxels))
    model = setup_model(df, df_filter_string, hierarchy_type, voxel_norm, sigma, sf_ecc_intercept,
                        sf_ecc_slope)
    if n_cores is None:
        n_cores = n_chains
    # n_cores cannot be larger than n_chains, or things get weird
    n_cores = min(n_cores, n_chains)
    with model:
        if save_path is not None:
            db = pm.backends.Text(os.path.splitext(save_path)[0])
        # not sure if this will work
        else:
            db = pm.backends.NDArray()
        if sampler == 'NUTS':
            trace = pm.sample(n_samples, chains=n_chains, cores=n_cores, random_seed=random_seed,
                              nuts_kwargs=nuts_kwargs, tune=1500, init=init, trace=db)
        elif sampler == 'Metropolis':
            trace = pm.sample(n_samples, step=pm.Metropolis(), chains=n_chains, cores=n_cores,
                              random_seed=random_seed, tune=10000, trace=db)
        post = pm.sample_posterior_predictive(trace, 500)
    inference_data = az.from_pymc3(trace, posterior_predictive=post)
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
    parser.add_argument("save_path",
                        help=("Path stem (with '.nc' extension) where we'll save the results"))
    parser.add_argument("first_level_results_path", nargs='+',
                        help=("Path to the first level results dataframe containing the data to "
                              "fit."))
    parser.add_argument("--random_seed", default=None, nargs='+', type=int,
                        help=("Random seed for the MCMC sampler. If None, we don't set it"))
    parser.add_argument("--n_samples", '-s', type=int, default=1000,
                        help=("The number of MCMC samples to draw"))
    parser.add_argument("--n_chains", '-c', type=int, default=4,
                        help=("The number of MCMC chains to run"))
    parser.add_argument("--n_cores", '-n', type=int, default=4,
                        help=("The number of cores to use when sampling."))
    parser.add_argument("--init", default='auto',
                        help=("How to initialize the sampler"))
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
    parser.add_argument("--hierarchy_type",
                        help="Hierarchy type: pooled, unpooled, or partially pooled")
    parser.add_argument("--sampler",
                        help=("The MCMC sampler to use: Metropolis or NUTS. Note if Metropolis, "
                              "will need way more samples (but each sample will be faster)"))
    args = vars(parser.parse_args())
    nuts_kwargs_tmp = args.pop('nuts_kwargs')
    nuts_kwargs = {}
    logger = logging.getLogger("pymc3")
    if nuts_kwargs_tmp is not None:
        for val in nuts_kwargs_tmp:
            try:
                k, v = val.split('=')
            except ValueError:
                logger.warn("Unable to get nuts_kwargs, assuming there are none")
                break
            else:
                try:
                    v = float(v)
                except ValueError:
                    try:
                        v = bool(v)
                    except ValueError:
                        pass
                nuts_kwargs[k] = v
    main(**args, **nuts_kwargs)
