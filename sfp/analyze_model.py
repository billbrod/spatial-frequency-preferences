#!/usr/bin/python
"""code to analyze the outputs of our 2d tuning model
"""
import matplotlib as mpl
# we do this because sometimes we run this without an X-server, and this backend doesn't need
# one. We set warn=False because the notebook uses a different backend and will spout out a big
# warning to that effect; that's unnecessarily alarming, so we hide it.
mpl.use('svg', warn=False)
import pandas as pd
import numpy as np
import torch
import re
import os
import argparse
import glob
import itertools
import warnings
from . import model as sfp_model


def load_single_model(save_path_stem, load_results_df=True):
    """load in the model, loss df, and model df found at the save_path_stem

    we also send the model to the appropriate device
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        if load_results_df:
            results_df = pd.read_csv(save_path_stem + '_results_df.csv')
        else:
            results_df = pd.read_csv(save_path_stem + '_results_df.csv', nrows=1)
    except FileNotFoundError as e:
        if load_results_df:
            raise e
        results_df = None
    loss_df = pd.read_csv(save_path_stem + '_loss.csv')
    # we try and infer model type from the path name, which we can do assuming we used the
    # Snakefile to generate saved model.
    vary_amps_label = save_path_stem.split('_')[-1]
    if vary_amps_label == 'vary':
        vary_amps = True
    elif vary_amps_label == 'constant':
        vary_amps = False
    ecc_type = save_path_stem.split('_')[-2]
    ori_type = save_path_stem.split('_')[-3]
    model = sfp_model.LogGaussianDonut(ori_type, ecc_type, vary_amps)
    model.load_state_dict(torch.load(save_path_stem + '_model.pt', map_location=device.type))
    model.eval()
    model.to(device)
    return model, loss_df, results_df


def combine_models(base_path_template, load_results_df=True):
    """load in many models and combine into dataframes

    returns: model_df, loss_df, results_df

    base_path_template: path template where we should find the results. should contain no string
    formatting symbols (e.g., "{0}" or "%s") but should contain at least one '*' because we will
    use glob to find them (and therefore should point to an actual file when passed to glob, one
    of: the loss df, model df, or model paramters).

    load_results_df: boolean. Whether to load the results_df or not. Set False if your results_df
    are too big and you're worried about having them all in memory. In this case, the returned
    results_df will be None.
    """
    models = []
    loss_df = []
    results_df = []
    path_stems = []
    for p in glob.glob(base_path_template):
        path_stem = p.replace('_loss.csv', '').replace('_model.pt', '').replace('_results_df.csv', '')
        # we do this to make sure we're not loading in the outputs of a model twice (by finding
        # both its loss.csv and its results_df.csv, for example)
        if path_stem in path_stems:
            continue
        # based on how these are saved, we can make some assumptions and grab extra info from their
        # paths
        metadata = {}
        if 'tuning_2d_simulated' in path_stem:
            metadata['modeling_goal'] = path_stem.split(os.sep)[-2]
        elif 'tuning_2d_model' in path_stem:
            metadata['session'] = path_stem.split(os.sep)[-2]
            metadata['subject'] = path_stem.split(os.sep)[-3]
            metadata['modeling_goal'] = path_stem.split(os.sep)[-4]
            metadata['mat_type'] = path_stem.split(os.sep)[-5]
            metadata['atlas_type'] = path_stem.split(os.sep)[-6]
            metadata['task'] = re.search('_(task-[a-z0-9]+)_', path_stem).groups()[0]
        path_stems.append(path_stem)
        model, loss, results = load_single_model(path_stem, load_results_df=load_results_df)
        for k, v in metadata.items():
            if results is not None:
                results[k] = v
            loss[k] = v
        results_df.append(results)
        loss_df.append(loss)
        tmp = loss.head(1)
        tmp = tmp.drop(['epoch_num', 'batch_num', 'loss'], 1)
        tmp['model'] = model
        for name, val in model.named_parameters():
            tmper = tmp.copy()
            tmper['model_parameter'] = name
            tmper['fit_value'] = val.cpu().detach().numpy()
            if results is not None:
                if 'true_model_%s' % name in results.columns:
                    tmper['true_value'] = results['true_model_%s' % name].unique()[0]
            models.append(tmper)
    loss_df = pd.concat(loss_df).reset_index(drop=True)
    if load_results_df:
        results_df = pd.concat(results_df).reset_index(drop=True).drop('index', 1)
    else:
        results_df = None
    models = pd.concat(models)
    return models, loss_df, results_df


def create_feature_df(models, eccen_range=(.01, 11), orientation_range=(0, np.pi),
                      orientation_n_steps=8, retinotopic_angle_n_steps=4, **identity_kwargs):
    """create dataframe with preferred period and amplitude for given models

    will do so at multiple (stimulus) orientations, retinotopic angles, and retinotopic
    eccentricities

    identity_kwargs: the values of the key, values pairs here should be lists the same length as
    models, which contain extra values to add to the features_df in order to identify the models
    (e.g. test_subset, etc)
    """
    features = []
    eccen = np.linspace(*eccen_range, num=10)
    orientations = np.linspace(*orientation_range, num=orientation_n_steps, endpoint=False)
    angles = np.linspace(*orientation_range, num=retinotopic_angle_n_steps, endpoint=False)
    for i, m in enumerate(models):
        for o, a in itertools.product(orientations, angles):
            period = m.preferred_period(o, eccen, a).detach().cpu().numpy()
            max_amp = m.max_amplitude(o, a).detach().cpu().numpy()
            data_dict = {'preferred_period': period, 'max_amplitude': max_amp,
                         'retinotopic_angle': a, 'fit_model_type': m.model_type,
                         'eccentricity': eccen, 'orientation': o}
            for k, v in identity_kwargs.items():
                data_dict[k] = v[i]
            features.append(pd.DataFrame(data_dict))
    return pd.concat(features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Load in a bunch of model results dataframes and save them. When run from "
                     "the command-line, we will not save out the combined results_df. If you want "
                     "to do that, run the combine_models function of this directly"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("base_path_template",
                        help=("path template where we should find the results. should contain no "
                              "string  formatting symbols (e.g., '{0}' or '%s') but should contain"
                              " at least one '*' because we will use glob to find them (and "
                              "therefore should point to an actual file when passed to glob, one"
                              " of: the loss df, model df, or model parameters)."))
    parser.add_argument("save_path_stem",
                        help=("Path stem (no extension) where we'll save the results"))
    args = vars(parser.parse_args())
    models, loss_df, results_df = combine_models(args['base_path_template'], False)
    models.to_csv(args['save_path_stem'] + "_model.csv")
    loss_df.to_csv(args['save_path_stem'] + "_loss.csv")
    models = models.drop_duplicates('model')
    features = create_feature_df(models.model.values, orientation_n_steps=50,
                                 retinotopic_angle_n_steps=50,
                                 test_subset=models.test_subset.values)
    features.to_csv(args['save_path_stem'] + '_features.csv')
