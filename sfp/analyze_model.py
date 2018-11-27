#!/usr/bin/python
"""code to analyze the outputs of our 2d tuning model
"""
import matplotlib as mpl
# we do this because sometimes we run this without an X-server, and this backend doesn't need one
mpl.use('svg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
import glob
import model as sfp_model


def load_single_model(save_path_stem, model_type=None, load_results_df=True):
    """load in the model, loss df, and model df found at the save_path_stem

    we also send the model to the appropriate device
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if load_results_df:
        results_df = pd.read_csv(save_path_stem + '_model_df.csv')
    else:
        results_df = pd.read_csv(save_path_stem + '_model_df.csv', nrows=1)
    loss_df = pd.read_csv(save_path_stem + '_loss.csv')
    if model_type is None:
        # then we try and infer it from the path name, which we can do assuming we used the
        # Snakefile to generate saved model.
        model_type = save_path_stem.split('_')[-1]
    if model_type == 'full-absolute':
        model = sfp_model.LogGaussianDonut(.4, orientation_type='absolute')
    elif model_type == 'full-relative':
        model = sfp_model.LogGaussianDonut(.4, orientation_type='relative')
    elif model_type == 'constant':
        model = sfp_model.ConstantIsoLogGaussianDonut(.4)
    elif model_type == 'scaling':
        model = sfp_model.ScalingIsoLogGaussianDonut(.4)
    elif model_type == 'iso':
        model = sfp_model.IsoLogGaussianDonut(.4)
    else:
        raise Exception("Don't know how to handle model_type %s!" % model_type)
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
        path_stem = p.replace('_loss.csv', '').replace('_model.pt', '').replace('_model_df.csv', '')
        # we do this to make sure we're not loading in the outputs of a model twice (by finding
        # both its loss.csv and its model_df.csv, for example)
        if path_stem in path_stems:
            continue
        path_stems.append(path_stem)
        model, loss, results = load_single_model(path_stem, load_results_df=load_results_df)
        results_df.append(results)
        loss_df.append(loss)
        tmp = loss.head(1)
        tmp = tmp.drop(['epoch_num', 'batch_num', 'loss'], 1)
        tmp['model'] = model
        for name, val in model.named_parameters():
            tmper = tmp.copy()
            tmper['model_parameter'] = name
            tmper['fit_value'] = val.cpu().detach().numpy()
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
