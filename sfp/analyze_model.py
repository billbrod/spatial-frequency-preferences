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
import functools
import os
import argparse
import glob
import itertools
import warnings
from . import model as sfp_model


def load_LogGaussianDonut(save_path_stem):
    """this loads and returns the actual model, given the saved parameters, for analysis
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    return model


def load_single_model(save_path_stem, load_results_df=True):
    """load in the model, loss df, and model df found at the save_path_stem

    we also send the model to the appropriate device
    """
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
    model_history_df = pd.read_csv(save_path_stem + "_model_history.csv")
    if 'test_subset' not in loss_df.columns or 'test_subset' not in model_history_df.columns:
        # unclear why this happens, it's really strange
        assert save_path_stem.split('_')[-4].startswith('c'), "Can't grab test_subset from path!"
        # this will give it the same spacing as the original version
        test_subset = ', '.join(save_path_stem.split('_')[-4][1:].split(','))
        if "test_subset" not in loss_df.columns:
            loss_df['test_subset'] = test_subset
        if "test_subset" not in model_history_df.columns:
            model_history_df['test_subset'] = test_subset
    model = load_LogGaussianDonut(save_path_stem)
    return model, loss_df, results_df, model_history_df


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
    model_history_df = []
    path_stems = []
    for p in glob.glob(base_path_template):
        path_stem = (p.replace('_loss.csv', '').replace('_model.pt', '')
                     .replace('_results_df.csv', '').replace('_model_history.csv', ''))
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
        model, loss, results, model_history = load_single_model(path_stem,
                                                                load_results_df=load_results_df)
        for k, v in metadata.items():
            if results is not None:
                results[k] = v
            loss[k] = v
        results_df.append(results)
        loss_df.append(loss)
        model_history_df.append(model_history)
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
    model_history_df = pd.concat(model_history_df).reset_index(drop=True)
    if load_results_df:
        results_df = pd.concat(results_df).reset_index(drop=True).drop('index', 1)
    else:
        results_df = None
    models = pd.concat(models)
    return models, loss_df, results_df, model_history_df


def _finish_feature_df(df, reference_frame='absolute'):
    if isinstance(df, list):
        df = pd.concat(df).reset_index(drop=True)
    df['reference_frame'] = reference_frame
    angle_ref = np.linspace(0, np.pi, 4, endpoint=False)
    angle_labels = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$']
    rel_labels = ['radial', 'forward spiral', 'angular', 'reverse spiral']
    abs_labels = ['vertical', 'forward diagonal', 'horizontal', 'reverse diagonal']
    if np.array_equiv(angle_ref, df["Retinotopic angle (rad)"].unique()):
        df["Retinotopic angle (rad)"] = df["Retinotopic angle (rad)"].map(dict((k, v) for k, v in
                                                                               zip(angle_ref,
                                                                                   angle_labels)))
    if np.array_equiv(angle_ref, df["Orientation (rad)"].unique()):
        if reference_frame == 'relative':
            df["Stimulus type"] = df["Orientation (rad)"].map(dict((k, v) for k, v in
                                                                   zip(angle_ref, rel_labels)))
        elif reference_frame == 'absolute':
            df["Stimulus type"] = df["Orientation (rad)"].map(dict((k, v) for k, v in
                                                                   zip(angle_ref, abs_labels)))
    return df


def create_preferred_period_df(model, retinotopic_angle=np.linspace(0, np.pi, 4, endpoint=False),
                               orientation=np.linspace(0, np.pi, 4, endpoint=False),
                               eccentricity=np.linspace(0, 11, 11), reference_frame='absolute'):
    df = []
    for o in orientation:
        if reference_frame == 'absolute':
            tmp = model.preferred_period(eccentricity, retinotopic_angle, o)
        elif reference_frame == 'relative':
            tmp = model.preferred_period(eccentricity, retinotopic_angle, rel_sf_angle=o)
        tmp = pd.DataFrame(tmp.detach().numpy(), index=retinotopic_angle, columns=eccentricity)
        tmp = tmp.reset_index().rename(columns={'index': 'Retinotopic angle (rad)'})
        tmp['Orientation (rad)'] = o
        df.append(pd.melt(tmp, ['Retinotopic angle (rad)', 'Orientation (rad)'],
                          var_name='Eccentricity (deg)', value_name='Preferred period (dpc)'))
    return _finish_feature_df(df, reference_frame)


def create_preferred_period_contour_df(model,
                                       retinotopic_angle=np.linspace(0, np.pi, 48, endpoint=False),
                                       orientation=np.linspace(0, np.pi, 4, endpoint=False),
                                       period_target=[.5, 1, 1.5], reference_frame='absolute'):
    df = []
    for p in period_target:
        if reference_frame == 'absolute':
            tmp = model.preferred_period_contour(p, retinotopic_angle, orientation)
        elif reference_frame == 'relative':
            tmp = model.preferred_period_contour(p, retinotopic_angle, rel_sf_angle=orientation)
        tmp = pd.DataFrame(tmp.detach().numpy(), index=retinotopic_angle, columns=orientation)
        tmp = tmp.reset_index().rename(columns={'index': 'Retinotopic angle (rad)'})
        tmp['Preferred period (dpc)'] = p
        df.append(pd.melt(tmp, ['Retinotopic angle (rad)', 'Preferred period (dpc)'],
                          var_name='Orientation (rad)', value_name='Eccentricity (deg)'))
    return _finish_feature_df(df, reference_frame)


def create_max_amplitude_df(model, retinotopic_angle=np.linspace(0, np.pi, 48, endpoint=False),
                            orientation=np.linspace(0, np.pi, 4, endpoint=False),
                            reference_frame='absolute'):
    if reference_frame == 'absolute':
        tmp = model.max_amplitude(retinotopic_angle, orientation).detach().numpy()
    elif reference_frame == 'relative':
        tmp = model.max_amplitude(retinotopic_angle, rel_sf_angle=orientation).detach().numpy()
    tmp = pd.DataFrame(tmp, index=retinotopic_angle, columns=orientation)
    tmp = tmp.reset_index().rename(columns={'index': 'Retinotopic angle (rad)'})
    df = pd.melt(tmp, ['Retinotopic angle (rad)'], var_name='Orientation (rad)',
                 value_name='Max amplitude')
    return _finish_feature_df(df, reference_frame)


def create_feature_df(models, feature_type='preferred_period', reference_frame='absolute',
                      retinotopic_angle=np.linspace(0, np.pi, 4, endpoint=False),
                      orientation=np.linspace(0, np.pi, 4, endpoint=False),
                      eccentricity=np.linspace(0, 11, 11), period_target=[.5, 1, 1.5]):
    df = []
    for ind in models.indicator.unique():
        m = sfp_model.LogGaussianDonut.init_from_df(models.query('indicator==@ind'))
        if feature_type == 'preferred_period':
            df.append(create_preferred_period_df(m, retinotopic_angle, orientation, eccentricity,
                                                 reference_frame))
        elif feature_type == 'preferred_period_contour':
            df.append(create_preferred_period_contour_df(m, retinotopic_angle, orientation,
                                                         period_target, reference_frame))
        elif feature_type == 'max_amplitude':
            df.append(create_max_amplitude_df(m, retinotopic_angle, orientation, reference_frame))
        df[-1]['indicator'] = ind
    return pd.concat(df).reset_index(drop=True)


def bootstrap_features(feature_df, n_bootstraps, value_name='Preferred period (dpc)',
                       groupby_cols=['indicator', 'Retinotopic angle (rad)', 'Orientation (rad)',
                                     'Eccentricity (deg)']):
    assert groupby_cols[0] == 'indicator', "First groupby_cols value must be indicator!"
    # first use groupby to get this to a ndarray, from
    # https://stackoverflow.com/questions/47715300/convert-a-pandas-dataframe-to-a-multidimensional-ndarray
    grouped = feature_df.groupby(groupby_cols)[value_name].mean()
    # create an empty array of NaN of the right dimensions
    shape = tuple(map(len, grouped.index.levels))
    all_data = np.full(shape, np.nan)
    # fill it using Numpy's advanced indexing
    all_data[tuple(grouped.index.codes)] = grouped.values.flat
    if functools.reduce(lambda x, y: x*y, shape) != len(feature_df):
        raise Exception("groupby_cols does not completely cover the columns of feature_df!")
    bootstraps = np.random.randint(0, feature_df.indicator.nunique(),
                                   size=(n_bootstraps, feature_df.indicator.nunique()))
    bootstrapped = np.empty((n_bootstraps, *shape[1:]))
    for i, b in enumerate(bootstraps):
        bootstrapped[i] = np.mean(all_data[b], 0)
    return bootstrapped


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
                              " of: the loss df, model history df, results df, or model "
                              "parameters)."))
    parser.add_argument("save_path_stem",
                        help=("Path stem (no extension) where we'll save the results"))
    args = vars(parser.parse_args())
    models, loss_df, results_df, model_history_df = combine_models(args['base_path_template'],
                                                                   False)
    models.to_csv(args['save_path_stem'] + "_model.csv", index=False)
    model_history_df.to_csv(args['save_path_stem'] + "_model_history.csv", index=False)
    loss_df.to_csv(args['save_path_stem'] + "_loss.csv", index=False)
    models = models.drop_duplicates('model')
