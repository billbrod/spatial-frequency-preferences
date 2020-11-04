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
from sklearn import metrics
from torch.utils import data as torchdata
from . import model as sfp_model
from tqdm import tqdm


def load_LogGaussianDonut(save_path_stem):
    """this loads and returns the actual model, given the saved parameters, for analysis
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # we try and infer model type from the path name, which we can do assuming we used the
    # Snakefile to generate saved model.
    vary_amps = save_path_stem.split('_')[-1]
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
        if not save_path_stem.split('_')[-4].startswith('c'):
            raise Exception("Can't grab test_subset from path %s!" % save_path_stem)
        # this will give it the same spacing as the original version
        test_subset = ', '.join(save_path_stem.split('_')[-4][1:].split(','))
        if "test_subset" not in loss_df.columns:
            loss_df['test_subset'] = test_subset
        if "test_subset" not in model_history_df.columns:
            model_history_df['test_subset'] = test_subset
    model = load_LogGaussianDonut(save_path_stem)
    return model, loss_df, results_df, model_history_df


def combine_models(base_path_template, load_results_df=True, groupaverage=False):
    """load in many models and combine into dataframes

    returns: model_df, loss_df, results_df

    base_path_template: path template where we should find the results. should contain no string
    formatting symbols (e.g., "{0}" or "%s") but should contain at least one '*' because we will
    use glob to find them (and therefore should point to an actual file when passed to glob, one
    of: the loss df, model df, or model paramters).

    load_results_df: boolean. Whether to load the results_df or not. Set False if your results_df
    are too big and you're worried about having them all in memory. In this case, the returned
    results_df will be None.

    groupaverage : boolean. Whether to grab the individual subject fits
    or the sub-groupaverage subject (which is a bootstrapped average
    subject)

    """
    models = []
    loss_df = []
    results_df = []
    model_history_df = []
    path_stems = []
    for p in glob.glob(base_path_template):
        if groupaverage and 'sub-groupaverage' not in p:
            continue
        if not groupaverage and 'sub-groupaverage' in p:
            continue
        path_stem = (p.replace('_loss.csv', '').replace('_model.pt', '')
                     .replace('_results_df.csv', '').replace('_model_history.csv', '')
                     .replace('_preds.pt', ''))
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
            metadata['subject'] = path_stem.split(os.sep)[-3]
            if not groupaverage:
                metadata['session'] = path_stem.split(os.sep)[-2]
            else:
                session_dir = path_stem.split(os.sep)[-2]
                metadata['session'] = session_dir.split('_')[0]
                metadata['groupaverage_seed'] = session_dir.split('_')[-1]
            metadata['modeling_goal'] = path_stem.split(os.sep)[-4]
            metadata['mat_type'] = path_stem.split(os.sep)[-5]
            metadata['atlas_type'] = path_stem.split(os.sep)[-6]
            metadata['task'] = re.search('_(task-[a-z0-9]+)_', path_stem).groups()[0]
            metadata['indicator'] = str((metadata['subject'], metadata['session'], metadata['task'])).replace("'", "")
        path_stems.append(path_stem)
        model, loss, results, model_history = load_single_model(path_stem,
                                                                load_results_df=load_results_df)
        for k, v in metadata.items():
            if results is not None:
                results[k] = v
            loss[k] = v
            model_history[k] = v
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
    """helper function to clean up the feature dataframes

    This helper function cleans up the feature dataframes so that they
    can be more easily used for plotting with feature_df_plot and
    feature_df_polar_plot functions. It performs the following actions:

    1. Adds reference_frame as column.

    2. Converts retinotopic angles to human-readable labels (only if
       default retinotopic angles used).

    3. Adds "Stimulus type" as column, giving human-readable labels
       based on "Orientation" columns.

    Parameters
    ----------
    df : pd.DataFrame
        The feature dataframe to finish up
    reference_frame : {'absolute, 'relative'}
        The reference frame of df

    Returns
    -------
    df : pd.DataFrame
        The cleaned up dataframe

    """
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
    if reference_frame == 'relative':
        df["Stimulus type"] = df["Orientation (rad)"].map(dict((k, v) for k, v in
                                                               zip(angle_ref, rel_labels)))
    elif reference_frame == 'absolute':
        df["Stimulus type"] = df["Orientation (rad)"].map(dict((k, v) for k, v in
                                                               zip(angle_ref, abs_labels)))
    return df


def create_preferred_period_df(model, reference_frame='absolute',
                               retinotopic_angle=np.linspace(0, np.pi, 4, endpoint=False),
                               orientation=np.linspace(0, np.pi, 4, endpoint=False),
                               eccentricity=np.linspace(0, 11, 11)):
    """Create dataframe summarizing preferred period as function of eccentricity

    Generally, you should not call this function directly, but use
    create_feature_df. Differences from that function: this functions
    requires the initialized model and only creates the info for a
    single model, while create_feature_df uses the models dataframe to
    initialize models itself, and combines the outputs across multiple
    scanning sessions.

    This function creates a dataframe summarizing the specified model's
    preferred period as a function of eccentricity, for multiple
    stimulus orientations (in either absolute or relative reference
    frames) and retinotopic angles. This dataframe is then used for
    creating plots to summarize the model.

    You can also use this function to create the information necessary
    to plot preferred period as a function of retinotopic angle at a
    specific eccentricity. You can do that by reducing the number of
    eccentricities (e.g., eccentricity=[5]) and increasing the number of
    retinotopic angles (e.g., np.linspace(0, 2*np.pi, 49)).

    Unless you have something specific in mind, you can trust the
    default options for retinotopic_angle, orientation, and
    eccentricity.

    Parameters
    ----------
    model : sfp.model.LogGaussianDonut
        a single, initialized model, which we will summarize.
    reference_frame : {"absolute", "relative"}, optional
        Whether we use the absolute or relative reference frame in the
        feature dataframe; that is whether we consider conventional
        gratings (absolute, orientation is relative to
        vertical/horizontal), or our log-polar gratings (relative,
        orientation is relative to fovea).
    retinotopic_angle : np.array, optional
        Array specifying which retinotopic angles to find the preferred
        period for. If you don't care about retinotopic angle and just
        want to summarize the model's overall features, you should use
        the default (which includes all angles where the model can have
        different preferences, based on its parametrization) and then
        average over them.
    orientation : np.array, optional
        Array specifying which stimulus orientations to find the
        preferred period for. Note that the meaning of these
        orientations will differ depending on the value of
        reference_frame; you should most likely plot and interpret the
        output based on the "Stimulus type" column instead (which
        include strings like 'vertical'/'horizontal' or
        'radial'/'angular'). However, this mapping can only happen if
        the values in orientation line up with our stimuli (0, pi/4,
        pi/2, 3*pi/2), and thus it's especially recommended that you use
        the default value for this argument. If you don't care about
        orientation and just want to summarize the model's overall
        features, you should use the default (which includes all
        orientations where the model can have different preferences,
        based on its parametrization) and then average over them.
    eccentricity : np.array, optional
        Array specifying which eccentricities to find the preferred
        period for. The default values span the range of measurements
        for our experiment, but you can certainly go higher if you
        wish. Note that, for creating the plot of preferred period as a
        function of eccentricity, the model's predictions will always be
        linear and so you most likely only need 2 points. More are
        included because you may want to examine the preferred period at
        specific eccentricities

    Returns
    -------
    preferred_period_df : pd.DataFrame
        Dataframe containing preferred period of the model, to use with
        sfp.plotting.feature_df_plot for plotting preferred period as a
        function of eccentricity.

    """
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
                          var_name='Eccentricity (deg)', value_name='Preferred period (deg)'))
    return _finish_feature_df(df, reference_frame)


def create_preferred_period_contour_df(model, reference_frame='absolute',
                                       retinotopic_angle=np.linspace(0, 2*np.pi, 49),
                                       orientation=np.linspace(0, np.pi, 4, endpoint=False),
                                       period_target=[.5, 1, 1.5], ):
    """Create dataframe summarizing preferred period as function of retinotopic angle

    Generally, you should not call this function directly, but use
    create_feature_df. Differences from that function: this functions
    requires the initialized model and only creates the info for a
    single model, while create_feature_df uses the models dataframe to
    initialize models itself, and combines the outputs across multiple
    scanning sessions.

    This function creates a dataframe summarizing the specified model's
    preferred period as a function of retinotopic angle, for multiple
    stimulus orientations (in either absolute or relative reference
    frames) and target periods. That is, it contains information showing
    at what eccentricity the model's preferred period is, e.g., 1 for a
    range of retinotopic angles and stimulus orientation. This dataframe
    is then used for creating plots to summarize the model.

    So this function creates information to plot iso-preferred period
    lines. If you want to plot preferred period as a function of
    retinotopic angle for a specific eccentricity, you can do that with
    create_preferred_period_df, by reducing the number of eccentricities
    (e.g., eccentricity=[5]) and increasing the number of retinotopic
    angles (e.g., np.linspace(0, 2*np.pi, 49))

    Unless you have something specific in mind, you can trust the
    default options for retinotopic_angle, orientation, and
    period_target

    Parameters
    ----------
    model : sfp.model.LogGaussianDonut
        a single, initialized model, which we will summarize.
    reference_frame : {"absolute", "relative"}, optional
        Whether we use the absolute or relative reference frame in the
        feature dataframe; that is whether we consider conventional
        gratings (absolute, orientation is relative to
        vertical/horizontal), or our log-polar gratings (relative,
        orientation is relative to fovea).
    retinotopic_angle : np.array, optional
        Array specifying which retinotopic angles to find the preferred
        period for. Note that the sampling of retinotopic angle is much
        finer than for create_preferred_period_df (and goes all the way
        to 2*pi), because this is what we will use as the dependent
        variable in our plotsl
    orientation : np.array, optional
        Array specifying which stimulus orientations to find the
        preferred period for. Note that the meaning of these
        orientations will differ depending on the value of
        reference_frame; you should most likely plot and interpret the
        output based on the "Stimulus type" column instead (which
        include strings like 'vertical'/'horizontal' or
        'radial'/'angular'). However, this mapping can only happen if
        the values in orientation line up with our stimuli (0, pi/4,
        pi/2, 3*pi/2), and thus it's especially recommended that you use
        the default value for this argument. If you don't care about
        orientation and just want to summarize the model's overall
        features, you should use the default (which includes all
        orientations where the model can have different preferences,
        based on its parametrization) and then average over them.
    period_target : np.array, optional
        Array specifying which the target periods for the model. The
        intended use of this dataframe is to plot contour plots showing
        at what eccentricity the model will have a specified preferred
        period (for a range of angles and orientations), and this
        argument specifies those periods.

    Returns
    -------
    preferred_period_contour_df : pd.DataFrame
        Dataframe containing preferred period of the model, to use with
        sfp.plotting.feature_df_polar_plot for plotting preferred period
        as a function of retinotopic angle.

    """
    df = []
    for p in period_target:
        if reference_frame == 'absolute':
            tmp = model.preferred_period_contour(p, retinotopic_angle, orientation)
        elif reference_frame == 'relative':
            tmp = model.preferred_period_contour(p, retinotopic_angle, rel_sf_angle=orientation)
        tmp = pd.DataFrame(tmp.detach().numpy(), index=retinotopic_angle, columns=orientation)
        tmp = tmp.reset_index().rename(columns={'index': 'Retinotopic angle (rad)'})
        tmp['Preferred period (deg)'] = p
        df.append(pd.melt(tmp, ['Retinotopic angle (rad)', 'Preferred period (deg)'],
                          var_name='Orientation (rad)', value_name='Eccentricity (deg)'))
    return _finish_feature_df(df, reference_frame)


def create_max_amplitude_df(model, reference_frame='absolute',
                            retinotopic_angle=np.linspace(0, 2*np.pi, 49),
                            orientation=np.linspace(0, np.pi, 4, endpoint=False)):
    """Create dataframe summarizing max amplitude as function of retinotopic angle

    Generally, you should not call this function directly, but use
    create_feature_df. Differences from that function: this functions
    requires the initialized model and only creates the info for a
    single model, while create_feature_df uses the models dataframe to
    initialize models itself, and combines the outputs across multiple
    scanning sessions.

    This function creates a dataframe summarizing the specified model's
    maximum amplitude as a function of retinotopic angle, for multiple
    stimulus orientations (in either absolute or relative reference
    frames). This dataframe is then used for creating plots to summarize
    the model.

    Unless you have something specific in mind, you can trust the
    default options for retinotopic_angle and orientation.

    Parameters
    ----------
    model : sfp.model.LogGaussianDonut
        a single, initialized model, which we will summarize.
    reference_frame : {"absolute", "relative"}, optional
        Whether we use the absolute or relative reference frame in the
        feature dataframe; that is whether we consider conventional
        gratings (absolute, orientation is relative to
        vertical/horizontal), or our log-polar gratings (relative,
        orientation is relative to fovea).
    retinotopic_angle : np.array, optional
        Array specifying which retinotopic angles to find the preferred
        period for. Note that the sampling of retinotopic angle is much
        finer than for create_preferred_period_df (and goes all the way
        to 2*pi), because this is what we will use as the dependent
        variable in our plotsl
    orientation : np.array, optional
        Array specifying which stimulus orientations to find the
        preferred period for. Note that the meaning of these
        orientations will differ depending on the value of
        reference_frame; you should most likely plot and interpret the
        output based on the "Stimulus type" column instead (which
        include strings like 'vertical'/'horizontal' or
        'radial'/'angular'). However, this mapping can only happen if
        the values in orientation line up with our stimuli (0, pi/4,
        pi/2, 3*pi/2), and thus it's especially recommended that you use
        the default value for this argument. If you don't care about
        orientation and just want to summarize the model's overall
        features, you should use the default (which includes all
        orientations where the model can have different preferences,
        based on its parametrization) and then average over them.

    Returns
    -------
    max_amplitude_df : pd.DataFrame
        Dataframe containing maximum amplitude of the model, to use with
        sfp.plotting.feature_df_polar_plot for plotting max amplitude as
        a function of retinotopic angle.

    """
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
                      gb_cols=['subject', 'bootstrap_num'], **kwargs):
    """Create dataframe to summarize the predictions made by our models

    The point of this dataframe is to generate plots (using
    plotting.feature_df_plot and plotting.feature_df_polar_plot) to
    easily visualize what the parameters of our model mean, either for
    demonstrative purposes or with the parameters fit to actual data.

    This is used to create a feature data frame that combines info
    across multiple models, using the columns indicated in gb_cols to
    separate them, and serves as a wrapper around three other functions:
    create_preferred_period_df, create_preferred_period_contour_df, and
    create_max_amplitude_df (based on the value of the feature_type
    arg). We loop through the unique subsets of the data given by
    models.groupby(gb_cols) in the models dataframe and instantiate a
    model for each one (thus, each subset must only have one associated
    model). We then create dataframes summarizing the relevant features,
    add the identifying information, and, concatenate.

    The intended use of these dataframes is to create plots showing the
    models' predictions for (using bootstraps to get confidence
    intervals to show variability across subjects):
    
    1. preferred period as a function of eccentricity:

    ```
    pref_period = create_feature_df(models, feature_type='preferred_period')
    sfp.plotting.feature_df_plot(pref_period)
    # average over retinotopic angle
    sfp.plotting.feature_df_plot(pref_period, col=None, 
                                 pre_boot_gb_func=np.mean)
    ```

    2. preferred period as a function of a function of retinotopic
       angle of stimulus orientation at a given eccentricity:

    ```
    pref_period = create_feature_df(models, feature_type='preferred_period',
                                    eccentricity=[5],
                                    retinotopic_angle=np.linspace(0, 2*np.pi, 49))
    sfp.plotting.feature_df_polar_plot(pref_period, col='Eccentricity (deg)', 
                                       r='Preferred period (deg)')
    ```

    3. iso-preferred period lines as a function of retinotopic angle and
       stimulus orientation (i.e., at what eccentricity do you have a
       preferred period of 1 for this angle and orientation):

    ```
    pref_period_contour = create_feature_df(models, 
                                            feature_type='preferred_period_contour')
    sfp.plotting.feature_df_polar_plot(pref_period_contour)
    ```

    4. max amplitude as a function of retinotopic angle and stimulus
       orientation:

    ```
    max_amp = create_feature_df(models, feature_type='max_amplitude')
    sfp.plotting.feature_df_polar_plot(max_amp, col=None, r='Max amplitude')
    ```

    Parameters
    ----------
    models : pd.DataFrame
        dataframe summarizing model fits across many subjects / sessions
        (as created by analyze_model.combine_models function). Must
        contain the columns indicated in gb_cols and a row for each of
        the model's 11 parameters
    feature_type : {"preferred_period", "preferred_period_contour", "max_amplitude"}, optional
        Which feature dataframe to create. Determines which function we
        call, from create_preferred_period_df,
        create_preferred_period_contour_df, and create_max_amplitude_df
    reference_frame : {"absolute", "relative"}, optional
        Whether we use the absolute or relative reference frame in the
        feature dataframe; that is whether we consider conventional
        gratings (absolute, orientation is relative to
        vertical/horizontal), or our log-polar gratings (relative,
        orientation is relative to fovea).
    gb_cols : list, optional
        list of strs indicating columns in the df to groupby. when we
        groupby these columns, each subset should give a single
        model. Thus, this should be something like ['subject'],
        ['subject', 'bootstrap_num'], or ['subject', 'session']
        (depending on the contents of your df)
    kwargs : {retinotopic_angle, orientation, eccentricity, period_target}
        passed to the various create_*_df functions. See their
        docstrings for more info. if not set, use the defaults.

    Returns
    -------
    feature_df : pd.DataFrame
        Dataframe containing specified feature info

    """
    df = []
    for n, g in models.groupby(gb_cols):
        m = sfp_model.LogGaussianDonut.init_from_df(g)
        if feature_type == 'preferred_period':
            df.append(create_preferred_period_df(m, reference_frame, **kwargs))
        elif feature_type == 'preferred_period_contour':
            df.append(create_preferred_period_contour_df(m, reference_frame, **kwargs))
        elif feature_type == 'max_amplitude':
            df.append(create_max_amplitude_df(m, reference_frame, **kwargs))
        # in this case, gb_cols is a list with one element, so n will
        # just be a single element (probably a str). In order for the
        # following dict(zip(gb_cols, n)) call to work correctly, both
        # have to be lists with the same length, so this ensures that.
        if len(gb_cols) == 1:
            n = [n]
        df[-1] = df[-1].assign(**dict(zip(gb_cols, n)))
    return pd.concat(df).reset_index(drop=True)


def collect_final_loss(paths):
    """collect up the loss files, add some metadata, and concat

    We loop through the paths, loading each in, grab the last epoch, add
    some metadata by parsing the path, and then concatenate and return
    the resulting df

    Note that the assumed use here is collecting the loss.csv files
    created by the different folds of cross-validation, but we don't
    explicitly check for that and so this maybe useful in other contexts

    Parameters
    ----------
    paths : list
        list of strs giving the paths to the loss files. we attempt to
        parse these strings to find the subject, session, and task, and
        will raise an Exception if we can't do so

    Returns
    -------
    df : pd.DataFrame
        the collected loss

    """
    df = []
    print(f"Loading in {len(paths)} total paths")
    pbar = tqdm(range(len(paths)))
    for i in pbar:
        p = paths[i]
        regexes = [r'(sub-[a-z0-9]+)', r'(ses-[a-z0-9]+)', r'(task-[a-z0-9]+)']
        regex_names = ['subject', 'session', 'task']
        if 'sub-groupaverage' in p:
            regex_names.append('groupaverage_seed')
            regexes.append(r'(_s[0-9]+)')
        pbar.set_postfix(path=os.path.split(p)[-1])
        tmp = pd.read_csv(p)
        last_epoch = tmp.epoch_num.unique().max()
        tmp = tmp.query("epoch_num == @last_epoch")
        for n, regex in zip(regex_names, regexes):
            res = re.findall(regex, p)
            if len(set(res)) != 1:
                raise Exception(f"Unable to infer {n} from path {p}!")
            tmp[n] = res[0]
        df.append(tmp)
    return pd.concat(df)


def _calc_loss(preds, targets, loss_func, average=True):
    """Compute loss from preds and targets.

    Parameters
    ----------
    preds : torch.tensor
        The torch tensor containing the predictions
    targets : torch.tensor
        The torch tensor containing the targets
    loss_func : str
        The loss function to compute. One of: {'weighted_normed_loss',
        'crosscorrelation', 'normed_loss', 'explained_variance_score',
        'cosine_distance', 'cosine_distance_scaled'}.
    average : bool, optional
        If True, we average the cv loss so we have only one value. If False, we
        return one value per voxel.

    Returns
    -------
    loss : array or float
        The loss, either overall or per voxel.

    """
    if loss_func == 'crosscorrelation':
        # targets[..., 0] contains the actual targets, targets[..., 1]
        # contains the precision, unimportant right here
        corr = np.corrcoef(targets[..., 0].cpu().detach().numpy(),
                           preds.cpu().detach().numpy())
        cv_loss = corr[0, 1]
        if not average:
            raise Exception("crosscorrelation must be averaged!")
    elif 'normed_loss' in loss_func:
        if loss_func.startswith('weighted'):
            weighted = True
        else:
            weighted = False
        cv_loss = sfp_model.weighted_normed_loss(preds, targets, weighted=weighted,
                                                 average=average)
        if not average:
            cv_loss = cv_loss.cpu().detach().numpy().mean(1)
        else:
            cv_loss = cv_loss.item()
    elif loss_func == 'explained_variance_score':
        # targets[..., 0] contains the actual targets, targets[..., 1]
        # contains the precision, unimportant right here
        cv_loss = metrics.explained_variance_score(targets[..., 0].cpu().detach().numpy(),
                                                   preds.cpu().detach().numpy(),
                                                   multioutput='uniform_average')
        if not average:
            raise Exception("explained variance score must be averaged!")
    elif loss_func.startswith('cosine_distance'):
        cv_loss = metrics.pairwise.cosine_distances(targets[..., 0].cpu().detach().numpy(),
                                                    preds.cpu().detach().numpy())
        # for some reason, this returns a matrix of distances, giving the
        # distance between each sample in X and Y. in our case, that means the
        # distance between the targets of each voxel and the prediction of each
        # voxel. We just want the diagonal, which is the distance between
        # voxel's target and its own predictions
        cv_loss = np.diag(cv_loss)
        if loss_func.endswith('_scaled'):
            # see paper / notebook for derivation, but I determined that
            # our normed loss (without precision-weighting) is 2/n times
            # the cosine distance (where n is the number of classes, 48 in
            # our case, so that's equal to 1/24)
            cv_loss *= 1/24
        if average:
            cv_loss = cv_loss.mean()
    return cv_loss


def calc_cv_error(loss_files, dataset_path, wildcards, outputs,
                  df_filter_string='drop_voxels_with_negative_amplitudes,drop_voxels_near_border'):
    """Calculate cross-validated loss and save as new dataframe

    We use 12-fold cross-validation to determine the mode that best fits
    the data for each scanning session. To do that, we fit the model to
    a subset of the data (the subset contains all responses to 44 out of
    the 48 stimulus classes, none to the other 4). When fitting the
    cross-validation models, we follow the same procedure we use when
    fitting all the data, but we need to use something else for
    evaluation: we get each cross-validation model's predictions for the
    4 classes it *didn't* see, concatenating together these predictions
    for all 12 of them, then compare this against the full dataset (all
    voxels, all stimuli). We then create a dataframe containing this
    loss, as well as other identifying information, and save it at the
    specified path.

    We also save out the predictions and targets tensors.

    The arguments for this function are a bit strange because it's
    expressly meant to be called by a snakemake rule and not directly
    from a python interpreter (it gets called by the rules calc_cv_error
    and calc_simulated_cv_error)

    Parameters
    ----------
    loss_files : list
        list of strings giving the paths to loss files for the
        cross-validation models. each one contains, among other things
        the specific test subset for this model, and there should be an
        associated model.pt file saved in the same folder.
    dataset_path : str
        The path to the first_level_analysis dataframe, saved as a csv,
        which contains the actual data our model was fit to predict
    wildcards : dict
        dictionary of wildcards, information used to identify this model
        (e.g., subject, session, crossvalidation seed, model type,
        etc). Automatically put together by snakemake
    outputs : list
        list containing two strings, the paths to save the loss dataframe (as a
        csv) and the predictions / targets tensors (as a pt)
    df_filter_string : str or None, optional
        a str specifying how to filter the voxels in the dataset. see
        the docstrings for sfp.model.FirstLevelDataset and
        sfp.model.construct_df_filter for more details. If None, we
        won't filter. Should probably use the default, which is what all
        models are trained using.

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if df_filter_string:
        df_filter = sfp_model.construct_df_filter(df_filter_string)
    else:
        df_filter = None
    ds = sfp_model.FirstLevelDataset(dataset_path, device=device, df_filter=df_filter)
    dl = torchdata.DataLoader(ds, len(ds))
    features, targets = next(iter(dl))
    preds = torch.empty(targets.shape[:2], dtype=targets.dtype)
    for path in loss_files:
        m, l, _, _ = load_single_model(path.replace('_loss.csv', ''), False)
        test_subset = l.test_subset.unique()
        test_subset = [int(i) for i in test_subset[0].split(',')]
        pred = m(features[:, test_subset, :])
        preds[:, test_subset] = pred
    torch.save({'predictions': preds, 'targets': targets}, outputs[1])
    data = dict(wildcards)
    data.pop('model_type')
    data['loss_func'] = []
    data['cv_loss'] = []
    for loss_func in ['weighted_normed_loss', 'crosscorrelation', 'normed_loss',
                      'explained_variance_score', 'cosine_distance',
                      'cosine_distance_scaled']:
        cv_loss = _calc_loss(preds, targets, loss_func, True)
        data['loss_func'].append(loss_func)
        data['cv_loss'].append(cv_loss)
    data['dataset_df_path'] = dataset_path
    data['fit_model_type'] = l.fit_model_type.unique()[0]
    if 'true_model_type' in l.columns:
        data['true_model_type'] = l.true_model_type.unique()[0]
    cv_loss_csv = pd.DataFrame(data)
    cv_loss_csv.to_csv(outputs[0], index=False)


def gather_results(base_path, outputs, metadata, cv_loss_files=None, groupaverage=False):
    """Combine model dataframes

    We fit a huge number of models as part of this analysis pipeline. In
    order to make examining their collective results easier, we need to
    combine them in a meaningful way, throwing away the unnecessary
    info. This function uses the combine_models function to load in the
    models, loss, and models_history dataframes (not the results one,
    which is the largest). We then use df.groupby(metadata) and some
    well-placed funtions to summarize the information and then save them
    out.

    This was written to be called by snakemake rules, not from a python
    interpeter directly

    Parameters
    ----------
    base_path : str
        path template where we should find the results. Should contain
        no string formatting symbols, but should contain at least one
        '*' because we will use glob to find them. We do not search for
        them recursively, so you will need multiple '*'s if you want to
        combine dataframes contained in different folders
    outputs : list
        list of 5 or 6 strs giving the paths to save models,
        grouped_loss, timing_df, diff_df, model_history, and
        (optionally) cv_loss to.
    metadata : list
        list of strs giving the columns in the individual models, loss,
        and model_history dataframes that we will groupby in order to
        summarize them.
    cv_loss_files : list, optional
        either None or list of cross-validated loss dataframes (as
        cretated by calc_cv_error). If not None, outputs must contain 6
        strs. because of how these dataframes were constructed, we
        simply concatenate them, doing none fo the fancy groupby we do
        for the other dataframes
    groupaverage : bool, optional
        whether to grab the individual subject fits or the
        sub-groupaverage subject (which is a bootstrapped average
        subject)

    """
    models, loss_df, _, model_history = combine_models(base_path, False, groupaverage)
    timing_df = loss_df.groupby(metadata + ['epoch_num']).time.max().reset_index()
    grouped_loss = loss_df.groupby(metadata + ['epoch_num', 'time']).loss.mean().reset_index()
    grouped_loss = grouped_loss.groupby(metadata).last().reset_index()
    final_model_history = model_history.groupby(metadata + ['parameter']).last().reset_index().rename(columns={'parameter': 'model_parameter'})
    models = pd.merge(models, final_model_history[metadata + ['model_parameter', 'hessian']])
    models = models.fillna(0)
    diff_df = loss_df.groupby(metadata + ['epoch_num'])[['loss', 'time']].mean().reset_index()
    diff_df['loss_diff'] = diff_df.groupby(metadata)['loss'].diff()
    diff_df['time_diff'] = diff_df.groupby(metadata)['time'].diff()
    model_history['value_diff'] = model_history.groupby(metadata + ['parameter'])['value'].diff()
    models.to_csv(outputs[0], index=False)
    grouped_loss.to_csv(outputs[1], index=False)
    timing_df.to_csv(outputs[2], index=False)
    diff_df.to_csv(outputs[3], index=False)
    model_history.to_csv(outputs[4], index=False)
    if cv_loss_files is not None:
        cv_loss = []
        for path in cv_loss_files:
            cv_loss.append(pd.read_csv(path))
        cv_loss = pd.concat(cv_loss)
        cv_loss.to_csv(outputs[-1], index=False)


def combine_summarized_results(base_template, outputs, cv_loss_flag=True):
    """Combine model dataframes (second-order)

    This function combined model dataframes that have already been
    combined (that is, are the outputs of gather_results). As such, we
    don't do anything more than load them all in and concatenate them
    (no groupby, adding new columns, or anything else).

    This was written to be called by snakemake rules, not from a python
    interpeter directly

    Parameters
    ----------
    base_template : list
        list of strs, each of which is a path template where we should
        find the results. Unlike gather_results's base_path, this
        shouldn't contain any '*' (nor any string formatting symbols),
        but should just be the path to a single _all_models.csv, with
        that removed (see snakemake rule summarize_gathered_resutls,
        params.base_template for an example). For each p in
        base_template, we'll load in: p+'_all_models.csv',
        p+'_all_loss.csv', p+'_all_timing.csv', and (if cv_loss_flag is
        True) p+'_all_cv_loss.csv'.
    outputs : list
        list of 3 or 4 strs, the paths to save out our combined models,
        grouped_loss, timing, and (if cv_loss_flag is True) cv_loss
        dataframes
    cv_loss_flag : bool, optional
        whether we load in and save out the cross-validated loss
        dataframes

    """
    models = []
    grouped_loss_df = []
    timing_df = []
    cv_loss = []
    for p in base_template:
        models.append(pd.read_csv(p+'_all_models.csv'))
        grouped_loss_df.append(pd.read_csv(p+'_all_loss.csv'))
        timing_df.append(pd.read_csv(p+'_all_timing.csv'))
        if cv_loss_flag:
            cv_loss.append(pd.read_csv(p+'_all_cv_loss.csv'))
    models = pd.concat(models, sort=False)
    grouped_loss_df = pd.concat(grouped_loss_df, sort=False)
    timing_df = pd.concat(timing_df, sort=False)
    models.to_csv(outputs[0], index=False)
    grouped_loss_df.to_csv(outputs[1], index=False)
    timing_df.to_csv(outputs[2], index=False)
    if cv_loss_flag:
        cv_loss = pd.concat(cv_loss, sort=False)
        cv_loss.to_csv(outputs[3], index=False)


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
