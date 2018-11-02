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


def load_single_model(save_path_stem, model_type=None):
    """load in the model, loss df, and model df found at the save_path_stem

    we also send the model to the appropriate device
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    first_df = pd.read_csv(save_path_stem + '_model_df.csv')
    loss_df = pd.read_csv(save_path_stem + '_loss.csv')
    if model_type is None:
        # then we try and infer it from the path name, which we can do assuming we used the
        # Snakefile to generate saved model.
        model_type = save_path_stem.split('_')[-1]
    if model_type == 'full':
        model = LogGaussianDonut(1, 2, .4)
    elif model_type == 'constant':
        model = ConstantLogGaussianDonut(1, 2, .4)
    elif model_type == 'scaling':
        model = ScalingLogGaussianDonut(1, 2, .4)
    else:
        raise Exception("Don't know how to handle model_type %s!" % model_type)
    model.load_state_dict(torch.load(save_path_stem + '_model.pt', map_location=device.type))
    model.eval()
    model.to(device)    
    return model, loss_df, first_df


def combine_models(base_path_template):
    """load in many models and combine into dataframes

    returns: model_df, loss_df, data_df

    base_path_template: path template where we should find the results.
    """
    models = []
    loss_df = []
    data_df = []
    for sim in ['constant', 'scaling', 'full']:
        for fit in ['constant', 'scaling', 'full']:
            m_files = glob.glob('/home/billbrod/Data/spatial_frequency_preferences/derivatives/tuning_2d_simulated/sim-{0}/absolute/sim-{0}_*_{1}_loss.csv'.format(sim, fit))
            for m_f in m_files:
                m, l, f = sfp.model.load_model(m_f.replace('_loss.csv', ''))
                models[("sim-%s"%sim, fit, f.learning_rate.unique()[0], f.batch_size.unique()[0])] = m
                data_df.append(f)
                # I think this is as it should be
                l['true_model'] = sim
                l['fit_model'] = fit
                # can remove this soon
                l['learning_rate'] = f.learning_rate.unique()[0]
                loss_df.append(l)
    loss_df = pd.concat(loss_df).reset_index(drop=True)
    data_df = pd.concat(data_df).reset_index(drop=True).drop('index', 1)
models_df_lr = []
for i, (k, m) in enumerate(models_lr.iteritems()):
    models_df_lr.append(pd.DataFrame(index=[i], data={'true_model': k[0], 'fit_model': k[1], 'learning_rate': k[2], 'batch_size': k[3], 'amplitude': m.amplitude.detach().numpy(), 'mode': m.mode.detach().numpy(), 
                                                      'sigma': m.sigma.detach().numpy(), 'sf_ecc_slope': m.sf_ecc_slope.detach().numpy(), 'sf_ecc_intercept': m.sf_ecc_intercept.detach().numpy()}))
models_df_lr = pd.concat(models_df_lr)
models_df_lr=models_df_lr.melt(['true_model', 'fit_model', 'learning_rate', 'batch_size'], var_name='model_parameter').reset_index(drop=True)    
