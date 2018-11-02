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

    base_path_template: path template where we should find the results. should contain no string
    formatting symbols (e.g., "{0}" or "%s") but should contain at least one '*' because we will
    use glob to find them (and therefore should point to an actual file when passed to glob, one
    of: the loss df, model df, or model paramters).
    """
    models = []
    loss_df = []
    data_df = []
    params = []
    for p in glob.glob(base_path_template):
        m, l, f = load_single_model(p.replace('_loss.csv', '').replace('_model.pt', '').replace('_model_df.csv', ''))
        data_df.append(f)
        loss_df.append(l)
        tmp = l.head(1)
        tmp = tmp.drop(['epoch_num', 'batch_num', 'loss'])
        for name, val in m.named_parameters():
            tmp[name] = val.cpu().detach().numpy()
            if name not in params:
                params.append(name)
        tmp['model'] = m
        models.append(tmp)
    loss_df = pd.concat(loss_df).reset_index(drop=True)
    data_df = pd.concat(data_df).reset_index(drop=True).drop('index', 1)
    models = pd.concat(models)
    models=models.melt([i for i in models.columns if i not in params], params,
                       var_name='model_parameter').reset_index(drop=True)    
    return models, loss_df, data_df
