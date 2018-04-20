#!/usr/bin/python
"""consolidate tuning curve results across subjects / sessions
"""
import argparse
import os
import re
import numpy as np
import pandas as pd

PATH_TEMPLATE = ("tuning_curves/(?P<mat_type>[a-z0-9_]+)/(?P<atlas_type>[a-z_]+)/"
                 "(?P<subject>sub-[a-z0-9]+)/(?P<session>ses-[a-z0-9]+)/(?P=subject)_(?P=session)"
                 "_(?P<task>task-[a-z0-9]+).*(?P<df_mode>summary|full)\.csv")


def main(root_dir, save_path=None, **kwargs):
    """finds all tuning curve dataframes under root_dir and consolidates them

    kwargs can be any of mat_type, atlas_type, subject, session, task, or df_mode, and will limit
    which tuning curve dataframes we consolidate to only those whose values for that field match
    the specified value(s)
    """
    limit_kwargs = {}
    for k, v in kwargs.iteritems():
        if isinstance(v, basestring) or not hasattr(v, '__iter__'):
            limit_kwargs[k] = [v]
        else:
            limit_kwargs[k] = v
    root_dir = os.path.abspath(root_dir)
    walker = os.walk(root_dir)
    csv_paths = [os.path.join(root, f) for root, _, files in walker for f in files if 'csv' in f]
    df = []
    duplicate_check_cols = ['varea', 'eccen', 'stimulus_superclass', 'frequency_type']
    for p in csv_paths:
        info_dict = re.search(PATH_TEMPLATE, p).groupdict()
        # the [True] here ensures that if limit_kwargs is empty, there will be one True and thus
        # np.all will return True.
        in_limit_kwargs = [True] + [info_dict[k] in v for k, v in limit_kwargs.iteritems()]
        if not np.all(in_limit_kwargs):
            continue
        tmp_df = pd.read_csv(p)
        if 'bootstrap_num' in tmp_df.columns:
            tmp_df = tmp_df.drop_duplicates(duplicate_check_cols + ['bootstrap_num'])
        else:
            tmp_df = tmp_df.drop_duplicates(duplicate_check_cols)
        tmp_df = tmp_df.assign(**info_dict)
        tmp_df['eccen'] = tmp_df.eccen.apply(lambda x: np.mean([float(i) for i in x.split('-')]))
        df.append(tmp_df)
    df = pd.concat(df).reset_index(drop=True)
    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df


if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        description=("Load in the tuning curve dataframes found underneath the root directory and"
                     " save their important parameters in the specified save_path. Currently, "
                     "we only do this for summary.csv dataframes"),
        formatter_class=CustomFormatter)
    parser.add_argument("root_dir",
                        help="Root of directory tree that we'll find everything underneath")
    parser.add_argument("save_path", help="Path to save resulting consolidated dataframe at")
    args = vars(parser.parse_args())
    main(df_mode='summary', **args)
