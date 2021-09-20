#!/usr/bin/env python3

import argparse
import subprocess
import os.path as op
import yaml


def main(target_dataset):
    """Download dataset from OpenNeuro or OSF.

    Parameters
    ----------
    target_dataset : {'preprocessed', 'partially', 'fully'}
        Which dataset to download. See project README for more info.

    """
    with open(op.join(op.dirname(op.realpath(__file__)), 'config.yml')) as f:
        config = yaml.safe_load(f)
    deriv_folder = op.join(config['DATA_DIR'], 'derivatives')
    print(f"Using {config['DATA_DIR']} as data root directory.")
    if target_dataset == 'fully':
        print("Downloading fully-processed data.")
        subprocess.call(["curl", "-O", "-J", "-L", "https://osf.io/djak4/download"])
        subprocess.call(["tar", "xf", "sfp_fully_processed_data.tar.gz"])
        subprocess.call(["rsync", "-avPLuz", "derivatives/", f"{deriv_folder}/"])
        subprocess.call(["rm", "-r", "derivatives/"])
        subprocess.call(["rm", "sfp_fully_processed_data.tar.gz"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Download the spatial frequency preferences dataset, to reproduce the results.")
    )
    parser.add_argument("target_dataset", choices=['preprocessed', 'partially', 'fully'],
                        help="Which dataset to download, see project README for details.")
    args = vars(parser.parse_args())
    main(**args)
