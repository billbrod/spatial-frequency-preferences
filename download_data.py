#!/usr/bin/env python3

import argparse
import subprocess
import os
import os.path as op
import json
from glob import glob


def main(target_dataset, preprocessed_version='1.0.0'):
    """Download dataset from OpenNeuro or OSF.

    Parameters
    ----------
    target_dataset : {'preprocessed', 'partially-processed',
                      'fully-processed', 'supplemental'}
        Which dataset to download. See project README for more info.
    preprocessed_version : str, optional
        Which version of the preprocessed data to download. See
        https://openneuro.org/datasets/ds003812 for possible choices.

    """
    with open(op.join(op.dirname(op.realpath(__file__)), 'config.json')) as f:
        config = json.load(f)
    if op.split(config['DATA_DIR'])[-1].lower() != op.split(config['DATA_DIR'])[-1]:
        raise Exception(f"Name of your DATA_DIR must be all lowercase! But got {config['DATA_DIR']}")
    deriv_folder = op.join(config['DATA_DIR'], 'derivatives')
    stim_folder = op.join(config['DATA_DIR'], 'stimuli')
    os.makedirs(deriv_folder, exist_ok=True)
    os.makedirs(stim_folder, exist_ok=True)
    print(f"Using {config['DATA_DIR']} as data root directory.")
    targets = ['preprocessed', 'fully-processed', 'supplemental', 'partially-processed']
    check_dirs = ['preprocessed', 'tuning_2d_model', 'freesurfer', 'GLMdenoise']
    yesno = 'y'
    for tar, check, size in zip(targets, check_dirs, ['41GB', '500MB', '5GB', '60GB']):
        if target_dataset == tar:
            if op.exists(op.join(deriv_folder, check)):
                yesno = input("Previous data found, do you wish to download the data anyway? [y/n] ")
                while yesno not in ['y', 'n']:
                    print("Please enter y or n")
                    yesno = input("Previous data found, do you wish to download the data anyway? [y/n] ")
            yesno = input(f"{tar} dataset will be approximately {size}, do you wish to download it? [y/n] ")
            while yesno not in ['y', 'n']:
                print("Please enter y or n")
                yesno = input(f"{tar} dataset will be approximately {size}, do you wish to download it? [y/n] ")
    if yesno == 'n':
        print("Exiting...")
        exit(0)
    if target_dataset == 'preprocessed':
        print("Downloading preprocessed data.")
        try:
            subprocess.call(['openneuro', 'download', '--snapshot', preprocessed_version,
                             'ds003812', config["DATA_DIR"]])
            # there can be an issue with the timestamps when downloading using
            # openneuro that confuses snakemake, so we make sure it knows these
            # don't need to be re-generated
            all_preproc_files = glob(op.join(config['DATA_DIR'], 'derivatives',
                                             'preprocessed', '*', 'ses-04', '*nii.gz'))
            subprocess.call(['snakemake', '-j', '1', '--touch'] + all_preproc_files)
        except FileNotFoundError:
            raise Exception("openneuro command-line interface is not installed on your "
                            "path, please install it first!")
    elif target_dataset == 'partially-processed':
        print("Downloading partially-processed data, each subject is approximately 5GB")
        GLMdenoise_dir = op.join(deriv_folder, 'GLMdenoise', 'stim_class', 'bayesian_posterior')
        for i, sub in enumerate([1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121]):
            print(f"Downloading subject sub-wlsubj{sub:03d}")
            subprocess.call(['curl', '-k', '-L',
                             f'https://archive.nyu.edu/rest/bitstreams/{128495+i}/retrieve',
                             '-o', f'sub-wlsubj{sub:03d}_ses-04_task-sfprescaled_results.mat'])
            subprocess.call(['mkdir', '-p', op.join(GLMdenoise_dir, f'sub-wlsubj{sub:03d}', 'ses-04')])
            subprocess.call(['mv', f'sub-wlsubj{sub:03d}_ses-04_task-sfprescaled_results.mat',
                             op.join(GLMdenoise_dir, f'sub-wlsubj{sub:03d}', 'ses-04')])
    elif target_dataset == 'fully-processed':
        print("Downloading fully-processed data.")
        subprocess.call(["curl", "-O", "-J", "-L", "https://osf.io/djak4/download"])
        subprocess.call(["tar", "xf", "sfp_fully_processed_data.tar.gz"])
        subprocess.call(["rsync", "-avPLuz", "derivatives/", f"{deriv_folder}/"])
        subprocess.call(["rsync", "-avPLuz", "stimuli/", f"{stim_folder}/"])
        subprocess.call(["rm", "-r", "derivatives/"])
        subprocess.call(["rm", "-r", "stimuli/"])
        subprocess.call(["rm", "sfp_fully_processed_data.tar.gz"])
    elif target_dataset == 'supplemental':
        print("Downloading data required for supplemental figures.")
        subprocess.call(["curl", "-O", "-J", "-L", "https://osf.io/d94ue/download"])
        subprocess.call(["tar", "xf", "sfp_supplemental_data.tar.gz"])
        subprocess.call(["rsync", "-avPLuz", "derivatives/", f"{deriv_folder}/"])
        subprocess.call(["rm", "-r", "derivatives/"])
        subprocess.call(["rm", "sfp_supplemental_data.tar.gz"])
    subprocess.call(['chmod', '-R', '777', config['DATA_DIR']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Download the spatial frequency preferences dataset, to reproduce the results.")
    )
    parser.add_argument("target_dataset", choices=['preprocessed',
                                                   'partially-processed',
                                                   'fully-processed',
                                                   'supplemental'],
                        help="Which dataset to download, see project README for details.")
    parser.add_argument("--preprocessed_version", default='1.0.0',
                        help="Which version fo the preprocessed dataset from openneuro to download")
    args = vars(parser.parse_args())
    main(**args)
