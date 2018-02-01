#!/usr/bin/python
"""functions to create the design matrices used in our first-level MRI analysis.
"""
import matplotlib as mpl
# we do this because sometimes we run this without an X-server, and this backend doesn't need one
mpl.use('svg')
import numpy as np
import argparse
import warnings
import nibabel as nib
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from bids.grabbids import BIDSLayout
from collections import Counter


def _discover_class_size(values):
    class_size = 0
    break_out = False
    while not break_out:
        class_size += 1
        tmp = np.abs(values[:-class_size:class_size] - values[class_size::class_size])
        class_changes = np.nonzero(tmp)[0]
        indices = np.array(range(len(tmp)))
        if len(class_changes) == len(indices):
            break_out = np.equal(class_changes, indices).all()
    return class_size


def _find_stim_class_length(value, class_size):
    """helper function to find the length of one stimulus class / trial type, in seconds
    """
    lengths = (value[1:] - value[:-1]).astype(float)
    counts = Counter(np.round(lengths, 1))
    real_length = counts.most_common()[0][0]
    counts.pop(real_length)
    counts.pop(real_length + class_size * real_length)
    for i in counts.keys():
        if (np.abs(i - real_length)/real_length > .005):
            perc_diff = (np.abs(lengths - real_length) / real_length) * 100
            warnings.warn("One of your stimuli lengths is greater than .5 percent different than the "
                          "assumed length of %s! It differs by %.02f percent" %
                          (real_length, perc_diff))
    return real_length * class_size


def create_design_matrix(design_df, n_TRs):
    """create and return the design matrix for a run

    design_df: pandas DataFrame describing the design of the experiment and the stimulus classes
    (created by create_design_df function)
    """
    # because the values are 0-indexed
    design_matrix = np.zeros((n_TRs, design_df.trial_type.max()+1))
    for i, row in design_df.iterrows():
        row = row[['Onset time (TR)', 'trial_type']].astype(int)
        design_matrix[row['Onset time (TR)'], row['trial_type']] = 1
    return design_matrix


def check_design_matrix(design_matrix, run_num=None):
    """quick soundness test to double-check design matrices

    this just checks to make sure that each event happens exactly once and that each TR has 0 or 1
    events.
    """
    if not (design_matrix.sum(0) == 1).all():
        raise Exception("There's a problem with the design matrix%s, at least one event doesn't"
                        " show up once!" % {None: ''}.get(run_num, " for run %s" % run_num))
    if not ((design_matrix.sum(1) == 0) + (design_matrix.sum(1) == 1)).all():
        raise Exception("There's a problem with the design matrix%s, at least one TR doesn't have"
                        " 0 or 1 events!" % {None: ''}.get(run_num, " for run %s" % run_num))


def plot_design_matrix(design_matrix, title, save_path=None):
    """plot design matrix and, if save_path is set, save the resulting image
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(design_matrix, 'gray')
    ax.axes.grid(False)
    plt.xlabel("Stimulus class")
    plt.ylabel("TR")
    plt.title(title)
    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches='tight')


def create_all_design_matrices(input_path, mat_type="stim_class", permuted=False,
                               save_path="data/MRI_first_level/run-%s_design_matrix.tsv"):
    """create and save design matrices for all runs

    input_path should be a path to a BIDS directory containing one scanning session. we will then
    construct a design matrix for each events.tsv file. all runs must have the same TR for
    GLMdenoise, so we'll through an exception if that's not the case.

    save_path should contain some string formatting symbol (e.g., %s, %02d) that can indicate the
    run number and should end in .tsv

    mat_type: {"stim_class", "all_visual"}. What design matrix to make. stim_class has each
    stimulus class as a separate regressor and is our actual design matrix for the
    experiment. all_visual has every stimulus class combined into regressor (so that that
    regressors represents whenever anything is on the screen) and is used to check that things are
    working as expected, since every voxel in the visual cortex should then show increased
    activation relative to baseline.

    permuted: boolean, default False. Whether to permute the run labels or not. The reason to do
    this is to double-check your results: your R2 values should be much lower when permuted,
    because you're basically breaking your hypothesized connection between the GLM model and brain
    activity.
    """
    assert mat_type in ["stim_class", "all_visual"], "Don't know how to handle mat_type %s!" % mat_type
    layout = BIDSLayout(input_path)
    run_nums = layout.get_runs()
    stim_lengths = []
    TR_lengths = []
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    run_details_save_path = os.path.join(os.path.dirname(save_path), "run_details_%s.json" % mat_type)
    save_labels = np.array(run_nums).copy()
    if permuted is True:
        if 'permuted' not in save_path:
            save_path = save_path.replace('.tsv', '_permuted.tsv')
        # this shuffles in place, ensuring that every value is moved:
        while len(np.where(save_labels == run_nums)[0]) != 0:
            np.random.shuffle(save_labels)
        run_details_save_path = run_details_save_path.replace('.json', '_permuted.json')
    for run_num, save_num in zip(run_nums, save_labels):
        tsv_file = layout.get(type='events', run=run_num)
        if len(tsv_file) != 1:
            raise IOError("Need one tsv for run %s, but found %d!" % (run_num, len(tsv_file)))
        tsv_df = pd.read_csv(tsv_file[0].filename, sep='\t')
        class_size = _discover_class_size(tsv_df.trial_type.values)
        stim = _find_stim_class_length(tsv_df.onset.values, class_size)
        tsv_df = tsv_df[::class_size]
        nii_file = layout.get(type='bold', run=run_num)
        if len(nii_file) != 1:
            raise IOError("Need one nifti for run %s, but found %d!" % (run_num, len(nii_file)))
        nii = nib.load(nii_file[0].filename)
        n_TRs = nii.shape[3]
        TR = layout.get_metadata(nii_file[0].filename)['RepetitionTime']
        stim_times = tsv_df.onset.values
        stim_times = np.repeat(np.expand_dims(stim_times, 1), n_TRs, 1)
        TR_times = [TR * i for i in range(n_TRs)]
        time_from_TR = np.round(stim_times - TR_times)
        tsv_df['Onset time (TR)'] = np.where(time_from_TR == 0)[1]
        design_mat = create_design_matrix(tsv_df, n_TRs)
        stim_lengths.append(stim)
        TR_lengths.append(TR)
        check_design_matrix(design_mat, run_num)
        if mat_type == "all_visual":
            design_mat = design_mat.sum(1).reshape((design_mat.shape[0], 1))
        plot_design_matrix(design_mat, "Design matrix for run %s" % save_num,
                           save_path.replace('.tsv', '.png') % save_num)
        np.savetxt(save_path % save_num, design_mat, '%d', '\t')
    assert ((np.array(stim_lengths) - stim_lengths[0]) == 0).all(), "You have different stim lengths!"
    assert ((np.array(TR_lengths) - TR_lengths[0]) == 0).all(), "You have different TR lengths!"
    with open(run_details_save_path, 'w') as f:
        run_details = {"stim_length": stim_lengths[0], 'TR_length': TR_lengths[0],
                       'save_labels': list(save_labels), 'run_numbers': list(run_nums)}
        json.dump(run_details, f)


if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass
    parser = argparse.ArgumentParser(description=("Create and save design matrices for all non-em"
                                                  "pty runs found in the behavioral results file"),
                                     formatter_class=CustomFormatter)
    parser.add_argument("input_path",
                        help=("path to a BIDS directory containing one scanning session. we will "
                              "then construct a design matrix for each events.tsv file. all runs "
                              "must have the same TR for GLMdenoise, so we'll through an exception"
                              " if that's not the case."))
    parser.add_argument("--save_path",
                        default="data/MRI_first_level/run-%s_design_matrix.tsv",
                        help=("Template path that we should save the resulting design matrices in."
                              "Must contain at least one string formatting signal (to indicate run"
                              "number) and must end in .tsv."))
    parser.add_argument("--mat_type", default="stim_class",
                        help=("{'stim_class', 'all_visual'}. What design matrix to make. stim_class"
                              " has each stimulus class as a separate regressor and is our actual "
                              "design matrix for the experiment. all_visual has every stimulus "
                              "class combined into regressor (so that that regressors represents "
                              "whenever anything is on the screen) and is used to check that things"
                              " are working as expected, since every voxel in the visual cortex "
                              "should then show increased activation relative to baseline."))
    parser.add_argument("--permuted", '-p', action="store_true",
                        help=("Whether to permute the run labels or not. The reason to do this is"
                              " to double-check your results: your R2 values should be much lower "
                              "when permuted, because you're basically breaking your hypothesized"
                              " connection between the GLM model and brain activity."))
    args = vars(parser.parse_args())
    create_all_design_matrices(**args)
