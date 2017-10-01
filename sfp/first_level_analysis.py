#!/usr/bin/python
"""functions to run first-level MRI analyses
"""
import pandas as pd
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import warnings

# TODO:
# - if main block for command line


def _discover_class_size(df):
    class_size = 0
    w_a = df['w_a'].values.copy()
    w_r = df['w_r'].values.copy()
    break_out = False
    while not break_out:
        class_size += 1
        # we replace the NaNs with zeros for this calculation -- we want them be different than all
        # the other classes (and technically, the blank stimuli do have 0s in both w_r and w_a)
        nan_replace = 0
        w_r[np.isnan(w_r)] = nan_replace
        w_a[np.isnan(w_a)] = nan_replace
        tmp = (np.abs(w_r[:-class_size:class_size] - w_r[class_size::class_size]) +
               np.abs(w_a[:-class_size:class_size] - w_a[class_size::class_size]))
        class_changes = np.nonzero(tmp)[0]
        indices = np.array(range(len(tmp)))
        if len(class_changes) == len(indices):
            break_out = np.equal(class_changes, indices).all()
    return class_size


def _find_times(value, name):
    """helper function to find lengths of time
    """
    lengths = (value[1:] - value[:-1]).astype(float)
    real_length = np.round(lengths[0])
    if (np.abs(lengths - real_length)/real_length > .005).any():
        perc_diff = (np.abs(lengths - real_length) / real_length).max()
        warnings.warn("One of your %s lenghts is greater than .5% different than the assumed "
                      "length of %s! It differs by %.02f percent" % (name, real_length, perc_diff))
    return real_length


def find_lengths(design_df):
    """this uses design_df to find the length of the stimuli and of the TRs in seconds
    """
    stim_length = _find_times(design_df['Onset time (sec)'].values.copy(), "stimuli")
    TR_length = _find_times(design_df['Onset time (TR)'].values.copy(), "TR")
    TR_length = stim_length / TR_length
    return stim_length, TR_length


def design_matrix(behavioral_results, unshuffled_stim_description, run_num):
    """create and return the design matrix for a run

    behavioral_results: h5py File (not the path) containing behavioral results

    unshuffled_stim_description: dataframe containing info on the stimuli

    run_num: int, which run (as used in behavioral_results) to examine
    """
    df = unshuffled_stim_description[['w_r', 'w_a', 'index']].set_index('index')
    df = df.reindex(behavioral_results['run_%02d_shuffled_indices' % run_num].value)
    class_size = _discover_class_size(df)
    df['class_idx'] = df.index / class_size
    timing = behavioral_results['run_%02d_timing_data' % run_num].value
    # Because we want to skip the first one and drop the last nblanks * 2 (since for each stimuli
    # we have two entries: one for on, one for off). Finally, we only grab every other because we
    # only want the on timing
    timing = timing[1:-behavioral_results['run_%02d_nblanks' % run_num].value*2:2]
    # Now we get rid of the first TR
    initial_TR_time = float(behavioral_results['run_%02d_button_presses' % run_num].value[0][1])
    timing = [float(i[2]) - initial_TR_time for i in timing]
    # and add to our dataframe
    df['Onset time (sec)'] = timing
    # we only look at the class transitions
    design_df = df[::class_size]
    # 5 indicates a backtick from the scanner
    TR_times = np.array([float(i[1]) for i in behavioral_results['run_%02d_button_presses' % run_num].value if '5' == i[0]])
    TR_times -= TR_times[0]
    stim_times = design_df['Onset time (sec)'].values
    stim_times = np.expand_dims(stim_times, 1)
    stim_times = np.repeat(stim_times, len(TR_times), 1)
    time_from_TR = np.round(stim_times - TR_times)
    design_df['Onset time (TR)'] = np.where(time_from_TR == 0)[1]
    # we want to find the length of the stimuli and TRs, in seconds
    stim_length, TR_length = find_lengths(design_df)
    # Our blanks show up as having nan values, and we don't want to model them in our GLM, so we
    # drop them
    design_df = design_df.dropna()
    # because the values are 0-indexed
    design_matrix = np.zeros((len(TR_times), design_df.class_idx.max()+1))
    for i, row in design_df.iterrows():
        row = row.astype(int)
        design_matrix[row['Onset time (TR)'], row['class_idx']] = 1
    return design_matrix, stim_length, TR_length


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
    ax = plt.imshow(design_matrix, 'gray', aspect='auto')
    ax.axes.grid(False)
    plt.xlabel("Stimulus class")
    plt.ylabel("TR")
    plt.title(title)
    if save_path is not None:
        ax.figure.savefig(save_path)


def create_all_design_matrices(behavioral_results_path, unshuffled_stim_descriptions_path,
                               save_path="data/MRI_first_level/run_%02d_design_matrix.mat",
                               mat_type="stim_class"):
    """create and save design matrices for all runs

    we do this for all non-empty runs in the h5py File found at behavioral_results_path

    save_path should contain some string formatting symbol (e.g., %s, %02d) that can indicate the
    run number and should end in .mat

    mat_type: {"stim_class", "all_visual"}. What design matrix to make. stim_class has each
    stimulus class as a separate regressor and is our actual design matrix for the
    experiment. all_visual has every stimulus class combined into regressor (so that that
    regressors represents whenever anything is on the screen) and is used to check that things are
    working as expected, since every voxel in the visual cortex should then show increased
    activation relative to baseline.
    """
    results = h5py.File(behavioral_results_path)
    df = pd.read_csv(unshuffled_stim_descriptions_path)
    assert mat_type in ["stim_class", "all_visual"], "Don't know how to handle mat_type %s!" % mat_type
    run_num = 0
    stim_lengths = []
    TR_lengths = []
    if not os.path.exists(os.path.split(save_path)[0]):
        os.makedirs(os.path.split(save_path)[0])
    while "run_%02d_button_presses" % run_num in results.keys():
        n_TRs = sum(['5' in i[0] for i in results['run_%02d_button_presses' % run_num].value])
        if n_TRs > 0:
            design_mat, stim, TR = design_matrix(results, df, run_num)
            stim_lengths.append(stim)
            TR_lengths.append(TR)
            check_design_matrix(design_mat, run_num)
            if mat_type == "all_visual":
                design_mat = design_mat.sum(1).reshape((design_mat.shape[0], 1))
            plot_design_matrix(design_mat, "Design matrix for run %02d" % run_num,
                               save_path.replace('.mat', '.png') % run_num)
            sio.savemat(save_path % run_num, {"design_matrix_run_%02d" % run_num: design_mat})
        run_num += 1
    assert ((np.array(stim_lengths) - stim_lengths[0]) == 0).all(), "You have different stim lengths!"
    assert ((np.array(TR_lengths) - TR_lengths[0]) == 0).all(), "You have different TR lengths!"
    sio.savemat(os.path.join(os.path.split(save_path)[0], "run_details.mat"),
                {"stim_length": stim_lengths[0], 'TR_length': TR_lengths[0]})
