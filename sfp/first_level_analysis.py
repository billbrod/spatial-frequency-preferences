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
import pyPyrTools as ppt
import nibabel as nib
import itertools

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
        perc_diff = (np.abs(lengths - real_length) / real_length).max() * 100
        warnings.warn("One of your %s lenghts is greater than .5 percent different than the assumed"
                      " length of %s! It differs by %.02f percent" % (name, real_length, perc_diff))
    return real_length


def find_lengths(design_df):
    """this uses design_df to find the length of the stimuli and of the TRs in seconds
    """
    stim_length = _find_times(design_df['Onset time (sec)'].values.copy(), "stimuli")
    TR_length = _find_times(design_df['Onset time (TR)'].values.copy(), "TR")
    TR_length = stim_length / TR_length
    return stim_length, TR_length


def create_design_df(behavioral_results, unshuffled_stim_description, run_num, drop_blanks=True):
    """create and return the design df for a run, which describes stimulus classes

    behavioral_results: h5py File (not the path) containing behavioral results

    unshuffled_stim_description: dataframe containing info on the stimuli

    run_num: int, which run (as used in behavioral_results) to examine

    returns the design dataframe, lengths of stimuli and TRs (in seconds)
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
    # need to do this before dropping the blanks, or the times get messed up
    stim, TR = find_lengths(design_df)
    if drop_blanks:
        # Our blanks show up as having nan values, and we don't want to model them in our GLM, so we
        # drop them
        design_df = design_df.dropna()
    return design_df, stim, TR


def create_design_matrix(design_df, behavioral_results, run_num):
    """create and return the design matrix for a run

    design_df: pandas DataFrame describing the design of the experiment and the stimulus classes
    (created by create_design_df function)

    behavioral_results: h5py File (not the path) containing behavioral results

    run_num: int, which run (as used in behavioral_results) to examine
    """
    # 5 indicates a backtick from the scanner
    TR_times = np.array([float(i[1]) for i in behavioral_results['run_%02d_button_presses' % run_num].value if '5' == i[0]])
    TR_times -= TR_times[0]
    # because the values are 0-indexed
    design_matrix = np.zeros((len(TR_times), design_df.class_idx.max()+1))
    for i, row in design_df.iterrows():
        row = row.astype(int)
        design_matrix[row['Onset time (TR)'], row['class_idx']] = 1
    return design_matrix


def find_ecc_range_in_degrees(stim, rad_in_degrees):
    """find the min and max eccentricity of the stimulus, in degrees

    all of our stimuli have a central aperture where nothing is presented and an outside limit,
    beyond which nothing is presented. In order to make sure we're not looking at voxels whose pRFs
    lie outside the stimulus, we want to know the extent of the stimulus annulus, in degrees

    this assumes the fixation is in the center of the stimulus, will have to re-think things if
    it's not. also assumes that the "middle / zero value", which corresponds to no stimulus, is 127

    rad_in_degrees: int or float, the radius of the viewing screen, in degrees.

    returns min, max
    """
    if stim.ndim == 3:
        stim = stim[0, :, :]
    R = ppt.mkR(stim.shape)
    factor = R.max() / float(rad_in_degrees)
    # 127 is the middle value.
    x, y = np.where(stim != 127)
    return R[x, y].min() / factor, R[x, y].max() / factor


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
                               mat_type="stim_class", permuted=False):
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

    permuted: boolean, default False. Whether to permute the run labels or not. The reason to do
    this is to double-check your results: your R2 values should be much lower when permuted,
    because you're basically breaking your hypothesized connection between the GLM model and brain
    activity.
    """
    results = h5py.File(behavioral_results_path)
    df = pd.read_csv(unshuffled_stim_descriptions_path)
    assert mat_type in ["stim_class", "all_visual"], "Don't know how to handle mat_type %s!" % mat_type
    run_nums = []
    run_num = 0
    stim_lengths = []
    TR_lengths = []
    if not os.path.exists(os.path.split(save_path)[0]):
        os.makedirs(os.path.split(save_path)[0])
    while "run_%02d_button_presses" % run_num in results.keys():
        n_TRs = sum(['5' in i[0] for i in results['run_%02d_button_presses' % run_num].value])
        if n_TRs > 0:
            run_nums.append(run_num)
        run_num += 1
    run_details_save_path = os.path.join(os.path.split(save_path)[0], "run_details_%s.mat" % mat_type)
    save_labels = np.array(run_nums).copy()
    if permuted is True:
        if 'permuted' not in save_path:
            save_path = save_path.replace('.mat', '_permuted.mat')
        # this shuffles in place, ensuring that every value is moved:
        while len(np.where(save_labels == run_nums)[0]) != 0:
            np.random.shuffle(save_labels)
        run_details_save_path = run_details_save_path.replace('.mat', '_permuted.mat')
    for run_num, save_num in zip(run_nums, save_labels):
        design_df, stim, TR = create_design_df(results, df, run_num)
        design_mat = create_design_matrix(design_df, results, run_num)
        stim_lengths.append(stim)
        TR_lengths.append(TR)
        check_design_matrix(design_mat, run_num)
        if mat_type == "all_visual":
            design_mat = design_mat.sum(1).reshape((design_mat.shape[0], 1))
        plot_design_matrix(design_mat, "Design matrix for run %02d" % save_num,
                           save_path.replace('.mat', '.png') % save_num)
        sio.savemat(save_path % save_num, {"design_matrix_run_%02d" % save_num: design_mat})
    assert ((np.array(stim_lengths) - stim_lengths[0]) == 0).all(), "You have different stim lengths!"
    assert ((np.array(TR_lengths) - TR_lengths[0]) == 0).all(), "You have different TR lengths!"
    sio.savemat(run_details_save_path,
                {"stim_length": stim_lengths[0], 'TR_length': TR_lengths[0],
                 'save_labels': save_labels, 'run_numbers': run_nums})


def _load_mgz(path):
    """load and reshape mgz so it's either 1 or 2d, instead of 3 or 4d
    """
    # see http://pandas.pydata.org/pandas-docs/version/0.19.1/gotchas.html#byte-ordering-issues
    tmp = nib.load(path).get_data().byteswap().newbyteorder()
    if tmp.ndim == 3:
        return tmp.reshape(max(tmp.shape))
    elif tmp.ndim == 4:
        return tmp.reshape(max(tmp.shape), sorted(tmp.shape)[-2])


def _unfold_4d_mgz(mgz, value_name, hemi):
    tmp = pd.DataFrame(mgz)
    tmp = pd.melt(tmp.reset_index(), id_vars='index')
    tmp['hemi'] = hemi
    tmp = tmp.rename(columns={'index': 'voxel', 'variable': 'stimulus_class', 'value': value_name})
    return tmp.set_index(['voxel', 'stimulus_class'])


def create_GLM_result_df(design_df, benson_template_path, results_template_path,
                         results_names=['modelse', 'modelmd', 'R2']):
    """this loads in the realigned mgz files and creates a dataframe of their values

    This only returns those voxels that lie within visual areas outlined by the Benson14 varea mgz

    this should be run after GLMdenoise and after realign.py. The mgz files you give the path to
    should be surfaces, not volumes. this may take up to 2 minutes to run.

    design_df: output of create_design_df

    benson_template_path: template path to the Benson14 mgz files, containing two string formatting
    symbols (%s; one for hemisphere, one for variable [angle, varea, eccen]),
    e.g. /mnt/Acadia/Freesurfer_subjects/wl_subj042/surf/%s.benson14_%s.mgz

    results_template_path: template path to the results mgz files (outputs of realign.py),
    containing two string formatting symbols (%s; one for hemisphere, one for results_names)
    """
    ##### MAKE OWN FUNCTION
    mgzs = {}
    varea = {'lh': _load_mgz(benson_template_path % ('lh', 'varea'))}
    varea['rh'] = _load_mgz(benson_template_path % ('rh', 'varea'))
    for hemi, var in itertools.product(['lh', 'rh'], ['varea', 'angle', 'eccen']):
        tmp = _load_mgz(benson_template_path % (hemi, var))
        mgzs['%s-%s' % (var, hemi)] = tmp[varea['%s' % hemi] != 0]

    for hemi, res in itertools.product(['lh', 'rh'], results_names):
        tmp = _load_mgz(results_template_path % (res, hemi))
        if tmp.ndim == 1:
            mgzs['%s-%s' % (res, hemi)] = tmp[varea['%s' % hemi] != 0]
        # some will be 2d, not 1d (since they start with 4 dimensions)
        elif tmp.ndim == 2:
            mgzs['%s-%s' % (res, hemi)] = tmp[varea['%s' % hemi] != 0, :]

    ##### MAKE OWN FUNCTION
    df = {}
    for hemi in ['lh', 'rh']:
        df[hemi] = _unfold_4d_mgz(mgzs['modelse-%s' % hemi], 'model_std_error', hemi)

        tmp = _unfold_4d_mgz(mgzs['modelmd-%s' % hemi], 'model_ampl_est', hemi)
        df[hemi]['model_ampl_est'] = tmp['model_ampl_est']

        df[hemi] = df[hemi].reset_index('stimulus_class')
        for k in ['varea', 'angle', 'eccen', 'R2']:
            tmp = pd.DataFrame(mgzs['%s-%s' % (k, hemi)])
            tmp.index.rename('voxel', True)
            df[hemi][k] = tmp[0]

        df[hemi] = df[hemi].reset_index()

    df['rh'].voxel = df['rh'].voxel + df['lh'].voxel.max()+1
    df = pd.concat(df).reset_index(0, drop=True)

    # Add the stimulus frequency information
    df = df.set_index('stimulus_class')
    df = df.join(design_df)
    df = df.reset_index().rename(columns={'index': 'stimulus_class'})

    ##### MAKE OWN FUNCTION

    # stimuli belong to five super classes, or paths through the frequency space: w_r=0; w_a=0;
    # w_r=w_a; w_r=-w_a; and sqrt(w_r^2 + w_a^)=32. We want to be able to look at them separately,
    # so we label them (this is inefficient but works). We also want to get some other identifying
    # values. We do this all at once because the major time cost comes from applying this to all
    # rows, not the computations themselves
    def freq_identifier(x):
        if x.w_r == 0 and x.w_a != 0:
            sc = 'radial'
        elif x.w_r != 0 and x.w_a == 0:
            sc = 'circular'
        elif x.w_r == x.w_a:
            sc = 'forward spiral'
        elif x.w_r == -x.w_a:
            sc = 'reverse spiral'
        else:
            sc = 'mixtures'
        try:
            ang = np.arctan(x.w_r / x.w_a)
        except ZeroDivisionError:
            ang = np.arctanh(x.w_a / x.w_r)
        return sc, ang, np.sqrt(x.w_r**2 + x.w_a**2)

    properties_list = df[['w_r', 'w_a']].apply(freq_identifier, 1)
    sc = pd.Series([i[0] for i in properties_list.values], properties_list.index)
    ang = pd.Series([i[1] for i in properties_list.values], properties_list.index)
    dist = pd.Series([i[2] for i in properties_list.values], properties_list.index)

    df['stimulus_superclass'] = sc
    df['freq_space_angle'] = ang
    df['freq_space_distance'] = dist

    return df


def _find_closest_to(a, bs):
    idx = np.argmin(np.abs(np.array(bs) - a))
    return bs[idx]


def round_freq_space_distance(df, core_distances=[6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181]):
    df['rounded_freq_space_distance'] = df.freq_space_distance.apply(_find_closest_to,
                                                                     bs=core_distances)
    return df

# Make wrapper function that does above, loading in design_df and maybe grabbing it for different
# results? and then combining them.