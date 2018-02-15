#!/usr/bin/python
"""functions to run first-level MRI analyses
"""
import argparse
import pandas as pd
import numpy as np
import os
import warnings
import nibabel as nib
import itertools
import re
from matplotlib import pyplot as plt
import stimuli as sfp_stimuli
import pyPyrTools as ppt
import utils


def _load_mgz(path):
    """load and reshape mgz so it's either 1 or 2d, instead of 3 or 4d
    """
    # see http://pandas.pydata.org/pandas-docs/version/0.19.1/gotchas.html#byte-ordering-issues
    tmp = nib.load(path).get_data().byteswap().newbyteorder()
    # with my change to nibabel, all mgzs will have 4d, even if the data is basically 3d.
    if tmp.ndim == 3 or tmp.shape[-1] == 1:
        return tmp.reshape(max(tmp.shape))
    elif tmp.ndim == 4:
        return tmp.reshape(max(tmp.shape), sorted(tmp.shape)[-2])


def _arrange_mgzs_into_dict(benson_template_path, results_template_path, results_names, vareas,
                            eccen_range, benson_template_names={'varea': 'varea', 'angle': 'angle',
                                                                'eccen': 'eccen'}):
    """load in the mgzs, put in a dictionary, and return that dictionary

    vareas: list of ints. which visual areas (as defined in the Benson visual area template) to
    include. all others will be discarded.

    eccen_range: 2-tuple of ints or floats. What range of eccentricities to include (as specified
    in the Benson eccentricity template).

    benson_template_names: dictionary between the labels we use for the different Benson templates
    (varea, angle, eccen) and the ones that are actually found in the filename. Doing this because
    the names for the Benson templates are different depending on when they're run (and so differ
    for different subjects).
    """
    if sorted(benson_template_names.keys()) != ['angle', 'eccen', 'varea']:
        raise Exception("The keys of benson_template_names MUST be angle, eccen, and varea!")
    mgzs = {}

    varea_mask = {}
    eccen_mask = {}
    for hemi in ['lh', 'rh']:
        varea_mask[hemi] = _load_mgz(benson_template_path % (hemi, benson_template_names['varea']))
        varea_mask[hemi] = np.isin(varea_mask[hemi], vareas)
        eccen_mask[hemi] = _load_mgz(benson_template_path % (hemi, benson_template_names['eccen']))
        eccen_mask[hemi] = (eccen_mask[hemi] > eccen_range[0]) & (eccen_mask[hemi] < eccen_range[1])

    for hemi, var in itertools.product(['lh', 'rh'], ['varea', 'angle', 'eccen']):
        tmp = _load_mgz(benson_template_path % (hemi, benson_template_names[var]))
        mgzs['%s-%s' % (var, hemi)] = tmp[(varea_mask[hemi]) & (eccen_mask[hemi])]

    for hemi, res in itertools.product(['lh', 'rh'], results_names):
        tmp = _load_mgz(results_template_path % (hemi, res))
        res_name = os.path.split(res)[-1]
        if tmp.ndim == 1:
            mgzs['%s-%s' % (res_name, hemi)] = tmp[(varea_mask[hemi]) & (eccen_mask[hemi])]
        # some will be 2d, not 1d (since they start with 4 dimensions)
        elif tmp.ndim == 2:
            mgzs['%s-%s' % (res_name, hemi)] = tmp[(varea_mask[hemi]) & (eccen_mask[hemi]), :]

    return mgzs


def _bin_mgzs_dict(mgzs, results_names, eccen_range, vareas, hemi_bin=True):
    """bins by eccentricity and, optionally, hemisphere (if hemi_bin is true)

    vareas: list of ints. which visual areas (as defined in the Benson visual area template) to
    include. all others will be discarded.

    eccen_range: 2-tuple of ints or floats. What range of eccentricities to include (as specified
    in the Benson eccentricity template).
    """
    for hemi in ['lh', 'rh']:
        masks = []
        for area in vareas:
            for i in range(*eccen_range):
                masks.append((mgzs['eccen-%s' % hemi] > i) & (mgzs['eccen-%s' % hemi] < i+1) & (mgzs['varea-%s' % hemi] == area))
        for res in results_names + ['varea', 'angle', 'eccen']:
            res_name = os.path.split(res)[-1]
            tmp = mgzs['%s-%s' % (res_name, hemi)]
            mgzs['%s-%s' % (res_name, hemi)] = np.array([tmp[m].mean(0) for m in masks])
    if hemi_bin:
        mgzs_tmp = {}
        for res in results_names + ['varea', 'angle', 'eccen']:
            res_name = os.path.split(res)[-1]
            mgzs_tmp[res_name] = np.mean([mgzs['%s-lh' % res_name], mgzs['%s-rh' % res_name]], 0)
        mgzs = mgzs_tmp
    return mgzs


def _unfold_2d_mgz(mgz, value_name, variable_name, mgz_name, hemi=None):
    tmp = pd.DataFrame(mgz)
    tmp = pd.melt(tmp.reset_index(), id_vars='index')
    if hemi is not None:
        tmp['hemi'] = hemi
    tmp = tmp.rename(columns={'index': 'voxel', 'variable': variable_name, 'value': value_name})
    if 'models_class' in mgz_name:
        # then the value name contains which stimulus class this and the actual value_name is
        # amplitude_estimate
        class_idx = re.search('models_class_([0-9]+)', mgz_name).groups()
        assert len(class_idx) == 1, "models_class title %s should only contain one number, to identify stimulus class!" % value_name
        tmp['stimulus_class'] = int(class_idx[0])
    return tmp


def _add_freq_metainfo(stim_df):
    """this function takes the stim_df and adds some metainfo based on the stimulus frequency

    right now these are: stimulus_superclass (radial, circular, etc), freq_space_angle (the angle
    in our 2d frequency space) and freq_space_distance (distance from the origin in our 2d
    frequency space)
    """
    # stimuli belong to five super classes, or paths through the frequency space: w_r=0; w_a=0;
    # w_r=w_a; w_r=-w_a; and sqrt(w_r^2 + w_a^)=32. We want to be able to look at them separately,
    # so we label them (this is inefficient but works). We also want to get some other identifying
    # values. We do this all at once because the major time cost comes from applying this to all
    # rows, not the computations themselves
    def freq_identifier_logpolar(x):
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
        return sc, np.arctan2(x.w_a, x.w_r), np.sqrt(x.w_r**2 + x.w_a**2)

    def freq_identifier_constant(x):
        if x.w_x == 0 and x.w_y != 0:
            sc = 'horizontal'
        elif x.w_x != 0 and x.w_y == 0:
            sc = 'vertical'
        elif x.w_x == x.w_y:
            sc = 'forward diagonal'
        elif x.w_x == -x.w_y:
            sc = 'reverse diagonal'
        else:
            sc = 'off-diagonal'
        return sc, np.arctan2(x.w_y, x.w_x), np.sqrt(x.w_x**2 + x.w_y**2)

    try:
        properties_list = stim_df[['w_r', 'w_a']].apply(freq_identifier_logpolar, 1)
    except KeyError:
        properties_list = stim_df[['w_x', 'w_y']].apply(freq_identifier_constant, 1)
    sc = pd.Series([i[0] for i in properties_list.values], properties_list.index)
    ang = pd.Series([i[1] for i in properties_list.values], properties_list.index)
    dist = pd.Series([i[2] for i in properties_list.values], properties_list.index)

    stim_df['stimulus_superclass'] = sc
    stim_df['freq_space_angle'] = ang
    stim_df['freq_space_distance'] = dist
    return stim_df


def _setup_mgzs_for_df(mgzs, results_names, df_mode, hemi=None):
    df = None
    if hemi is None:
        mgz_key = '%s'
    else:
        mgz_key = '%s-{}'.format(hemi)
    for brain_name in results_names:
        if df_mode == 'summary':
            value_name = {'modelmd': 'amplitude_estimate_median',
                          'modelse': 'amplitude_estimate_std_error'}.get(brain_name)
            tmp = _unfold_2d_mgz(mgzs[mgz_key % brain_name], value_name,
                                 'stimulus_class', brain_name, hemi)
        elif df_mode == 'full':
            tmp = _unfold_2d_mgz(mgzs[mgz_key % brain_name], 'amplitude_estimate',
                                 'bootstrap_num', brain_name, hemi)
        if df is None:
            df = tmp
        else:
            if df_mode == 'summary':
                df = df.set_index(['voxel', 'stimulus_class'])
                tmp = tmp.set_index(['voxel', 'stimulus_class'])
                df[value_name] = tmp[value_name]
                df = df.reset_index()
            elif df_mode == 'full':
                df = pd.concat([df, tmp])

    df = df.set_index('voxel')
    for brain_name in ['varea', 'eccen', 'angle', 'R2']:
        tmp = pd.DataFrame(mgzs[mgz_key % brain_name])
        tmp.index.rename('voxel', True)
        df[brain_name] = tmp[0]

    df = df.reset_index()
    return df


def _put_mgzs_dict_into_df(mgzs, stim_df, results_names, df_mode, eccen_bin=True, hemi_bin=True):
    if not hemi_bin:
        df = {}
        for hemi in ['lh', 'rh']:
            df[hemi] = _setup_mgzs_for_df(mgzs, results_names, df_mode, hemi)

        # because python 0-indexes, the minimum voxel number is 0. thus if we were to just add the
        # max, the min in the right hemi would be the same as the max in the left hemi
        df['rh'].voxel = df['rh'].voxel + df['lh'].voxel.max()+1
        df = pd.concat(df).reset_index(0, drop=True)
    else:
        df = _setup_mgzs_for_df(mgzs, results_names, df_mode, None)

    # Add the stimulus frequency information
    stim_df = _add_freq_metainfo(stim_df)

    df = df.set_index('stimulus_class')
    df = df.join(stim_df)
    df = df.reset_index().rename(columns={'index': 'stimulus_class'})

    if eccen_bin:
        df['eccen'] = df['eccen'].apply(lambda x: '%i-%i' % (np.floor(x), np.ceil(x)))
    return df


def _find_closest_to(a, bs):
    idx = np.argmin(np.abs(np.array(bs) - a))
    return bs[idx]


def _round_freq_space_distance(df, core_distances=[6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181]):
    df['rounded_freq_space_distance'] = df.freq_space_distance.apply(_find_closest_to,
                                                                     bs=core_distances)
    return df


def find_ecc_range_in_pixels(stim, mid_val=128):
    """find the min and max eccentricity of the stimulus, in pixels

    all of our stimuli have a central aperture where nothing is presented and an outside limit,
    beyond which nothing is presented.

    this assumes the fixation is in the center of the stimulus, will have to re-think things if
    it's not. also assumes that the "middle / zero value", which corresponds to no stimulus, is 127

    returns min, max
    """
    if stim.ndim == 3:
        stim = stim[0, :, :]
    R = ppt.mkR(stim.shape)
    x, y = np.where(stim != mid_val)
    return R[x, y].min(), R[x, y].max()


def find_ecc_range_in_degrees(stim, stim_rad_deg, mid_val=128):
    """find the min and max eccentricity of the stimulus, in degrees

    all of our stimuli have a central aperture where nothing is presented and an outside limit,
    beyond which nothing is presented. In order to make sure we're not looking at voxels whose pRFs
    lie outside the stimulus, we want to know the extent of the stimulus annulus, in degrees

    this assumes the fixation is in the center of the stimulus, will have to re-think things if
    it's not. also assumes that the "middle / zero value", which corresponds to no stimulus, is 127

    stim_rad_deg: int or float, the radius of the stimulus, in degrees.

    returns min, max
    """
    if stim.ndim == 3:
        stim = stim[0, :, :]
    Rmin, Rmax = find_ecc_range_in_pixels(stim, mid_val)
    R = ppt.mkR(stim.shape)
    # if stim_rad_deg corresponds to the max vertical/horizontal extent, the actual max will be
    # np.sqrt(2*stim_rad_deg**2) (this corresponds to the far corner). this should be the radius of
    # the screen, because R starts from the center and goes to the edge
    factor = R.max() / np.sqrt(2*stim_rad_deg**2)
    return Rmin / factor, Rmax / factor


def calculate_stim_local_sf(stim, w_1, w_2, stim_type, stim_rad_deg=12, eccen_bin=True,
                            eccen_range=(1, 12), eccens=[], plot_flag=False, mid_val=128):
    """calculate the local spatial frequency for a specified stimulus and screen size

    NOTE: this assumes that the local spatial frequency does not depend on angle, only on
    eccentricity. this is true for the log polar stimuli and the constant grating stimuli created
    for this experiment.

    This works slightly differently if you are binning by eccentricity or not (i.e., depending on
    whether eccen_bin is True or False). If binning, then we take the annulus with inner edge i
    degrees from the origin and outer edge i+1 degrees (where i runs from eccen_range[0] to
    eccen_range[1]-1) and average together the local spatial frequency found there.

    If not binning, eccen_range is ignored, and eccens must be specified. We then look for the
    spatial frequency at the closest eccentricity value we have in the
    distance(-in-degrees)-from-origin map (made by rescaling the output of ppt.mkR()) for each
    value in eccens.

    stim: 2d array of floats. an example stimulus. used to determine where the stimuli are masked
    and to mask the calculated spatial frequency in the same way.

    w_1, w_2: ints or floats. the first and second components of the stimulus's spatial
    frequency. if stim_type is 'logarpolar' or 'pilot', this should be the radial and angular
    components (in that order!); if stim_type is 'constant', this should be the x and y components
    (in that order!)

    stim_type: {'logpolar', 'constant', 'pilot'}. which type of stimuli were used in the session
    we're analyzing. This matters because it changes the local spatial frequency and, since that is
    determined analytically and not directly from the stimuli, we have no way of telling otherwise.
    
    stim_rad_deg: float, the radius of the stimulus, in degrees of visual angle

    plot_flag: boolean, optional, default False. Whether to create a plot showing the local spatial
    frequency vs eccentricity for the specified stimulus

    mid_val: int. the value of mid-grey in the stimuli, should be 127 (for pilot stimuli) or 128
    (for actual stimuli)
    """
    if stim_type in ['logpolar', 'pilot']:
        mag = sfp_stimuli.create_sf_maps_cpd(stim.shape[0], stim_rad_deg*2, stim_type=stim_type,
                                             w_r=w_1, w_a=w_2)
    elif stim_type == 'constant':
        mag = sfp_stimuli.create_sf_maps_cpd(stim.shape[0], stim_rad_deg*2, stim_type=stim_type,
                                             w_x=w_1, w_y=w_2)
    R = ppt.mkR(stim.shape[0])

    # this limits the frequency maps to only where our stimulus has a grating.
    mag = utils.mask_array_like_grating(stim, mag, mid_val)

    # if stim_rad_deg corresponds to the max vertical/horizontal extent, the actual max will be
    # np.sqrt(2*stim_rad_deg**2) (this corresponds to the far corner). this should be the radius of
    # the screen, because R starts from the center and goes to the edge
    R = R/(R.max()/np.sqrt(2*stim_rad_deg**2))

    if eccen_bin:
        # create masks that look at each degree
        bin_masks = []
        eccen_idx = []
        for i in range(*eccen_range):
            bin_masks.append((R > i) & (R < (i+1)))
            eccen_idx.append('%s-%s' % (i, i+1))

        eccen_local_freqs = []
        eccens = []
        for m in bin_masks:
            eccens.append(R[m].mean())
            eccen_local_freqs.append(mag[m].mean())
    else:
        eccen_idx = eccens
        eccen_local_freqs = [mag.flatten()[abs(R - e).argmin()] for e in eccens]

    if plot_flag:
        plt.plot(eccens, eccen_local_freqs)
        ax = plt.gca()
        ax.set_title('Spatial frequency vs eccentricity')
        ax.set_xlabel('Eccentricity (degrees)')
        ax.set_ylabel('Local spatial frequency (cpd)')

    return pd.Series(eccen_local_freqs, eccen_idx)


def _add_local_sf_to_df(df, eccen_bin, eccen_range, stim, stim_type, stim_rad_deg=12, mid_val=128):
    """Adds local spatial frequency information for all stimuli to the df
    """
    try:
        freqs = df.drop_duplicates(['w_r', 'w_a'])[['w_r', 'w_a', 'stimulus_superclass']]
        freq_labels = ['w_r', 'w_a']
    except KeyError:
        freqs = df.drop_duplicates(['w_x', 'w_y'])[['w_x', 'w_y', 'stimulus_superclass']]
        freq_labels = ['w_x', 'w_y']
    sfs = []

    for w_1, w_2, stim_class in freqs.values:
        tmp = calculate_stim_local_sf(stim, w_1, w_2, stim_type, stim_rad_deg, eccen_bin,
                                      eccen_range, df.eccen.unique(), mid_val=mid_val)
        tmp = pd.DataFrame(tmp, columns=['Local spatial frequency (cpd)'])
        tmp.index.name = 'eccen'
        tmp[freq_labels[0]] = w_1
        tmp[freq_labels[1]] = w_2
        tmp['stimulus_superclass'] = stim_class
        sfs.append(tmp)

    sfs = pd.concat(sfs)
    sfs = sfs.reset_index().set_index(['stimulus_superclass', freq_labels[0], freq_labels[1], 'eccen'])
    df = df.set_index(['stimulus_superclass', freq_labels[0], freq_labels[1], 'eccen'])
    df = df.join(sfs)

    return df.reset_index()


def main(benson_template_path, results_template_path, df_mode='summary', stim_type='logpolar',
         save_path=None, class_nums=xrange(48), vareas=[1], eccen_range=(1, 12), eccen_bin=True,
         hemi_bin=True, stim_rad_deg=12, unshuffled_stim_path="../data/stimuli/unshuffled.npy",
         unshuffled_stim_descriptions_path="../data/stimuli/unshuffled_stim_description.csv",
         mid_val=128):
    """this loads in the realigned mgz files and creates a dataframe of their values

    This only returns those voxels that lie within visual areas outlined by the Benson14 varea mgz

    this should be run after GLMdenoise and after realign.py. The mgz files you give the path to
    should be surfaces, not volumes. this will take a while to run, which is why it's recommended
    to provide save_path so the resulting dataframe can be saved.

    benson_template_path: template path to the Benson14 mgz files, containing two string formatting
    symbols (%s; one for hemisphere, one for variable [angle, varea, eccen]),
    e.g. /mnt/Acadia/Freesurfer_subjects/wl_subj042/surf/%s.benson14_%s.mgz

    results_template_path: template path to the results mgz files (outputs of realign.py),
    containing two string formatting symbols (%s; one for hemisphere, one for results_names)

    df_mode: {'summary', 'full'}. If 'summary', will load in the 'modelmd' and 'modelse' mgz files,
    using those calculated summary values. If 'full', will load in the 'models_class_##' mgz files,
    containing the info to calculate central tendency and spread directly. In both cases, 'R2' will
    also be loaded in. Assumes modelmd and modelse lie directly in results_template_path and that
    models_class_## files lie within the subfolder models_niftis

    stim_type: {'logpolar', 'constant', 'pilot'}. which type of stimuli were used in the session
    we're analyzing. This matters because it changes the local spatial frequency and, since that is
    determined analytically and not directly from the stimuli, we have no way of telling otherwise.

    save_path: None or str. if str, will save the GLM_result_df at this location

    class_nums: list of ints. if df_mode=='full', which classes to load in. If df_mode=='summary',
    then this is ignored.

    vareas: list of ints. Which visual areas to include. the Benson14 template numbers vertices 0
    (not a visual area), -3, -2 (V3v and V2v, respectively), and 1 through 7.

    eccen_range: 2-tuple of ints or floats. What range of eccentricities to include.

    eccen_bin: boolean, default True. Whether to bin the eccentricities in integer
    increments. HIGHLY RECOMMENDED to be True if df_mode=='full', otherwise this will take much
    longer and the resulting DataFrame will be absurdly large and unwieldy.

    hemi_bin: boolean, default True. Does nothing if eccen_bin is False, but if eccen_bin is True,
    average corresponding eccentricity ROIs across the two hemispheres. Generally, this is what you
    want, unless you also to examine differences between the two hemispheres.

    stim_rad_deg: float, the radius of the stimulus, in degrees of visual angle

    unshuffled_stim_path: path to the unshuffled stimuli.

    unshuffled_stim_descriptions_path: path to the unshuffled stimulus description csv, as saved
    during the creation of the stimuli

    mid_val: int. the value of mid-grey in the stimuli, should be 127 (for pilot stimuli) or 128
    (for actual stimuli)
    """
    # This contains the information on each stimulus, allowing us to determine whether some stimuli
    # are part of the same class or a separate one.
    stim_df = pd.read_csv(unshuffled_stim_descriptions_path)
    stim_df = stim_df.dropna()
    stim_df.class_idx = stim_df.class_idx.astype(int)
    stim_df = stim_df.drop_duplicates('class_idx').set_index('class_idx')
    stim_df = stim_df.rename(columns={'index': 'stimulus_index'})
    # we only need one stimulus, because all of them have the same masks, which is what we're
    # interested in here
    stim = np.load(unshuffled_stim_path)[0, :, :]
    if df_mode == 'summary':
        results_names = ['modelse', 'modelmd']
    elif df_mode == 'full':
        results_names = ['models_class_%02d' % i for i in class_nums]
        if not eccen_bin:
            warnings.warn("Not binning by eccentricities while constructing the full DataFrame is "
                          "NOT recommended! This may fail because you run out of memory!")
    else:
        raise Exception("Don't know how to construct df with df_mode %s!" % df_mode)
    if hemi_bin and not eccen_bin:
        warnings.warn("You set eccen_bin to False but hemi_bin to True. I can only bin across "
                      "hemispheres if also binning eccentricities!")
        hemi_bin = False
    if os.path.isfile(benson_template_path % ('lh', 'varea')):
        mgzs = _arrange_mgzs_into_dict(benson_template_path, results_template_path,
                                       results_names+['R2'], vareas, eccen_range)
    elif os.path.isfile(benson_template_path % ('lh', 'areas')):
        mgzs = _arrange_mgzs_into_dict(benson_template_path, results_template_path,
                                       results_names+['R2'], vareas, eccen_range,
                                       {'varea': 'areas', 'angle': 'angle', 'eccen': 'eccen'})
    else:
        raise Exception("Unable to find the Benson visual areas template! Check your "
                        "benson_template_path!")
    if eccen_bin:
        mgzs = _bin_mgzs_dict(mgzs, results_names+['R2'], eccen_range, vareas, hemi_bin)

    results_names = [os.path.split(i)[-1] for i in results_names]

    df = _put_mgzs_dict_into_df(mgzs, stim_df, results_names, df_mode, eccen_bin, hemi_bin)
    df.varea = df.varea.astype(int)
    core_dists = df[df.stimulus_superclass == 'radial'].freq_space_distance.unique()
    if stim_type in ['logpolar', 'pilot']:
        df = _round_freq_space_distance(df, core_dists)
    df = _add_local_sf_to_df(df, eccen_bin, eccen_range, stim, stim_type, stim_rad_deg, mid_val)

    if save_path is not None:
        df.to_csv(save_path, index=False)

    return df


if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        description=("Load in relevant data and create a DataFrame summarizing the first-level "
                     "results for a given subject. Note that this can take a rather long time, "
                     "especially if you are not binning by eccentricity."),
        formatter_class=CustomFormatter)
    parser.add_argument("results_template_path",
                        help=("template path to the results mgz files (outputs of realign.py), "
                              "containing two string formatting symbols (one for hemisphere, "
                              "one specifying results type). Can contain any environment"
                              "al variable (in all caps, contained within curly brackets, e.g., "
                              "{SUBJECTS_DIR})"))
    parser.add_argument("benson_template_path",
                        help=("template path to the Benson14 mgz files, containing two string "
                              "formatting symbols (one for hemisphere, one for variable [angle"
                              ", varea, eccen]). Can contain any environmental variable (in all "
                              "caps, contained within curly brackets, e.g., {SUBJECTS_DIR})"))
    parser.add_argument("--stim_type", default='logpolar',
                        help=("{'logpolar', 'constant', 'pilot'}. which type of stimuli were used "
                              "in the session we're analyzing. This matters because it changes the"
                              " local spatial frequency and, since that is determined analytically"
                              " and not directly from the stimuli, we have no way of telling "
                              "otherwise."))
    parser.add_argument("--save_dir", default="data/MRI_first_level",
                        help=("directory to save the GLM result DataFrame in. The DataFrame will "
                              "be saved in a sub-directory (named for the subject) of this as a "
                              "csv with some identifying information in the path."))
    parser.add_argument("--df_mode", default='summary',
                        help=("{summary, full}. If summary, will load in the 'modelmd' and "
                              "'modelse' mgz files, and use those calculated summary values. If "
                              "full, will load in the 'models_class_##' mgz files, which contain "
                              "the info to calculate central tendency and spread directly. In both"
                              " cases, 'R2' will also be loaded in. Assumes modelmd and modelse "
                              "lie directly in results_template_path and that models_class_## "
                              "files lie within the subfolder models_niftis"))
    parser.add_argument("--class_nums", "-c", default=48, type=int,
                        help=("int. if df_mode=='full', will load classes in range(class_nums). If "
                              "df_mode=='summary', then this is ignored."))
    parser.add_argument("--vareas", "-v", nargs='+', default=[1], type=int,
                        help=("list of ints. Which visual areas to include. the Benson14 template "
                              "numbers vertices 0 (not a visual area), -3, -2 (V3v and V2v, "
                              "respectively), and 1 through 7."))
    parser.add_argument("--eccen_range", "-r", nargs=2, default=(1, 12), type=int,
                        help=("2-tuple of ints or floats. What range of eccentricities to "
                              "include."))
    parser.add_argument("--eccen_bin", action="store_true",
                        help=("Whether to bin the eccentricities in integer"
                              "increments. HIGHLY RECOMMENDED to be True if df_mode=='full', "
                              "otherwise this will take much longer and the resulting DataFrame "
                              "will be absurdly large and unwieldy."))
    parser.add_argument("--hemi_bin", action="store_true",
                        help=("Does nothing if eccen_bin is False, but if "
                              "eccen_bin is True, average corresponding eccentricity ROIs across "
                              "the two hemispheres. Generally, this is what you want, unless you "
                              "also to examine differences between the two hemispheres."))
    parser.add_argument("--stim_rad_deg", default=12, type=float,
                        help="float, the radius of the stimulus, in degrees of visual angle")
    parser.add_argument("--unshuffled_stim_descriptions_path", "-d",
                        default="data/stimuli/unshuffled_stim_description.csv",
                        help=("Path to the unshuffled_stim_descriptions.csv file that contains the"
                              " pandas Dataframe that specifies each stimulus's frequency"))
    parser.add_argument("--unshuffled_stim_path", "-s",
                        default="data/stimuli/unshuffled.npy",
                        help=("Path to the unshuffled.npy file that contains the numpy array with"
                              "the stimuli used in the experiment"))
    parser.add_argument("--save_stem", default="",
                        help=("String to prefix the filename of output csv with. Useful for making"
                              " this BIDS-like"))
    args = vars(parser.parse_args())
    save_dir = args.pop('save_dir')
    save_stem = args.pop('save_stem')
    save_dict = {'df_mode': args['df_mode'], 'vareas': '-'.join(str(i) for i in args['vareas']),
                 'eccen': '-'.join(str(i) for i in args['eccen_range'])}
    if args['eccen_bin']:
        save_dict['eccen_bin'] = '_eccen_bin'
    else:
        save_dict['eccen_bin'] = ''
    if args['hemi_bin']:
        save_dict['hemi_bin'] = '_hemi_bin'
    else:
        save_dict['hemi_bin'] = ''
    save_name = "{df_mode}_v{vareas}_e{eccen}{eccen_bin}{hemi_bin}.csv".format(**save_dict)
    args['save_path'] = os.path.join(save_dir, save_stem+save_name)
    args['class_nums'] = xrange(args['class_nums'])
    if not os.path.isdir(os.path.dirname(args['save_path'])):
        os.makedirs(os.path.dirname(args['save_path']))
    main(**args)
