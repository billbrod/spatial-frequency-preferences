#!/usr/bin/python
"""functions to run first-level MRI analyses
"""
import pandas as pd
import numpy as np
import os
import warnings
import nibabel as nib
import itertools
import re
from matplotlib import pyplot as plt
import stimuli
import pyPyrTools as ppt


def _load_mgz(path):
    """load and reshape mgz so it's either 1 or 2d, instead of 3 or 4d
    """
    # see http://pandas.pydata.org/pandas-docs/version/0.19.1/gotchas.html#byte-ordering-issues
    tmp = nib.load(path).get_data().byteswap().newbyteorder()
    if tmp.ndim == 3:
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
        tmp = _load_mgz(results_template_path % (res, hemi))
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


def _add_freq_metainfo(design_df):
    """this function takes the design_df and adds some metainfo based on the stimulus frequency

    right now these are: stimulus_superclass (radial, circular, etc), freq_space_angle (the angle
    in our 2d frequency space) and freq_space_distance (distance from the origin in our 2d
    frequency space)
    """
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

    properties_list = design_df[['w_r', 'w_a']].apply(freq_identifier, 1)
    sc = pd.Series([i[0] for i in properties_list.values], properties_list.index)
    ang = pd.Series([i[1] for i in properties_list.values], properties_list.index)
    dist = pd.Series([i[2] for i in properties_list.values], properties_list.index)

    design_df['stimulus_superclass'] = sc
    design_df['freq_space_angle'] = ang
    design_df['freq_space_distance'] = dist
    return design_df


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


def _put_mgzs_dict_into_df(mgzs, design_df, results_names, df_mode, eccen_bin=True, hemi_bin=True):
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
    design_df = _add_freq_metainfo(design_df)

    df = df.set_index('stimulus_class')
    df = df.join(design_df)
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


def find_ecc_range_in_pixels(stim):
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
    # 127 is the middle value.
    x, y = np.where(stim != 127)
    return R[x, y].min(), R[x, y].max()


def find_ecc_range_in_degrees(stim, stim_rad_deg):
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
    Rmin, Rmax = find_ecc_range_in_pixels(stim)
    R = ppt.mkR(stim.shape)
    # if stim_rad_deg corresponds to the max vertical/horizontal extent, the actual max will be
    # np.sqrt(2*stim_rad_deg**2) (this corresponds to the far corner). this should be the radius of
    # the screen, because R starts from the center and goes to the edge
    factor = R.max() / np.sqrt(2*stim_rad_deg**2)
    return Rmin / factor, Rmax / factor


def calculate_stim_local_sf(stim, w_r, w_a=0, alpha=50, stim_size_pix=1080, stim_rad_deg=12,
                            eccen_bin=True, eccen_range=(2, 8), eccens=[], plot_flag=False):
    """calculate the local spatial frequency for a specified stimulus and screen size

    WARNING: Currently this only works for circular stimuli.

    NOTE: this assumes that the local spatial frequency does not depend on angle, only on
    eccentricity.

    This works slightly differently if you are binning by eccentricity or not (i.e., depending on
    whether eccen_bin is True or False). If binning, then we take the annulus with inner edge i
    degrees from the origin and outer edge i+1 degrees (where i runs from eccen_range[0] to
    eccen_range[1]-1) and average together the local spatial frequency found there.

    If not binning, eccen_range is ignored, and eccens must be specified. We then look for the
    spatial frequency at the closest eccentricity value we have in the
    distance(-in-degrees)-from-origin map (made by rescaling the output of ppt.mkR()) for each
    value in eccens.

    w_r, w_a: ints or floats. the radial and angular components, respectively, of the stimulus's
    spatial frequency

    alpha: int, radius (in pixel spacing) of the "fovea".  IE: log_rad = log(r^2 + alpha^2)

    stim_size_pix: int, the size (diameter) of the stimulus, in pixels

    stim_rad_deg: float, the radius of the stimulus, in degrees of visual angle

    eccen_bin:

    plot_flag: boolean, optional, default False. Whether to create a plot showing the local spatial
    frequency vs eccentricity for the specified stimulus
    """
    mag = stimuli.create_sf_maps_cpd(stim_size_pix, alpha, stim_rad_deg*2, w_r, w_a)
    R = ppt.mkR(stim_size_pix)

    # this limits the frequency maps to only where our stimulus has a grating.
    x, y = np.where(stim != 127)
    Rmin, Rmax = R[x, y].min(), R[x, y].max()
    mag[R < Rmin] = 0
    mag[R > Rmax] = 0

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


def _add_local_sf_to_df(df, eccen_bin, eccen_range, stimuli, alpha=50, stim_size_pix=1080,
                        stim_rad_deg=12):
    """Adds local spatial frequency information for all stimuli to the df
    """
    freqs = df.drop_duplicates(['w_r', 'w_a'])[['w_r', 'w_a', 'stimulus_superclass']]
    sfs = []

    for w_r, w_a, stim_class in freqs.values:
        # we only need one stimulus, because all of them have the same masks, which is what we're
        # interested in here
        tmp = calculate_stim_local_sf(stimuli[0, :, :], w_r, w_a, alpha, stim_size_pix,
                                      stim_rad_deg, eccen_bin, eccen_range, df.eccen.unique())
        tmp = pd.DataFrame(tmp, columns=['Local spatial frequency (cpd)'])
        tmp.index.name = 'eccen'
        tmp['w_r'] = w_r
        tmp['w_a'] = w_a
        tmp['stimulus_superclass'] = stim_class
        sfs.append(tmp)

    sfs = pd.concat(sfs)
    sfs = sfs.reset_index()

    sfs = sfs.set_index(['stimulus_superclass', 'w_a', 'w_r', 'eccen'])
    df = df.set_index(['stimulus_superclass', 'w_a', 'w_r', 'eccen'])
    df = df.join(sfs)

    return df.reset_index()


def create_GLM_result_df(design_df, stimuli, benson_template_path, results_template_path,
                         df_mode='summary', save_path=None, class_nums=xrange(52), vareas=[1],
                         eccen_range=(2, 8), eccen_bin=True, hemi_bin=True, stim_rad_deg=12):
    """this loads in the realigned mgz files and creates a dataframe of their values

    This only returns those voxels that lie within visual areas outlined by the Benson14 varea mgz

    this should be run after GLMdenoise and after realign.py. The mgz files you give the path to
    should be surfaces, not volumes. this will take a while to run, which is why it's recommended
    to provide save_path so the resulting dataframe can be saved.

    design_df: output of create_design_df

    stimuli: numpy array of unpermuted stimuli used in the experiment.

    benson_template_path: template path to the Benson14 mgz files, containing two string formatting
    symbols (%s; one for hemisphere, one for variable [angle, varea, eccen]),
    e.g. /mnt/Acadia/Freesurfer_subjects/wl_subj042/surf/%s.benson14_%s.mgz

    results_template_path: template path to the results mgz files (outputs of realign.py),
    containing two string formatting symbols (%s; one for hemisphere, one for results_names)

    df_mode: 'summary' or 'full'. If 'summary', will load in the 'modelmd' and 'modelse' mgz files,
    using those calculated summary values. If 'full', will load in the 'models_class_##' mgz files,
    containing the info to calculate central tendency and spread directly. In both cases, 'R2' will
    also be loaded in. Assumes modelmd and modelse lie directly in results_template_path and that
    models_class_## files lie within the subfolder models_niftis

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
    """
    if df_mode == 'summary':
        results_names = ['modelse', 'modelmd']
    elif df_mode == 'full':
        results_names = ['models_niftis/models_class_%02d' % i for i in class_nums]
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

    df = _put_mgzs_dict_into_df(mgzs, design_df, results_names, df_mode, eccen_bin, hemi_bin)
    core_dists = df[df.stimulus_superclass == 'radial'].freq_space_distance.unique()
    df = _round_freq_space_distance(df, core_dists)
    df = _add_local_sf_to_df(df, eccen_bin, eccen_range, stimuli, design_df.alpha.unique()[0],
                             design_df.res.unique()[0], stim_rad_deg)

    if save_path is not None:
        df.to_csv(save_path)

    return df


# Make wrapper function that does above, loading in design_df and maybe grabbing it for different
# results? and then combining them.
def main(behavioral_results_path, benson_template_path, results_template_path,
         unshuffled_stim_descriptions_path="../data/stimuli/unshuffled_stim_description.csv"):
    """wrapper function that loads in relevant bits of information and calls relevant functions

    Ends up creating and saving the dataframe containing the first level results.
    """
