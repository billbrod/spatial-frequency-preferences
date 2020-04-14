#!/usr/bin/python
"""arranges results mgzs into a dataframe for further analyses
"""
import matplotlib as mpl
# we do this because sometimes we run this without an X-server, and this backend doesn't need
# one. We set warn=False because the notebook uses a different backend and will spout out a big
# warning to that effect; that's unnecessarily alarming, so we hide it.
mpl.use('svg', warn=False)
import argparse
import warnings
import seaborn as sns
import pandas as pd
import numpy as np
import os
import nibabel as nib
import itertools
import re
import h5py
from matplotlib import pyplot as plt
from . import stimuli as sfp_stimuli


def _load_mgz(path):
    """load and reshape mgz so it's either 1d, instead of 3d

    this will also make an mgz 2d instead of 4d, but we also want to rearrange the data some, which
    this doesn't do

    """
    # see http://pandas.pydata.org/pandas-docs/version/0.19.1/gotchas.html#byte-ordering-issues
    return nib.load(path).get_data().byteswap().newbyteorder().squeeze()


def _arrange_helper(prf_dir, hemi, name, template, varea_mask, eccen_mask):
    """this small helper function is just to be called in a generator by _arrange_mgzs_into_dict
    """
    tmp = _load_mgz(template % (prf_dir, hemi, name))
    if tmp.ndim == 1:
        tmp = tmp[(varea_mask[hemi]) & (eccen_mask[hemi])]
    elif tmp.ndim == 2:
        tmp = tmp[(varea_mask[hemi]) & (eccen_mask[hemi]), :]
    if os.sep in name:
        res_name = os.path.split(name)[-1]
    elif '_' in name:
        res_name = name.split('_')[-1]
    elif '-' in name:
        res_name = name.split('-')[-1]
    else:
        res_name = name
    return "%s-%s" % (res_name, hemi), tmp


def _load_mat_file(path, results_names, varea_mask, eccen_mask):
    """load and reshape data from .mat file, so it's either 1 or 2d instead of 3 or 4d

    this will open the mat file (we assume it's saved by matlab v7.3 and so use h5py to do so). we
    then need to know the name of the mat_field (e.g., 'models', 'modelmd', 'modelse') to
    grab. then, some times we don't want to grab the entire corresponding array but only some
    subset (e.g., we don't want all bootstraps, all stimulus classes, we only want one stimulus
    class, all bootstraps), so index, if non-None, specifies the index along the second dimension
    (that's the one that indexes the stimulus classes) to take.

    """
    mgzs = {}
    with h5py.File(path, 'r') as f:
        for var, index in results_names:
            tmp_ref = f['results'][var]
            if tmp_ref.shape == (2, 1):
                # this is the case for all the models fields of the .mat file (modelse, modelmd,
                # models). [0, 0] contains the hrf, and [1, 0] contains the actual results.
                res = f[tmp_ref[1, 0]][:]
            else:
                # that reference thing is only necessary for those models fields, because I think
                # they're matlab structs
                res = tmp_ref[:]
            for idx in index:
                res_name = var
                if idx is None:
                    tmp = res
                else:
                    tmp = res[:, idx]
                    res_name += '_%02d' % idx
                tmp = tmp.squeeze()
                # in this case, the data is stimulus classes or bootstraps by voxels, and we want
                # voxels first, so we transpose.
                if tmp.ndim == 2:
                    tmp = tmp.transpose()
                # because of how bidsGetPreprocData.m loads in the surface files, we know this is the left
                # and right hemisphere concatenated together, in that order
                for hemi in ['lh', 'rh']:
                    if hemi == 'lh':
                        tmper = tmp[:varea_mask['lh'].shape[0]]
                    else:
                        tmper = tmp[-varea_mask['rh'].shape[0]:]
                    if tmper.ndim == 1:
                        tmper = tmper[(varea_mask[hemi]) & (eccen_mask[hemi])]
                    elif tmper.ndim == 2:
                        tmper = tmper[(varea_mask[hemi]) & (eccen_mask[hemi]), :]
                    mgzs['%s-%s' % (res_name, hemi)] = tmper
    return mgzs


def _arrange_mgzs_into_dict(benson_template_path, results_path, results_names, vareas,
                            eccen_range, benson_template_names=['varea', 'angle', 'eccen'],
                            prf_data_names=['sigma'], benson_atlas_type='bayesian_retinotopy'):
    """load in the mgzs, put in a dictionary, and return that dictionary

    vareas: list of ints. which visual areas (as defined in the Benson visual area template) to
    include. all others will be discarded.

    eccen_range: 2-tuple of ints or floats. What range of eccentricities to include (as specified
    in the Benson eccentricity template).

    benson_template_names: list of labels that specify which output files to get from the Benson
    retinotopy. The complete list is the default, ['varea', 'angle', 'sigma', 'eccen']. For this
    analysis to work, must contain 'varea' and 'eccen'.
    """
    varea_name = [i for i in benson_template_names if 'varea' in i]
    eccen_name = [i for i in benson_template_names if 'eccen' in i]
    if len(varea_name) != 1 or len(eccen_name) != 1:
        raise Exception("Need Benson retinotopy files 'eccen' and 'varea'!")
    mgzs = {}

    varea_mask = {}
    eccen_mask = {}
    for hemi in ['lh', 'rh']:
        varea_mask[hemi] = _load_mgz(benson_template_path % (benson_atlas_type, hemi, varea_name[0]))
        varea_mask[hemi] = np.isin(varea_mask[hemi], vareas)
        eccen_mask[hemi] = _load_mgz(benson_template_path % (benson_atlas_type, hemi, eccen_name[0]))
        eccen_mask[hemi] = (eccen_mask[hemi] > eccen_range[0]) & (eccen_mask[hemi] < eccen_range[1])

    # these are all mgzs
    for hemi, var in itertools.product(['lh', 'rh'], benson_template_names):
        k, v = _arrange_helper(benson_atlas_type, hemi, var, benson_template_path, varea_mask,
                               eccen_mask)
        mgzs[k] = v

    # these are all mgzs
    for hemi, var in itertools.product(['lh', 'rh'], prf_data_names):
        k, v = _arrange_helper('data', hemi, var, benson_template_path, varea_mask, eccen_mask)
        mgzs[k] = v

    # these all live in the results.mat file produced by GLMdenoise
    mgzs.update(_load_mat_file(results_path, results_names, varea_mask, eccen_mask))
    return mgzs


def _unfold_2d_mgz(mgz, value_name, variable_name, mgz_name, hemi=None):
    tmp = pd.DataFrame(mgz)
    tmp = pd.melt(tmp.reset_index(), id_vars='index')
    if hemi is not None:
        tmp['hemi'] = hemi
    tmp = tmp.rename(columns={'index': 'voxel', 'variable': variable_name, 'value': value_name})
    if 'models_' in mgz_name:
        # then the value name contains which stimulus class this and the actual value_name is
        # amplitude_estimate
        class_idx = re.search('models_([0-9]+)', mgz_name).groups()
        assert len(class_idx) == 1, "models title %s should only contain one number, to identify stimulus class!" % value_name
        tmp['stimulus_class'] = int(class_idx[0])
    return tmp


def _add_freq_metainfo(stim_df):
    """this function takes the stim_df and adds some metainfo based on the stimulus frequency

    right now these are: stimulus_superclass (angular, radial, etc), freq_space_angle (the angle
    in our 2d frequency space) and freq_space_distance (distance from the origin in our 2d
    frequency space)
    """
    # stimuli belong to five super classes, or paths through the frequency space: w_r=0; w_a=0;
    # w_r=w_a; w_r=-w_a; and sqrt(w_r^2 + w_a^)=32. We want to be able to look at them separately,
    # so we label them (this is inefficient but works). We also want to get some other identifying
    # values. We do this all at once because the major time cost comes from applying this to all
    # rows, not the computations themselves
    def freq_identifier_logpolar(x):
        if x.w_r == 0 and x.w_a == 0:
            sc = 'baseline'
        elif x.w_r == 0 and x.w_a != 0:
            sc = 'angular'
        elif x.w_r != 0 and x.w_a == 0:
            sc = 'radial'
        elif x.w_r == x.w_a:
            sc = 'forward spiral'
        elif x.w_r == -x.w_a:
            sc = 'reverse spiral'
        else:
            sc = 'mixtures'
        return sc, np.arctan2(x.w_a, x.w_r), np.sqrt(x.w_r**2 + x.w_a**2)

    def freq_identifier_constant(x):
        if x.w_x == 0 and x.w_y == 0:
            sc = 'baseline'
        elif x.w_x == 0 and x.w_y != 0:
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
        stim_df.loc[(stim_df['w_r'].isnull()) & (stim_df['w_a'].isnull()), ['w_r', 'w_a']] = (0, 0)
        properties_list = stim_df[['w_r', 'w_a']].apply(freq_identifier_logpolar, 1)
    except KeyError:
        stim_df.loc[(stim_df['w_x'].isnull()) & (stim_df['w_y'].isnull()), ['w_x', 'w_y']] = (0, 0)
        properties_list = stim_df[['w_x', 'w_y']].apply(freq_identifier_constant, 1)
    sc = pd.Series([i[0] for i in properties_list.values], properties_list.index)
    ang = pd.Series([i[1] for i in properties_list.values], properties_list.index)
    dist = pd.Series([i[2] for i in properties_list.values], properties_list.index)

    stim_df['stimulus_superclass'] = sc
    # get these between 0 and 2*pi, like the local spatial frequency angles
    stim_df['freq_space_angle'] = np.mod(ang, 2*np.pi)
    stim_df['freq_space_distance'] = dist
    return stim_df


def _setup_mgzs_for_df(mgzs, results_names, df_mode, hemi=None,
                       benson_template_names=['varea', 'angle', 'eccen'],
                       prf_data_names=['sigma']):
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

    for brain_name_full in benson_template_names + ['R2'] + prf_data_names:
        brain_name = brain_name_full.replace('inferred_', '').replace('benson14_', '')
        brain_name = brain_name.replace('all00-', '').replace('full-', '')
        if brain_name == 'R2':
            df_name = 'GLM_R2'
        elif brain_name == 'vexpl':
            df_name = 'prf_vexpl'
        else:
            df_name = brain_name
        try:
            tmp = pd.DataFrame(mgzs[mgz_key % brain_name])
            tmp = tmp.reset_index().rename(columns={'index': 'voxel', 0: df_name})
            df = df.merge(tmp)
        except ValueError:
            # see http://pandas.pydata.org/pandas-docs/version/0.19.1/gotchas.html#byte-ordering-issues
            warnings.warn("%s had that big-endian error" % brain_name)
            tmp = pd.DataFrame(mgzs[mgz_key % brain_name].byteswap().newbyteorder())
            tmp = tmp.reset_index().rename(columns={'index': 'voxel', 0: df_name})
            df = df.merge(tmp)

    return df


def _put_mgzs_dict_into_df(mgzs, stim_df, results_names, df_mode,
                           benson_template_names=['varea', 'angle', 'eccen'],
                           prf_data_names=['sigma']):
    df = {}
    for hemi in ['lh', 'rh']:
        df[hemi] = _setup_mgzs_for_df(mgzs, results_names, df_mode, hemi, benson_template_names,
                                      prf_data_names)

    # because python 0-indexes, the minimum voxel number is 0. thus if we were to just add the
    # max, the min in the right hemi would be the same as the max in the left hemi
    df['rh'].voxel = df['rh'].voxel + df['lh'].voxel.max()+1
    df = pd.concat(df).reset_index(0, drop=True)

    df = df.set_index('stimulus_class')
    df = df.join(stim_df)
    df = df.reset_index().rename(columns={'index': 'stimulus_class'})
    # Add the stimulus frequency information
    df = _add_freq_metainfo(df)

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
    R = sfp_stimuli.mkR(stim.shape)
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
    R = sfp_stimuli.mkR(stim.shape)
    # if stim_rad_deg corresponds to the max vertical/horizontal extent, the actual max will be
    # np.sqrt(2*stim_rad_deg**2) (this corresponds to the far corner). this should be the radius of
    # the screen, because R starts from the center and goes to the edge
    factor = R.max() / np.sqrt(2*stim_rad_deg**2)
    return Rmin / factor, Rmax / factor


def calculate_stim_local_sf(stim, w_1, w_2, stim_type, eccens, angles, stim_rad_deg=12,
                            plot_flag=False, mid_val=128):
    """calculate the local spatial frequency for a specified stimulus and screen size

    stim: 2d array of floats. an example stimulus. used to determine where the stimuli are masked
    (and thus where the spatial frequency is zero).

    w_1, w_2: ints or floats. the first and second components of the stimulus's spatial
    frequency. if stim_type is 'logarpolar' or 'pilot', this should be the radial and angular
    components (in that order!); if stim_type is 'constant', this should be the x and y components
    (in that order!)

    stim_type: {'logpolar', 'constant', 'pilot'}. which type of stimuli were used in the session
    we're analyzing. This matters because it changes the local spatial frequency and, since that is
    determined analytically and not directly from the stimuli, we have no way of telling otherwise.

    eccens, angles: lists of floats. these are the eccentricities and angles we want to find
    local spatial frequency for.

    stim_rad_deg: float, the radius of the stimulus, in degrees of visual angle

    plot_flag: boolean, optional, default False. Whether to create a plot showing the local spatial
    frequency vs eccentricity for the specified stimulus

    mid_val: int. the value of mid-grey in the stimuli, should be 127 (for pilot stimuli) or 128
    (for actual stimuli)
    """
    eccen_min, eccen_max = find_ecc_range_in_degrees(stim, stim_rad_deg, mid_val)
    eccen_local_freqs = []
    for i, (e, a) in enumerate(zip(eccens, angles)):
        if stim_type in ['logpolar', 'pilot']:
            dx, dy, mag, direc = sfp_stimuli.sf_cpd(stim.shape[0], stim_rad_deg*2, e, a,
                                                    stim_type=stim_type, w_r=w_1, w_a=w_2)
            dr, da, new_angle = sfp_stimuli.sf_origin_polar_cpd(stim.shape[0], stim_rad_deg*2, e,
                                                                a, stim_type=stim_type, w_r=w_1,
                                                                w_a=w_2)
        elif stim_type == 'constant':
            dx, dy, mag, direc = sfp_stimuli.sf_cpd(stim.shape[0], stim_rad_deg*2, e, a,
                                                    stim_type=stim_type, w_x=w_1, w_y=w_2)
            dr, da, new_angle = sfp_stimuli.sf_origin_polar_cpd(stim.shape[0], stim_rad_deg*2, e,
                                                                a, stim_type=stim_type, w_x=w_1,
                                                                w_y=w_2)
        eccen_local_freqs.append(pd.DataFrame(
            {'local_w_x': dx, 'local_w_y': dy, 'local_w_r': dr, 'local_w_a': da, 'eccen': e,
             'angle': a, 'local_sf_magnitude': mag, 'local_sf_xy_direction': direc,
             'local_sf_ra_direction': new_angle}, [i]))
    eccen_local_freqs = pd.concat(eccen_local_freqs)

    if plot_flag:
        plt.plot(eccen_local_freqs['eccen'], eccen_local_freqs['local_sf_magnitude'])
        ax = plt.gca()
        ax.set_title('Spatial frequency vs eccentricity')
        ax.set_xlabel('Eccentricity (degrees)')
        ax.set_ylabel('Local spatial frequency (cpd)')

    return eccen_local_freqs


def _add_local_sf_to_df(df, stim, stim_type, stim_rad_deg=12, mid_val=128):
    """Adds local spatial frequency information for all stimuli to the df
    """
    try:
        freqs = df.drop_duplicates(['w_r', 'w_a'])[['w_r', 'w_a', 'stimulus_superclass']]
        freq_labels = ['w_r', 'w_a']
    except KeyError:
        freqs = df.drop_duplicates(['w_x', 'w_y'])[['w_x', 'w_y', 'stimulus_superclass']]
        freq_labels = ['w_x', 'w_y']
    sfs = []

    # this gets us the unique pairs of (eccen, angle). It will also include a column that gives the
    # number of times each pair exists in the dataframe, but we ignore that.
    df_eccens_angles = df.groupby(['eccen', 'angle']).size().reset_index()
    for w_1, w_2, stim_class in freqs.values:
        tmp = calculate_stim_local_sf(stim, w_1, w_2, stim_type, df_eccens_angles.eccen.values,
                                      df_eccens_angles.angle.values, stim_rad_deg, mid_val=mid_val)
        tmp[freq_labels[0]] = w_1
        tmp[freq_labels[1]] = w_2
        tmp['stimulus_superclass'] = stim_class
        sfs.append(tmp)

    sfs = pd.concat(sfs)
    sfs = sfs.set_index(['stimulus_superclass', freq_labels[0], freq_labels[1], 'eccen', 'angle'])
    df = df.set_index(['stimulus_superclass', freq_labels[0], freq_labels[1], 'eccen', 'angle'])
    df = df.join(sfs)

    return df.reset_index()


def _add_baseline(df):
    if 'baseline' not in df.stimulus_superclass.unique():
        return df.assign(baseline=0)
    else:
        new_df = []
        for n, g in df.groupby(['varea', 'eccen']):
            try:
                baseline = g[g.stimulus_superclass == 'baseline'].amplitude_estimate.median()
            except AttributeError:
                baseline = g[g.stimulus_superclass == 'baseline'].amplitude_estimate_median.median()
            new_df.append(g.assign(baseline=baseline))
        return pd.concat(new_df)


def _transform_angle(x):
    """transform angle from Benson14 convention to our convention

    The Benson atlases' convention for angle in visual field is: zero is the upper vertical
    meridian, angle is in degrees, the left and right hemisphere both run from 0 to 180 from the
    upper to lower meridian (so they increase as you go clockwise and counter-clockwise,
    respectively). For our calculations, we need the following convention: zero is the right
    horizontal meridian, angle is in radians (and lie between 0 and 2*pi, rather than -pi and pi),
    angle increases as you go clockwise, and each angle is unique (refers to one point on the
    visual field; we don't have the same number in the left and right hemispheres)
    """
    ang = x.angle
    if x.hemi == 'rh':
        # we want to remap the right hemisphere angles to negative. Noah says this is the
        # convention, but I have seen positive values there, so maybe it changed at one point.
        if ang > 0:
            ang = -ang
    return np.mod(np.radians(ang - 90), 2*np.pi)


def _precision_dist(x, axis=None):
    """get precision from a distribution of values (inverse of variance)
    """
    cis = np.percentile(x, [16, 84], axis=axis)
    std_dev = abs(cis[0] - cis[1]) / 2.
    return 1. / (std_dev**2)


def _append_precision_col(df):
    """calculate precision and add to the dataframe

    the precision is the inverse of the variance and can be used as weights when combining across
    voxels. here, for each voxel, we calculate the precision for each stimulus class's estimate and
    then average across all stimulus classes to get a single precision estimate for each voxel.
    """
    df = df.copy()
    if 'amplitude_estimate_std_error' in df.columns:
        df['precision'] = 1. / (df.amplitude_estimate_std_error ** 2)
    else:
        gb = df.groupby(['varea', 'voxel', 'stimulus_class'])
        df = df.set_index(['varea', 'voxel', 'stimulus_class'])
        df['precision'] = gb.amplitude_estimate.apply(_precision_dist)
        df = df.reset_index()
    gb = df.groupby(['varea', 'voxel'])
    df = df.set_index(['varea', 'voxel'])
    df['precision'] = gb.precision.mean()
    return df.reset_index()


def _normalize_amplitude_estimate(df, norm_order=2):
    """calculates the norm of the ampltiude estimates, and normalizes by that

    by default, this is the L2-norm (as calculated by np.linalg.norm). Specify norm_order to change
    this, see https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.norm.html
    for possible values.
    """
    gb = df.groupby(['varea', 'voxel'])
    df = df.set_index(['varea', 'voxel'])
    if 'amplitude_estimate_median' in df.columns:
        df['amplitude_estimate_norm'] = gb.amplitude_estimate_median.apply(np.linalg.norm,
                                                                           norm_order)
        df = df.reset_index()
        for col in ['amplitude_estimate_median', 'amplitude_estimate_std_error']:
            df['%s_normed' % col] = df[col] / df.amplitude_estimate_norm
    else:
        df['amplitude_estimate_norm'] = gb.amplitude_estimate.apply(np.linalg.norm, norm_order)
        df = df.reset_index()
        df['amplitude_estimate_normed'] = df.amplitude_estimate / df.amplitude_estimate_norm
    return df


def main(benson_template_path, results_path, df_mode='summary', stim_type='logpolar',
         save_path=None, class_nums=range(48), vareas=[1], eccen_range=(1, 12), stim_rad_deg=12,
         benson_template_names=['inferred_varea', 'inferred_angle', 'inferred_eccen'],
         benson_atlas_type='bayesian_posterior', prf_data_names=['all00-sigma'],
         unshuffled_stim_path="../data/stimuli/task-sfp_stimuli.npy",
         unshuffled_stim_descriptions_path="../data/stimuli/task-sfp_stim_description.csv",
         mid_val=128):
    """this loads in the realigned mgz files and creates a dataframe of their values

    This only returns those voxels that lie within visual areas outlined by the Benson14 varea mgz

    this should be run after GLMdenoise and after realign.py. The mgz files you give the path to
    should be surfaces, not volumes. this will take a while to run, which is why it's recommended
    to provide save_path so the resulting dataframe can be saved.

    benson_template_path: template path to the Benson14 mgz files, containing three string
    formatting symbols (%s; one for retinotopy type [data, atlas, bayesian_posterior], one for
    hemisphere, one for variable [angle, varea, eccen, sigma]),
    e.g. /mnt/winawerlab/Projects/spatial_frequency_preferences/BIDS/derivatives/prf_solutions/sub-wlsubj042/%s/%s.%s.mgz

    results_path: path to the results.mat file (output of GLMdenoise)

    df_mode: {'summary', 'full'}. If 'summary', will load in the 'modelmd' and 'modelse' results
    fields, using those calculated summary values. If 'full', will load in the bootstrapped
    'models' results field, containing the info to calculate central tendency and spread
    directly. In both cases, 'R2' will also be loaded in.

    stim_type: {'logpolar', 'constant', 'pilot'}. which type of stimuli were used in the session
    we're analyzing. This matters because it changes the local spatial frequency and, since that is
    determined analytically and not directly from the stimuli, we have no way of telling otherwise.

    save_path: None or str. if str, will save the GLM_result_df at this location

    class_nums: list of ints. if df_mode=='full', which classes to load in. If df_mode=='summary',
    then this is ignored.

    vareas: list of ints. Which visual areas to include. the Benson14 template numbers vertices 0
    (not a visual area), -3, -2 (V3v and V2v, respectively), and 1 through 7.

    eccen_range: 2-tuple of ints or floats. What range of eccentricities to include.

    stim_rad_deg: float, the radius of the stimulus, in degrees of visual angle

    benson_template_names: list of labels that specify which output files to get from the Benson
    retinotopy. The complete list is ['varea', 'angle', 'sigma', 'eccen'] (plus either "benson14_"
    or "inferred_" beforehand). For this analysis to work, must contain 'varea' and 'eccen'.

    prf_data_names: list of labels that specify which output files to get from the pRF fits
    (without Bayesian retinotopy); we look for these in the directory found by inserting "data" as
    the first string for benson_template_path. The complete list is ['varea', 'angle', 'sigma',
    'eccen'] (plus either "all00-" or "full-" beforehand).

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
        results_names = [('modelse', [None]), ('modelmd', [None])]
    elif df_mode == 'full':
        results_names = [('models', class_nums)]
    else:
        raise Exception("Don't know how to construct df with df_mode %s!" % df_mode)
    for i in benson_template_names:
        if i in prf_data_names:
            raise Exception("Can only load variable from either Bayesian retinotopy or prf data, "
                            "not both! %s" % i)
        # we do this because it's possible that benson_template_names contains more than just
        # "varea", e.g., "inferred_varea"
        if 'varea' in i:
            if not os.path.isfile(benson_template_path % (benson_atlas_type, 'lh', i)):
                raise Exception("Unable to find the Benson visual areas template! Check your "
                                "benson_template_path! Checked %s" % (benson_template_path %
                                                                      (benson_atlas_type, 'lh', i)))
    mgzs = _arrange_mgzs_into_dict(benson_template_path, results_path,
                                   results_names+[('R2', [None])], vareas, eccen_range,
                                   benson_template_names, prf_data_names, benson_atlas_type)
    if save_path is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, hemi in zip(axes, ['lh', 'rh']):
            plot_data = mgzs['R2-%s' % hemi]
            num_nans = sum(np.isnan(plot_data))
            plot_data = plot_data[~np.isnan(plot_data)]
            sns.distplot(plot_data, ax=ax)
            ax.set_title("R2 for %s, data originally contained %s NaNs" % (hemi, num_nans))
        fig.savefig(save_path.replace('.csv', '_R2.svg'))

    if results_names[0][1][0] is None:
        # then all the results_names have None as their second value in the key tuple, so we can
        # ignore it
        results_names = [i[0] for i in results_names]
    else:
        # then every entry in results_names is a string followed by a list of
        # ints. itertools.product will give us what we want, but we need to wrap the string in a
        # list so that it doesn't iterate through the individual characters in the string
        results_names = ["%s_%02d" % (k, l) for i, j in results_names for k, l in itertools.product([i], j)]

    df = _put_mgzs_dict_into_df(mgzs, stim_df, results_names, df_mode, benson_template_names,
                                prf_data_names)
    df.varea = df.varea.astype(int)
    core_dists = df[df.stimulus_superclass == 'radial'].freq_space_distance.unique()
    if stim_type in ['logpolar', 'pilot']:
        df = _round_freq_space_distance(df, core_dists)
    df['angle'] = df.apply(_transform_angle, 1)
    df = _add_local_sf_to_df(df, stim, stim_type, stim_rad_deg, mid_val)
    df = _add_baseline(df)
    df = _append_precision_col(df)
    df = _normalize_amplitude_estimate(df)

    if save_path is not None:
        df.to_csv(save_path, index=False)

    return df


if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        description=("Load in relevant data and create a DataFrame summarizing the first-level "
                     "results for a given subject. Note that this can take a rather long time."),
        formatter_class=CustomFormatter)
    parser.add_argument("--results_path", required=True,
                        help=("path to the results.mat file (output of GLMdenoise) for a single session"))
    parser.add_argument("--benson_template_path", required=True,
                        help=("template path to the Benson14 mgz files, containing three string "
                              "formatting symbols (one for retinotopy type [data, atlas, bayesian_"
                              "posterior], one for hemisphere, one for variable [angle, varea, "
                              "eccen]). Can contain any environmental variable (in all caps, "
                              "contained within curly brackets, e.g., {SUBJECTS_DIR})"))
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
                              "'modelse' results fields, and use those calculated summary values. "
                              "If full, will load in the bootstrapped 'models' field, which "
                              "contains the info to calculate central tendency and spread directly"
                              ". In both cases, 'R2' will also be loaded in."))
    parser.add_argument("--class_nums", "-c", default=48, type=int,
                        help=("int. if df_mode=='full', will load classes in range(class_nums). If"
                              "df_mode=='summary', then this is ignored."))
    parser.add_argument("--vareas", "-v", nargs='+', default=[1], type=int,
                        help=("list of ints. Which visual areas to include. the Benson14 template "
                              "numbers vertices 0 (not a visual area), -3, -2 (V3v and V2v, "
                              "respectively), and 1 through 7."))
    parser.add_argument("--eccen_range", "-r", nargs=2, default=(1, 12), type=int,
                        help=("2-tuple of ints or floats. What range of eccentricities to "
                              "include."))
    parser.add_argument("--stim_rad_deg", default=12, type=float,
                        help="float, the radius of the stimulus, in degrees of visual angle")
    parser.add_argument("--benson_atlas_type", default="bayesian_posterior",
                        help=("{atlas, bayesian_posterior}. Type of Benson atlas. Will be the "
                              "first string inserted into benson_template_path"))
    parser.add_argument("--benson_template_names", nargs='+',
                        default=['inferred_varea', 'inferred_angle', 'inferred_eccen'],
                        help=("list of labels that specify which output files to get from the "
                              "Benson retinotopy. For this analysis to work, must contain 'varea'"
                              " and 'eccen'. Note that some subjects might not have sigma."))
    parser.add_argument("--prf_data_names", nargs='+',
                        default=['all00-sigma'],
                        help=("list of labels that specify which output files to get from the "
                              "pRF fits (will insert 'data' into the benson_template_path to find)"
                              "."))
    parser.add_argument("--unshuffled_stim_descriptions_path", "-d",
                        default="data/stimuli/task-sfp_stim_description.csv",
                        help=("Path to the csv file that contains the  pandas Dataframe that "
                              "specifies each stimulus's frequency"))
    parser.add_argument("--unshuffled_stim_path", "-s",
                        default="data/stimuli/task-sfp_stimuli.npy",
                        help=("Path to the npy file that contains the numpy array with the stimuli"
                              " used in the experiment"))
    parser.add_argument("--save_stem", default="",
                        help=("String to prefix the filename of output csv with. Useful for making"
                              " this BIDS-like"))
    parser.add_argument("--mid_val", default=128, type=int,
                        help=("The value of mid-grey in the stimuli. Should be 127 for pilot "
                              "stimuli, 128 for real experiment"))
    args = vars(parser.parse_args())
    save_dir = args.pop('save_dir')
    save_stem = args.pop('save_stem')
    save_dict = {'df_mode': args['df_mode'], 'vareas': '-'.join(str(i) for i in args['vareas']),
                 'eccen': '-'.join(str(i) for i in args['eccen_range'])}
    save_name = "v{vareas}_e{eccen}_{df_mode}.csv".format(**save_dict)
    args['save_path'] = os.path.join(save_dir, save_stem+save_name)
    args['class_nums'] = range(args['class_nums'])
    if not os.path.isdir(os.path.dirname(args['save_path'])):
        os.makedirs(os.path.dirname(args['save_path']))
    main(**args)
