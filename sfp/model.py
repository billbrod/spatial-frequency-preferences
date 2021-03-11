#!/usr/bin/python
"""2d tuning model
"""
import matplotlib as mpl
# we do this because sometimes we run this without an X-server, and this backend doesn't need
# one. We set warn=False because the notebook uses a different backend and will spout out a big
# warning to that effect; that's unnecessarily alarming, so we hide it.
mpl.use('svg', warn=False)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import warnings
import argparse
import itertools
import re
import functools
from scipy import stats
from torch.utils import data as torchdata
from hessian import hessian
from . import plotting


def reduce_num_voxels(df, n_voxels=200):
    """drop all but the first n_voxels

    this is just to speed things up for testing, it obviously shouldn't be used if you are actually
    trying to fit the data
    """
    return df[df.voxel < n_voxels]


def randomly_reduce_num_voxels(df, n_voxels=200):
    """drop voxels randomly so that we end up with n_voxels
    """
    voxels = np.random.choice(df.voxel.unique(), n_voxels, replace=False)
    return df.query('voxel in @voxels')


def drop_voxels_with_any_negative_amplitudes(df):
    """drop all voxels that have at least one negative amplitude
    """
    groupby_col = ['voxel']
    if 'indicator' in df.columns:
        groupby_col += ['indicator']
    try:
        # in this case, we have the full dataframe (with each bootstrap
        # separately), and we want to drop those voxels whose median
        # amplitude estimate is less than zero, not just a single
        # bootstrap. so first we compute the median amplitude estimates
        # for each voxel and stimulus class
        tmp = df.groupby(groupby_col + ['stimulus_class']).amplitude_estimate.median().reset_index()
        # then, doing like we do below, we only retain those voxels that
        # have all amplitude estimates above zero
        tmp = tmp.groupby(groupby_col).filter(lambda x: (x.amplitude_estimate >= 0).all())
        voxels = tmp.voxel.unique()
        df = df.query('voxel in @voxels')
    except AttributeError:
        df = df.groupby(groupby_col).filter(lambda x: (x.amplitude_estimate_median >= 0).all())
    return df


def drop_voxels_with_mean_negative_amplitudes(df):
    """drop all voxels that have an average negative amplitude (across stimuli)
    """
    groupby_col = ['voxel']
    if 'indicator' in df.columns:
        groupby_col += ['indicator']
    try:
        # in this case, we have the full dataframe (with each bootstrap
        # separately), and we want to drop those voxels whose median
        # amplitude estimate is less than zero, not just a single
        # bootstrap. so first we compute the median amplitude estimates
        # for each voxel and stimulus class
        tmp = df.groupby(groupby_col + ['stimulus_class']).amplitude_estimate.median().reset_index()
        # then we do a similar thing to get the average across stimulus classes.
        tmp = tmp.groupby(groupby_col).amplitude_estimate.mean().reset_index()
        tmp = tmp.groupby(groupby_col).filter(lambda x: (x.amplitude_estimate >= 0).all())
        voxels = tmp.voxel.unique()
        df = df.query('voxel in @voxels')
    except AttributeError:
        tmp = df.groupby(groupby_col).amplitude_estimate_median.mean().reset_index()
        tmp = tmp.groupby(groupby_col).filter(lambda x: (x.amplitude_estimate_median >= 0).all())
        voxels = tmp.voxel.unique()
        df = df.query('voxel in @voxels')
    return df


def drop_voxels_near_border(df, inner_border=.96, outer_border=12):
    """drop all voxels whose pRF center is one sigma away form the border

    where the sigma is the sigma of the Gaussian pRF
    """
    groupby_col = ['voxel']
    if 'indicator' in df.columns:
        groupby_col += ['indicator']
    df = df.groupby(groupby_col).filter(lambda x: (x.eccen + x.sigma <= outer_border).all())
    df = df.groupby(groupby_col).filter(lambda x: (x.eccen - x.sigma >= inner_border).all())
    return df


def restrict_to_part_of_visual_field(df, restriction):
    """restrict to voxels that lie in one part of visual field

    restriction: {"upper", "lower", "left", "right", "inner", "outer"}

    """
    pi = np.pi
    eccen_lim = df.eccen.max() / 2
    if restriction == 'right':
        restriction_str = 'angle<=@pi/2 | angle>=3*@pi/2'
    elif restriction == 'left':
        restriction_str = 'angle>@pi/2 & angle<3*@pi/2'
    elif restriction == 'upper':
        restriction_str = 'angle<@pi'
    elif restriction == 'lower':
        restriction_str = 'angle>=@pi'
    elif restriction == 'inner':
        restriction_str = 'eccen<@eccen_lim'
    elif restriction == 'outer':
        restriction_str = 'eccen>=@eccen_lim'
    elif restriction == 'horizontal':
        restriction_str = "angle<=@pi/4 | (angle > 3*@pi/4 & angle <= 5*@pi/4) | (angle > 7*@pi/4)"
    elif restriction == 'vertical':
        restriction_str = "(angle>@pi/4 & angle<=3*@pi/4) | (angle > 5*@pi/4 & angle <= 7*@pi/4)"
    return df.query(restriction_str)


def _cast_as_tensor(x):
    if type(x) == pd.Series:
        x = x.values
    # needs to be float32 to work with the Hessian calculations
    return torch.tensor(x, dtype=torch.float32)


def _cast_as_param(x, requires_grad=True):
    return torch.nn.Parameter(_cast_as_tensor(x), requires_grad=requires_grad)


def _cast_args_as_tensors(args, on_cuda=False):
    return_args = []
    for v in args:
        if not torch.is_tensor(v):
            v = _cast_as_tensor(v)
        if on_cuda:
            v = v.cuda()
        return_args.append(v)
    return return_args


def _check_and_reshape_tensors(x, y):
    if (x.ndimension() == 1 and y.ndimension() == 1) and (x.shape != y.shape):
        x = x.repeat(len(y), 1)
        y = y.repeat(x.shape[1], 1).transpose(0, 1)
    return x, y


class FirstLevelDataset(torchdata.Dataset):
    """Dataset for first level results

    the __getitem__ method here returns all (48) values for a single voxel, so keep that in mind
    when setting batch size. this is done because (in the loss function) we normalize the
    predictions and target so that that vector of length 48 has a norm of one.

    In addition the features and targets, we also return the precision

    df_filter: function or None. If not None, a function that takes a dataframe as input and
    returns one (most likely, a subset of the original) as output. See
    `drop_voxels_with_any_negative_amplitudes` for an example.

    stimulus_class: list of ints or None. What subset of the stimulus_class should be used. these
    are numbers between 0 and 47 (inclusive) and then the dataset will only include data from those
    stimulus classes. this is used for cross-validation purposes (i.e., train on 0 through 46, test
    on 47).

    bootstrap_num: list of ints or None. Which bootstrap(s) to select. If the dataframe
    contains multiple bootstraps and this is not set, we raise an exception; if the dataframe
    doesn't contain multiple bootstraps and this is set, we raise an exception.

    model_mode: {'tuning_curve', 'image-computable'}. whether we're fitting the tuning curve or
    image-computable version of the model (will change what info we return)

    """
    def __init__(self, df_path, device, df_filter=None, stimulus_class=None, bootstrap_num=None,
                 model_mode='tuning_curve'):
        df = pd.read_csv(df_path)
        if df_filter is not None:
            # we want the index to be reset so we can use iloc in get_single_item below. this
            # ensures that iloc and loc will return the same thing, which isn't otherwise the
            # case. and we want them to be the same because Dataloader assumes iloc but our custom
            # get_voxel needs loc.
            df = df_filter(df).reset_index()
        # in order to make sure that we can iterate through the dataset (as dataloader does), we
        # need to create a new "voxel" column. this column just relabels the voxel column, running
        # from 0 to df.voxel.nunique() while ensuring that voxel identity is preserved. if
        # df_filter is None, df.voxel_reindexed should just be a copy of df.voxel
        new_idx = pd.Series(range(df.voxel.nunique()), df.voxel.unique())
        df = df.set_index('voxel')
        df['voxel_reindexed'] = new_idx
        if stimulus_class is not None:
            df = df.query("stimulus_class in @stimulus_class")
        if bootstrap_num is not None:
            df = df.query("bootstrap_num in @bootstrap_num")
            if len(bootstrap_num) > 1:
                raise Exception("For now, bootstrap_num must only have 1 number in it. Major issue"
                                " is the construction of the performance_df in check_performance, "
                                "would need to get bootstrap_num as another column there (and set "
                                "it as an index when joining with results_df)")
        else:
            if 'bootstrap_num' in df.columns:
                raise Exception("Since dataframe contains multiple bootstraps, `bootstrap_num` arg"
                                " must be set!")
        if df.empty:
            raise Exception("Dataframe is empty!")
        self.df = df.reset_index()
        self.device = device
        self.df_path = df_path
        self.stimulus_class = df.stimulus_class.unique()
        self.bootstrap_num = bootstrap_num
        if model_mode not in ['tuning_curve', 'image-computable']:
            raise Exception("Don't know how to handle model_mode %s!" % model_mode)
        self.model_mode = model_mode

    def get_single_item(self, idx):
        row = self.df.iloc[idx]
        if self.model_mode == 'tuning_curve':
            vals = row[['local_sf_magnitude', 'local_sf_xy_direction', 'eccen', 'angle']].values
        elif self.model_mode == 'image-computable':
            # in this case, we'll grab spatial frequency information directly from the saved energy
            # arrays, so we need to tell it which stimulus class this belongs to and we need the
            # pRF size, sigma.
            vals = row[['stimulus_class', 'eccen', 'angle', 'sigma']].values
        feature = _cast_as_tensor(vals.astype(float))
        try:
            target = _cast_as_tensor(row['amplitude_estimate'])
        except KeyError:
            target = _cast_as_tensor(row['amplitude_estimate_median'])
        precision = _cast_as_tensor(row['precision'])
        return (feature.to(self.device),
                torch.stack([target.to(self.device), precision.to(self.device)], -1))

    def __getitem__(self, idx):
        vox_idx = self.df[self.df.voxel_reindexed == idx].index
        return self.get_single_item(vox_idx)

    def get_voxel(self, idx):
        vox_idx = self.df[self.df.voxel == idx].index
        return self.get_single_item(vox_idx)

    def __len__(self):
        return self.df.voxel.nunique()


def _check_log_gaussian_params(param_vals, train_params, period_orientation_type,
                               eccentricity_type, amplitude_orientation_type):
        if period_orientation_type in ['relative', 'iso']:
            for angle in ['cardinals', 'obliques']:
                if param_vals[f'abs_mode_{angle}'] != 0:
                    # when parsing from df, these can be nan. don't need
                    # to raise the warning in that case
                    if not np.isnan(param_vals[f'abs_mode_{angle}']):
                        warnings.warn(f"When period_orientation_type is {period_orientation_type}, "
                                      "all absolute variables must be 0, correcting this...")
                    param_vals[f'abs_mode_{angle}'] = 0
                train_params[f'abs_mode_{angle}'] = False
        if period_orientation_type in ['absolute', 'iso']:
            for angle in ['cardinals', 'obliques']:
                if param_vals[f'rel_mode_{angle}'] != 0:
                    # when parsing from df, these can be nan. don't need
                    # to raise the warning in that case
                    if not np.isnan(param_vals[f'rel_mode_{angle}']):
                        warnings.warn(f"When period_orientation_type is {period_orientation_type}, "
                                      "all relative variables must be 0, correcting this...")
                    param_vals[f'rel_mode_{angle}'] = 0
                train_params[f'rel_mode_{angle}'] = False
        if period_orientation_type not in ['relative', 'absolute', 'iso', 'full']:
            raise Exception("Don't know how to handle period_orientation_type "
                            f"{period_orientation_type}!")
        if amplitude_orientation_type in ['relative', 'iso']:
            for angle in ['cardinals', 'obliques']:
                if param_vals[f'abs_amplitude_{angle}'] != 0:
                    # when parsing from df, these can be nan. don't need
                    # to raise the warning in that case
                    if not np.isnan(param_vals[f'abs_amplitude_{angle}']):
                        warnings.warn(f"When amplitude_orientation_type is {amplitude_orientation_type}, "
                                      "all absolute variables must be 0, correcting this...")
                    param_vals[f'abs_amplitude_{angle}'] = 0
                train_params[f'abs_amplitude_{angle}'] = False
        if amplitude_orientation_type in ['absolute', 'iso']:
            for angle in ['cardinals', 'obliques']:
                if param_vals[f'rel_amplitude_{angle}'] != 0:
                    # when parsing from df, these can be nan. don't need
                    # to raise the warning in that case
                    if not np.isnan(param_vals[f'rel_amplitude_{angle}']):
                        warnings.warn(f"When amplitude_orientation_type is {amplitude_orientation_type}, "
                                      "all relative variables must be 0, correcting this...")
                    param_vals[f'rel_amplitude_{angle}'] = 0
                train_params[f'rel_amplitude_{angle}'] = False
        if amplitude_orientation_type not in ['relative', 'absolute', 'iso', 'full']:
            raise Exception("Don't know how to handle amplitude_orientation_type "
                            f"{amplitude_orientation_type}!")
        if eccentricity_type == 'scaling':
            if param_vals['sf_ecc_intercept'] != 0:
                # when parsing from df, these can be nan. don't need
                # to raise the warning in that case
                if not np.isnan(param_vals[f'sf_ecc_intercept']):
                    warnings.warn("When eccentricity_type is scaling, sf_ecc_intercept must be 0! "
                                  "correcting...")
                param_vals['sf_ecc_intercept'] = 0
            train_params['sf_ecc_intercept'] = False
        elif eccentricity_type == 'constant':
            if param_vals['sf_ecc_slope'] != 0:
                # when parsing from df, these can be nan. don't need
                # to raise the warning in that case
                if not np.isnan(param_vals[f'sf_ecc_slope']):
                    warnings.warn("When eccentricity_type is constant, sf_ecc_slope must be 0! "
                                  "correcting...")
                param_vals['sf_ecc_slope'] = 0
            train_params['sf_ecc_slope'] = False
        elif eccentricity_type != 'full':
            raise Exception("Don't know how to handle eccentricity_type %s!" % eccentricity_type)
        return param_vals, train_params


class LogGaussianDonut(torch.nn.Module):
    """simple LogGaussianDonut in pytorch

    orientation_type, eccentricity_type, vary_amplitude: together specify what
    kind of model to train

    period_orientation_type: {iso, absolute, relative, full}.
        How we handle the effect of orientation on preferred period:
        - iso: model is isotropic, predictions identical for all orientations.
        - absolute: model can fit differences in absolute orientation, that is, in Cartesian
          coordinates, such that sf_angle=0 correponds to "to the right"
        - relative: model can fit differences in relative orientation, that is, in retinal polar
          coordinates, such that sf_angle=0 corresponds to "away from the fovea"
        - full: model can fit differences in both absolute and relative orientations

    eccentricity_type: {scaling, constant, full}.
        How we handle the effect of eccentricity on preferred period
        - scaling: model's relationship between preferred period and eccentricity is exactly scaling,
          that is, the preferred period is equal to the eccentricity.
        - constant: model's relationship between preferred period and eccentricity is exactly constant,
          that is, it does not change with eccentricity but is flat.
        - full: model discovers the relationship between eccentricity and preferred period, though it
          is constrained to be linear (i.e., model solves for a and b in $period = a * eccentricity +
          b$)

    amplitude_orientation_type: {iso, absolute, relative, full}.
        How we handle the effect of orientation on maximum amplitude:
        - iso: model is isotropic, predictions identical for all orientations.
        - absolute: model can fit differences in absolute orientation, that is, in Cartesian
          coordinates, such that sf_angle=0 correponds to "to the right"
        - relative: model can fit differences in relative orientation, that is, in retinal polar
          coordinates, such that sf_angle=0 corresponds to "away from the fovea"
        - full: model can fit differences in both absolute and relative orientations

    all other parameters are initial values. whether they will be fit or not (i.e., whether they
    have `requires_grad=True`) depends on the values of `orientation_type`, `eccentricity_type` and
    `vary_amplitude`

    when you call this model, sf_angle should be the (absolute) orientation of the grating, so that
    sf_angle=0 corresponds to "to the right". That is, regardless of whether the model considers
    the absolute orientation, relative orientation, neither or both to be important, you always
    call it with the absolute orientation.

    """
    def __init__(self, period_orientation_type='iso', eccentricity_type='full',
                 amplitude_orientation_type='iso', sigma=.4, sf_ecc_slope=1, sf_ecc_intercept=0,
                 abs_mode_cardinals=0, abs_mode_obliques=0, rel_mode_cardinals=0,
                 rel_mode_obliques=0, abs_amplitude_cardinals=0, abs_amplitude_obliques=0,
                 rel_amplitude_cardinals=0, rel_amplitude_obliques=0):
        super().__init__()
        train_kwargs = {}
        kwargs = {}
        for ori, param, angle in itertools.product(['abs', 'rel'], ['mode', 'amplitude'],
                                                   ['cardinals', 'obliques']):
            train_kwargs['%s_%s_%s' % (ori, param, angle)] = True
            kwargs['%s_%s_%s' % (ori, param, angle)] = eval('%s_%s_%s' % (ori, param, angle))
        for var in ['slope', 'intercept']:
            train_kwargs['sf_ecc_%s' % var] = True
            kwargs['sf_ecc_%s' % var] = eval("sf_ecc_%s" % var)
        kwargs, train_kwargs = _check_log_gaussian_params(kwargs, train_kwargs,
                                                          period_orientation_type,
                                                          eccentricity_type,
                                                          amplitude_orientation_type)

        self.period_orientation_type = period_orientation_type
        self.amplitude_orientation_type = amplitude_orientation_type
        self.eccentricity_type = eccentricity_type
        self.model_type = (f'{eccentricity_type}_donut_period-{period_orientation_type}_'
                           f'amps-{amplitude_orientation_type}')
        self.sigma = _cast_as_param(sigma)

        self.abs_amplitude_cardinals = _cast_as_param(kwargs['abs_amplitude_cardinals'],
                                                      train_kwargs['abs_amplitude_cardinals'])
        self.abs_amplitude_obliques = _cast_as_param(kwargs['abs_amplitude_obliques'],
                                                     train_kwargs['abs_amplitude_obliques'])
        self.rel_amplitude_cardinals = _cast_as_param(kwargs['rel_amplitude_cardinals'],
                                                      train_kwargs['rel_amplitude_cardinals'])
        self.rel_amplitude_obliques = _cast_as_param(kwargs['rel_amplitude_obliques'],
                                                     train_kwargs['rel_amplitude_obliques'])
        self.abs_mode_cardinals = _cast_as_param(kwargs['abs_mode_cardinals'],
                                                 train_kwargs['abs_mode_cardinals'])
        self.abs_mode_obliques = _cast_as_param(kwargs['abs_mode_obliques'],
                                                train_kwargs['abs_mode_obliques'])
        self.rel_mode_cardinals = _cast_as_param(kwargs['rel_mode_cardinals'],
                                                 train_kwargs['rel_mode_cardinals'])
        self.rel_mode_obliques = _cast_as_param(kwargs['rel_mode_obliques'],
                                                train_kwargs['rel_mode_obliques'])
        self.sf_ecc_slope = _cast_as_param(kwargs['sf_ecc_slope'],
                                           train_kwargs['sf_ecc_slope'])
        self.sf_ecc_intercept = _cast_as_param(kwargs['sf_ecc_intercept'],
                                               train_kwargs['sf_ecc_intercept'])

    @classmethod
    def init_from_df(cls, df):
        """initialize from the dataframe we make summarizing the models

        the df must only contain a single model (that is, it should only have 11 rows, one for each
        parameter value, and a unique value for the column fit_model_type)

        """
        fit_model_type = df.fit_model_type.unique()
        if len(fit_model_type) > 1 or len(df) != 11:
            raise Exception("df must contain exactly one model!")
        params = {}
        for i, row in df.iterrows():
            params[row.model_parameter] = row.fit_value
        # we may have renamed the model type to the version we want for
        # plotting. if so, this will map it back to the original version
        # so our re.findall will work as expected
        model_name_map = dict(zip(plotting.MODEL_PLOT_ORDER, plotting.MODEL_ORDER))
        fit_model_type = model_name_map.get(fit_model_type[0], fit_model_type[0])
        parse_string = r'([a-z]+)_donut_period-([a-z]+)_amps-([a-z]+)'
        ecc, period, amps = re.findall(parse_string, fit_model_type)[0]
        return cls(period, ecc, amps, **params)

    def __str__(self):
        # so we can see the parameters
        return ("{0}(sigma: {1:.03f}, sf_ecc_slope: {2:.03f}, sf_ecc_intercept: {3:.03f}, "
                "abs_amplitude_cardinals: {4:.03f}, abs_amplitude_obliques: {5:.03f}, "
                "abs_mode_cardinals: {6:.03f}, abs_mode_obliques: {7:.03f}, "
                "rel_amplitude_cardinals: {8:.03f}, rel_amplitude_obliques: {9:.03f}, "
                "rel_mode_cardinals: {10:.03f}, rel_mode_obliques: {11:.03f})").format(
                    type(self).__name__, self.sigma, self.sf_ecc_slope, self.sf_ecc_intercept,
                    self.abs_amplitude_cardinals, self.abs_amplitude_obliques,
                    self.abs_mode_cardinals, self.abs_mode_obliques, self.rel_amplitude_cardinals,
                    self.rel_amplitude_obliques, self.rel_mode_cardinals, self.rel_mode_obliques)

    def __repr__(self):
        return self.__str__()
    
    def prepare_image_computable(self, energy, filters, stim_radius_degree=12):
        """prepare for the image computable version of the model

        Parameters
        ----------
        energy : np.ndarray
            energy has shape (num_classes, n_scales, n_orientations, *img_size) and 
            contains the energy (square and absolute value the complex valued output of 
            SteerablePyramidFreq; equivalently, square and sum the output of the quadrature pair of 
            filters that make up the pyramid) for each image, at each scale and orientation. the energy
            has all been upsampled to the size of the initial image.
        filters : np.ndarray
            filters has shape (max_ht, n_orientations, *img_size) and is the fourier transform of the 
            filters at each scale and orientation, zero-padded so they all have the same size. we only 
            have one set of filters (instead of one per stimulus class) because the same pyramid was 
            used for each of them; we ensure this by getting the filters for each stimulus class and 
            checking that they're individually equal to the average across classes.
        stim_radius_degree : int
            the radius of the stimuli (in degrees), necessary for converting between pixels and 
            degrees.

        """
        self.stim_radius_degree = stim_radius_degree
        if energy.shape[-2] != energy.shape[-1]:
            raise Exception("For now, this only works on square input images!")
        self.image_size = energy.shape[-1]
        filters, energy = _cast_args_as_tensors([filters, energy], self.sigma.is_cuda)
        self.energy = energy.unsqueeze(0)
        # this is the l1 norm
        norm_weights = filters.abs().sum((2,3), keepdim=True)
        norm_weights = norm_weights[0] / norm_weights
        self.filters = filters * norm_weights
        x = np.linspace(-self.stim_radius_degree, self.stim_radius_degree, self.image_size)
        x, y = np.meshgrid(x, x)
        # we want to try and delete things to save memory
        del norm_weights, energy, filters
        self.visual_space = np.dstack((x, y))

    def _create_mag_angle(self, extent=(-10, 10), n_samps=1001):
        x = torch.linspace(extent[0], extent[1], n_samps)
        x, y = torch.meshgrid(x, x)
        r = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))
        th = torch.atan2(y, x)
        return r, th

    def create_image(self, vox_ecc, vox_angle, extent=None, n_samps=None):
        vox_ecc, vox_angle = _cast_args_as_tensors([vox_ecc, vox_angle], self.sigma.is_cuda)
        if vox_ecc.ndimension() == 0:
            vox_ecc = vox_ecc.unsqueeze(-1)
            vox_angle = vox_angle.unsqueeze(-1)
        if extent is None:
            extent = (-self.stim_radius_degree, self.stim_radius_degree)
        if n_samps is None:
            n_samps = self.image_size
        r, th = self._create_mag_angle(extent, n_samps)
        return self.evaluate(r.repeat(len(vox_ecc), 1, 1), th.repeat(len(vox_ecc), 1, 1),
                             vox_ecc, vox_angle)

    def preferred_period_contour(self, preferred_period, vox_angle, sf_angle=None,
                                 rel_sf_angle=None):
        """return eccentricity that has specified preferred_period for given sf_angle, vox_angle

        either sf_angle or rel_sf_angle can be set
        """
        if ((sf_angle is None and rel_sf_angle is None) or
            (sf_angle is not None and rel_sf_angle is not None)):
            raise Exception("Either sf_angle or rel_sf_angle must be set!")
        if sf_angle is None:
            sf_angle = rel_sf_angle
            rel_flag = True
        else:
            rel_flag = False
        preferred_period, sf_angle, vox_angle = _cast_args_as_tensors(
            [preferred_period, sf_angle, vox_angle], self.sigma.is_cuda)
        # we can allow up to two of these variables to be non-singletons.
        if sf_angle.ndimension() == 1 and preferred_period.ndimension() == 1 and vox_angle.ndimension() == 1:
            # if this is False, then all of them are the same shape and we have no issues
            if sf_angle.shape != preferred_period.shape != vox_angle.shape:
                raise Exception("Only two of these variables can be non-singletons!")
        else:
            sf_angle, preferred_period = _check_and_reshape_tensors(sf_angle, preferred_period)
            preferred_period, vox_angle = _check_and_reshape_tensors(preferred_period, vox_angle)
            sf_angle, vox_angle = _check_and_reshape_tensors(sf_angle, vox_angle)
        if not rel_flag:
            rel_sf_angle = sf_angle - vox_angle
        else:
            rel_sf_angle = sf_angle
            sf_angle = rel_sf_angle + vox_angle
        orientation_effect = (1 + self.abs_mode_cardinals * torch.cos(2 * sf_angle) +
                              self.abs_mode_obliques * torch.cos(4 * sf_angle) +
                              self.rel_mode_cardinals * torch.cos(2 * rel_sf_angle) +
                              self.rel_mode_obliques * torch.cos(4 * rel_sf_angle))
        return (preferred_period / orientation_effect - self.sf_ecc_intercept) / self.sf_ecc_slope

    def preferred_period(self, vox_ecc, vox_angle, sf_angle=None, rel_sf_angle=None):
        """return preferred period for specified voxel at given orientation
        """
        if ((sf_angle is None and rel_sf_angle is None) or
            (sf_angle is not None and rel_sf_angle is not None)):
            raise Exception("Either sf_angle or rel_sf_angle must be set!")
        if sf_angle is None:
            sf_angle = rel_sf_angle
            rel_flag = True
        else:
            rel_flag = False
        sf_angle, vox_ecc, vox_angle = _cast_args_as_tensors([sf_angle, vox_ecc, vox_angle],
                                                             self.sigma.is_cuda)
        # we can allow up to two of these variables to be non-singletons.
        if sf_angle.ndimension() == 1 and vox_ecc.ndimension() == 1 and vox_angle.ndimension() == 1:
            # if this is False, then all of them are the same shape and we have no issues
            if sf_angle.shape != vox_ecc.shape != vox_angle.shape:
                raise Exception("Only two of these variables can be non-singletons!")
        else:
            sf_angle, vox_ecc = _check_and_reshape_tensors(sf_angle, vox_ecc)
            vox_ecc, vox_angle = _check_and_reshape_tensors(vox_ecc, vox_angle)
            sf_angle, vox_angle = _check_and_reshape_tensors(sf_angle, vox_angle)
        if not rel_flag:
            rel_sf_angle = sf_angle - vox_angle
        else:
            rel_sf_angle = sf_angle
            sf_angle = rel_sf_angle + vox_angle
        eccentricity_effect = self.sf_ecc_slope * vox_ecc + self.sf_ecc_intercept
        orientation_effect = (1 + self.abs_mode_cardinals * torch.cos(2 * sf_angle) +
                              self.abs_mode_obliques * torch.cos(4 * sf_angle) +
                              self.rel_mode_cardinals * torch.cos(2 * rel_sf_angle) +
                              self.rel_mode_obliques * torch.cos(4 * rel_sf_angle))
        return torch.clamp(eccentricity_effect * orientation_effect, min=1e-6)

    def preferred_sf(self, sf_angle, vox_ecc, vox_angle):
        return 1. / self.preferred_period(vox_ecc, vox_angle, sf_angle)

    def max_amplitude(self, vox_angle, sf_angle=None, rel_sf_angle=None):
        if ((sf_angle is None and rel_sf_angle is None) or
            (sf_angle is not None and rel_sf_angle is not None)):
            raise Exception("Either sf_angle or rel_sf_angle must be set!")
        if sf_angle is None:
            sf_angle = rel_sf_angle
            rel_flag = True
        else:
            rel_flag = False
        sf_angle, vox_angle = _cast_args_as_tensors([sf_angle, vox_angle], self.sigma.is_cuda)
        sf_angle, vox_angle = _check_and_reshape_tensors(sf_angle, vox_angle)
        if not rel_flag:
            rel_sf_angle = sf_angle - vox_angle
        else:
            rel_sf_angle = sf_angle
            sf_angle = rel_sf_angle + vox_angle
        amplitude = (1 + self.abs_amplitude_cardinals * torch.cos(2*sf_angle) +
                     self.abs_amplitude_obliques * torch.cos(4*sf_angle) +
                     self.rel_amplitude_cardinals * torch.cos(2*rel_sf_angle) +
                     self.rel_amplitude_obliques * torch.cos(4*rel_sf_angle))
        return torch.clamp(amplitude, min=1e-6)

    def evaluate(self, sf_mag, sf_angle, vox_ecc, vox_angle):
        sf_mag, = _cast_args_as_tensors([sf_mag], self.sigma.is_cuda)
        # if ecc_effect is 0 or below, then log2(ecc_effect) is infinity or undefined
        # (respectively). to avoid that, we clamp it 1e-6. in practice, if a voxel ends up here
        # that means the model predicts 0 response for it.
        preferred_period = self.preferred_period(vox_ecc, vox_angle, sf_angle)
        pdf = torch.exp(-((torch.log2(sf_mag) + torch.log2(preferred_period))**2) /
                        (2*self.sigma**2))
        amplitude = self.max_amplitude(vox_angle, sf_angle)
        return amplitude * pdf

    def image_computable_weights(self, vox_ecc, vox_angle):
        vox_ecc, vox_angle = _cast_args_as_tensors([vox_ecc, vox_angle], self.sigma.is_cuda)
        vox_tuning = self.create_image(vox_ecc.unsqueeze(-1).unsqueeze(-1),
                                       vox_angle.unsqueeze(-1).unsqueeze(-1),
                                       (-self.stim_radius_degree, self.stim_radius_degree),
                                       self.image_size)
        vox_tuning = vox_tuning.unsqueeze(1).unsqueeze(1)
        return torch.sum(vox_tuning * self.filters, (-1, -2), keepdim=True).unsqueeze(1)

    def create_prfs(self, vox_ecc, vox_angle, vox_sigma):
        vox_ecc, vox_angle, vox_sigma = _cast_args_as_tensors([vox_ecc, vox_angle, vox_sigma],
                                                              self.sigma.is_cuda)
        if vox_ecc.ndimension() == 0:
            vox_ecc = vox_ecc.unsqueeze(-1)
        if vox_angle.ndimension() == 0:
            vox_angle = vox_angle.unsqueeze(-1)
        if vox_sigma.ndimension() == 0:
            vox_sigma = vox_sigma.unsqueeze(-1)
        vox_x = vox_ecc * np.cos(vox_angle)
        vox_y = vox_ecc * np.sin(vox_angle)
        prfs = []
        for x, y, s in zip(vox_x, vox_y, vox_sigma):
            prf = stats.multivariate_normal((x, y), s)
            prfs.append(prf.pdf(self.visual_space))
        return _cast_args_as_tensors([prfs], self.sigma.is_cuda)[0].unsqueeze(1)
    
    def image_computable(self, inputs):
        # the different features will always be indexed along the last axis (we don't know whether
        # this is 2d (stimulus_class, features) or 3d (voxels, stimulus_class, features))
        # to be used as index, must be long type
        stim_class = inputs.select(-1, 0)
        if stim_class.ndimension() == 2:
            stim_class = stim_class.mean(0)
        stim_class = torch.as_tensor(stim_class, dtype=torch.long)
        vox_ecc = inputs.select(-1, 1).mean(-1)
        if vox_ecc.ndimension() == 0:
            vox_ecc = vox_ecc.unsqueeze(-1)
        vox_angle = inputs.select(-1, 2).mean(-1)
        if vox_angle.ndimension() == 0:
            vox_angle = vox_angle.unsqueeze(-1)
        vox_sigma = inputs.select(-1, 3).mean(-1)
        if vox_sigma.ndimension() == 0:
            vox_sigma = vox_sigma.unsqueeze(-1)
        weights = self.image_computable_weights(vox_ecc, vox_angle)
        reweighted_energy = torch.sum(weights * self.energy[:, stim_class], (2, 3))

        prfs = self.create_prfs(vox_ecc, vox_angle, vox_sigma)
        predictions = torch.sum(prfs * reweighted_energy, (-1, -2))
        # reweighted_energy is big, so we want to delete it to try and save memory
        del reweighted_energy
        return predictions

    def forward(self, inputs):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # the different features will always be indexed along the last axis (we don't know whether
        # this is 2d (stimulus_class, features) or 3d (voxels, stimulus_class, features))
        sf_mag = inputs.select(-1, 0)
        sf_angle = inputs.select(-1, 1)
        vox_ecc = inputs.select(-1, 2)
        vox_angle = inputs.select(-1, 3)
        return self.evaluate(sf_mag, sf_angle, vox_ecc, vox_angle)


def show_image(donut, voxel_eccentricity=1, voxel_angle=0, extent=(-5, 5), n_samps=1001,
               cmap="Reds", show_colorbar=True, ax=None, **kwargs):
    """wrapper function to plot the image from a given donut

    This shows the spatial frequency selectivity implied by the donut at a given voxel eccentricity
    and angle, if appropriate (eccentricity and angle ignored if donut is ConstantLogGuassianDonut)

    donut: a LogGaussianDonut

    extent: 2-tuple of floats. the range of spatial frequencies to visualize `(min, max)`. this
    will be the same for x and y
    """
    if ax is None:
        plt.imshow(
            donut.create_image(voxel_eccentricity, voxel_angle, extent, n_samps=n_samps).detach()[0],
            extent=(extent[0], extent[1], extent[0], extent[1]), cmap=cmap,
            origin='lower', **kwargs)
        ax = plt.gca()
    else:
        ax.imshow(
            donut.create_image(voxel_eccentricity, voxel_angle, extent, n_samps=n_samps).detach()[0],
            extent=(extent[0], extent[1], extent[0], extent[1]), cmap=cmap,
            origin='lower', **kwargs)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_frame_on(False)
    if show_colorbar:
        plt.colorbar()
    return ax


def weighted_normed_loss(predictions, target, weighted=True, average=True):
    """takes in the predictions and target, returns weighted norm loss

    note all of these must be tensors, not numpy arrays

    target must contain both the targets and the precision (along the last axis)

    if we weren't multiplying by the precision, this would be equivalent to cosine distance (times
    a constant: num_classes / 2). set weighted=False to use this

    if average is True, we return a single value (averaging across voxels and
    stimulus classes). If average is False, we return the full `(n_voxels,
    n_classes)` matrix

    """
    precision = target.select(-1, 1)
    target = target.select(-1, 0)
    # we occasionally have an issue where the predictions are really small (like 1e-200), which
    # gives us a norm of 0 and thus a normed_predictions of infinity, and thus an infinite loss.
    # the point of renorming is that multiplying by a scale factor won't change our loss, so we do
    # that here to avoid this issue
    if 0 in predictions.norm(2, -1, True):
        warnings.warn("Predictions too small to normalize correctly, multiplying it be 1e100")
        predictions = predictions * 1e100
    # we norm / average along the last dimension, since that means we do it across all stimulus
    # classes for a given voxel. we don't know whether these tensors will be 1d (single voxel, as
    # returned by our FirstLevelDataset) or 2d (multiple voxels, as returned by the DataLoader)
    normed_predictions = predictions / predictions.norm(2, -1, True)
    normed_target = target / target.norm(2, -1, True)
    if weighted:
        # this isn't really necessary (all the values along that dimension
        # should be identical, based on how we calculated it), but just in
        # case. and this gets it in the right shape
        precision = precision.mean(-1, True)
        squared_error = precision * (normed_predictions - normed_target)**2
    else:
        squared_error = (normed_predictions - normed_target)**2
    if average:
        squared_error = squared_error.mean()
    return squared_error


def construct_history_df(history, var_name='batch_num', value_name='loss'):
    """constructs loss dataframe from array of loss or time history

    history: 2d list or array, as constructed in `train_model` for loss and time elapsed, n_epochs
    by batch_size

    """
    if np.array(history).ndim == 3:
        # then this is parameter value or Hessian history
        labels = np.array(history)[:, :, 0]
        assert (labels[0] == labels).all(), "%s history constructed incorrectly!" % value_name
        labels = labels[0]
        values = np.array(history)[:, :, 1].astype(np.float)
        df = pd.DataFrame(values, columns=labels)
    elif np.array(history).ndim == 2:
        df = pd.DataFrame(np.array(history))
    df = pd.melt(df.reset_index(), id_vars='index', var_name=var_name, value_name=value_name)
    return df.rename(columns={'index': 'epoch_num'})


def check_performance(trained_model, dataset, loss_func=weighted_normed_loss):
    """check performance of trained_model for each voxel in the dataset

    this assumes both model and dataset are on the same device
    """
    performance = []
    for i in dataset.df.voxel.unique():
        features, targets = dataset.get_voxel(i)
        predictions = trained_model(features)
        # targets[:, 0] contains the actual targets, targets[:, 1] contains the precision,
        # unimportant right here
        corr = np.corrcoef(targets[:, 0].cpu().detach().numpy(), predictions.cpu().detach().numpy())
        loss = loss_func(predictions, targets).item()
        performance.append(pd.DataFrame({'voxel': i, 'stimulus_class': range(len(targets)),
                                         'model_prediction_correlation': corr[0, 1],
                                         'model_prediction_loss': loss,
                                         'model_predictions': predictions.cpu().detach().numpy()}))
    return pd.concat(performance)


def combine_first_level_df_with_performance(first_level_df, performance_df):
    """combine results_df and performance_df, along the voxel, producing results_df
    """
    results_df = first_level_df.set_index(['voxel', 'stimulus_class'])
    performance_df = performance_df.set_index(['voxel', 'stimulus_class'])
    results_df = results_df.join(performance_df).reset_index()
    return results_df.reset_index()


def construct_dfs(model, dataset, train_loss_history, time_history, model_history, hessian_history,
                  max_epochs, batch_size, learning_rate, train_thresh, current_epoch,
                  loss_func=weighted_normed_loss):
    """construct the loss and results dataframes and add metadata
    """
    loss_df = construct_history_df(train_loss_history)
    time_df = construct_history_df(time_history, value_name='time')
    model_history_df = pd.merge(construct_history_df(model_history, 'parameter', 'value'),
                                construct_history_df(hessian_history, 'parameter', 'hessian'),
                                'outer')
    model_history_df = pd.merge(model_history_df,
                                time_df.groupby('epoch_num').time.max().reset_index())
    loss_df = pd.merge(loss_df, time_df)
    # we reload the first level dataframe because the one in dataset may be filtered in some way
    results_df = pd.read_csv(dataset.df_path)
    # however we still want to filter this by bootstrap_num if the dataset was filtered that way
    if dataset.bootstrap_num is not None:
        results_df = results_df.query("bootstrap_num in @dataset.bootstrap_num")
    results_df = combine_first_level_df_with_performance(
        results_df, check_performance(model, dataset, loss_func))
    if type(model) == torch.nn.DataParallel:
        # in this case, we need to access model.module in order to get the various custom
        # attributes we set in our LogGaussianDonut
        model = model.module
    # this is the case if the data is simulated
    for col in ['true_model_type', 'noise_level', 'noise_source_df']:
        if col in results_df.columns:
            loss_df[col] = results_df[col].unique()[0]
            model_history_df[col] = results_df[col].unique()[0]
    for name, val in model.named_parameters():
        results_df['fit_model_%s' % name] = val.cpu().detach().numpy()
    metadata_names = ['max_epochs', 'batch_size', 'learning_rate', 'train_thresh', 'loss_func',
                      'dataset_df_path', 'epochs_trained', 'fit_model_type']
    metadata_vals = [max_epochs, batch_size, learning_rate, train_thresh, loss_func.__name__,
                     dataset.df_path, current_epoch, model.model_type]
    for name, val in zip(metadata_names, metadata_vals):
        loss_df[name] = val
        results_df[name] = val
        model_history_df[name] = val
    return loss_df, results_df, model_history_df


def save_outputs(model, loss_df, results_df, model_history_df, save_path_stem):
    """save outputs (if save_path_stem is not None)

    results_df can be None, in which case we don't save it.
    """
    if type(model) == torch.nn.DataParallel:
        # in this case, we need to access model.module in order to just save the model
        model = model.module
    if save_path_stem is not None:
        torch.save(model.state_dict(), save_path_stem + "_model.pt")
        loss_df.to_csv(save_path_stem + "_loss.csv", index=False)
        model_history_df.to_csv(save_path_stem + "_model_history.csv", index=False)
        if results_df is not None:
            results_df.to_csv(save_path_stem + "_results_df.csv", index=False)


def _check_convergence(history, thresh):
    if len(history) > 3:
        if np.array(history).ndim == 3:
            history = np.array(history)[:, :, 1].astype(np.float)
            if (np.abs(np.diff(history, axis=0))[-3:] < thresh).all():
                return True
        else:
            history = np.array(history)
            if (np.abs(np.diff(np.mean(history, 1)))[-3:] < thresh).all():
                return True
    return False


def train_model(model, dataset, max_epochs=5, batch_size=10, train_thresh=1e-6, learning_rate=1e-3,
                save_path_stem=None, loss_func=weighted_normed_loss, cv_flag=False):
    """train the model
    """
    training_parameters = [p for p in model.parameters() if p.requires_grad]
    # AMSGrad argument here means we use a revised version that handles a bug people found where
    # it doesn't necessarily converge
    optimizer = torch.optim.Adam(training_parameters, lr=learning_rate, amsgrad=True)
    dataloader = torchdata.DataLoader(dataset, batch_size)
    loss_history = []
    start_time = time.time()
    time_history = []
    model_history = []
    hessian_history = []
    full_data, full_target = next(iter(torchdata.DataLoader(dataset, len(dataset))))
    for t in range(max_epochs):
        loss_history.append([])
        time_history.append([])
        for i, (features, target) in enumerate(dataloader):
            # these transposes get features from the dimensions (voxels, stimulus class, features)
            # into (features, voxels, stimulus class) so that the predictions are shape (voxels,
            # stimulus class), just like the targets are
            predictions = model(features)
            loss = loss_func(predictions, target)
            loss_history[t].append(loss.item())
            time_history[t].append(time.time() - start_time)
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                # we raise an exception here and then try again.
                raise Exception("Loss is nan or inf on epoch %s, batch %s!" % (t, i))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model_history.append([(k, v.clone().cpu().detach().numpy()) for k, v in model.named_parameters()])
        H = hessian(loss_func(model(full_data), full_target),
                    [p for p in model.parameters() if p.requires_grad])
        model.eval()
        # the inverse of the square root of the diagonal of the Hessian is an estimate of the
        # standard error of the maximum likelihood estimates of our parameters:
        # https://stats.stackexchange.com/a/68095
        hessian_item = tuple(zip([p[0] for p in model.named_parameters() if p[1].requires_grad],
                                 1./torch.sqrt(H.diag()).cpu().detach().numpy()))
        model.train()
        hessian_history.append(hessian_item)
        if (t % 100) == 99:
            loss_df, results_df, model_history_df = construct_dfs(
                model, dataset, loss_history, time_history, model_history, hessian_history,
                max_epochs, batch_size, learning_rate, train_thresh, t, loss_func=loss_func)
            if not cv_flag and dataset.bootstrap_num is not None:
                save_outputs(model, loss_df, results_df, model_history_df, save_path_stem)
            else:
                save_outputs(model, loss_df, None, model_history_df, save_path_stem)
        print("Average loss on epoch %s: %s" % (t, np.mean(loss_history[-1])))
        print(model)
        if _check_convergence(loss_history, train_thresh):
            if _check_convergence(model_history, train_thresh):
                print("Epoch loss and parameter values appear to have converged, so we stop "
                      "training")
                break
    loss_df, results_df, model_history_df = construct_dfs(
        model, dataset, loss_history, time_history, model_history, hessian_history, max_epochs,
        batch_size, learning_rate, train_thresh, t, loss_func=loss_func)
    return model, loss_df, results_df, model_history_df


def main(model_period_orientation_type, model_eccentricity_type, model_amplitude_orientation_type,
         first_level_results_path, random_seed=None, max_epochs=100, train_thresh=1e-6,
         batch_size=10, df_filter=None, learning_rate=1e-3, test_set_stimulus_class=None,
         bootstrap_num=None, save_path_stem="pytorch", loss_func=weighted_normed_loss):
    """create, train, and save a model on the given first_level_results dataframe

    model_period_orientation_type, model_eccentricity_type,
    model_amplitude_orientation_type: together specify what kind of
    model to train

    model_period_orientation_type: {iso, absolute, relative, full}.
        How we handle the effect of orientation on preferred period:
        - iso: model is isotropic, predictions identical for all orientations.
        - absolute: model can fit differences in absolute orientation, that is, in Cartesian
          coordinates, such that sf_angle=0 correponds to "to the right"
        - relative: model can fit differences in relative orientation, that is, in retinal polar
          coordinates, such that sf_angle=0 corresponds to "away from the fovea"
        - full: model can fit differences in both absolute and relative orientations

    model_eccentricity_type: {scaling, constant, full}.
        How we handle the effect of eccentricity on preferred period
        - scaling: model's relationship between preferred period and eccentricity is exactly scaling,
          that is, the preferred period is equal to the eccentricity.
        - constant: model's relationship between preferred period and eccentricity is exactly constant,
          that is, it does not change with eccentricity but is flat.
        - full: model discovers the relationship between eccentricity and preferred period, though it
          is constrained to be linear (i.e., model solves for a and b in $period = a * eccentricity +
          b$)

    model_amplitude_orientation_type: {iso, absolute, relative, full}.
        How we handle the effect of orientation on maximum amplitude:
        - iso: model is isotropic, predictions identical for all orientations.
        - absolute: model can fit differences in absolute orientation, that is, in Cartesian
          coordinates, such that sf_angle=0 correponds to "to the right"
        - relative: model can fit differences in relative orientation, that is, in retinal polar
          coordinates, such that sf_angle=0 corresponds to "away from the fovea"
        - full: model can fit differences in both absolute and relative orientations

    first_level_results_path: str. Path to the first level results dataframe containing the data to
    fit.

    random_seed: int or None. we initialize the model with random parameters in order to try and
    avoid local optima. we set the seed before generating all those random numbers.

    max_epochs: int. the max number of epochs to let the training run for. otherwise, we train
    until the loss changes by less than train_thresh for 3 epochs in a row.

    df_filter: function or None. If not None, a function that takes a dataframe as input and
    returns one (most likely, a subset of the original) as output. See
    `drop_voxels_with_any_negative_amplitudes` for an example.

    test_set_stimulus_class: list of ints or None. What subset of the stimulus_class should be
    considered the test set. these are numbers between 0 and 47 (inclusive) and then the test
    dataset will include data from those stimulus classes (train dataset will use the rest). this
    is used for cross-validation purposes (i.e., train on 0 through 46, test on 47). If None, will
    not have a test set, and will train on all data.

    bootstrap_num: list of ints or None. What subset of bootstrap_num we should fit the model
    to. Must be set if the first level results dataframe's df_mode "full", must not be set if its
    df_mode is "summary".

    save_path_stem: string or None. a string to save the trained model and loss_df at (should have
    no extension because we'll add it ourselves). If None, will not save the output.

    """
    # when we fit the model, we want to randomly initialize its starting parameters (for a given
    # seed) in order to help avoid local optima.
    if random_seed is not None:
        np.random.seed(int(random_seed))
    # all the parameters are bounded below by 0. they're not bounded above by anything. However,
    # they will probably be small, so we use a max of 1 (things get weird when the orientation
    # effect parameters get too large).
    param_inits = np.random.uniform(0, 1, 11)
    model = LogGaussianDonut(model_period_orientation_type, model_eccentricity_type,
                             model_amplitude_orientation_type, *param_inits)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1 and batch_size > torch.cuda.device_count():
        model = torch.nn.DataParallel(model)
    model.to(device)
    if test_set_stimulus_class is None:
        dataset = FirstLevelDataset(first_level_results_path, device, df_filter,
                                    bootstrap_num=bootstrap_num)
        print("Beginning training!")
        # use all stimulus classes
        model, loss_df, results_df, model_history_df = train_model(
            model, dataset, max_epochs, batch_size, train_thresh, learning_rate, save_path_stem,
            loss_func)
        test_subset = 'none'
    else:
        df = pd.read_csv(first_level_results_path)
        all_stimulus_class = df.stimulus_class.unique()
        train_set_stimulus_class = [i for i in all_stimulus_class if i not in
                                    test_set_stimulus_class]
        # split into test and train
        train_dataset = FirstLevelDataset(first_level_results_path, device, df_filter,
                                          train_set_stimulus_class, bootstrap_num)
        test_subset = test_set_stimulus_class
        print("Beginning training, treating stimulus classes %s as test set!" % test_subset)
        model, loss_df, results_df, model_history_df = train_model(
            model, train_dataset, max_epochs, batch_size, train_thresh, learning_rate,
            save_path_stem, loss_func, True)
    for name, metadata in zip(['test_subset', 'bootstrap_num'], [test_subset, bootstrap_num]):
        metadata_str = str(metadata).replace('[', '').replace(']', '')
        if len(metadata_str) == 1:
            metadata_str = int(metadata_str)
        results_df[name] = metadata_str
        loss_df[name] = metadata_str
        model_history_df[name] = metadata_str
    print("Finished training!")
    model.eval()
    if test_set_stimulus_class is None and bootstrap_num is None:
        save_outputs(model, loss_df, results_df, model_history_df, save_path_stem)
    else:
        save_outputs(model, loss_df, None, model_history_df, save_path_stem)
    return model, loss_df, results_df, model_history_df


def construct_df_filter(df_filter_string):
    """construct df_filter from string (as used in our command-line parser)

    the string should be a single string containing at least one of the following, separated by
    commas: 'drop_voxels_with_any_negative_amplitudes', 'drop_voxels_near_border',
    'reduce_num_voxels:n' (where n is an integer), 'None'. This will then construct the function
    that will chain them together in the order specified (if None is one of the entries, we will
    simply return None)

    """
    df_filters = []
    for f in df_filter_string.split(','):
        # this is a little bit weird, but it does what we want
        if f == 'drop_voxels_with_any_negative_amplitudes':
            df_filters.append(drop_voxels_with_any_negative_amplitudes)
        elif f == 'drop_voxels_with_mean_negative_amplitudes':
            df_filters.append(drop_voxels_with_mean_negative_amplitudes)
        elif f == 'drop_voxels_near_border':
            df_filters.append(drop_voxels_near_border)
        elif f == 'None' or f == 'none':
            df_filters = [None]
            break
        elif f.startswith('reduce_num_voxels:'):
            n_voxels = int(f.split(':')[-1])
            df_filters.append(lambda x: reduce_num_voxels(x, n_voxels))
        elif f.startswith('randomly_reduce_num_voxels:'):
            n_voxels = int(f.split(':')[-1])
            df_filters.append(lambda x: randomly_reduce_num_voxels(x, n_voxels))
        elif f.startswith('restrict_to_part_of_visual_field:'):
            location = f.split(':')[-1]
            df_filters.append(lambda x: restrict_to_part_of_visual_field(x, location))
        else:
            raise Exception("Don't know what to do with df_filter %s" % f)
    if len(df_filters) > 1:
        # from
        # https://stackoverflow.com/questions/11736407/apply-list-of-functions-on-an-object-in-python#11736719
        # and in python 3, reduce is replaced with functools.reduce
        df_filter = lambda x: functools.reduce(lambda o, func: func(o), df_filters, x)
    else:
        df_filter = df_filters[0]
    return df_filter


class NewLinesHelpFormatter(argparse.HelpFormatter):
    # add empty line if help ends with \n
    def _split_lines(self, text, width):
        text = text.split('\n')
        lines = []
        for t in text:
            lines.extend(super()._split_lines(t, width))
        return lines


if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                          NewLinesHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=("Load in the first level results Dataframe and train a 2d tuning model on it"
                     ". Will save the model parameters and loss information."))
    parser.add_argument("model_period_orientation_type",
                        help=("{iso, absolute, relative, full}\nEffect of orientation on "
                              "preferred period\n- iso: model is isotropic, "
                              "predictions identical for all orientations.\n- absolute: model can"
                              " fit differences in absolute orientation, that is, in Cartesian "
                              "coordinates, such that sf_angle=0 correponds to 'to the right'\n- "
                              "relative: model can fit differences in relative orientation, that "
                              "is, in retinal polar coordinates, such that sf_angle=0 corresponds"
                              " to 'away from the fovea'\n- full: model can fit differences in "
                              "both absolute and relative orientations"))
    parser.add_argument("model_eccentricity_type",
                        help=("{scaling, constant, full}\n- scaling: model's relationship between"
                              " preferred period and eccentricity is exactly scaling, that is, the"
                              " preferred period is equal to the eccentricity.\n- constant: model'"
                              "s relationship between preferred period and eccentricity is exactly"
                              " constant, that is, it does not change with eccentricity but is "
                              "flat.\n- full: model discovers the relationship between "
                              "eccentricity and preferred period, though it is constrained to be"
                              " linear (i.e., model solves for a and b in period = a * "
                              "eccentricity + b)"))
    parser.add_argument("model_amplitude_orientation_type",
                        help=("{iso, absolute, relative, full}\nEffect of orientation on "
                              "max_amplitude\n- iso: model is isotropic, "
                              "predictions identical for all orientations.\n- absolute: model can"
                              " fit differences in absolute orientation, that is, in Cartesian "
                              "coordinates, such that sf_angle=0 correponds to 'to the right'\n- "
                              "relative: model can fit differences in relative orientation, that "
                              "is, in retinal polar coordinates, such that sf_angle=0 corresponds"
                              " to 'away from the fovea'\n- full: model can fit differences in "
                              "both absolute and relative orientations"))
    parser.add_argument("first_level_results_path",
                        help=("Path to the first level results dataframe containing the data to "
                              "fit."))
    parser.add_argument("save_path_stem",
                        help=("Path stem (no extension) where we'll save the results: model state "
                              " dict (`save_path_stem`_model.pt), loss dataframe "
                              "(`save_path_stem`_loss.csv), and first level dataframe with "
                              "performance  (`save_path_stem`_results_df.csv)"))
    parser.add_argument("--random_seed", default=None,
                        help=("we initialize the model with random parameters in order to try and"
                              " avoid local optima. we set the seed before generating all those "
                              "random numbers. If not specified, then we don't set it."))
    parser.add_argument("--train_thresh", '-t', default=1e-6, type=float,
                        help=("How little the loss can change with successive epochs to be "
                              "considered done training."))
    parser.add_argument("--df_filter", '-d', default='drop_voxels_with_any_negative_amplitudes',
                        help=("{'drop_voxels_near_border', 'drop_voxels_with_any_negative_amplitudes',"
                              " 'drop_voxels_with_any_negative_amplitudes',"
                              " 'reduce_num_voxels:n', 'randomly_reduce_num_voxels:n', 'restrict_"
                              "to_part_of_visual_field:loc', 'None'}. How to filter the first "
                              "level dataframe. Can be multiple of these, separated by a comma, in"
                              " which case they will be chained in the order provided (so the "
                              "first one will be applied to the dataframe first). If 'drop_voxels_"
                              "near_border', will drop all voxels whose pRF center is one sigma "
                              "away from the stimulus borders. If 'drop_voxels_with_negative_"
                              "amplitudes', drop any voxel that has a negative response amplitude."
                              " If 'reduce_num_voxels:n', will drop all but the first n voxels. If"
                              " 'randomly_reduce_num_voxels:n', will randomly drop voxels so we "
                              "end up with n. If 'restrict_to_part_of_visual_field:loc', will drop"
                              " all voxels outside of loc, which must be one of {'upper', 'lower',"
                              " 'left', 'right', 'inner', 'outer'}. If 'None', fit on all data ("
                              "obviously, this cannot be chained with any of the others)."))
    parser.add_argument("--batch_size", "-b", default=10, type=int,
                        help=("Size of the batches for training"))
    parser.add_argument("--max_epochs", '-e', default=100, type=int,
                        help=("Maximum number of training epochs (full runs through the data)"))
    parser.add_argument("--learning_rate", '-r', default=1e-3, type=float,
                        help=("Learning rate for Adam optimizer (should change inversely with "
                              "batch size)."))
    parser.add_argument("--test_set_stimulus_class", '-c', default=None, nargs='+',
                        help=("Which stimulus class(es) to consider part of the test set. should "
                              "probably only be one, but should work if you pass more than one as "
                              "well."))
    parser.add_argument("--bootstrap_num", '-n', default=None, nargs='+',
                        help=("What subset of bootstrap_num we should fit the model to. Must be "
                              "set if the first level results dataframe's df_mode 'full', must not"
                              " be set if its df_mode is 'summary'."))
    args = vars(parser.parse_args())
    # test_set_stimulus_class can be either None or some ints. argparse will hand us a list, so we
    # have to parse it appropriately
    test_set_stimulus_class = args.pop('test_set_stimulus_class')
    try:
        test_set_stimulus_class = [int(i) for i in test_set_stimulus_class]
    except (TypeError, ValueError):
        # in this case, we can't cast one of the strs in the list to an int, so we assume it must
        # just contain None.
        test_set_stimulus_class = None
    bootstrap_num = args.pop('bootstrap_num')
    try:
        bootstrap_num = [int(i) for i in bootstrap_num]
    except (TypeError, ValueError):
        # in this case, we can't cast one of the strs in the list to an int, so we assume it must
        # just contain None.
        bootstrap_num = None
    df_filter = construct_df_filter(args.pop('df_filter'))
    main(test_set_stimulus_class=test_set_stimulus_class, bootstrap_num=bootstrap_num,
         df_filter=df_filter, **args)
