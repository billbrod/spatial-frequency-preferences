#!/usr/bin/python
"""2d tuning model
"""
import matplotlib as mpl
# we do this because sometimes we run this without an X-server, and this backend doesn't need one
mpl.use('svg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
import argparse
from torch.utils import data as torchdata


def reduce_num_voxels(df, n_voxels=200):
    """drop all but the first n_voxels

    this is just to speed things up for testing, it obviously shouldn't be used if you are actually
    trying to fit the data
    """
    return df[df.voxel < n_voxels]


def drop_voxels_with_negative_amplitudes(df):
    """drop all voxels that have at least one negative amplitude
    """
    try:
        df = df.groupby('voxel').filter(lambda x: (x.amplitude_estimate_normed>=0).all())
    except AttributeError:
        df = df.groupby('voxel').filter(lambda x: (x.amplitude_estimate_median_normed>=0).all())
    return df


def drop_voxels_near_border(df, inner_border=.96, outer_border=12):
    """drop all voxels whose pRF center is one sigma away form the border

    where the sigma is the sigma of the Gaussian pRF
    """
    df = df.groupby('voxel').filter(lambda x: (x.eccen + x.sigma <= outer_border).all())
    df = df.groupby('voxel').filter(lambda x: (x.eccen - x.sigma >= inner_border).all())
    return df


def _cast_as_tensor(x):
    if type(x) == pd.Series:
        x = x.values
    return torch.tensor(x, dtype=torch.float64)


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
    `drop_voxels_with_negative_amplitudes` for an example.

    stimulus_class: list of ints or None. What subset of the stimulus_class should be used. these
    are numbers between 0 and 47 (inclusive) and then the dataset will only include data from those
    stimulus classes. this is used for cross-validation purposes (i.e., train on 0 through 46, test
    on 47).
    """
    def __init__(self, df_path, device, df_filter=None, stimulus_class=None):
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
            df = df[df.stimulus_class.isin(stimulus_class)]
        if df.empty:
            raise Exception("Dataframe is empty!")
        self.df = df.reset_index()
        self.device = device
        self.df_path = df_path
        self.stimulus_class = df.stimulus_class.unique()
        
    def get_single_item(self, idx):
        row = self.df.iloc[idx]
        vals = row[['local_sf_magnitude', 'local_sf_xy_direction', 'eccen', 'angle']].values
        feature = _cast_as_tensor(vals.astype(float))
        try:
            target = _cast_as_tensor(row['amplitude_estimate'])
        except KeyError:
            target = _cast_as_tensor(row['amplitude_estimate_median'])
        precision = _cast_as_tensor(row['precision'])
        return (feature.to(self.device), target.to(self.device), precision.to(self.device))

    def __getitem__(self, idx):
        vox_idx = self.df[self.df.voxel_reindexed==idx].index
        return self.get_single_item(vox_idx)

    def get_voxel(self, idx):
        vox_idx = self.df[self.df.voxel==idx].index
        return self.get_single_item(vox_idx)

    def __len__(self):
        return self.df.voxel.nunique()


def torch_meshgrid(x, y=None):
    """from https://github.com/pytorch/pytorch/issues/7580"""
    if y is None:
        y = x
    x = torch.tensor(x, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    m, n = x.size(0), y.size(0)
    grid_x = x[None].expand(n, m)
    grid_y = y[:, None].expand(n, m)
    return grid_x, grid_y


class LogGaussianDonut(torch.nn.Module):
    """simple LogGaussianDonut in pytorch

    when you call this model, sf_angle should be the (absolute) orientation of the grating, so that
    sf_angle=0 corresponds to "to the right". if you want the model's prediction to be based on the
    relative orientation, set `orientation_type` parameter to 'relative', and then the angle will
    be remapped internally. NOTE THAT YOU CONTINUE TO CALL THIS MODEL WITH THE ABSOLUTE
    ORIENTATION.
    """
    def __init__(self, sigma, sf_ecc_slope=1, sf_ecc_intercept=0, mode_cardinals=0, mode_obliques=0,
                 amplitude_cardinals=0, amplitude_obliques=0, orientation_type='absolute',
                 train_sf_ecc_slope=True, train_sf_ecc_intercept=True, train_mode_cardinals=True,
                 train_mode_obliques=True, train_amplitude_cardinals=True,
                 train_amplitude_obliques=True):
        super(LogGaussianDonut,self).__init__()
        self.amplitude_cardinals = _cast_as_param(amplitude_cardinals, train_amplitude_cardinals)
        self.amplitude_obliques = _cast_as_param(amplitude_obliques, train_amplitude_obliques)
        self.sigma = _cast_as_param(sigma)
        self.sf_ecc_slope = _cast_as_param(sf_ecc_slope, train_sf_ecc_slope)
        self.sf_ecc_intercept = _cast_as_param(sf_ecc_intercept, train_sf_ecc_intercept)
        self.mode_cardinals = _cast_as_param(mode_cardinals, train_mode_cardinals)
        self.mode_obliques = _cast_as_param(mode_obliques, train_mode_obliques)
        if orientation_type not in ['relative', 'absolute']:
            raise Exception("Don't know how to handle orientation_type %s!" % orientation_type)
        self.orientation_type = orientation_type
        self.model_type = 'full_donut_%s' % orientation_type

    def __str__(self):
        # so we can see the parameters
        return "{0}({1:.03f}, {2:.03f}, {3:.03f}, {4:.03f}, {5:.03f}, {6:.03f}, {7:.03f}, {8})".format(
            type(self).__name__, self.sigma, self.sf_ecc_slope, self.sf_ecc_intercept,
            self.mode_cardinals, self.mode_obliques, self.amplitude_cardinals,
            self.amplitude_obliques, self.orientation_type)

    def __repr__(self):
        return self.__str__()

    def _create_mag_angle(self, extent=(-10, 10), n_samps=1001):
        x = torch.linspace(extent[0], extent[1], n_samps)
        x, y = torch_meshgrid(x)
        r = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))
        th = torch.atan2(y, x)
        return r, th
    
    def create_image(self, vox_ecc, vox_angle, extent=None, n_samps=1001):
        r, th = self._create_mag_angle(extent, n_samps)
        return self.evaluate(r, th, vox_ecc, vox_angle)
    
    def preferred_period(self, sf_angle, vox_ecc, vox_angle):
        """return preferred period for specified voxel at given orientation
        """
        sf_angle, vox_ecc, vox_angle = _cast_args_as_tensors([sf_angle, vox_ecc, vox_angle],
                                                             self.sigma.is_cuda)
        # we can allow up to two of these variables to be non-singletons.
        if sf_angle.ndimension() == 1 and vox_ecc.ndimension()==1 and vox_angle.ndimension()==1:
            # if this is False, then all of them are the same shape and we have no issues
            if sf_angle.shape != vox_ecc.shape != vox_angle.shape:
                raise Exception("Only two of these variables can be non-singletons!")
        else:
            sf_angle, vox_ecc = _check_and_reshape_tensors(sf_angle, vox_ecc)
            vox_ecc, vox_angle = _check_and_reshape_tensors(vox_ecc, vox_angle)
            sf_angle, vox_angle = _check_and_reshape_tensors(sf_angle, vox_angle)
        if self.orientation_type == 'relative':
            sf_angle = sf_angle - vox_angle
        eccentricity_effect = self.sf_ecc_slope * vox_ecc + self.sf_ecc_intercept
        orientation_effect = (1 + self.mode_cardinals * torch.cos(2 * sf_angle) +
                              self.mode_obliques * torch.cos(4 * sf_angle))
        return torch.clamp(eccentricity_effect * orientation_effect, min=1e-6)
        
    def preferred_sf(self, sf_angle, vox_ecc, vox_angle):
        return 1. / self.preferred_period(sf_angle, vox_ecc, vox_angle)
    
    def max_amplitude(self, sf_angle, vox_angle):
        sf_angle, vox_angle = _cast_args_as_tensors([sf_angle, vox_angle], self.sigma.is_cuda)
        sf_angle, vox_angle = _check_and_reshape_tensors(sf_angle, vox_angle)
        if self.orientation_type == 'relative':
            sf_angle = sf_angle - vox_angle
        amplitude = (1 + self.amplitude_cardinals * torch.cos(2*sf_angle) +
                     self.amplitude_obliques * torch.cos(4*sf_angle))
        return torch.clamp(amplitude, min=1e-6)
    
    def evaluate(self, sf_mag, sf_angle, vox_ecc, vox_angle):
        sf_mag, = _cast_args_as_tensors([sf_mag], self.sigma.is_cuda)
        # if ecc_effect is 0 or below, then log2(ecc_effect) is infinity or undefined
        # (respectively). to avoid that, we clamp it 1e-6. in practice, if a voxel ends up here
        # that means the model predicts 0 response for it.
        preferred_period = self.preferred_period(sf_angle, vox_ecc, vox_angle)
        pdf = torch.exp(-((torch.log2(sf_mag) + torch.log2(preferred_period))**2)/ (2*self.sigma**2))
        amplitude = self.max_amplitude(sf_angle, vox_angle)
        return amplitude * pdf

    def forward(self, spatial_frequency_magnitude, spatial_frequency_theta, voxel_eccentricity,
                voxel_angle):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.evaluate(spatial_frequency_magnitude, spatial_frequency_theta,
                             voxel_eccentricity, voxel_angle)


class ConstantIsoLogGaussianDonut(LogGaussianDonut):
    """Instantiation of the "constant" isotropic extreme possibility

    this version does not depend on voxel eccentricity or angle at all"""
    def __init__(self, sigma):
        super(ConstantIsoLogGaussianDonut, self).__init__(sigma, 0, 1, 0, 0, 0, 0, 'absolute', False,
                                                          False, False, False, False, False)
        self.model_type = 'constant_iso_donut'

    def create_image(self, extent=None, n_samps=1001):
        r, th = self._create_mag_angle(extent, n_samps)
        return self.evaluate(r, th)

    def evaluate(self, sf_mag, sf_angle):
        return super(ConstantIsoLogGaussianDonut, self).evaluate(sf_mag, sf_angle, 0, 0)
        
    def forward(self, spatial_frequency_magnitude, spatial_frequency_theta, voxel_eccentricity=None,
                voxel_angle=None):
        # we keep the same kwargs so that forward can be called the same way as the regular
        # LogGaussianDonut, but we do nothing with them.
        return self.evaluate(spatial_frequency_magnitude, spatial_frequency_theta)


class ScalingIsoLogGaussianDonut(LogGaussianDonut):
    """Instantiation of the "scaling" isotropic extreme possibility

    in this version, spatial frequency preferences scale *exactly* with eccentricity, so as to
    cancel out the scaling done in our stimuli creation
    """
    def __init__(self, sigma):
        super(ScalingIsoLogGaussianDonut, self).__init__(sigma, 1, 0, 0, 0, 0, 0, 'absolute', False,
                                                         False, False, False, False, False)
        self.model_type = 'scaling_iso_donut'


class IsoLogGaussianDonut(LogGaussianDonut):
    """Instantiation of the "full" isotropic donut

    In this version, there's no dependence on orientation / voxel angle, but we can learn how the
    preferences scale with eccentricity
    """
    def __init__(self, sigma, sf_ecc_slope=1, sf_ecc_intercept=0):
        super(IsoLogGaussianDonut, self).__init__(sigma, sf_ecc_slope, sf_ecc_intercept, 0, 0, 0,
                                                  0, 'absolute', True, True, False, False, False,
                                                  False)
        self.model_type = 'full_iso_donut'


def show_image(donut, voxel_eccentricity=1, voxel_angle=0, extent=(-5, 5), n_samps=1001,
               cmap="Reds", **kwargs):
    """wrapper function to plot the image from a given donut

    This shows the spatial frequency selectivity implied by the donut at a given voxel eccentricity
    and angle, if appropriate (eccentricity and angle ignored if donut is ConstantLogGuassianDonut)

    donut: a LogGaussianDonut

    extent: 2-tuple of floats. the range of spatial frequencies to visualize `(min, max)`. this
    will be the same for x and y
    """
    if isinstance(donut, ConstantIsoLogGaussianDonut):
        ax = plt.imshow(donut.create_image(extent, n_samps=n_samps).detach(),
                        extent=(extent[0], extent[1], extent[0], extent[1]), cmap=cmap, **kwargs)
    else:
        ax = plt.imshow(donut.create_image(voxel_eccentricity, voxel_angle, extent, n_samps=n_samps).detach(),
                        extent=(extent[0], extent[1], extent[0], extent[1]), cmap=cmap, **kwargs)
        
    plt.colorbar()
    return ax


def construct_loss_df(loss_history, subset='train'):
    """constructs loss dataframe from array of lost history

    loss_history: 2d list or array, as constructed in `train_model`, n_epochs by batch_size
    """
    loss_df = pd.DataFrame(np.array(loss_history))
    loss_df = pd.melt(loss_df.reset_index(), id_vars='index', var_name='batch_num',
                      value_name='loss')
    loss_df['data_subset'] = subset
    return loss_df.rename(columns={'index': 'epoch_num'})


def check_performance(trained_model, dataset, test_dataset=None):
    """check performance of trained_model for each voxel in the dataset

    this assumes both model and dataset are on the same device
    """
    performance = []
    for i in dataset.df.voxel.unique():
        features, targets, precision = dataset.get_voxel(i)
        predictions = trained_model(*features.transpose(1, 0))
        if test_dataset is not None:
            test_features, test_target, test_precision = test_dataset.get_voxel(i)
            test_predictions = trained_model(*test_features.transpose(1, 0))
            cv_loss = weighted_normed_loss(test_predictions, test_target, test_precision,
                                           torch.cat([test_predictions, predictions]),
                                           torch.cat([test_target, targets])).item()
        else:
            cv_loss = None
        corr = np.corrcoef(targets.cpu().detach().numpy(), predictions.cpu().detach().numpy())
        loss = weighted_normed_loss(predictions, targets, precision).item()
        performance.append(pd.DataFrame({'voxel': i, 'stimulus_class': range(len(targets)),
                                         'model_prediction_correlation': corr[0, 1],
                                         'model_prediction_loss': loss,
                                         'model_predictions': predictions.cpu().detach().numpy(),
                                         'model_prediction_cv_loss': cv_loss}))
    return pd.concat(performance)


def combine_first_level_df_with_performance(first_level_df, performance_df):
    """combine results_df and performance_df, along the voxel, producing results_df
    """
    results_df = first_level_df.set_index(['voxel', 'stimulus_class'])
    performance_df = performance_df.set_index(['voxel', 'stimulus_class'])
    results_df = results_df.join(performance_df).reset_index()
    return results_df.reset_index()    


def construct_dfs(model, dataset, train_loss_history, max_epochs, batch_size, learning_rate,
                  train_thresh, current_epoch, test_loss_history=None, test_dataset=None):
    """construct the loss and results dataframes and add metadata
    """
    loss_df = construct_loss_df(train_loss_history)
    if test_loss_history is not None:
        loss_df = pd.concat([loss_df, construct_loss_df(test_loss_history, 'test')])
    loss_df['max_epochs'] = max_epochs
    loss_df['batch_size'] = batch_size
    loss_df['learning_rate'] = learning_rate
    loss_df['train_thresh'] = train_thresh
    loss_df['epochs_trained'] = current_epoch
    # we reload the first level dataframe because the one in dataset may be filtered in some way
    results_df = combine_first_level_df_with_performance(pd.read_csv(dataset.df_path),
                                                         check_performance(model, dataset,
                                                                           test_dataset))
    if type(model) == torch.nn.DataParallel:
        # in this case, we need to access model.module in order to get the various custom
        # attributes we set in our LogGaussianDonut
        model = model.module
    results_df['fit_model_type'] = model.model_type
    loss_df['fit_model_type'] = model.model_type
    # this is the case if the data is simulated
    for col in ['true_model_type', 'noise_level', 'noise_source_df']:
        if col in results_df.columns:
            loss_df[col] = results_df[col].unique()[0]
    for name, val in model.named_parameters():
        results_df['fit_model_%s'%name] = val.cpu().detach().numpy()
    results_df['epochs_trained'] = current_epoch
    results_df['batch_size'] = batch_size
    results_df['learning_rate'] = learning_rate
    return loss_df, results_df    


def save_outputs(model, loss_df, results_df, save_path_stem):
    """save outputs (if save_path_stem is not None)
    """
    if type(model) == torch.nn.DataParallel:
        # in this case, we need to access model.module in order to just save the model
        model = model.module
    if save_path_stem is not None:
        torch.save(model.state_dict(), save_path_stem + "_model.pt")
        loss_df.to_csv(save_path_stem + "_loss.csv", index=False)
        results_df.to_csv(save_path_stem + "_model_df.csv", index=False)


def weighted_normed_loss(predictions, target, precision, predictions_for_norm=None,
                         target_for_norm=None):
    """takes in the predictions and target

    note all of these must be tensors, not numpy arrays

    predictions_for_norm, target_for_norm: normally, this should be called such that predictions
    and target each contain all the values for the voxels investigated. however, during
    cross-validation, predictions and target will contain a subset of the stimulus classes, so we
    need to pass the predictions and targets for all stimulus classes as well in order to normalize
    them properly (for an intuition as to why this is important, consider the extreme case: if both
    predictions and target have length 1 and are normalized with respect to themselves, the loss
    will always be 0)
    """
    # we occasionally have an issue where the predictions are really small (like 1e-200), which
    # gives us a norm of 0 and thus a normed_predictions of infinity, and thus an infinite loss.
    # the point of renorming is that multiplying by a scale factor won't change our loss, so we do
    # that here to avoid this issue
    if 0 in predictions.norm(2, -1, True):
        predictions = predictions * 1e100
    if predictions_for_norm is None:
        assert target_for_norm is None, "Both target_for_norm and predictions_for_norm must be unset"
        predictions_for_norm = predictions
        target_for_norm = target
    # we norm / average along the last dimension, since that means we do it across all stimulus
    # classes for a given voxel. we don't know whether these tensors will be 1d (single voxel, as
    # returned by our FirstLevelDataset) or 2d (multiple voxels, as returned by the DataLoader)
    normed_predictions = predictions / predictions_for_norm.norm(2, -1, True)
    normed_target = target / target_for_norm.norm(2, -1, True)
    # this isn't really necessary (all the values along that dimension should be identical, based
    # on how we calculated it), but just in case. and this gets it in the right shape
    precision = precision.mean(-1, True)
    squared_error = precision * (normed_predictions - normed_target)**2
    return squared_error.mean()


def train_model(model, dataset, max_epochs=5, batch_size=1, train_thresh=1e-8,
                learning_rate=1e-2, save_path_stem=None):
    """train the model
    """
    training_parameters = [p for p in model.parameters() if p.requires_grad]
    # AMSGrad argument here means we use a revised version that handles a bug people found where
    # it doesn't necessarily converge
    optimizer = torch.optim.Adam(training_parameters, lr=learning_rate, amsgrad=True)    
    dataloader = torchdata.DataLoader(dataset, batch_size)
    loss_history = []
    for t in range(max_epochs):
        loss_history.append([])
        for i, (features, target, precision) in enumerate(dataloader):
            # these transposes get features from the dimensions (voxels, stimulus class, features)
            # into (features, voxels, stimulus class) so that the predictions are shape (voxels,
            # stimulus class), just like the targets are
            predictions = model(*features.transpose(2, 0).transpose(2, 1))
            loss = weighted_normed_loss(predictions, target, precision)
            loss_history[t].append(loss.item())
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                print("Loss is nan or inf on epoch %s, batch %s! We won't update parameters on "
                      "this batch"% (t, i))
                print("Predictions are: %s" % predictions.detach())
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (t % 100) == 0:
            loss_df, results_df = construct_dfs(model, dataset, loss_history, max_epochs,
                                                batch_size, learning_rate, train_thresh, t)
            save_outputs(model, loss_df, results_df, save_path_stem)
        print("Average loss on epoch %s: %s" % (t, np.mean(loss_history[-1])))
        print(model)
        if len(loss_history) > 3:
            if ((np.abs(np.mean(loss_history[-1]) - np.mean(loss_history[-2])) < train_thresh) and
                (np.abs(np.mean(loss_history[-2]) - np.mean(loss_history[-3])) < train_thresh) and
                (np.abs(np.mean(loss_history[-3]) - np.mean(loss_history[-4])) < train_thresh)):
                print("Epoch loss appears to have stopped declining, so we stop training")
                break
    loss_df, results_df = construct_dfs(model, dataset, loss_history, max_epochs, batch_size,
                                        learning_rate, train_thresh, t)
    return model, loss_df, results_df


def train_model_traintest(model, train_dataset, test_dataset, full_dataset, max_epochs=5,
                          batch_size=1, train_thresh=1e-8, learning_rate=1e-2, save_path_stem=None):
    """train the model with separate train and test sets
    """
    training_parameters = [p for p in model.parameters() if p.requires_grad]
    # AMSGrad argument here means we use a revised version that handles a bug people found where
    # it doesn't necessarily converge
    optimizer = torch.optim.Adam(training_parameters, lr=learning_rate, amsgrad=True)    
    train_dataloader = torchdata.DataLoader(train_dataset, batch_size)
    test_dataloader = torchdata.DataLoader(test_dataset, batch_size)
    train_loss_history = []
    test_loss_history = []
    for t in range(max_epochs):
        train_loss_history.append([])
        for i, (train_stuff, test_stuff) in enumerate(zip(train_dataloader, test_dataloader)):
            features, target, precision = train_stuff
            test_features, test_target, _ = test_stuff
            # these transposes get features from the dimensions (voxels, stimulus class, features)
            # into (features, voxels, stimulus class) so that the predictions are shape (voxels,
            # stimulus class), just like the targets are
            predictions = model(*features.transpose(2, 0).transpose(2, 1))
            test_predictions = model(*test_features.transpose(2, 0).transpose(2, 1))
            loss = weighted_normed_loss(predictions, target, precision,
                                        torch.cat([test_predictions, predictions], 1),
                                        torch.cat([test_target, target], 1))
            train_loss_history[t].append(loss.item())
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                print("Loss is nan or inf on epoch %s, batch %s! We won't update parameters on "
                      "this batch"% (t, i))
                print("Predictions are: %s" % predictions.detach())
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        test_loss_history.append([])
        for v in test_dataset.df.voxel.unique():
            test_features, test_target, test_precision = test_dataset.get_voxel(v)
            train_features, train_target, _ = train_dataset.get_voxel(v)
            test_predictions = model(*test_features.transpose(1, 0))
            train_predictions = model(*train_features.transpose(1, 0))
            loss = weighted_normed_loss(test_predictions, test_target, test_precision,
                                        torch.cat([test_predictions, train_predictions]),
                                        torch.cat([test_target, train_target]))
            test_loss_history[t].append(loss.item())
        model.train()
        if (t % 100) == 0:
            loss_df, results_df = construct_dfs(model, full_dataset, train_loss_history, max_epochs,
                                                batch_size, learning_rate, train_thresh, t,
                                                test_loss_history, test_dataset)
            save_outputs(model, loss_df, results_df, save_path_stem)
        print("Average train loss on epoch %s: %s" % (t, np.mean(train_loss_history[-1])))
        print("Average test loss on epoch %s: %s" % (t, np.mean(test_loss_history[-1])))
        print(model)
        if len(train_loss_history) > 3:
            if ((np.abs(np.mean(train_loss_history[-1]) - np.mean(train_loss_history[-2])) < train_thresh) and
                (np.abs(np.mean(train_loss_history[-2]) - np.mean(train_loss_history[-3])) < train_thresh) and
                (np.abs(np.mean(train_loss_history[-3]) - np.mean(train_loss_history[-4])) < train_thresh)):
                print("Training loss appears to have stopped declining, so we stop training")
                break
    loss_df, results_df = construct_dfs(model, full_dataset, train_loss_history, max_epochs,
                                        batch_size, learning_rate, train_thresh, t,
                                        test_loss_history, test_dataset)
    return model, loss_df, results_df


def main(model_type, first_level_results_path, max_epochs=100, train_thresh=1e-8, batch_size=1,
         df_filter=None, learning_rate=1e-2, stimulus_class=None, save_path_stem="pytorch"):
    """create, train, and save a model on the given first_level_results dataframe

    model_type: {'full-absolute', 'full-relative', 'iso', 'scaling', 'constant'}. Which type of
    model to train. 'full_abslute' and 'full_relative' fit all parameters and so include the
    effects of orientation on the amplitude and mode; they differ in whether they consider
    orientation to be absolute (so that orientation=0 means "to the right") or relative (so that
    orientation=0 means "away from the fovea"). The other three models do not consider
    orientation. 'iso' is the LogGaussianDonut that can train its sf_ecc_intercept and
    sf_ecc_slope, while 'scaling' and 'constant' have those two parameters set (at 0,1 and 1,0,
    respectively).

    max_epochs: int. the max number of epochs to let the training run for. otherwise, we train
    until the loss changes by less than train_thresh for 3 epochs in a row.
    
    df_filter: function or None. If not None, a function that takes a dataframe as input and
    returns one (most likely, a subset of the original) as output. See
    `drop_voxels_with_negative_amplitudes` for an example.

    stimulus_class: list of ints or None. What subset of the stimulus_class should be used. these
    are numbers between 0 and 47 (inclusive) and then the dataset will only include data from those
    stimulus classes. this is used for cross-validation purposes (i.e., train on 0 through 46, test
    on 47).

    save_path_stem: string or None. a string to save the trained model and loss_df at (should have
    no extension because we'll add it ourselves). If None, will not save the output.
    """
    if model_type == 'full-absolute':
        model = LogGaussianDonut(.4, orientation_type='absolute')
    elif model_type == 'full-relative':
        model = LogGaussianDonut(.4, orientation_type='relative')
    elif model_type == 'constant':
        model = ConstantIsoLogGaussianDonut(.4)
    elif model_type == 'scaling':
        model = ScalingIsoLogGaussianDonut(.4)
    elif model_type == 'iso':
        model = IsoLogGaussianDonut(.4)
    else:
        raise Exception("Don't know how to handle model_type %s!" % model_type)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1 and batch_size > torch.cuda.device_count():
        model = torch.nn.DataParallel(model)
    model.to(device)
    dataset = FirstLevelDataset(first_level_results_path, device, df_filter)
    if stimulus_class is None:
        print("Beginning training!")
        # use all stimulus classes
        model, loss_df, results_df = train_model(model, dataset, max_epochs, batch_size,
                                                 train_thresh, learning_rate, save_path_stem)
        test_subset = 'none'
    else:
        df = pd.read_csv(first_level_results_path)
        all_stimulus_class = df.stimulus_class.unique()
        other_stimulus_class = [i for i in all_stimulus_class if i not in stimulus_class]
        # split into test and train
        if len(other_stimulus_class) > len(stimulus_class):
            # we assume that test set should be smaller than train
            train_dataset = FirstLevelDataset(first_level_results_path, device, df_filter,
                                              other_stimulus_class)
            test_dataset = FirstLevelDataset(first_level_results_path, device, df_filter,
                                             stimulus_class)
            test_subset = stimulus_class
        else:
            train_dataset = FirstLevelDataset(first_level_results_path, device, df_filter,
                                              stimulus_class)
            test_dataset = FirstLevelDataset(first_level_results_path, device, df_filter,
                                             other_stimulus_class)
            test_subset = other_stimulus_class
        print("Beginning training, treating stimulus classes %s as test "
              "set!"%test_subset)
        model, loss_df, results_df = train_model_traintest(model, train_dataset, test_dataset,
                                                           dataset, max_epochs, batch_size,
                                                           train_thresh, learning_rate, save_path_stem)
    test_subset = str(test_subset).replace('[', '').replace(']', '')
    if len(test_subset) == 1:
        test_subset = int(test_subset)
    results_df['test_subset'] = test_subset
    loss_df['test_subset'] = test_subset
    print("Finished training!")
    save_outputs(model, loss_df, results_df, save_path_stem)
    model.eval()
    return model, loss_df, results_df


def construct_df_filter(df_filter_string):
    """construct df_filter from string (as used in our command-line parser)

    the string should be a single string containing at least one of the following, separated by
    commas: 'drop_voxels_with_negative_amplitudes', 'reduce_num_voxels:n' (where n is an integer),
    'None'. This will then construct the function that will chain them together in the order
    specified (if None is one of the entries, we will simply return None)
    """
    df_filters = []    
    for f in df_filter_string.split(','):
        # this is a little bit weird, but it does what we want
        if f == 'drop_voxels_with_negative_amplitudes':
            df_filters.append(drop_voxels_with_negative_amplitudes)
        elif f == 'drop_voxels_near_border':
            df_filters.append(drop_voxels_near_border)
        elif f == 'None' or f == 'none':
            df_filters = [None]
            break
        elif f.startswith('reduce_num_voxels:'):
            n_voxels = int(f.split(':')[-1])
            df_filters.append(lambda x: reduce_num_voxels(x, n_voxels))
        else:
            raise Exception("Don't know what to do with df_filter %s" % f)
    if len(df_filters) > 1:
        # from
        # https://stackoverflow.com/questions/11736407/apply-list-of-functions-on-an-object-in-python#11736719
        df_filter = lambda x: reduce(lambda o, func: func(o), df_filters, x)
    else:
        df_filter = df_filters[0]
    return df_filter


if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=("Load in the first level results Dataframe and train a 2d tuning model on it"
                     ". Will save the model parameters and loss information."))
    parser.add_argument("model_type",
                        help=("{'full-absolute', 'full-relative', 'scaling', 'constant', 'iso'}."
                              " Which type of model to train"))
    parser.add_argument("first_level_results_path",
                        help=("Path to the first level results dataframe containing the data to "
                              "fit."))
    parser.add_argument("save_path_stem",
                        help=("Path stem (no extension) where we'll save the results: model state "
                              " dict (`save_path_stem`_model.pt), loss dataframe "
                              "(`save_path_stem`_loss.csv), and first level dataframe with "
                              "performance  (`save_path_stem`_model_df.csv)"))
    parser.add_argument("--train_thresh", '-t', default=1e-8, type=float,
                        help=("How little the loss can change with successive epochs to be "
                              "considered done training."))
    parser.add_argument("--df_filter", '-d', default='drop_voxels_with_negative_amplitudes',
                        help=("{'drop_voxels_near_border', 'drop_voxels_with_negative_amplitudes',"
                              " 'reduce_num_voxels:n', 'None'}."
                              " How to filter the first level dataframe. Can be multiple of these,"
                              " separated by a comma, in which case they will be chained in the "
                              "order provided (so the first one will be applied to the dataframe "
                              "first). If 'drop_voxels_with_negative_amplitudes', "
                              "drop any voxel that has a negative response amplitude. If "
                              "'reduce_num_voxels:n', will drop all but the first n voxels. If "
                              "'None', fit on all data (obviously, this cannot be chained with any"
                              " of the others)."))
    parser.add_argument("--batch_size", "-b", default=1, type=int,
                        help=("Size of the batches for training"))
    parser.add_argument("--max_epochs", '-e', default=100, type=int,
                        help=("Maximum number of training epochs (full runs through the data)"))
    parser.add_argument("--learning_rate", '-r', default=1e-2, type=float,
                        help=("Learning rate for Adam optimizer (should change inversely with "
                              "batch size)."))
    parser.add_argument("--stimulus_class", '-s', type=int, default=None, nargs='+',
                        help=("Which stimulus class(es) to consider part of the test set. should "
                              "probably only be one, but should work if you pass more than one as "
                              "well"))
    args = vars(parser.parse_args())
    df_filter = construct_df_filter(args.pop('df_filter'))
    main(df_filter=df_filter, **args)
