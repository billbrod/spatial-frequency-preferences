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


def _cast_as_tensor(x):
    if type(x) == pd.Series:
        x = x.values
    return torch.tensor(x, dtype=torch.float64)


def _cast_as_param(x, requires_grad=True):
    return torch.nn.Parameter(_cast_as_tensor(x), requires_grad=requires_grad)


class FirstLevelDataset(torchdata.Dataset):
    """Dataset for first level results

    the __getitem__ method here returns all (48) values for a single voxel, so keep that in mind
    when setting batch size. this is done because (in the loss function) we normalize the
    predictions and target so that that vector of length 48 has a norm of one.

    In addition the features and targets, we also return the precision

    df_filter: function or None. If not None, a function that takes a dataframe as input and
    returns one (most likely, a subset of the original) as output. See `drop_voxels_with_negative_amplitudes`
    for an example.
    """
    def __init__(self, df_path, device, direction_type='absolute', df_filter=None, normed=True):
        df = pd.read_csv(df_path)
        if df_filter is not None:
            # we want the index to be reset so we can use iloc in __getitem__ below. this ensures
            # that iloc and loc will return the same thing, which isn't otherwise the case. and we
            # want them to be the same because Dataloader assumes iloc but our custom get_voxel
            # needs loc.
            self.df = df_filter(df).reset_index()
        else:
            self.df = df
        if direction_type not in ['relative', 'absolute']:
            raise Exception("Don't know how to handle direction_type %s!" % direction_type)
        self.direction_type = direction_type
        self.device = device
        self.normed = normed
        self.df_path = df_path
        
    def get_single_item(self, idx):
        row = self.df.iloc[idx]
        if self.direction_type == 'relative':
            vals = row[['local_sf_magnitude', 'local_sf_ra_direction', 'eccen', 'angle']].values
        elif self.direction_type == 'absolute':
            vals = row[['local_sf_magnitude', 'local_sf_xy_direction', 'eccen', 'angle']].values
        feature = _cast_as_tensor(vals.astype(float))
        if self.normed:
            try:
                target = _cast_as_tensor(row['amplitude_estimate_normed'])
            except KeyError:
                target = _cast_as_tensor(row['amplitude_estimate_median_normed'])
        else:
            try:
                target = _cast_as_tensor(row['amplitude_estimate'])
            except KeyError:
                target = _cast_as_tensor(row['amplitude_estimate_median'])
        precision = _cast_as_tensor(row['precision'])
        return (feature.to(self.device), target.to(self.device), precision.to(self.device))
    
    def __getitem__(self, idx):
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
    """
    def __init__(self, amplitude, mode, sigma, sf_ecc_slope=1, sf_ecc_intercept=0,
                 train_sf_ecc_slope=True, train_sf_ecc_intercept=True):
        super(LogGaussianDonut,self).__init__()
        # we don't train the amplitude because our loss function is amplitude-independent
        self.amplitude = _cast_as_param(amplitude, requires_grad=False)
        self.mode = _cast_as_param(mode)
        self.sigma = _cast_as_param(sigma)
        self.sf_ecc_slope = _cast_as_param(sf_ecc_slope, train_sf_ecc_slope)
        self.sf_ecc_intercept = _cast_as_param(sf_ecc_intercept, train_sf_ecc_intercept)
        self.model_type = 'full_donut'

    def __str__(self):
        # so we can see the parameters
        return "{0}({1:.03f}, {2:.03f}, {3:.03f}, {4:.03f}, {5:.03f})".format(
            type(self).__name__, self.amplitude, self.mode, self.sigma, self.sf_ecc_slope,
            self.sf_ecc_intercept)

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
    
    def log_norm_pdf_1d(self, x):
        """the pdf of a one-dimensional log normal distribution, with a scale factor

        we parameterize this using the mode instead of mu because we place constraints on the mode
        during optimization

        this is the same function as sfp.tuning_curves.log_norm_pdf, but in pytorch (and without
        the normalization by the max; the function in tuning_curves is always called on a range of
        x values, so that's fine -- this one is occasionally called on a single value, so it's not)
        """
        mu = torch.log(self.mode) + torch.pow(self.sigma, 2)
        pdf = (1/(x*self.sigma*np.sqrt(2*np.pi))) * torch.exp(-(torch.log(x)-mu)**2/(2*self.sigma**2))
        return self.amplitude * pdf

    def evaluate(self, sf_mag, sf_angle, vox_ecc, vox_angle):
        variables = {'sf_mag': sf_mag, 'sf_angle': sf_angle, 'vox_ecc': vox_ecc, 'vox_angle': vox_angle}
        # this is messy
        for k, v in variables.iteritems():
            if not torch.is_tensor(v):
                v = torch.tensor(v, dtype=torch.float64)
            if self.amplitude.is_cuda:
                v = v.cuda()
            variables[k] = v
        relative_freq = variables['sf_mag'] * (self.sf_ecc_slope * variables['vox_ecc'] + self.sf_ecc_intercept)
        relative_freq = torch.clamp(relative_freq, min=1e-6)
        return self.log_norm_pdf_1d(relative_freq)

    def forward(self, spatial_frequency_magnitude, spatial_frequency_theta, voxel_eccentricity, voxel_angle):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.evaluate(spatial_frequency_magnitude, spatial_frequency_theta, voxel_eccentricity, voxel_angle)


class ConstantLogGaussianDonut(LogGaussianDonut):
    """Instantiation of the "constant" extreme possibility

    this version does not depend on voxel eccentricity or angle at all"""
    def __init__(self, amplitude, mode, sigma):
        # this way the "relative frequency" is sf_mag * (0*voxel_ecc + 1) = sf_mag
        super(ConstantLogGaussianDonut, self).__init__(amplitude, mode, sigma, 0, 1, False, False)
        self.model_type = 'constant_donut'
        
    def create_image(self, extent=None, n_samps=1001):
        r, th = self._create_mag_angle(extent, n_samps)
        return self.evaluate(r, th)
        
    def evaluate(self, sf_mag, sf_angle):
        return super(ConstantLogGaussianDonut, self).evaluate(sf_mag, sf_angle, 0, 0)
        
    def forward(self, spatial_frequency_magnitude, spatial_frequency_theta, voxel_eccentricity=None,
                voxel_angle=None):
        # we keep the same kwargs so that forward can be called the same way as the regular
        # LogGaussianDonut, but we do nothing with them.
        return self.evaluate(spatial_frequency_magnitude, spatial_frequency_theta)


class ScalingLogGaussianDonut(LogGaussianDonut):
    """Instantiation of the "scaling" extreme possibility

    in this version, spatial frequency preferences scale *exactly* with eccentricity, so as to
    cancel out the scaling done in our stimuli creation
    """
    def __init__(self, amplitude, mode, sigma):
        # this way the "relative frequency" is sf_mag * (1*voxel_ecc + 0) = sf_mag * voxel_ecc
        super(ScalingLogGaussianDonut, self).__init__(amplitude, mode, sigma, 1, 0, False, False)
        self.model_type = 'scaling_donut'


def show_image(donut, voxel_eccentricity=1, voxel_angle=0, extent=(-5, 5), n_samps=1001,
               cmap="Reds", **kwargs):
    """wrapper function to plot the image from a given donut

    This shows the spatial frequency selectivity implied by the donut at a given voxel eccentricity
    and angle, if appropriate (eccentricity and angle ignored if donut is ConstantLogGuassianDonut)

    donut: a LogGaussianDonut

    extent: 2-tuple of floats. the range of spatial frequencies to visualize `(min, max)`. this
    will be the same for x and y
    """
    if isinstance(donut, ConstantLogGaussianDonut):
        ax = plt.imshow(donut.create_image(extent, n_samps=n_samps).detach(),
                        extent=(extent[0], extent[1], extent[0], extent[1]), cmap=cmap, **kwargs)
    else:
        ax = plt.imshow(donut.create_image(voxel_eccentricity, voxel_angle, extent, n_samps=n_samps).detach(),
                        extent=(extent[0], extent[1], extent[0], extent[1]), cmap=cmap, **kwargs)
        
    plt.colorbar()
    return ax


def construct_loss_df(loss_history):
    """constructs loss dataframe from array of lost history

    loss_history: 2d list or array, as constructed in `train_model`, n_epochs by batch_size
    """
    loss_df = pd.DataFrame(np.array(loss_history))
    loss_df = pd.melt(loss_df.reset_index(), id_vars='index', var_name='batch_num',
                      value_name='loss')
    return loss_df.rename(columns={'index': 'epoch_num'})


def check_performance(trained_model, dataset):
    """check performance of trained_model for each voxel in the dataset

    this assumes both model and dataset are on the same device
    """
    performance = []
    for i in dataset.df.voxel.unique():
        features, targets, precision = dataset[i]
        predictions = trained_model(*features.transpose(1, 0))
        corr = np.corrcoef(targets.cpu().detach().numpy(), predictions.cpu().detach().numpy())
        loss = weighted_normed_loss(predictions, targets, precision).item()
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


def construct_dfs(model, dataset, loss_history, max_epochs, batch_size, learning_rate, train_thresh,
                  current_epoch, normalize_voxels):
    """construct the loss and results dataframes and add metadata
    """
    loss_df = construct_loss_df(loss_history)
    loss_df['max_epochs'] = max_epochs
    loss_df['batch_size'] = batch_size
    loss_df['learning_rate'] = learning_rate
    loss_df['train_thresh'] = train_thresh
    loss_df['epochs_trained'] = current_epoch
    # we reload the first level dataframe because the one in dataset may be filtered in some way
    results_df = combine_first_level_df_with_performance(pd.read_csv(dataset.df_path),
                                                         check_performance(model, dataset))
    if type(model) == torch.nn.DataParallel:
        # in this case, we need to access model.module in order to get the various custom
        # attributes we set in our LogGaussianDonut
        model = model.module
    results_df['fit_model_type'] = model.model_type
    loss_df['fit_model_type'] = model.model_type
    loss_df['predict_normalized_voxels'] = normalize_voxels
    # this is the case if the data is simulated
    for col in ['true_model_type', 'noise_level', 'noise_source_df']:
        if col in results_df.columns:
            loss_df[col] = results_df[col].unique()[0]
    for name, val in model.named_parameters():
        results_df['fit_model_%s'%name] = val.cpu().detach().numpy()
    results_df['epochs_trained'] = current_epoch
    results_df['batch_size'] = batch_size
    results_df['learning_rate'] = learning_rate
    results_df['predict_normalized_voxels'] = normalize_voxels
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


def weighted_normed_loss(predictions, target, precision):
    """takes in the predictions and target

    note all of these must be tensors, not numpy arrays
    """
    # we occasionally have an issue where the predictions are really small (like 1e-200), which
    # gives us a norm of 0 and thus a normed_predictions of infinity, and thus an infinite loss.
    # the point of renorming is that multiplying by a scale factor won't change our loss, so we do
    # that here to avoid this issue
    if 0 in predictions.norm(2, -1, True):
        predictions = predictions * 1e100
    # we norm / average along the last dimension, since that means we do it across all stimulus
    # classes for a given voxel. we don't know whether these tensors will be 1d (single voxel, as
    # returned by our FirstLevelDataset) or 2d (multiple voxels, as returned by the DataLoader)
    normed_predictions = predictions / predictions.norm(2, -1, True)
    normed_target = target / target.norm(2, -1, True)
    # this isn't really necessary (all the values along that dimension should be identical, based
    # on how we calculated it), but just in case. and this gets it in the right shape
    precision = precision.mean(-1, True)
    squared_error = precision * (normed_predictions - normed_target)**2
    return squared_error.mean()


def train_model(model, dataset, max_epochs=5, batch_size=1, train_thresh=1e-8,
                learning_rate=1e-2, save_path_stem=None, normalize_voxels=False):
    """train the model
    """
    # AMSGrad argument here means we use a revised version that handles a bug people found where
    # it doesn't necessarily converge
    training_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(training_parameters, lr=learning_rate, amsgrad=True)    
    dataloader = torchdata.DataLoader(dataset, batch_size,)# shuffle=True)
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
                print("Loss is nan or inf on epoch %s, batch %s! We won't update parameters on this batch"% (t, i))
                print("Predictions are: %s" % predictions.detach())
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (t % 100) == 0:
            loss_df, results_df = construct_dfs(model, dataset, loss_history, max_epochs,
                                                batch_size, learning_rate, train_thresh, t,
                                                normalize_voxels)
            save_outputs(model, loss_df, results_df, save_path_stem)
        print("Average loss on epoch %s: %s" % (t, np.mean(loss_history[-1])))
        print(model)
        if len(loss_history) > 3:
            if ((np.abs(np.mean(loss_history[-1]) - np.mean(loss_history[-2])) < train_thresh) and
                (np.abs(np.mean(loss_history[-2]) - np.mean(loss_history[-3])) < train_thresh) and
                (np.abs(np.mean(loss_history[-3]) - np.mean(loss_history[-4])) < train_thresh)):
                print("Epoch loss appears to have stopped declining, so we stop training")
                break
    loss_df, results_df = construct_dfs(model, dataset, loss_history, max_epochs, batch_size, learning_rate,
                                        train_thresh, t, normalize_voxels)
    return model, loss_df, results_df


def main(model_type, first_level_results_path, max_epochs=100, train_thresh=1e-8, batch_size=1,
         df_filter=None, learning_rate=1e-2, save_path_stem="pytorch", normalize_voxels=False):
    """create, train, and save a model on the given first_level_results dataframe

    model_type: {'full', 'scaling', 'constant'}. Which type of model to train. 'full' is the
    LogGaussianDonut that can train its sf_ecc_intercept and sf_ecc_slope, while 'scaling' and
    'constant' have those two parameters set (at 0,1 and 1,0, respectively).

    max_epochs: int. the max number of epochs to let the training run for. otherwise, we train
    until the loss changes by less than train_thresh for 3 epochs in a row.
    
    df_filter: function or None. If not None, a function that takes a dataframe as input and
    returns one (most likely, a subset of the original) as output. See
    `drop_voxels_with_negative_amplitudes` for an example.

    save_path_stem: string or None. a string to save the trained model and loss_df at (should have
    no extension because we'll add it ourselves). If None, will not save the output.
    """
    if model_type == 'full':
        model = LogGaussianDonut(1, 2, .4)
    elif model_type == 'constant':
        model = ConstantLogGaussianDonut(1, 2, .4)
    elif model_type == 'scaling':
        model = ScalingLogGaussianDonut(1, 2, .4)
    else:
        raise Exception("Don't know how to handle model_type %s!" % model_type)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1 and batch_size > torch.cuda.device_count():
        model = torch.nn.DataParallel(model)
    model.to(device)
    dataset = FirstLevelDataset(first_level_results_path, device, df_filter=df_filter,
                                normed=normalize_voxels)
    print("Beginning training!")
    model, loss_df, results_df = train_model(model, dataset, max_epochs, batch_size, train_thresh,
                                             learning_rate, save_path_stem, normalize_voxels)
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
                        help=("{'full', 'scaling', 'constant'}. Which type of model to train"))
    parser.add_argument("first_level_results_path",
                        help=("Path to the first level results dataframe containing the data to "
                              "fit."))
    parser.add_argument("save_path_stem",
                        help=("Path stem (no extension) where we'll save the results: model state "
                              " dict (`save_path_stem`_model.pt), loss dataframe "
                              "(`save_path_stem`_loss.csv), and first level dataframe with "
                              "performance  (`save_path_stem`_model_df.csv)"))
    parser.add_argument("--train_thresh", '-t', default=.1, type=float,
                        help=("How little the loss can change with successive epochs to be "
                              "considered done training."))
    parser.add_argument("--df_filter", '-d', default='drop_voxels_with_negative_amplitudes',
                        help=("{'drop_voxels_with_negative_amplitudes', 'reduce_num_voxels:n', 'None'}."
                              " How to filter the first level dataframe. Can be multiple of these,"
                              " separated by a comma, in which case they will be chained in the "
                              "order provided (so the first one will be applied to the dataframe "
                              "first). If 'drop_voxels_with_negative_amplitudes', "
                              "drop any voxel that has a negative response amplitude. If "
                              "'reduce_num_voxels:n', will drop all but the first n voxels. If "
                              "'None', fit on all data (obviously, this cannot be chained with any"
                              " of the others)."))
    parser.add_argument("--batch_size", "-b", default=2000, type=int,
                        help=("Size of the batches for training"))
    parser.add_argument("--max_epochs", '-e', default=100, type=int,
                        help=("Maximum number of training epochs (full runs through the data)"))
    parser.add_argument("--learning_rate", '-r', default=1e-3, type=float,
                        help=("Learning rate for Adam optimizer (should change inversely with "
                              "batch size)."))
    parser.add_argument("--normalize_voxels", '-n', action='store_true',
                        help=("Whether to normalize the voxel responses or not"))
    args = vars(parser.parse_args())
    df_filter = construct_df_filter(args.pop('df_filter'))
    main(df_filter=df_filter, **args)
