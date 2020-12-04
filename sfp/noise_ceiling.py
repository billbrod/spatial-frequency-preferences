"""stuff to help with computing the noise ceiling
"""
import matplotlib as mpl
# we do this because sometimes we run this without an X-server, and this backend doesn't need
# one. We set warn=False because the notebook uses a different backend and will spout out a big
# warning to that effect; that's unnecessarily alarming, so we hide it.
mpl.use('svg', warn=False)
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from torch.utils import data as torchdata
from . import model as sfp_model


def sample_df(df, seed=0,
              df_filter_string='drop_voxels_with_negative_amplitudes,drop_voxels_near_border',
              is_simulated=False):
    """Sample df to get info for necessary computing Monte Carlo noise ceiling

    This is the df we use to compute the monte carlo noise ceiling,
    where we're comparing the amplitude estimates computed for different
    bootstraps on the full data. We pick two bootstraps (without
    replacement), and query the dataframe to grab only these
    bootstraps. One of them becomes the feature and one the
    target. Because this uses all the data, it does not need to be
    corrected to compare against our actual models. In this file, we
    also provide functionality to compute the split-half noise ceiling,
    where we're comparing the amplitude estimates computed on two
    separate halves of the data (which does need a correction).

    Parameters
    ----------
    df : pd.DataFrame
        The full df created by first_level_analysis for the GLMdenoise
        run on all the data
    seed : int
        random seed to use (used to set numpy's RNG)
    df_filter_string : str or None, optional
        a str specifying how to filter the voxels in the dataset. see
        the docstrings for sfp.model.FirstLevelDataset and
        sfp.model.construct_df_filter for more details. If None, we
        won't filter. Should probably use the default, which is what all
        models are trained using.
    is_simulated : bool, optional
        Whether this is simulated data or actual data (changes which columns we
        merge on).

    Returns
    -------
    df : pd.DataFrame
        re-sampled dataframe with one row per (voxel, stimulus) pair,
        where row has two values for the columns: bootstrap_num,
        amplitude_estimate, amplitude_estimate_norm, and
        amplitude_estimate_normed (with suffixes "_1" and "_2"), where
        the two values come from two separate bootstraps (all
        bootstrap_num_1 vals will be identical, as will all
        bootstrap_num_2).

    """
    if df_filter_string is not None:
        df_filter = sfp_model.construct_df_filter(df_filter_string)
        df = df_filter(df).reset_index()
    np.random.seed(seed)
    bootstraps = np.random.choice(100, 2, False)
    tmp = [df.query("bootstrap_num == @b") for b in bootstraps]
    # then combine_dfs
    if not is_simulated:
        cols = ['varea', 'voxel', 'stimulus_superclass', 'w_r', 'w_a', 'eccen', 'angle',
                'stimulus_class', 'hemi', 'sigma', 'prf_vexpl', 'phi', 'res', 'stimulus_index',
                'freq_space_angle', 'freq_space_distance', 'rounded_freq_space_distance', 'local_w_x',
                'local_w_y', 'local_w_r', 'local_w_a', 'local_sf_magnitude', 'local_sf_xy_direction',
                'local_sf_ra_direction', 'precision', 'baseline', 'GLM_R2']
    else:
        cols = ['varea', 'voxel', 'eccen', 'angle', 'stimulus_class',
                'local_sf_magnitude', 'local_sf_xy_direction', 'noise_level',
                'noise_source_df', 'period_orientation_type', 'eccentricity_type',
                'amplitude_orientation_type']
        cols += [c for c in df.columns if c.startswith('true_m')]
    df = pd.merge(*tmp, on=cols, suffixes=['_1', '_2'], validate='1:1')
    df['noise_ceiling_seed'] = seed
    return df


def combine_dfs(first_half, second_half, all_data):
    """combine dfs to get all the info necessary for computing split-half noise ceiling

    This is the df we use to compute the split-half noise ceiling, where
    we're comparing the amplitude estimates computed on two separate
    halves of the data. Split-half noise ceiling needs a correction to
    account for the fact that, unlike the actual models we fit, it only
    uses half the data. In this file, we also provide functionality to
    compute the Monte Carlo noise ceiling, where we sample from the
    existing distribution of amplitude estimates we get when fitting
    GLMdenoise to the full dataset (because we use bootstraps across
    runs to estimate the variability of the amplitude estimates). Monte
    Carlo noise ceiling does not need this dataset-size correction

    We want our dataset to only take a single df as the input, as our
    FirstLevelDataset does. However, the info required for this analysis
    is contained in several different dfs, so this function combines
    them. We want the two amplitude estimates (from the split halves)
    and then we want the precision from the data fit to all the data.

    We merge the two halves, combining them on the various identifying
    columns (varea, voxel, stimulus_class, frequency info related to the
    stimuli), and keeping the ones related to the GLM fit separate
    (amplitude_estimate, precision, etc.); these will all have the
    suffix "_1" (from first_half) and "_2" (from second_half). We add a
    new column, 'overall_precision', which contains the precision from
    all_data

    Parameters
    ----------
    first_half : pd.DataFrame
        The summary df created by first_level_analysis for the
        GLMdenoise run on one half of the runs
    second_half : pd.DataFrame
        The summary df created by first_level_analysis for the
        GLMdenoise run on the second half of the runs
    all_data : pd.DataFrame
        The summary df created by first_level_analysis for the
        GLMdenoise run on all runs

    Returns
    -------
    df : pd.DataFrame
        The merged dataframe

    """
    cols = ['varea', 'voxel', 'stimulus_superclass', 'w_r', 'w_a', 'eccen', 'angle',
            'stimulus_class', 'hemi', 'sigma', 'prf_vexpl', 'phi', 'res', 'stimulus_index',
            'freq_space_angle', 'freq_space_distance', 'rounded_freq_space_distance', 'local_w_x',
            'local_w_y', 'local_w_r', 'local_w_a', 'local_sf_magnitude', 'local_sf_xy_direction',
            'local_sf_ra_direction']
    if sorted(first_half.voxel.unique()) != sorted(second_half.voxel.unique()):
        raise Exception("the two dataframes must have same stimulus classes!")
    if sorted(first_half.stimulus_class.unique()) != sorted(second_half.stimulus_class.unique()):
        raise Exception("the two dataframes must have same voxels!")
    df = pd.merge(first_half, second_half, on=cols, suffixes=['_1', '_2'], validate='1:1')
    df = df.set_index('voxel')
    all_data = all_data.set_index('voxel')
    df['overall_precision'] = all_data['precision']
    return df


class NoiseCeilingDataset(torchdata.Dataset):
    """Dataset for computing noise ceiling

    the __getitem__ method here returns all (48) values for a single
    voxel, so keep that in mind when setting batch size. this is done
    because (in the loss function) we normalize the predictions and
    target so that that vector of length 48 has a norm of one.

    In addition the features and targets, we also return the precision

    This dataset returns two sets estimates of the amplitude, as given
    by GLMdenoise; one set form the features, and the second (along with
    the precision) form the target. There are two modes for this
    dataset, depending on the input df (this is saved as the attribute
    ds.mode):
    
    - 'split_half': df created by combine_dfs(). Then features are the
      amplitude estimates from GLMdenoise fit to one half the data and
      the targets are the amplitude estimates from GLMdenoise fit to the
      other half (without replacement). The precision is from GLMdenoise
      fit to all the data.

    - 'monte_carlo': df created by sample_df(). Then features are
      amplitude estimate from one bootstrap from GLMdenoise fit to all
      data, and targets are from another bootstrap (selected without
      replacement so they will never be the same bootstrap; each (voxel,
      stimulus) pair is selected independently). The precision is from
      GLMdenoise fit to all the data.

    Parameters
    ----------
    df : str or pd.DataFrame
        the df or the path to the df to use for this dataset, as created
        by sfp.noise_ceiling.combine_dfs or sfp.noise_ceiling.sample_dfs
    device : str or torch.device
        the device this dataset should live, either cpu or a specific
        gpu
    df_filter : function or None, optional. 
        If not None, a function that takes a dataframe as input and
        returns one (most likely, a subset of the original) as
        output. See `drop_voxels_with_negative_amplitudes` for an
        example.

    """
    def __init__(self, df, device, df_filter=None,):
        try:
            df_path = df
            df = pd.read_csv(df)
        except ValueError:
            df_path = None
            pass
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
        if 'bootstrap_num' in df.columns:
            raise Exception("Either this should be the split-half df, in which case it must be "
                            "computed on the summarized df, which has no 'bootstrap_num' columns, "
                            "or this should be the monte carlo df, in which case it must have "
                            "'bootstrap_num_1' and 'bootstrap_num_2' columns!")
        if df.empty:
            raise Exception("Dataframe is empty!")
        self.df = df.reset_index()
        self.device = device
        self.df_path = df_path
        self.stimulus_class = sorted(df.stimulus_class.unique())
        self.bootstrap_num = None
        if 'overall_precision' in df.columns:
            self.mode = 'split_half'
        else:
            self.mode = 'monte_carlo'

    def get_single_item(self, idx):
        row = self.df.iloc[idx]
        try:
            # this is for the split-half noise ceiling
            feature = row[['amplitude_estimate_median_1']].values.astype(float)
            target = row[['amplitude_estimate_median_2', 'overall_precision']].values.astype(float)
        except KeyError:
            # this is for the Monte Carlo noise ceiling
            feature = row[['amplitude_estimate_1']].values.astype(float)
            target = row[['amplitude_estimate_2', 'precision']].values.astype(float)
        feature = sfp_model._cast_as_tensor(feature)
        target = sfp_model._cast_as_tensor(target)
        return feature.to(self.device), target.to(self.device)

    def __getitem__(self, idx):
        vox_idx = self.df[self.df.voxel_reindexed == idx].index
        return self.get_single_item(vox_idx)

    def get_voxel(self, idx):
        vox_idx = self.df[self.df.voxel == idx].index
        return self.get_single_item(vox_idx)

    def __len__(self):
        return self.df.voxel.nunique()


class NoiseCeiling(torch.nn.Module):
    """simple linear model for computing the noise ceiling

    This is the simplest possible model: we're just trying to fit the
    line giving the relationship between the amplitudes estimated from
    two halves of the data. If they were identical, then slope=1, and
    intercept=0. 

    Our model predicts that:

    Y = slope * X + intercept

    where Y is the amplitudes estimated from the second half of the
    dataset, X is those from the first, and slope and intercept are the
    two (scalar) parameters.

    On initialization, either parameter can be set to None, in which
    case they will be drawn from a uniform distribution between 0 and 1.

    Default parameters predict that the two split halves are identical,
    which is probably what you want to use

    Model parameters
    ----------------
    slope : float
        the slope of the linear relationship between the amplitude
        estimates from the two halves
    intercept : float
        the intercept of the linear relationship between the amplitude
        estimates from the two halves

    """
    def __init__(self, slope=1, intercept=0):
        super().__init__()
        if slope is None:
            slope = torch.rand(1)[0]
        if intercept is None:
            intercept = torch.rand(1)[0]
        self.slope = sfp_model._cast_as_param(slope)
        self.intercept = sfp_model._cast_as_param(intercept)
        self.model_type = 'noise_ceiling'

    def __str__(self):
        return (f"NoiseCeiling({self.slope:.03f} X + {self.intercept:.03f})")

    def __repr__(self):
        return self.__str__()

    def evaluate(self, first_half):
        """generate predictions for second_half from first_half
        """
        return self.slope * first_half + self.intercept

    def forward(self, inputs):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.evaluate(inputs.select(-1, 0))


def plot_noise_ceiling_model(model, df, overall_loss=None):
    """Plot model's predictions with the data

    this just creates a simple scatterplot with the amplitudes estimated
    from the first half (the features) on the x-axis, and those from the
    second half (the targets) on the y-axis, with a red dashed line
    showing the prediction of model for this range of values

    Parameters
    ----------
    model : NoiseCeiling
        A trained sfp.noise_ceiling.NoiseCeiling model
    df : pd.DataFrame
        The dataframe created by sfp.noise_ceiling.combined_dfs, which
        contains the columns amplitude_estiamte_median_1 and
        amplitude_estiamte_median_2
    overall_loss : float or None, optional
        The overall loss of this model, as computed by
        get_overall_loss(). If not None, will add to the plot. If None,
        will not.

    Returns
    -------
    fig : plt.Figure
        figure containing the plot

    """
    if 'amplitude_estimate_1' in df.columns:
        ampl_col_name = 'amplitude_estimate'
    else:
        ampl_col_name = 'amplitude_estimate_median'
    ax = sns.scatterplot(f'{ampl_col_name}_1', f'{ampl_col_name}_2', data=df)
    x = np.linspace(df[f'{ampl_col_name}_1'].min(), df[f'{ampl_col_name}_1'].max(),
                    1000)
    ax.plot(x, model.slope.detach().numpy() * x + model.intercept.detach().numpy(), 'r--')
    ax.set_title(f'Predictions for {model}')
    ax.axhline(color='gray', linestyle='dashed')
    ax.axvline(color='gray', linestyle='dashed')
    text = ""
    if overall_loss is not None:
        text += f'Overall loss:\n{overall_loss:.05f}\n\n'
    text += (f"Seed: {df.noise_ceiling_seed.unique()[0]}\nBootstraps: "
             f"[{df.bootstrap_num_1.unique()[0]}, {df.bootstrap_num_2.unique()[0]}]")
    ax.text(1.01, .5, text, transform=ax.transAxes, va='center')
    return ax.figure


def get_overall_loss(model, ds):
    """Compute the loss of model on the full dataset

    This computes the loss of model on the full dataset and is used to
    get a final sense of how well the model performed

    Parameters
    ----------
    model : sfp.noise_ceiling.NoiseCeiling
        A trained sfp.noise_ceiling.NoiseCeiling() model
    ds : sfp.noise_ceiling.NoiseCeilingDataset
        The dataset to evaluate the model on

    Returns
    -------
    loss : torch.tensor
        single-element tensor containing the loss of the model on the
        full dataset

    """
    dl = torchdata.DataLoader(ds, len(ds))
    features, targets = next(iter(dl))
    return sfp_model.weighted_normed_loss(model(features), targets)


def split_half(df_path, save_stem, seed=0, batch_size=10, learning_rate=.1, max_epochs=100, gpus=0):
    """find the split-half noise ceiling for a single scanning session and save the output

    In addition to the standard sfp_model outputs, we also save a figure
    showing the predictions and loss of the final noise ceiling model

    The outputs will be saved at `save_stem` plus the following
    suffixes: "_loss.csv", "_results_df.csv", "_model.pt",
    "_model_history.csv", "_predictions.png"

    Parameters
    ----------
    df_path : str
        The path where the merged df is saved (as created by
        sfp.noise_ceiling.combine_dfs)
    save_stem : str
        the stem of the path to save things at (i.e., should not end in
        the extension)
    seed : int, optional
        random seed to use (used to set both torch and numpy's RNG)
    batch_size : int, optional
        The batch size for training the model (in number of voxels)
    learning_rate : float, optional
        The learning rate for the optimization algorithm
    max_epochs : int, optional
        The number of epochs to train for
    gpus : {0, 1}, optional
        How many gpus to use

    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpus == 1:
        device = torch.device('cuda:0')
    elif gpus == 0:
        device = torch.device('cpu')
    else:
        raise Exception(f"Only 0 and 1 gpus supported right now, not {gpus}")
    ds = NoiseCeilingDataset(df_path, device)
    model = NoiseCeiling(None, None).to(device)
    model, loss, results, history = sfp_model.train_model(model, ds, max_epochs, batch_size,
                                                          learning_rate=learning_rate,
                                                          save_path_stem=save_stem)
    model.eval()
    sfp_model.save_outputs(model, loss, results, history, save_stem)

    overall_loss = get_overall_loss(model, ds)
    with sns.axes_style('white'):
        fig = plot_noise_ceiling_model(model, pd.read_csv(df_path), overall_loss.item())
        fig.savefig(save_stem+"_predictions.png", bbox_inches='tight')


def monte_carlo(df, save_stem, **metadata):
    """find the Monte Carlo noise ceiling for a single scanning session and save the output

    Note that this doesn't involve training the model at all, we simply
    see whether the two values are identical (i.e., we use the model
    NoiseCeiling(1, 0)).

    Because we don't train the model, the outputs are a little different:

    - save_stem+"_loss.csv" is a single-row pd.Dataframe containing the
      loss and the values passed as metadata

    - save_stem+"_predictions.png" is a a figure showing the predictions
      and loss of the final noise ceiling model.

    Parameters
    ----------
    df : pd.DataFrame
        The sampled df (as created by sfp.noise_ceiling.sample_dfs)
    save_stem : str
        the stem of the path to save things at (i.e., should not end in
        the extension)
    metadata:
        Extra key=value pairs to add to the loss.csv output

    """
    device = torch.device('cpu')
    orig_df = df.copy(deep=True)
    ds = NoiseCeilingDataset(df, device)
    model = NoiseCeiling(1, 0).to(device)
    model.eval()
    overall_loss = get_overall_loss(model, ds)
    metadata['loss'] = overall_loss.item()
    loss_df = pd.DataFrame(metadata, index=[0])
    loss_df.to_csv(save_stem + "_loss.csv", index=False)

    with sns.axes_style('white'):
        fig = plot_noise_ceiling_model(model, orig_df, overall_loss.item())
        fig.savefig(save_stem+"_predictions.png", bbox_inches='tight')
