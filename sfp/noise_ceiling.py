"""stuff to help with computing the noise ceiling
"""
import matplotlib as mpl
# we do this because sometimes we run this without an X-server, and this backend doesn't need
# one. We set warn=False because the notebook uses a different backend and will spout out a big
# warning to that effect; that's unnecessarily alarming, so we hide it.
mpl.use('svg', warn=False)
import torch
import pandas as pd
from torch.utils import data as torchdata
from . import model as sfp_model


def combine_dfs(first_half, second_half, all_data):
    """combine dfs to get all the info necessary for computing noise ceiling

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
        The df created by first_level_analysis for the GLMdenoise run on
        one half of the runs
    second_half : pd.DataFrame
        The df created by first_level_analysis for the GLMdenoise run on
        the second half of the runs
    all_data : pd.DataFrame
        The df created by first_level_analysis for the GLMdenoise run on
        all runs

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

    For this dataset, the features are the amplitude estimates from
    GLMdenoise fit to one half the data and the targets are the
    amplitude estimates from GLMdenoise fit to the other half (without
    replacement). The precision is from GLMdenoise fit to all the data.

    Parameters
    ----------
    df_path : str
        path to the df to use for this dataset, as created by
        sfp.noise_ceiling.combine_dfs
    device : str or torch.device
        the device this dataset should live, either cpu or a specific
        gpu
    df_filter : function or None, optional. 
        If not None, a function that takes a dataframe as input and
        returns one (most likely, a subset of the original) as
        output. See `drop_voxels_with_negative_amplitudes` for an
        example.

    """
    def __init__(self, df_path, device, df_filter=None,):
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
        if 'bootstrap_num' in df.columns:
            raise Exception("We only work on the 'summarized' dataset, which doesn't have individual bootstraps!")
        if df.empty:
            raise Exception("Dataframe is empty!")
        self.df = df.reset_index()
        self.device = device
        self.df_path = df_path
        self.stimulus_class = sorted(df.stimulus_class.unique())
        self.bootstrap_num = None

    def get_single_item(self, idx):
        row = self.df.iloc[idx]
        feature = row[['amplitude_estimate_median_1']].values.astype(float)
        feature = sfp_model._cast_as_tensor(feature)
        target = row[['amplitude_estimate_median_2', 'overall_precision']].values.astype(float)
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

    Model parameters
    ----------------
    slope : float
        the slope of the linear relationship between the amplitude
        estimates from the two halves
    intercept : float
        the intercept of the linear relationship between the amplitude
        estimates from the two halves

    """
    def __init__(self):
        super().__init__()
        self.slope = sfp_model._cast_as_param(torch.rand(1)[0])
        self.intercept = sfp_model._cast_as_param(torch.rand(1)[0])
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


def plot_noise_ceiling_model(model, df):
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

    Returns
    -------
    fig : plt.Figure
        figure containing the plot

    """
    ax = sns.scatterplot('amplitude_estimate_median_1', 'amplitude_estimate_median_2', data=df)
    x = np.linspace(df.amplitude_estimate_median_1.min(), df.amplitude_estimate_median_1.max(),
                    1000)
    ax.plot(x, model.slope.detach().numpy() * x + model.intercept.detach().numpy(), 'r--')
    ax.set_title(f'Predictions for {model}')
    ax.axhline(color='gray', linestyle='dashed')
    ax.axvline(color='gray', linestyle='dashed')
    return ax.figure
