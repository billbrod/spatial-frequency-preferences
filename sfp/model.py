#!/usr/bin/python
"""2d tuning model
"""
import matplotlib as mpl
# we do this because sometimes we run this without an X-server, and this backend doesn't need one
mpl.use('svg')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import warnings
from torch.utils import data as torchdata


class FirstLevelDataset(torchdata.Dataset):
    """Dataset for first level results
    """
    def __init__(self, df_path, device, direction_type='absolute'):
        self.df = pd.read_csv(df_path)
        if direction_type not in ['relative', 'absolute']:
            raise Exception("Don't know how to handle direction_type %s!" % direction_type)
        self.direction_type = direction_type
        self.device = device
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.direction_type == 'relative':
            vals = row[['local_sf_magnitude', 'local_sf_ra_direction', 'eccen', 'angle']].values
        elif self.direction_type == 'absolute':
            vals = row[['local_sf_magnitude', 'local_sf_xy_direction', 'eccen', 'angle']].values
        feature = torch.tensor(vals.astype(float), dtype=torch.float64)
        try:
            target = torch.tensor(row['amplitude_estimate_normed'], dtype=torch.float64)
        except KeyError:
            target = torch.tensor(row['amplitude_estimate_median_normed'], dtype=torch.float64)
        return feature.to(self.device), target.to(self.device)
    
    def get_voxel(self, idx):
        vox_idx = self.df[self.df.voxel==idx].index
        return self[vox_idx]
            
    def __len__(self):
        return self.df.shape[0]


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


def _cast_as_tensor(x):
    return torch.tensor(x, dtype=torch.float64)


def _cast_as_param(x):
    return torch.nn.Parameter(_cast_as_tensor(x))


class LogGaussianDonut(torch.nn.Module):
    """simple LogGaussianDonut in pytorch
    """
    def __init__(self, amplitude, mode, sigma, sf_ecc_slope=1, sf_ecc_intercept=0,
                 train_sf_ecc_slope=True, train_sf_ecc_intercept=True):
        super(LogGaussianDonut,self).__init__()
        self.amplitude = _cast_as_param(amplitude)
        self.mode = _cast_as_param(mode)
        self.sigma = _cast_as_param(sigma)
        if train_sf_ecc_slope:
            self.sf_ecc_slope = _cast_as_param(sf_ecc_slope)
        else:
            self.sf_ecc_slope = _cast_as_tensor(sf_ecc_slope)
        if train_sf_ecc_intercept:
            self.sf_ecc_intercept = _cast_as_param(sf_ecc_intercept)
        else:
            self.sf_ecc_intercept = _cast_as_tensor(sf_ecc_intercept)
        
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
        relative_freq = torch.clamp(relative_freq, min=1e-12)
        return self.log_norm_pdf_1d(relative_freq)

    def forward(self, spatial_frequency_magnitude, spatial_frequency_theta, voxel_eccentricity, voxel_angle):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.evaluate(spatial_frequency_magnitude, spatial_frequency_theta, voxel_eccentricity, voxel_angle)
    
    def to(self, device):
        super(LogGaussianDonut, self).to(device)
        self.sf_ecc_intercept = self.sf_ecc_intercept.to(device)
        self.sf_ecc_slope = self.sf_ecc_slope.to(device)


class ConstantLogGaussianDonut(LogGaussianDonut):
    """Instantiation of the "constant" extreme possibility

    this version does not depend on voxel eccentricity or angle at all"""
    def __init__(self, amplitude, mode, sigma):
        # this way the "relative frequency" is sf_mag * (0*voxel_ecc + 1) = sf_mag
        super(ConstantLogGaussianDonut, self).__init__(amplitude, mode, sigma, 0, 1, False, False)
        
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


def train_model(donut, dataset, device, n_epochs=5, batch_size=2000):
    """train the model
    """
    donut.to(device)
    loss_fn = torch.nn.MSELoss(False)
    # AMSGrad argument here means we use a revised version that handles a bug people found where
    # it doesn't necessarily converge
    optimizer = torch.optim.Adam(donut.parameters(), lr=1e-3, amsgrad=True)    
    dataloader = torchdata.DataLoader(dataset, batch_size,)# shuffle=True)
    loss_history = []
    for j, t in enumerate(range(n_epochs)):
        loss_history.append([])
        for i, (features, target) in enumerate(dataloader):
            predictions = donut(*features.transpose(1, 0))
            loss = loss_fn(predictions, target)
            if i % 10 == 0:
                print(i, loss.item())
            if np.isnan(loss.item()):
                print(i)
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history[j].append(loss.item())
        print("Average epoch loss:" % np.mean(loss_history[-1]))
    return donut, loss_history
