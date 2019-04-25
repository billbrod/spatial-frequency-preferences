#!/usr/bin/python
"""code to help run the image-computable version of the model

we're using this primarily to check the effect of vignetting, but this does make our project
image-computable (though it's a linear model and so will fail in some trivial cases)

"""
import itertools
import argparse
import numpy as np
import pandas as pd
import pyrtools as pt
from scipy import interpolate


def upsample(signal, target_shape):
    """upsample a signal to target_shape

    this uses scipy's interpolate.interp2d (and so will end up with a smoothed signal)
    """
    x = np.linspace(-(signal.shape[0]-1)/2, (signal.shape[0]-1)/2, num=signal.shape[0])
    y = np.linspace(-(signal.shape[1]-1)/2, (signal.shape[1]-1)/2, num=signal.shape[1])
    f = interpolate.interp2d(x, y, signal)
    x = np.linspace(-(signal.shape[0]-1)/2, (signal.shape[0]-1)/2, num=target_shape[0])
    y = np.linspace(-(signal.shape[1]-1)/2, (signal.shape[1]-1)/2, num=target_shape[1])
    return f(x,y)


def calc_energy_and_filters(stim, stim_df, n_orientations=6, save_path_template=None):
    """this creates the energy and filter arrays

    We assume the stimuli have natural groups, here indexed by the "class_idx" column in stim_df,
    and all stimuli within these groups should be considered the same stimuli, that is, we sum the
    energy across all of them. for the spatial frequency project, these are the different phases of
    the gratings (because of how we structure our experiment, we estimate a response amplitude to
    all phases together).

    Note that this will take a while to run (~10 or 20 minutes). Since it only needs to run once
    per experiment, didn't bother to make it efficient at all.

    Parameters
    ----------
    stim : np.ndarray
        The stimuli to produce energy for. Should have shape (n, *img_size), where n is the number 
        of total stimuli.
    stim_df : pd.DataFrame
        The DataFrame describing the stimuli. Must contain the column "class_idx", which indexes 
        the different stimulus classes (see above)
    n_orientations : int
        the number of orientations in the steerable pyramid. 6 is the number used to model fMRI
        voxels in Roth, Z. N., Heeger, D., & Merriam, E. (2018). Stimulus vignetting and 
        orientation selectivity in human visual cortex. bioRxiv.
    save_path_template : str or None
        the template string for the save path we'll use for energy and filters. should end in .npy 
        and contain one %s, which we'll replace with "energy" and "filters".

    Returns
    -------
    energy : np.ndarray
        energy has shape (stim_df.class_idx.nunique(), max_ht, n_orientations, *img_size) and 
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

    """
    img_size = stim.shape[1:]
    # this computation comes from the SteerablePyramidFreq code
    max_ht = int(np.floor(np.log2(min(img_size))) - 2)
    energy = np.zeros((stim_df.class_idx.nunique(), max_ht, n_orientations, *img_size))
    filters = np.zeros_like(energy)
    for i, g in stim_df.groupby('class_idx'):
        idx = g.index
        filled_filters = False
        for j in idx:
            pyr = pt.pyramids.SteerablePyramidFreq(stim[j], order=n_orientations-1, is_complex=True)
            for k, l in itertools.product(range(max_ht), range(n_orientations)):
                energy[int(i),k,l,:,:] = upsample(np.abs(pyr.pyr_coeffs[(k, l)])**2, img_size)
                # we only want to run this once per stimulus class
                if not filled_filters:
                    if k > 0:
                        lomask = pyr._lomasks[k-1]
                    else:
                        lomask = pyr._lo0mask
                    filt = pyr._anglemasks[k][l] * pyr._himasks[k] * lomask
                    pad_num = []
                    for m in range(2):
                        pad_num.append([(img_size[m] - filt.shape[m])//2, (img_size[m] - filt.shape[m])//2])
                        if filt.shape[m] + 2*pad_num[m][0] != img_size[m]:
                            pad_num[m][0] += img_size[m] - (filt.shape[m] + 2*pad_num[m][0])
                    filters[int(i), k, l, :, :] = np.pad(filt, pad_num, 'constant', constant_values=0)
            filled_filters = True
    filter_mean = np.mean(filters, 0)
    for i in range(filters.shape[0]):
        if not(np.allclose(filter_mean, filters[i,:,:,:,:])):
            raise Exception("Something has gone terribly wrong, the filters for stim class %d are different than the rest!" % i)
    filters = filter_mean
    if save_path_template is not None:
        np.save(save_path_template % "energy", energy)
        np.save(save_path_template % "filters", filters)
    return energy, filters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Calculate and save the energy for each stimulus class, as well as the Fourier"
                     " transform of the filters of the steerable pyramid we use to get this. For "
                     "use with image-computable version of this model"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("stimuli",
                        help=("Path to the stimulus .npy file."))
    parser.add_argument("stimuli_description_df",
                        help=("Path to the stimulus description dataframe .csv file."))
    parser.add_argument("save_path_template",
                        help=("Path template (with .npy extension) where we'll save the results. "
                              "Should contain one %s."))
    parser.add_argument('--n_orientations', '-n', default=6, type=int,
                        help=("The number of orientations in the steerable pyramid used here."))
    args = vars(parser.parse_args())
    stim = np.load(args.pop('stimuli'))
    stim_df = pd.read_csv(args.pop('stimuli_description_df'))
    calc_energy_and_filters(stim, stim_df, **args)
