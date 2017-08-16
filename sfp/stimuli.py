#!/usr/bin/python
"""script to generate stimuli
"""
import pyPyrTools as ppt
import numpy as np
from matplotlib import pyplot as plt
import itertools
import pandas as pd
import seaborn as sns
import warnings


def log_polar_grating(size, alpha, w_r=0, w_a=0, phi=0, ampl=1, origin=None, scale_factor=1):
    """Make a sinusoidal grating in logPolar space.

    this allows for the easy creation of stimuli whose spatial frequency decreases with
    eccentricity, as the peak spatial frequency of neurons in the early visual cortex does.

    Examples
    ============

    circular: `log_polar_grating(512, 4, 10)`

    radial: `log_polar_grating(512, 4, w_a=10)`

    spiral: `log_polar_grating(512, 4, 10, 10)`

    plaid: `log_polar_grating(512, 4, 10) + log_polar_grating(512, 4, w_a=10)`


    Parameters
    =============

    size: scalar. size of the image (only square images permitted).

    alpha: int, radius (in pixel spacing) of the "fovea".  IE: log_rad = log(r^2 + alpha^2)

    w_r: int, logRadial frequency.  Units are matched to those of the angular frequency (`w_a`).

    w_a: int, angular frequency.  Units are cycles per revolution around the origin.

    phi: int, phase (in radians).

    ampl: int, amplitude

    origin: 2-tuple of floats, the origin of the image, from which all distances will be measured
    and angles will be relative to. By default, the center of the image

    scale_factor: int or float. how to scale the distance from the origin before computing the
    grating. this is most often done for checking aliasing; e.g., set size_2 = 100*size_1 and
    scale_factor_2 = 100*scale_factor_1. then the two gratings will have the same pattern, just
    sampled differently
    """
    assert not hasattr(size, '__iter__'), "Only square images permitted, size must be a scalar!"
    rad = ppt.mkR(size, origin=origin)/scale_factor
    # if the origin is set such that it lies directly on a pixel, then one of the pixels will have
    # distance 0 and, if alpha is also 0, that means we'll have a -inf out of np.log2 and thus a
    # nan from the cosine. this little hack avoids that issue.
    if alpha == 0 and 0 in rad:
        rad += 1e-12
    lrad = np.log2(rad**2 + alpha**2)
    theta = ppt.mkAngle(size, origin=origin)

    return ampl * np.cos((w_r/np.pi) * lrad + w_a * theta + phi)


def _create_better_sampled_grating(orig_size, alpha, w_r=0, w_a=0, phi=0, ampl=1, orig_origin=None,
                                   orig_scale_factor=1, check_scale_factor=99):
    if check_scale_factor % 2 == 0:
        raise Exception("For this aliasing check to work, the check_scale_factor must be odd!")
    if orig_origin is None:
        origin = None
    else:
        # this preserves origin's shape, regardless of whether it's an iterable or a scalar
        origin = np.array(orig_origin) * check_scale_factor - (check_scale_factor - 1)/2
    return log_polar_grating(orig_size*check_scale_factor, alpha, w_r, w_a, phi, ampl, origin,
                             orig_scale_factor*check_scale_factor)


def aliasing_plot(better_sampled_stim, stim, slices_to_check=None, axes=None, **kwargs):
    """Plot to to check aliasing.

    This does not create the stimuli, only plots them (see `check_aliasing` or `check_aliasing_with
    mask` for functions that create the stimuli and then call this to plot them)

    to add to an existing figure, pass axes (else a new one will be created)
    """
    size = stim.shape[0]
    check_scale_factor = better_sampled_stim.shape[0] / size
    if slices_to_check is None:
        slices_to_check = [(size+1)/2]
    elif not hasattr(slices_to_check, '__iter__'):
        slices_to_check = [slices_to_check]
    if axes is None:
        fig, axes = plt.subplots(ncols=len(slices_to_check), squeeze=False,
                                 figsize=(5*len(slices_to_check), 5), **kwargs)
        # with squeeze=False, this will always be a 2d array, but because we only set ncols, it
        # will only have axes in one dimension
        axes = axes[0]
    x0 = np.array(range(size)) / float(size) + 1./(size*2)
    x1 = np.array(range(better_sampled_stim.shape[0])) / float(better_sampled_stim.shape[0]) + 1./(better_sampled_stim.shape[0]*2)
    for i, ax in enumerate(axes):
        ax.plot(x1, better_sampled_stim[:, check_scale_factor*slices_to_check[i] + (check_scale_factor - 1)/2])
        ax.plot(x0, stim[:, slices_to_check[i]], 'o:')


def check_aliasing(size, alpha, w_r=0, w_a=0, phi=0, ampl=1, origin=None, scale_factor=1,
                   slices_to_check=None, check_scale_factor=99):
    """Create a simple plot to visualize aliasing

    arguments are mostly the same as for log_polar_grating. this creates both the specified
    stimulus, `orig_stim`, and a `better_sampled_stim`, which has `check_scale_factor` more points
    in each direction. both gratings are returned and a quick plot is generated.

    NOTE that because this requires creating a much larger gradient, it can take a while. Reduce
    `check_scale_factor` to speed it up (at the risk of your "ground truth" becoming aliased)

    slices_to_check: list, None, or int. slices of the stimulus to plot. if None, will plot
    center
    """
    orig_stim = log_polar_grating(size, alpha, w_r, w_a, phi, ampl, origin, scale_factor)
    better_sampled_stim = _create_better_sampled_grating(size, alpha, w_r, w_a, phi, ampl, origin,
                                                         scale_factor, check_scale_factor)
    aliasing_plot(better_sampled_stim, orig_stim, slices_to_check)
    return orig_stim, better_sampled_stim


def _fade_mask(mask, number_of_fade_pixels, origin=None):
    """note that mask must contain 0s where you want to mask out, 1s elsewhere
    """
    # if there's no False in mask, then we don't need to mask anything out
    if False not in mask or number_of_fade_pixels == 0:
        return mask
    size = mask.shape[0]
    rad = ppt.mkR(size, origin=origin)
    alias_rad = (~mask*rad).max()
    # in order to get the right number of pixels to act as transition, we set the frequency based
    # on the specified number_of_fade_pixels
    fade_freq = (size/2.) / (2*number_of_fade_pixels)
    fade_freq = (size/2.) / (2*number_of_fade_pixels)

    def fade(x):
        return (-np.cos(fade_freq*2*np.pi*(x-alias_rad) / (size/2.))+1)/2

    faded_mask = np.piecewise(rad,
                              [rad < alias_rad,
                               (rad > alias_rad) & (rad < (alias_rad + number_of_fade_pixels)),
                               rad > (alias_rad + number_of_fade_pixels)],
                              [0, fade, 1])
    return faded_mask


def create_sf_maps_cpp(size, alpha, w_r=0, w_a=0, origin=None, scale_factor=1):
    """Create maps of spatial frequency in cycles per pixel.

    returns two maps: the angular spatial frequency map and the radial one.
    """
    assert not hasattr(size, '__iter__'), "Only square images permitted, size must be a scalar!"
    rad = ppt.mkR(size, origin=origin)/scale_factor
    # if the origin is set such that it lies directly on a pixel, then one of the pixels will have
    # distance 0 and that means we'll have a divide by zero coming up. this little hack avoids that
    # issue.
    if 0 in rad:
        rad += 1e-12
    # the angular spatial frequency drops by 1/r as you move away from the origin, where r is the
    # distance to the origin
    a_sfmap = w_a / rad
    # the logRadial spatial frequency drops by r/(r^2+alpha^2). this is proportional to d/dr
    # log(r^2 + alpha^2)
    r_sfmap = w_r * rad / (rad**2 + alpha**2)
    # in both cases above, we don't scale the spatial frequency map appropriately (the derivative
    # of log(r) is only 1/r when the log's base is e; here it's 2 so we need to scale it). this
    # scaling constant was found experimentally; it's the value required to get the a_sfmap created
    # from log_polar_grating(8, 0, w_a=2, phase=45) to have values of .5 in the center four voxels
    # (if you create this grating, you'll be able to clearly see the center four pixels go from 1
    # to -1 to 1 to -1 and thus is at the limit of aliasing).
    a_sfmap *= (.5/2.82842712)
    r_sfmap *= (.5/2.82842712)
    return a_sfmap, r_sfmap


def create_sf_maps_cpd(size, alpha, max_visual_angle, w_r=0, w_a=0, origin=None, scale_factor=1):
    """Create maps of the spatial frequency in cycles per degree of visual angle

    returns two maps: the angular spatial frequency map and the radial one.

    Parameters
    ============

    max_visual_angle: int, the visual angle (in degrees) corresponding to the largest dimension of
    the full image (on NYU CBI's prisma scanner and the set up the Winawer lab uses, this is 28)
    """
    a_sfmap, r_sfmap = create_sf_maps_cpp(size, alpha, w_r, w_a, origin, scale_factor)
    if hasattr(size, '__iter__'):
        size = max(size)
    size = float(size)
    return a_sfmap / (max_visual_angle / size), r_sfmap / (max_visual_angle / size)


def create_mask(size, alpha, w_r=0, w_a=0, origin=None, number_of_fade_pixels=3, scale_factor=1):
    """Create mask to hide aliasing

    Because of how our stimuli are created, they have higher spatial frequency at the origin
    (probably center of the image) than at the edge of the image. This makes it a little harder to
    determine where aliasing will happen, which is made more complicated by the addition of alpha
    (the radius of the region where the frequency will be held constant, basically) and the
    possibility of mixing angular and logRadial frequencies. for the specified arguments, this will
    create the mask that will hide the aliasing of the grating(s) with these arguments. *NOTE* that
    this means they must have the same of all these arguments: a circular grating with this
    specified w_r and w_a=0 and a radial one with this w_a and w_r=0 need two different masks.

    the mask will not be strictly binary, there will a `number_of_fade_pixels` where it transitions
    from 0 to 1. this transition is half of a cosine.

    returns both the faded_mask and the binary mask.
    """
    a_sfmap, r_sfmap = create_sf_maps_cpp(size, alpha, w_r, w_a, origin, scale_factor)
    nyq_freq = .5
    a_mask = a_sfmap < nyq_freq
    r_mask = r_sfmap < nyq_freq
    # the two masks created above are 0 where there's aliasing and 1 everywhere else. logical_and
    # then gives us a 1 only where both have 1s; i.e., we mask out anywhere that *either* mask says
    # will alias.
    mask = np.logical_and(a_mask, r_mask)
    faded_mask = _fade_mask(mask, number_of_fade_pixels, origin)
    return faded_mask, mask


def check_aliasing_with_mask(size, alpha, w_r=0, w_a=0, phi=0, ampl=1, origin=None, scale_factor=1,
                             number_of_fade_pixels=3, slices_to_check=None):
    """check the aliasing when mask is applied
    """
    stim = log_polar_grating(size, alpha, w_r, w_a, phi, ampl, origin, scale_factor)
    fmask, mask = create_mask(size, alpha, w_r, w_a, origin)
    better_sampled_stim = _create_better_sampled_grating(size, alpha, w_r, w_a, phi, ampl, origin,
                                                         scale_factor, 99)
    big_fmask = fmask.repeat(99, 0).repeat(99, 1)
    big_mask = mask.repeat(99, 0).repeat(99, 1)
    if slices_to_check is None:
        slices_to_check = [(size+1)/2]
    fig, axes = plt.subplots(ncols=3, nrows=len(slices_to_check), squeeze=False,
                             figsize=(15, 5*len(slices_to_check)))
    aliasing_plot(better_sampled_stim, stim, slices_to_check, axes[:, 0])
    aliasing_plot(big_fmask*better_sampled_stim, fmask*stim, slices_to_check, axes[:, 1])
    aliasing_plot(big_mask*better_sampled_stim, mask*stim, slices_to_check, axes[:, 2])
    axes[0, 0].set_title("Slices of un-masked stimulus")
    axes[0, 1].set_title("Slices of fade-masked stimulus")
    axes[0, 2].set_title("Slices of binary-masked stimulus")
    return stim, fmask, mask, better_sampled_stim, big_fmask, big_mask


def check_stim_properties(size, origin, max_visual_angle, alpha=0, w_r=0, w_a=range(10)):
    """Creates a dataframe with data on several stimulus properties, based on the specified arguments

    the properties examined are:
    - mask radius (in pixels)
    - max frequency in cycles per pixel
    - min frequency in cycles per pixel
    - max frequency in cycles per degree
    - min frequency in cycles per degree
    - max masked frequency in cycles per pixel
    - max masked frequency in cycles per degree

    note that we don't calculate the min masked frequency because that will always be zero (because
    we zero out the center of the image, where the frequency is at its highest).

    note that size, origin, and max_visual_angle must have only one value, the others can be lists
    or single values (and all combinations of them will be checked)
    """
    if hasattr(size, '__iter__'):
        raise Exception("size must *not* be iterable! All generated stimuli must be the same size")
    if hasattr(origin, '__iter__'):
        raise Exception("only one value of origin at a time!")
    if hasattr(max_visual_angle, '__iter__'):
        raise Exception("only one value of max_visual_angle at a time!")
    if not hasattr(alpha, '__iter__'):
        alpha = [alpha]
    if not hasattr(w_r, '__iter__'):
        w_r = [w_r]
    if not hasattr(w_a, '__iter__'):
        w_a = [w_a]
    rad = ppt.mkR(size, origin=origin)
    mask_df = []
    for i, (a, f_r, f_a) in enumerate(itertools.product(alpha, w_r, w_a)):
        fmask, mask = create_mask(size, a, f_r, f_a, origin, 0)
        a_sfmap_cpp, r_sfmap_cpp = create_sf_maps_cpp(size, a, f_r, f_a, origin)
        a_sfmap_cpd, r_sfmap_cpd = create_sf_maps_cpd(size, a, max_visual_angle, f_r, f_a, origin)
        data = {'mask_radius': (~mask*rad).max(), 'w_r': f_r, 'w_a': f_a, 'alpha': a, }
        for name, (a_sfmap, r_sfmap) in zip(['cpp', 'cpd'],
                                            [(a_sfmap_cpp, r_sfmap_cpp), (a_sfmap_cpd, r_sfmap_cpd)]):
            data[name + "_max"] = max(a_sfmap.max(), r_sfmap.max())
            data[name + "_min"] = min(a_sfmap.min(), r_sfmap.min())
            data[name + "_masked_max"] = max((fmask*a_sfmap).max(), (fmask*r_sfmap).max())
        mask_df.append(pd.DataFrame(data, index=[i]))
    return pd.concat(mask_df)


def _set_ticklabels(datashape):
    xticklabels = datashape[1]/10
    if xticklabels == 0 or xticklabels == 1:
        xticklabels = True
    yticklabels = datashape[0]/10
    if yticklabels == 0 or yticklabels == 1:
        yticklabels = True
    return xticklabels, yticklabels


def plot_stim_properties(mask_df, x='w_a', y='w_r', col='alpha', data_label='mask_radius',
                         title_text="Mask radius in pixels",
                         fancy_labels={"w_a": r"$\omega_a$", "w_r": r"$\omega_r$",
                                       "alpha": r"$\alpha$"},
                         **kwargs):
    """plot the mask_df created by check_mask_radius, to visualize how mask radius depends on args.

    fancy_labels is a dict of mask_df columns to nice (latex) ways of labeling them on the plot.
    """
    def facet_heatmap(x, y, data_label, **kwargs):
        data = kwargs.pop('data').pivot(y, x, data_label)
        xticks, yticks = _set_ticklabels(data.shape)
        sns.heatmap(data, xticklabels=xticks, yticklabels=yticks, **kwargs).invert_yaxis()

    cmap = kwargs.pop('cmap', 'Blues')
    font_scale = kwargs.pop('font_scale', 1.5)
    plotting_context = kwargs.pop('plotting_context', 'notebook')
    size = kwargs.pop('size', 3)
    with sns.plotting_context(plotting_context, font_scale=font_scale):
        g = sns.FacetGrid(mask_df, col=col, col_wrap=min(4, mask_df[col].nunique()), size=size)
        cbar_ax = g.fig.add_axes([.92, .3, .02, .4])
        g.map_dataframe(facet_heatmap, x, y, data_label, vmin=0,
                        vmax=mask_df[data_label].max(), cmap=cmap, cbar_ax=cbar_ax, **kwargs)
        g.fig.suptitle(title_text)
        g.fig.tight_layout(rect=[0, 0, .9, .95])
        g.set_axis_labels(fancy_labels[x], fancy_labels[y])
        g.set_titles(r"%s={col_name}" % fancy_labels[col])


def main(size, alpha, w_r=[0], w_a=[0], phi=[0], ampl=[1], origin=None, number_of_fade_pixels=3,
         combo_stimuli_type=['spiral'], filename=None):
    """Create the specified stimuli and apply the anti-aliasing mask

    this function creates the specified stimuli, calculates what their anti-aliasing masks should
    be, and applies the largest of those masks to all stimuli. Each argument (except size, origin,
    and number_of_fade_pixels) should be a list and stimuli will be made from all combinations of
    these arguments

    Note that this function should be run *last*, after you've determined your parameters and
    checked to make sure the aliasing is taken care of.

    If you want to save the created stimuli (for running an experiment, you probably do), then set
    filename to a string (with extension .npy) and an array containing the masked stimuli will be
    saved there. It's recommended you place them within the data/stimuli directory

    Parameters
    =============

    combo_stimuli_type: list with possible elements {'spiral', 'plaid'}. type of stimuli to create
    when both w_r and w_a are nonzero, as described in the docstring for log_polar_grating (to
    create circular and radial stimuli, just include 0 in w_a or w_r, respectively).

    Returns
    =============

    masked and unmasked stimuli
    """
    # we need to make sure that size, origin, and number_of_fade_pixels are not iterable and the
    # other arguments are
    if hasattr(size, '__iter__'):
        raise Exception("size must *not* be iterable! All generated stimuli must be the same size")
    if hasattr(origin, '__iter__'):
        raise Exception("origin must *not* be iterable! All generated stimuli must have the same "
                        " origin")
    if hasattr(number_of_fade_pixels, '__iter__'):
        raise Exception("number_of_fade_pixels must *not* be iterable! It's a property of the mask"
                        " and we want to apply the same mask to all stimuli.")
    if not hasattr(alpha, '__iter__'):
        alpha = [alpha]
    if not hasattr(w_r, '__iter__'):
        w_r = [w_r]
    if not hasattr(w_a, '__iter__'):
        w_a = [w_a]
    if not hasattr(phi, '__iter__'):
        phi = [phi]
    if not hasattr(ampl, '__iter__'):
        ampl = [ampl]
    if not hasattr(combo_stimuli_type, '__iter__'):
        combo_stimuli_type = [combo_stimuli_type]
    stimuli = []
    masked_stimuli = []
    mask = []
    for a, f_r, f_a in itertools.product(alpha, w_r, w_a):
        _, tmp_mask = create_mask(size, a, f_r, f_a, origin, number_of_fade_pixels)
        mask.append(tmp_mask)
    if len(mask) > 1:
        mask = np.logical_and.reduce(mask)
    else:
        mask = mask[0]
    mask = _fade_mask(mask, number_of_fade_pixels, origin)
    for a, f_r, f_a, p, A in itertools.product(alpha, w_r, w_a, phi, ampl):
        if f_r == 0 and f_a == 0:
            # this is the empty stimulus
            continue
        if f_r == 0 and len(alpha) > 1 and a != alpha[0]:
            # if we're making a radial grating (i.e., f_r is 0), then the alpha argument has no
            # effect. so if we're making a radial grating, there's more than one alpha, and this is
            # not the first one, we skip, because we've already made this stimulus.
            continue
        if 0 in [f_r, f_a] or 'spiral' in combo_stimuli_type:
            stimuli.append(log_polar_grating(size, a, f_r, f_a, p, A, origin))
            masked_stimuli.append(stimuli[-1]*mask)
        if 'plaid' in combo_stimuli_type and 0 not in [f_r, f_a]:
            stimuli.append(log_polar_grating(size, a, f_r, 0, p, A, origin) +
                           log_polar_grating(size, a, 0, f_a, p, A, origin))
            masked_stimuli.append(stimuli[-1]*mask)
    if filename is not None:
        if not filename.endswith('.npy'):
            warnings.warn("filename %s does not end in .npy, so adding that extension" % filename)
            filename += ".npy"
        np.save(filename, masked_stimuli)
    return masked_stimuli, stimuli
