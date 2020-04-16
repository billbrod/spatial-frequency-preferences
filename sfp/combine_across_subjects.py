"""functions to combine GLMdenoise outputs across subjects

several steps to this

1. Interpolate all subjects' GLMdenoise models outputs (that is, the
   amplitude estimate bootstraps) to fsaverage prior space (so all
   subjects have same number of vertices)

2. Calculate each subject's precision (precision is the inverse of
   variance, so estimate variance for each voxel, each stimulus classes,
   then average across stimulus classes, then average across voxels)

3. Perform a precision-weighted average to get the 'groupaverage'
   subject's amplitude estimates for each bootstrap.

4. Compute the median and standard error across bootstraps.

5. Save this in a hdf5 file that looks like the GLMdenoise output
   (reshaping the above correctly), so it can be passed to the rest of
   our analysis pipeline

"""
import h5py
import os
import neuropythy as ny
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .plotting import MidpointNormalize
from .first_level_analysis import _precision_dist


def get_fsaverage_coords(hemi, target_varea=1, reverse_rh=False):
    """load in the fsaverage retinotopic prior coordinates

    we need to get the x and y coordinates for the fsaverage retinotopic
    prior, so we know where to interpolate to. this function will load
    them in (neuropythy has them built in) for one visual area and
    return them

    we also return the `prior_varea` array, which has not been reduced
    in size. we do this so that we can work backwards and create an
    array with entries for vertex in the brain and then insert our
    interpolated values into the correct locations

    Parameters
    ----------
    hemi : {'lh', 'rh', 'both'}
        which hemisphere to load in. If both, we concatenate them in the
        order lh, rh.
    target_varea : int, optional
        which visual area to restrict ourselves to
    reverse_rh : bool, optional
        whether to reverse the right hemisphere coordinates (x and
        angle, respectively). You want to do this for plotting (so they
        show up in the left visual field) but not for interpolating

    Returns
    -------
    prior_x, prior_y : np.array
        the x and y coordinates for the fsaverage vertices, restricted
        to `target_varea`
    prior_varea : np.array
        the visual area for each fsaverage vertex, for the whole brain
        (thus, this will be larger than `prior_x` and `prior_y`)
    prior_angle, prior_ecc : np.arraecc
        the angle and eccentricity coordinates for the fsaverage
        vertices, restricted to `target_varea`

    """
    fsaverage_surf = os.path.join(os.path.dirname(ny.__file__), 'lib', 'data', 'fsaverage', 'surf')
    if hemi == 'both':
        hemi = ['lh', 'rh']
    else:
        hemi = [hemi]
    prior_angle, prior_ecc = [], []
    prior_varea = []
    prior_x, prior_y = [], []
    for h in hemi:
        prior_angle_tmp = ny.load(os.path.join(fsaverage_surf, f'{h}.benson14_angle.v4_0.mgz'))
        prior_ecc_tmp = ny.load(os.path.join(fsaverage_surf, f'{h}.benson14_eccen.v4_0.mgz'))
        prior_varea_tmp = ny.load(os.path.join(fsaverage_surf, f'{h}.benson14_varea.v4_0.mgz'))

        prior_angle_tmp = prior_angle_tmp[prior_varea_tmp==target_varea]
        prior_ecc_tmp = prior_ecc_tmp[prior_varea_tmp==target_varea]

        prior_x_tmp, prior_y_tmp = ny.as_retinotopy({'eccentricity': prior_ecc_tmp,
                                                     'polar_angle': prior_angle_tmp},
                                                    'geographical')

        if h == 'rh' and reverse_rh:
            multiplier = -1
        else:
            multiplier = 1

        prior_x.append(multiplier * prior_x_tmp)
        prior_y.append(prior_y_tmp)
        prior_angle.append(multiplier * prior_angle_tmp)
        prior_ecc.append(prior_ecc_tmp)
        prior_varea.append(prior_varea_tmp)

    prior_x = np.concatenate(prior_x)
    prior_y = np.concatenate(prior_y)
    prior_angle = np.concatenate(prior_angle)
    prior_ecc = np.concatenate(prior_ecc)
    prior_varea = np.concatenate(prior_varea)

    return prior_x, prior_y, prior_varea, prior_angle, prior_ecc


def plot_amplitudes(x, y, amplitudes, hemi, plot_content, prf_space, class_num=0, ax=None,
                    vmin=None, vmax=None, annotate=True):
    """plot amplitude estimates as function of location in visual field

    Parameters
    ----------
    x, y : np.array
        the x and y locations of the vertices in a single varea
    amplitudes : np.array
        the amplitude estimates, with shape (num_vertices, num_classes),
        for each vertex in a single varea. should be the same shape as x
        and y.
    hemi : {'lh', 'rh'}
        whether these vertices are in the right or left hemisphere. used
        to title the plot and to make sure they end up in the right side
        of the visual field
    plot_content : str
        what is plotted here, for titling the axis. e.g., "bootstrap 0",
        "median", "standard error"
    prf_space : str
        which pRF space we're plotting here (fsaverage or subject, most
        likely), only used for titling the plot
    class_num : int, optional
        which class number to plot. used to select the proper slice from
        amplitudes and to title the plot
    ax : plt.Axes or None, optional
        if not None, the axis to create this plot on. if None, we'll
        create a new axis with figsize (7.5, 7.5) and equal aspect
    vmin, vmax : float or None, optional
        the minimum and maximum values for the colormap. If None (the
        default), will use the min and max from `amplitudes[:,
        class_num]`
    annotate : bool, optional
        whether to add text saying the number of vertices in the plot

    Returns
    -------
    fig : plt.Figure
        the figure containing the plot

    """
    if hemi == 'rh':
        # then this is the right hemisphere = left visual field, and we
        # thus want the x values to be negative. need to copy this
        # otherwise we mess up the prior_x array
        x = x.copy() * -1
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5), subplot_kw={'aspect': 1})
    amplitudes = amplitudes[:, class_num]
    # there may be some NaNs, don't plot those
    x = x[~np.isnan(amplitudes)]
    y = y[~np.isnan(amplitudes)]
    amplitudes = amplitudes[~np.isnan(amplitudes)]
    if vmin is None:
        vmin = amplitudes.min()
    if vmax is None:
        vmax = amplitudes.max()
    sc = ax.scatter(x, y, c=amplitudes, alpha=.5, cmap='RdBu',
                    norm=MidpointNormalize(vmin, vmax, 0))
    ax.set_title(f'{hemi}, {prf_space}-space: {plot_content} amplitude \n estimates for '
                 f'class {class_num} projected onto visual field')
    ax.figure.colorbar(sc, ax=ax)
    if annotate:
        ax.text(1.3, .5, f"{x.shape[0]} vertices", transform=ax.transAxes, va='center')
    return ax.figure


def plot_zero_check(amplitudes, properties, vars=['polar_angle', 'eccentricity'], hue='hemi',
                    sum_idx=1, nan_check=False):
    """plot properties of voxels with zero or NaN amplitude

    after interpolation, some voxels may end up with zero amplitude for
    some reason, including because their coordinates were incorrect
    (e.g., they had negative x values). this plot allows you to easily
    check the propreties of those voxels in order to see if there's
    anything wrong with them.

    If the nan_check arg is True, we check for NaNs instead.

    Properties
    ----------
    amplitudes : dict
        dict containing the keys ['lh', 'rh'] containing the amplitudes
        amplitude estimates for each vertex in a single varea as an
        array.
    properties : dict
        dict containing keys ['lh', 'rh'], each value of which is
        another dict containing keys of different voxel properties. Must
        have same number of voxels as `amplitudes`.
    vars : list, optional
        list of strs specifying the keys in `properties` that you wish
        to plot in the pairplot
    hue : str, optional
        either 'hemi' or a str of a key in `properties`, this is the
        variable to plot as the hue dimension in pairplot
    sum_idx : int or tuple, optional
        either an int or a tuple of ints, this is the dimensions over
        which to sum `amplitude` over so that we end up with something
        one dimensional, with each element corresponding to a different
        voxel. For example, if `amplitude` has shape `(num_voxels,
        num_classes)`, `sum_idx` should equal `1`, so that we sum over
        all the classes in order to determine which voxels have any zero
        amplitudes. If `amplitude` has shape `(num_bootstraps,
        num_classes, num_voxels)`, this should be `(0, 1)`.
    nan_check : bool, optional
        if True, we check for (any) NaNs instead of all zeros. If False,
        we check for zeros

    Returns
    -------
    fig : plt.Figure or str
        the figure containing the plot or, if no voxels had zero
        amplitude, the str 'No voxels have amplitude zero'

    """
    df = []
    keys_to_copy = vars
    if hue != 'hemi':
        keys_to_copy += [hue]
    for hemi in ['lh', 'rh']:
        if not nan_check:
            text = 'zero'
            zero_idx = np.where((amplitudes[hemi]==0).sum(sum_idx))[0]
        if nan_check:
            text = 'NaN'
            zero_idx = np.where(np.isnan(amplitudes[hemi].sum(sum_idx)))[0]
        d = {'hemi': hemi}
        d.update(dict((k, properties[hemi][k][zero_idx]) for k in keys_to_copy))
        df.append(pd.DataFrame(d))
    df = pd.concat(df)
    # first any checks if there's any value in each column, second is if
    # there's any value in any column
    if not df.any().any():
        # then there's nothing to plot -- this happens when zero_idx is
        # empty
        return f"No voxels have amplitude {text}"
    else:
        g = sns.pairplot(df, hue, vars=vars, height=5)
        g.fig.suptitle(f"pRF Locations of voxels with amplitude {text}")
        return g.fig


def add_GLMdenoise_field_to_props(GLMdenoise_path, props, GLMdenoise_field='models'):
    """load in the GLMdenoise results.mat file and add a field to props dict

    This will only work if GLMdenoise was run on the surface (not in the
    volume), because we make some strong assumptions about how
    everything is shaped

    Parameters
    ----------
    GLMdenoise_path : str
        path to the results.mat file created by GLMdenoise for this
        subject/session
    props : dict
        dictionary with keys 'lh' and 'rh' to add the GLMdenoise field
        to.
    GLMdenoise_field : str, optional
        the field from the reuslts.mat to load in. this should work for
        models, modelmd, or modelmd, but others are unknown, and only
        models is recommended

    Returns
    -------
    props : dict
        updated props dict

    """
    with h5py.File(GLMdenoise_path, 'r') as f:
        tmp_ref = f['results'][GLMdenoise_field]
        if tmp_ref.shape == (2, 1):
            # this is the case for all the models fields of the .mat
            # file (modelse, modelmd, models). [0, 0] contains the hrf,
            # and [1, 0] contains the actual results.
            res = f[tmp_ref[1, 0]][:]
        else:
            # that reference thing is only necessary for those models
            # fields, because I think they're matlab structs
            res = tmp_ref[:]

    res = res.squeeze()
    if res.ndim == 3:
        # then this has bootstraps in it and we want res.shape to be
        # (bootstraps, voxels, model_class), so we reshape
        res = res.transpose(0, 2, 1)
        for i, r in enumerate(res):
            for h in ['lh', 'rh']:
                # because of how bidsGetPreprocData.m loads in the
                # surface files, we know this is the left and right
                # hemisphere concatenated together, in that order
                if h == 'lh':
                    tmp = r[:props['lh']['visual_area'].shape[0]]
                else:
                    tmp = r[-props['rh']['visual_area'].shape[0]:]
                props[h][f'{GLMdenoise_field}_bootstrap_{i:02d}'] = tmp
    elif res.ndim == 2:
        # then this doesn't have bootstraps in it and we want res.shape
        # to be (voxels, model_class), so we reshape
        res = res.transpose(1, 0)
        for h in ['lh', 'rh']:
            # because of how bidsGetPreprocData.m loads in the surface
            # files, we know this is the left and right hemisphere
            # concatenated together, in that order
            if h == 'lh':
                tmp = res[:props['lh']['visual_area'].shape[0]]
            else:
                tmp = res[-props['rh']['visual_area'].shape[0]:]
            props[h][f'{GLMdenoise_field}'] = tmp
    else:
        raise Exception(f"results is shape {res.shape} for GLMdenoise field {GLMdenoise_field}, but "
                        "I only know how to handle 2 and 3 dimensional results!")
    return props


def interpolate_GLMdenoise_to_fsaverage_prior(freesurfer_sub, prf_props, save_stem,
                                              GLMdenoise_path=None, plot_class=0, plot_bootstrap=0,
                                              target_varea=1, interp_method='linear'):
    """interpolate a scanning session's GLMdenoise models results to fsaverage space

    In order to combine data across subjects, we need them to have
    equivalent vertices (that is, vertices we can consider 'the same'
    and average together). We follow the method done by Benson et al,
    2019's analysis of the retinotopic data in the Human Connectome
    Project: interpolate each subject's results to the locations in
    fsaverage, in the visual field (the Benson et al, 2014 retinotopic
    atlas defines the retinotopic coordinates for fsaverage).

    For the subject's retinotopic information, you should almost
    certainly pass the outputs of the Bayesian retinotopy, as a
    dictionary. For the paths used in this project, the following is how
    to create this dictionary (setting the BIDS_DIR and subject
    variables beforehand):

    ```
    template = (f'{BIDS_dir}/derivatives/prf_solutions/{subject}/bayesian_posterior/'
                '{hemi}.inferred_{data}.mgz')
    prf_props = {}
    for h in ['lh', 'rh']:
        prf_props[h] = {}
        names = zip(['varea', 'eccen', 'angle'], ['visual_area', 'eccentricity', 'polar_angle'])
        for k, prop in names:
            prf_props[h][prop] = ny.load(template.format(hemi=h, data=k))
    ```

    The following steps are taken:

    - grab and shape the 'models' field from the GLMdenoise results.mat
      file, add to the prf_props dict

    - for each hemisphere:
    
      - add all properties from the prf_props dict to the neuropythy
        mesh
    
      - grab the fsaverage retinotopic prior (from the neuropythy package)

      - for each bootstrap:

        - interpolate the amplitude estimates for all models from the
          subject's retinotopic space to the fsaverage one

      - insert all these interpolated estimates into a properly-sized
        array

    - concatenate this array across hemispheres and save as an hdf5 file
      (the array is now the right size for the GLMdenoise results field,
      but doesn't look like quite right because the GLMdenoise results
      field also contains the fitted HRF)

    The main output is:

    - save_stem+"_models.hdf5": a HDF5 file containing the array (as
      field 'models') with shape (num_bootstraps, num_classes, 1,
      num_vertices, 1) containing the subject/session's amplitude
      estimates (for each bootstrap and class) interpolate to the
      fsaverage retinotopic prior space. It has this shape because
      that's the shape of the GLMdenoise output, and we'll want to mimic
      that. We use a HDF5 file because this will be very large, and a
      HDF5 file is more compact than a .npy file

    We also produce several outputs to help check what's going on.

    The first two are plots which show the same amplitude estimates, one
    in the subject's original retinotopic space, and one interpolated to
    the fsaverage retinotopic prior space. These two should look like
    they're conveying the same information, just sampling at different
    locations.
    
    - save_stem+"_models_b{plot_bootstrap}_c{plot_class}_space-subject.png":
      a plot showing the amplitude estimates for the stimulus class
      `plot_class` and the bootstrap `plot_bootstrap` as a scatter plot,
      with x, y locations coming from the subject's pRFs and the values
      from the output of GLMdenoise

    - save_stem+"_models_b{plot_bootstrap}_c{plot_class}_space-prior.png":
      a plot showing the amplitude estimates for the stimulus class
      `plot_class` and the bootstrap `plot_bootstrap` as a scatter plot,
      with x, y locations coming from the fsaverage pRF prior and the
      interpolated values.

    We then produce four outputs to examine any voxels that have zero
    amplitudes. GLMdenoise shouldn't produce voxels that have an
    amplitude estimate of exactly zero, so this is often a sign that
    something has gotten messed up. For each of the following, if there
    are no voxels with zero amplitude, we create a text file (replacing
    the .png extension with .txt) that contains the string "No voxels
    have amplitude zero" instead of the plot.

    - save_stem+"_zero_check_b{plot_bootstrap}_coords-polar_space-subject":
      a seaborn pairplot showing the polar angle and eccentricity
      locations of all voxels that have any zero amplitudes prior to
      interpolation.

    - save_stem+"_zero_check_b{plot_bootstrap}_coords-cartesian_space-subject":
      a seaborn pairplot showing the x and y locations of all voxels
      that have any zero amplitudes prior to interpolation.

    - save_stem+"_zero_check_b{plot_bootstrap}_coords-polar_space-prior":
      a seaborn pairplot showing the polar angle and eccentricity
      locations of all voxels that have any zero amplitudes after
      interpolation

    - save_stem+"_zero_check_b{plot_bootstrap}_coords-polar_space-prior":
      a seaborn pairplot showing the x and y locations of all voxels
      that have any zero amplitudes after interpolation.

    The expectation is:

    - There should never be any voxels with amplitude zero prior to
      interpolation (so none of the `space-subject` plots should be
      created)

    - if `interp_method='linear'`, the only voxels with amplitude zero
      after interpolation should be at the extremes of the visual field
      (so along the visual meridian and far periphery / with min and max
      possible eccentricity values)

    - if `interp_method='nearest'`, no voxels should have amplitude zero
      after interpolation

    Parameters
    ----------
    freesurfer_sub : str
        The freesurfer subject to use. This can be either the name
        (e.g., wlsubj045; in which case the environmental variable
        SUBJECTS_DIR must be set) or a path to the freesurfer folder. It
        will be passed directly to neuropythy.freesurfer_subject, so see
        the docstring of that function for more details
    prf_props : dict
        dictionary containing the arrays with prf properties to add to
        the neuropythy freesurfer subject. This should contain two keys,
        'lh' and 'rh', corresponding to the left and right hemispheres,
        respectively. Each of those should have a dictionary containing
        identical keys, which should be some subset of 'visual_area',
        'eccentricity', and 'polar_angle'. If any of those are not
        included in prf_props, we will use the corresponding property
        from the freesurfer directory (and if they aren't present there,
        this function will fail). The intended use is that this will
        contain the results of the Bayesian retinotopy, which we'll use
        as the pRF parameters in subject-space.
    save_stem : str
        the stem of the path to save things at (i.e., should not end in
        the extension)
    GLMdenoise_path : str or None, optional
        path to the results.mat file created by GLMdenoise for this
        subject/session. If None, we assume prf_props already contains
        the 'models_bootstrap_{i:02d}' keys
    plot_class : int, optional
        we create a plot showing the amplitudes for one class, one
        bootstrap. this specifies which class to plot.
    plot_bootstrap : int, optional
        we create a plot showing the amplitudes for one class, one
        bootstrap. this specifies which bootstrap to plot.
    target_varea : int, optional
        The visual area we're interpolating. because we interpolate in
        the visual field, we can only do one visual area at a time
        (because otherwise they'll interfere with each other)
    interp_method : {'nearest', 'linear'}, optional
        whether to use linear or nearest-neighbor interpolation. See the
        docstring of `neuropythy.mesh.interpolate` for more details

    Returns
    -------
    interp_all : np.array
        the numpy array containing the interpolated amplitude estimates,
        of shape (num_bootstraps, num_classes, 1, num_vertices, 1). note
        that num_vertices here is the number of vertices in the entire
        fsaverage brain, not just `target_varea` (but all vertices not
        in that visual area will be 0).

    """
    sub = ny.freesurfer_subject(freesurfer_sub)

    if GLMdenoise_path is not None:
        prf_props = add_GLMdenoise_field_to_props(GLMdenoise_path, prf_props)
    num_bootstraps = len([b for b in prf_props['lh'].keys() if 'bootstrap' in b])
    if num_bootstraps != 100:
        raise Exception(f"There should be 100 bootstraps, but there are {num_bootstraps}!")

    priors = {}
    idx = {}
    submesh = {}
    for hemi in ['lh', 'rh']:
        priors[hemi] = dict(zip(['x', 'y', 'varea', 'polar_angle', 'eccentricity'],
                                get_fsaverage_coords(hemi, target_varea)))
        # we need to figure out which vertices correspond to our
        # targeted visual area for constructing the overall array (which
        # should mimic the results of GLMdenoise run on the full
        # brain). we grab the first element of np.where because this is
        # a 1d array
        idx[hemi] = np.where(priors[hemi]['varea'] == target_varea)[0]
        if hemi == 'lh':
            mesh = sub.lh.with_prop(**prf_props['lh'])
        else:
            mesh = sub.rh.with_prop(**prf_props['rh'])
        submesh[hemi] = mesh.white_surface.submesh(mesh.white_surface.mask(('visual_area',
                                                                            target_varea)))

    # grab the vmin and vmax, for the target varea, in the plotted
    # bootstrap, across both hemispheres and all classes. We use 1st and
    # 99th percnetile because the min/max are often much larger than the
    # rest of the distribution
    vmin = min(np.percentile(submesh['lh'].properties[f'models_bootstrap_{plot_bootstrap:02d}'][:, plot_class], 1),
               np.percentile(submesh['rh'].properties[f'models_bootstrap_{plot_bootstrap:02d}'][:, plot_class], 1))
    vmax = max(np.percentile(submesh['lh'].properties[f'models_bootstrap_{plot_bootstrap:02d}'][:, plot_class], 99),
               np.percentile(submesh['rh'].properties[f'models_bootstrap_{plot_bootstrap:02d}'][:, plot_class], 99))

    interpolated_all = []
    zero_check_data = {'submesh': {}, 'interpolated': {}, 'original': {}}
    for hemi in ['lh', 'rh']:
        # this should be of shape (num_bootstraps, num_classes, 1,
        # num_vertices, 1), in order to mimic the output of
        # GLMdenoise. num_vertices will be different between the two
        # hemispheres, everything else will be the same. Note that we
        # use priors[hemi][varea] to get the number of vertices, NOT
        # prf_props[hemi]['models_bootstrap_00'], because we want the
        # number in fsaverage-space, not in subject-space
        _, num_classes = prf_props[hemi]['models_bootstrap_00'].shape
        interpolated_hemi = np.zeros((num_bootstraps, num_classes, 1,
                                      priors[hemi]['varea'].shape[0], 1))

        x, y = ny.as_retinotopy(submesh[hemi], 'geographical')
        submesh_tmp = submesh[hemi].copy(coordinates=[x, y])

        zero_check_data['submesh'][hemi] = submesh_tmp.with_prop(x=x, y=y).properties

        # neuropythy's interpolate can only work with 2d arrays, so we
        # need to do each bootstrap separate
        for i in range(num_bootstraps):
            interp_models = submesh_tmp.interpolate([priors[hemi]['x'], priors[hemi]['y']],
                                                    f'models_bootstrap_{i:02d}', method=interp_method)
            # for now, there's a bug where neuropythy isn't putting
            # inserting NaNs in the extrapolated locations, so we do
            # that manually. they'll be exactly 0
            interp_models[interp_models.sum(1)==0] = np.nan
            interpolated_hemi[i, :, 0, idx[hemi], 0] = interp_models

            if i == plot_bootstrap:
                fig = plot_amplitudes(x, y, submesh_tmp.properties[f'models_bootstrap_{i:02d}'],
                                      hemi, f'bootstrap {i}', 'subject', plot_class, vmin=vmin,
                                      vmax=vmax)
                fig.savefig(save_stem + f"_models_{hemi}_b{i:02d}_c{plot_class:02d}_space-subject.png")

                fig = plot_amplitudes(priors[hemi]['x'], priors[hemi]['y'], interp_models, hemi,
                                      f'bootstrap {i}', 'fsaverage', plot_class, vmin=vmin,
                                      vmax=vmax)
                fig.savefig(save_stem + f"_models_{hemi}_b{i:02d}_c{plot_class:02d}_space-prior.png")
                zero_check_data['interpolated'][hemi] = interp_models
                zero_check_data['original'][hemi] = submesh_tmp.properties[f'models_bootstrap_{i:02d}']

        interpolated_all.append(interpolated_hemi)
    for a, p, s, n in zip([zero_check_data['original'], zero_check_data['interpolated']],
                          [zero_check_data['submesh'], priors], ['subject', 'prior'],
                          ['zero', 'nan']):
        for v, c in zip([['polar_angle', 'eccentricity'], ['x', 'y']], ['polar', 'cartesian']):
            fig = plot_zero_check(a, p, v, nan_check=(n == 'nan'))
            if not isinstance(fig, str):
                fig.savefig(save_stem + f"_{n}_check_b{i:02d}_coords-{c}_space-{s}.png")
            else:
                print(fig)
                print(fig, file=open(save_stem + f"_zero_check_b{i:02d}_coords-{c}_space-{s}.txt", 'w'))
    # concatenate into one array (vertices are on dimension 3)
    interpolated_all = np.concatenate(interpolated_all, 3)
    # and save
    with h5py.File(save_stem + '_models.hdf5', 'w') as f:
        f.create_dataset('results/models', data=interpolated_all, compression='gzip')
    return interpolated_all


def check_nans(subjects):
    """handle NaNs in subjects

    `compute_groupaverage` handles NaNs well: it ignores them when
    computing the weighted average across subjects. however, there will
    be some voxels that are NaNs in all, or almost all, subjects. For
    those voxels, we just want to ignore them completely.

    we find those voxels that have NaNs in more than half of the
    subjects and replace all their values with NaNs. this means they
    will also be NaNs in the groupaverage (and will need to be ignored
    later on).

    we also double-check that any voxel that has NaN values has NaN
    values for all bootstraps, all classes (in a given subject), because
    NaNs should only show up because of the interpolation. We raise an
    Exception if this is not true

    Parameters
    ----------
    subjects : np.array
        an array with shape (subjects, bootstraps, classes, voxels),
        contains the interpolated amplitude estimates for several
        subjects (assumed to be restricted to a single visual area, but
        this probably works for any)

    Returns
    -------
    subjects : np.array
        The subjects array, same shape as before, modified as described
        above: with some extra np.nan inserted
    okay_idx : array
        a 1d boolean array with length equal to `subjects.shape[-1]`,
        contains a True for the voxels we did not modify

    """
    # this is the total number of beta values / amplitude estimates that
    # a single voxel will have: 1 per bootstrap per class
    n_betas = subjects.shape[1] * subjects.shape[2]
    # sum over subjects, bootstraps, and classes
    nans = np.isnan(subjects).sum((0, 1, 2))
    # by dividing by n_betas, the values here will show how many
    # subjects have al NaNs at that voxel
    nan_nums = nans / n_betas
    # do a quick check here: voxels should either have all or no NaNs,
    # not e.g., half NaNs
    if (nan_nums.astype(int) != nan_nums).any():
        raise Exception("There are voxels that have some NaNs after interpolation, which "
                        "shouldn't be the case. All voxels should have either all or no NaNs")
    # find those voxels where more than half of the subjects have NaNs
    drop_idx = np.where(nan_nums > (subjects.shape[0] / 2))[0]
    # and fill them completely with NaNs, so we ignore them
    subjects[..., drop_idx] = np.nan
    idx = np.ones_like(nans).astype(bool)
    idx[drop_idx] = False
    return subjects, idx


def compute_groupaverage(interpolated_models, save_stem, seed=None, plot_class=0, plot_bootstrap=0,
                         target_varea=1):
    """Computer sub-groupaverage from interpolated GLMdenoise outputs

    After interpolating individual subjects' GLMdenoise outputs to the
    fsaverage space, we can combine them into a new groupaverage subject
    for fitting the model to.

    We take the following steps here:

    1. Load in interpolated outputs (if necessary) and restrict to
       `target_varea`

    2. Calculate each subject's precision: take the inverse of the
       variance across bootstraps and then average across all voxels and
       classes

    3. Bootstrap to select subjects (randomly select n with
       replacement), then perform a precision-weighted average across
       subjects to get an amplitude estimate per voxel per class per
       (GLMdenoise) bootstrap

    4. Compute median and standard error across the GLMdenoise
       bootstraps

    5. Create some plots to help check that these look reasonable, then
       save these in a hdf5 file that looks like the GLMdenoise output
       (and so looks like the whole brain, though it will contain 0s
       everywhere else) (the file has a .mat extension because that's
       what the next steps expect; a .mat file is atually a hdf5 file)

    The outputs will be saved at:

    - save_stem+"_results.mat": GLMdenoise-like hdf5 file for
      sub-groupaverage with this bootstrap.

    - save_stem+"_b{plot_bootstrap}_c{plot_class}_models.png": figure
      showing the amplitude estimates for a single bootstrap of a single
      class, the median for that class, and the standard error for that
      class, across the whole visual field

    Parameters
    ----------
    interpolated_models : list
        list of strs or list of arrays. If strs, we assume these are the
        paths to the hdf5 files created by the
        `interpolate_GLMdenoise_to_fsaverage_prior` function, one per
        subject. If arrays, we assume these are the `results/models`
        field from those hdf5 files (and we check if it's the whole
        brain or not -- if so, we restrict to `target_varea`, if not, we
        assume it's already been restricted)
    save_stem : str
        the stem of the path to save things at (i.e., should not end in
        the extension). Should include the seed (we won't add it).
    seed : int or None, optional
        seed to pass to random number generator for reproducibility
    plot_class : int, optional
        we create a plot showing the amplitudes for one class, one
        bootstrap. this specifies which class to plot.
    plot_bootstrap : int, optional
        we create a plot showing the amplitudes for one class, one
        bootstrap. this specifies which bootstrap to plot.
    target_varea : int, optional
        The visual area we're interpolating. because we interpolate in
        the visual field, we can only do one visual area at a time
        (because otherwise they'll interfere with each other)

    """
    if seed is not None:
        np.random.seed(seed)
    x, y, varea, _, _ = get_fsaverage_coords('both', target_varea, True)
    varea_idx = np.where(varea==target_varea)[0]
    if isinstance(interpolated_models[0], str):
        # then these are a list of paths to the hdf5 files
        subjects = []
        for p in interpolated_models:
            with h5py.File(p, 'r') as f:
                # and squeeze and restirct to our target varea
                subjects.append(f['results']['models'][:].squeeze()[..., varea_idx])
        subjects = np.stack(subjects)
    else:
        # then this is a list or array of the results
        subjects = np.stack(interpolated_models).squeeze()
        # check if they've been restricted to our target varae
        if subjects.shape[-1] == len(varea):
            # if in here, they haven't been, so restrict now
            subjects = subjects[..., varea_idx]
    # subjects now has shape (subject, bootstraps, classes, target varea
    # voxels), so we take the precision (inverse of variance) across
    # bootstraps so we have one per voxel per class per subject, then
    # average across voxels and classes so we have one per subject. we
    # use nanmean because there may be some NaNs (places where linear
    # interpolate couldn't reasonably place values, i.e. extrapolated
    # values)
    precision = np.nanmean(_precision_dist(subjects, 1), (-1, -2))
    # bootstrap: choose (with replacement and with uniform
    # probabilities) n subjects, where n is the number of subjects we
    # have
    choice_idx = np.random.choice(range(len(precision)), len(precision))
    # we may have NaNs in subjects, depending on the
    # interpolation. First, we find those voxels where more than half of
    # the subjects have NaNs, then fill them completely with NaNs (this
    # effectively means we ignore them; it would be easier to drop them,
    # but then things get weird when we reinsert them into the whole
    # brain array)
    subjects, okay_idx = check_nans(subjects)
    # then we do the weighted average, ignoring NaNs. This comes from
    # https://stackoverflow.com/a/35758345 (normal np.nanmean can't be
    # weighted)
    masked_subjects = np.ma.masked_array(subjects, np.isnan(subjects))
    tmp = np.ma.average(masked_subjects[choice_idx], axis=0, weights=precision[choice_idx])
    # this converts it back to a non-masked array, but there shouldn't
    # be any NaNs. this can get messed up if we do it in place, so make
    # sure to assign the output to a new variable
    models_tmp = tmp.filled(np.nan)
    # we check to make sure that there are no NaNs in the okay voxels
    # (those that we didn't cast to all nans in the check_nans
    # call). This can happen if you don't have enough subjects. at the
    # extreme, if you have two subjects and choice_idx is the same
    # number repeated (e.g., [0, 0]), then this will be True
    if np.isnan(models_tmp[..., okay_idx]).any():
        raise Exception("Somehow we have NaNs after the weighted average")
    modelmd_tmp = np.median(models_tmp, 0)
    modelse_tmp = np.diff(np.percentile(models_tmp, [16, 84], 0), axis=0) / 2
    # this is the shape we want to save it as: (bootstraps, classes, 1,
    # all brain voxels, 1)
    models = np.empty((*subjects.shape[1:3], 1, len(varea), 1))
    # these two have the shape: (classes, 1, all brain voxels, 1)
    modelmd = np.empty((subjects.shape[2], 1, len(varea), 1))
    modelse = np.empty((subjects.shape[2], 1, len(varea), 1))
    models[:, :, 0, varea_idx, 0] = models_tmp
    modelmd[:, 0, varea_idx, 0] = modelmd_tmp
    # this mimics how standard error is computed by GLMdenoise: take the
    # 16th and 84th percentile across bootstraps, take their difference,
    # and then divide by 2
    modelse[:, 0, varea_idx, 0] = modelse_tmp
    fig, axes = plt.subplots(1, 3, figsize=(22.5, 7.5), subplot_kw={'aspect': 1})
    for ax, data, c  in zip(axes, [models_tmp[plot_bootstrap], modelmd_tmp, modelse_tmp],
                            [f'bootstrap {plot_bootstrap}', 'median', 'standard error']):
        # plot_amplitudes expects the data to be (voxels, classes) but
        # these are (classes, voxels)
        plot_amplitudes(x, y, data.squeeze().transpose(), 'both', c, 'fsaverage', plot_class,
                        ax, annotate=False)
    fig.savefig(save_stem + f"_b{plot_bootstrap:02d}_c{plot_class:02d}_models.png")
    with h5py.File(save_stem + '_results.mat', 'w') as f:
        f.create_dataset('results/models', data=models, compression='gzip')
        f.create_dataset('results/modelse', data=modelse, compression='gzip')
        f.create_dataset('results/modelmd', data=modelmd, compression='gzip')
        # first_level_analysis expects an R2 field, but we have nothing to put there
        f.create_dataset('results/R2', data=np.nan * np.ones((models.shape[-2], 1)),
                         compression='gzip')
    return models
