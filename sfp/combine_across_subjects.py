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


def get_fsaverage_coords(hemi, target_varea=1):
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
    hemi : {'lh', 'rh'}
        which hemisphere to load in
    target_varea : int, optional
        which visual area to restrict ourselves to

    Returns
    -------
    prior_x, prior_y : np.array
        the x and y coordinates for the fsaverage vertices, restricted
        to `target_varea`
    prior_varea : np.array
        the visual area for each fsaverage vertex, for the whole brain
        (thus, this will be larger than `prior_x` and `prior_y`)

    """
    fsaverage_surf = os.path.join(os.path.dirname(ny.__file__), 'lib', 'data', 'fsaverage', 'surf')
    prior_angle = ny.load(os.path.join(fsaverage_surf, f'{hemi}.benson14_angle.v4_0.mgz'))
    prior_ecc = ny.load(os.path.join(fsaverage_surf, f'{hemi}.benson14_eccen.v4_0.mgz'))
    prior_varea = ny.load(os.path.join(fsaverage_surf, f'{hemi}.benson14_varea.v4_0.mgz'))

    prior_angle = prior_angle[prior_varea==target_varea]
    prior_ecc = prior_ecc[prior_varea==target_varea]

    prior_x, prior_y = ny.as_retinotopy({'eccentricity': prior_ecc, 'polar_angle': prior_angle},
                                        'geographical')

    return prior_x, prior_y, prior_varea


def plot_amplitudes(x, y, amplitudes, hemi, bootstrap, prf_space, class_num=0, ax=None):
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
    bootstrap : int
        which bootstrap these estimates come from, only used for titling
        the plot
    prf_space : str
        which pRF space we're plotting here (fsaverage or subject, most
        likely), only used for titling the plot
    class_num : int, optional
        which class number to plot. used to select the proper slice from
        amplitudes and to title the plot
    ax : plt.Axes or None, optional
        if not None, the axis to create this plot on. if None, we'll
        create a new axis with figsize (7.5, 7.5) and equal aspect

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
    ax.scatter(x, y, c=amplitudes[:, class_num], alpha=.5)
    ax.set_title(f'{hemi}, {prf_space}-space: Amplitude estimates for\n bootstrap {bootstrap}, '
                 f'class {class_num} projected onto visual field')
    ax.text(1.01, .5, f"{x.shape[0]} vertices", transform=ax.transAxes, va='center')
    return ax.figure


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
                                              target_varea=1):
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

    We also produce two plots, which show the same amplitude estiamtes,
    one in the subject's original retinotopic space, and one
    interpolated to the fsaverage retinotopic prior space. These two
    should look like they're conveying the same information, just
    sampling at different locations.
    
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

    interpolated_all = []
    for hemi in ['lh', 'rh']:
        if hemi == 'lh':
            mesh = sub.lh.with_prop(**prf_props['lh'])
        else:
            mesh = sub.rh.with_prop(**prf_props['rh'])

        prior_x, prior_y, prior_varea = get_fsaverage_coords(hemi, target_varea)
        # we need to figure out which vertices correspond to our
        # targeted visual area for constructing the overall array (which
        # should mimic the results of GLMdenoise run on the full
        # brain). we grab the first element of np.where because this is
        # a 1d array
        idx = np.where(prior_varea == target_varea)[0]
        # this should be of shape (num_bootstraps, num_classes, 1,
        # num_vertices, 1), in order to mimic the output of
        # GLMdenoise. num_vertices will be different between the two
        # hemispheres, everything else will be the same. Note that we
        # use prior_varea to get the number of vertices, NOT
        # prf_props[hemi]['models_bootstrap_00'], because we want the
        # number in fsaverage-space, not in subject-space
        _, num_classes = prf_props[hemi]['models_bootstrap_00'].shape
        interpolated_hemi = np.zeros((num_bootstraps, num_classes, 1, prior_varea.shape[0], 1))

        submesh = mesh.white_surface.submesh(mesh.white_surface.mask(('visual_area', target_varea)))
        x, y = ny.as_retinotopy(submesh, 'geographical')
        submesh = submesh.copy(coordinates=[x, y])

        # neuropythy's interpolate can only work with 2d arrays, so we
        # need to do each bootstrap separate
        for i in range(num_bootstraps):
            interp_models = submesh.interpolate([prior_x, prior_y], f'models_bootstrap_{i:02d}',
                                                method='linear')
            interpolated_hemi[i, :, 0, idx, 0] = interp_models

            if i == plot_bootstrap:
                fig = plot_amplitudes(x, y, submesh.properties[f'models_bootstrap_{i:02d}'],
                                      hemi, i, 'subject', plot_class)
                fig.savefig(save_stem + f"_models_{hemi}_b{i:02d}_c{plot_class:02d}_space-subject.png")

                fig = plot_amplitudes(prior_x, prior_y, interp_models, hemi, i, 'fsaverage',
                                      plot_class)
                fig.savefig(save_stem + f"_models_{hemi}_b{i:02d}_c{plot_class:02d}_space-prior.png")

        interpolated_all.append(interpolated_hemi)
    # concatenate into one array (vertices are on dimension 3)
    interpolated_all = np.concatenate(interpolated_all, 3)
    # and save
    with h5py.File(save_stem + '_models.hdf5', 'w') as f:
        f.create_dataset('results/models', data=interpolated_all, compression='gzip')
    return interpolated_all
