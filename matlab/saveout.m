function saveout(resultsPath, exampleBoldPath, outputDir, saveStem, fsPath)
% function saveout(resultsPath, exampleBoldPath, saveTemplatePath, fsPath)
% 
% Saves out nifti versions of the models field from results. This
% is 5d, with an amplitude per voxel per bootstrap per condition,
% and each condition is saved into a separate nifti file.
% 
% requires Freesurfer
% 
% <resultsPath> string, path to the results.mat file that
% GLMdenoise puts out
% 
% <exampleBoldPath> string, path to a single nifti. We will load
% this nifti in and replace its data with the relevant stats from
% resultsPath (so our outputted data will have the same header as
% this file)
%
% <outputDir> path. Directory to save results in. Must exist.
% 
% <saveStem> string, optional. If set, will prefix all of the results
% saved in *this function* (so the various nifti outputs, not the ones
% put out by GLMdenoisedata) with this string (outputDir
% unchanged). useful for making the outputs BIDS-like.
% 
% <fsPath> string, path to the freesurfer matlab functions (e.g.,
% freesurfer/6.0.0/matlab).
    
    addpath(genpath(fsPath));

    load(resultsPath, 'models');

    nii = MRIread(exampleBoldPath);

    for ii=1:size(models{2}, 4)
        nii.vol = squeeze(models{2}(:, :, :, ii, :));
        MRIwrite(nii, fullfile(outputDir, strcat(saveStem, sprintf('models_class_%02d.nii.gz', ii-1))));
    end
    
    display('Saved result models niftis');
end
