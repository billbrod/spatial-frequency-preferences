function saveout(saveN, resultsPath, exampleBoldPath, outputDir, saveStem, fsPath)
% function saveout(resultsPath, exampleBoldPath, saveTemplatePath, fsPath)
% 
% Saves out nifti versions of one condition from the models field from
% results.mat. models is 5d, with an amplitude per voxel per bootstrap
% per condition, and we want each condition in a separate field. Note
% that saving each condition in a separate call is inefficient (the
% big overhead is in loading results.mat), but we do it this way
% because it makes things easier for Snakemake and, since we massively
% parallelize it, the loss of efficiency isn't too much of an
% issue. Also note that we will use the outputs in python, so we
% save the nifti as models_class_{n}, where n=saveN-1
% 
% requires Freesurfer
% 
% <saveN> integer or list of integers, which model number(s) to save
% out.
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

    for ii=1:length(saveN)
        n = saveN(ii);
        nii.vol = squeeze(models{2}(:, :, :, n, :));
        MRIwrite(nii, fullfile(outputDir, strcat(saveStem, sprintf('models_class_%02d.nii.gz', n-1))));
    end
    
    display('Saved result models niftis');
end
