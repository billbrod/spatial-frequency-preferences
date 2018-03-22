function runGLM(designMatPathTemplate, boldPathTemplate, behavRuns, boldRuns, runDetailsPath, fsPath, glmDenoisePath, seed, outputDir, saveStem, hrfJSONpath)
% function runGLM(designMatPathTemplate, boldPathTemplate, runs, runDetailsPath, fsPath, glmDenoisePath, seed)
% 
% Loads in the design matrices and BOLD data, arranges them in proper
% format, runs Kendrick Kay's GLMdenoise, saves the output, and then
% also saves niftis for R2, R2run, modelse, and modelmd. If you want
% others, you'll have to load them from results.mat and save them
% yourself.
% 
% requires GLMdenoise and Freesurfer
%
% <designMatPathTemplate> string, template path to the design matrices
% (as .mat). Must include a string formatting symbol (e.g., %s, %02d)
% so we can find the different runs. This script assumes each run's
% design matrix is stored separately. It's assumed that the variables
% saved in these mat files are named design_matrix_run_## where ## is
% the run number in %02d format (e.g., 01, 02, 10, etc)
%
% <boldPathTemplate> string, template path to the bold data (stored as
% nifti files). Must include a string formatting symbol (e.g., %s,
% %02d) so we can find the different runs.
%
% <behavRuns> vector of ints. Which runs we should load behavioral
% data for (will fill in the string formatting symbol for
% designMatPathTemplate and boldPathTemplate). Must be the same length
% as boldRuns and it's assumed that there's a 1-to-1 correspondence
% (first behavRun corresponds to first boldRun, etc.); these are
% separate because the numbering may be different.
% 
% <boldRuns> vector of ints. Which runs we should load BOLD data for
% (will fill in the string formatting symbol for designMatPathTemplate
% and boldPathTemplate). Must be the same length as behavRuns and it's
% assumed that there's a 1-to-1 correspondence (first behavRun
% corresponds to first boldRun, etc.); these are separate because the
% numbering may be different.
% 
% <runDetailsPath> string, path to the (single) .mat file that
% includes two variables: stim_length and TR_length, which we pass
% to GlMdenoise data as stimdur and tr, respectively
% 
% <fsPath> string, path to the freesurfer matlab functions (e.g.,
% freesurfer/6.0.0/matlab).
% 
% <glmDenoisePath> string, path to the GLMdenoise matlab toolbox.
% 
% <seed> integer. random number seed to pass to GLMdenoise.
% 
% <outputDir> path. Directory to save results in. Does not need to
% exist
% 
% <saveStem> string, optional. If set, will prefix all of the results
% saved in *this function* (so the various nifti outputs, not the ones
% put out by GLMdenoisedata) with this string (outputDir
% unchanged). useful for making the outputs BIDS-like.
% 
% <hrfJSONpath> string, optional. If set, should be a path to a json file
% containing the hrf (with variable name `hrf`, like the one saved by
% this script), in which case, we'll use that hrf instead of
% optimizing for it. If unset, we'll optimize for the hrf.

    if nargin < 10
        saveStem = '';
    end
    if nargin < 11
        hrfType = 'optimize';
        hrf = [];
    else
        hrfType = 'assume';
        fid = fopen(hrfJSONpath);
        hrfJSON = jsondecode(char(fread(fid, inf)'));
        fclose(fid);
        hrf = hrfJSON.hrf;
    end
    
    addpath(genpath(fsPath));
    addpath(genpath(glmDenoisePath));

    fid = fopen(runDetailsPath);
    runDetails = jsondecode(char(fread(fid, inf)'));
    fclose(fid);

    if length(behavRuns) ~= length(boldRuns)
        error('You have different numbers of behavioral and bold runs!')
    end

    design = cell(1, length(behavRuns));
    bold = cell(1, length(boldRuns));
    for ii=1:length(behavRuns)
        design{ii} = dlmread(sprintf(designMatPathTemplate, behavRuns(ii)), '\t');
        boldTmp = MRIread(sprintf(boldPathTemplate, boldRuns(ii)));
        bold{ii} = single(boldTmp.vol);
    end

    [results, denoiseddata] = GLMdenoisedata(design, bold, runDetails.stim_length, runDetails.TR_length, hrfType, hrf, struct('seed', seed), outputDir)

    boldTmp.vol = results.modelmd{2};
    MRIwrite(boldTmp, fullfile(outputDir, strcat(saveStem, 'modelmd.nii.gz')));

    boldTmp.vol = results.modelse{2};
    MRIwrite(boldTmp, fullfile(outputDir, strcat(saveStem, 'modelse.nii.gz')));

    boldTmp.vol = results.R2;
    MRIwrite(boldTmp, fullfile(outputDir, strcat(saveStem, 'R2.nii.gz')));

    boldTmp.vol = results.R2run;
    MRIwrite(boldTmp, fullfile(outputDir, strcat(saveStem, 'R2run.nii.gz')));

    display('Saved results (non-models)  niftis')

    save(fullfile(outputDir, strcat(saveStem, 'results.mat')), '-struct', 'results', '-v7.3')
    display('Saved results.mat');
    save(fullfile(outputDir, strcat(saveStem, 'denoiseddata.mat')), 'denoiseddata', '-v7.3')
    display('Saved denoiseddata.mat');

    tosave.hrf = results.modelmd{1};
    tosave.R2 = median(results.pcR2final(results.pcvoxels));
    fid = fopen(fullfile(outputDir, strcat(saveStem, 'hrf.json')), 'w');
    fprintf(fid, jsonencode(tosave));
    fclose(fid);
    display('Saved HRF info')
end
