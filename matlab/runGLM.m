function runGLM(designMatPathTemplate, boldPathTemplate, behavRuns, boldRuns, runDetailsPath, fsPath, glmDenoisePath, seed, outputDir)
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

    addpath(genpath(fsPath));
    addpath(genpath(glmDenoisePath));

    load(runDetailsPath);

    if length(behavRuns) ~= length(boldRuns)
        error('You have different numbers of behavioral and bold runs!')
    end

    design = cell(1, length(behavRuns));
    bold = cell(1, length(boldRuns));
    for ii=1:length(behavRuns)
        load(sprintf(designMatPathTemplate, behavRuns(ii)));
        design{ii} = eval(sprintf('design_matrix_run_%02d', behavRuns(ii)));
        boldTmp = MRIread(sprintf(boldPathTemplate, boldRuns(ii)));
        bold{ii} = single(boldTmp.vol);
    end

    [results, denoiseddata] = GLMdenoisedata(design, bold, stim_length, TR_length, [], [], struct('seed', seed), outputDir)

    boldTmp.vol = results.modelmd{2};
    MRIwrite(boldTmp, fullfile(outputDir, 'modelmd.nii.gz'));

    boldTmp.vol = results.modelse{2};
    MRIwrite(boldTmp, fullfile(outputDir, 'modelse.nii.gz'));

    boldTmp.vol = results.R2;
    MRIwrite(boldTmp, fullfile(outputDir, 'R2.nii.gz'));

    boldTmp.vol = results.R2run;
    MRIwrite(boldTmp, fullfile(outputDir, 'R2run.nii.gz'));

    for ii=1:size(model{2}, 4)
        boldTmp.vol = squeeze(results.models{2}(:, :, :, ii, :));
        MRIwrite(boldTmp, fullfile(outputDir, sprintf('models_class_%02d.nii.gz', ii)));
    end

    display('Saved result niftis');

    save(fullfile(outputDir, 'results.mat'), '-struct', 'results', '-v7.3')
    display('Saved results.mat');
    save(fullfile(outputDir, 'denoiseddata.mat'), 'denoiseddata', '-v7.3')
    display('Saved denoiseddata.mat');

endp