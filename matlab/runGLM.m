% runGLM.m
%
% by William F. Broderick
%
% loads in the design matrices and BOLD data, arranges them into
% the proper format and then runs Kendrick Kay's GLMdenoise on the
% data to run the GLM on the data.
%
% requires vistaSoft (which requires SPM) and GLMdenoise

function runGLM(designMatPathTemplate, boldPathTemplate, behavRuns, boldRuns, runDetailsPath, vistaSoftPath, spmPath, glmDenoisePath, seed, outputDir)
% function runGLM(designMatPathTemplate, boldPathTemplate, runs, runDetailsPath, vistaSoftPath, spmPath, glmDenoisePath, seed)
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
% <vistaSoftPath> string, path to the vistaSoft matlab toolbox.
% 
% <spmPath> string, path to the SPM matlab toolbox.
% 
% <glmDenoisePath> string, path to the GLMdenoise matlab toolbox.
% 
% <seed> integer. random number seed to pass to GLMdenoise.
% 
% <outputDir> path. Directory to save results in. Does not need to
% exist

    addpath(genpath(vistaSoftPath));
    addpath(genpath(spmPath));
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
        boldTmp = niftiRead(sprintf(boldPathTemplate, boldRuns(ii)));
        bold{ii} = boldTmp.data;
    end

    [results, denoiseddata] = GLMdenoisedata(design, bold, stim_length, TR_length, [], [], struct('seed', seed), outputDir)

    save(fullfile(outputDir, 'results.mat'), '-struct', 'results')
    save(fullfile(outputDir, 'denoiseddata.mat'), 'denoiseddata')

end