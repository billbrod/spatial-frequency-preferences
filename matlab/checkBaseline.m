addpath(genpath('/share/apps/freesurfer/6.0.0'))
addpath(genpath('/home/wfb229/matlab-toolboxes/GLMdenoise'))
subjects = {'subj014', 'subj045'};
sessions = {'ses-03', 'ses-02'};


for jj=2:length(subjects)

    fid = fopen(sprintf('/scratch/wfb229/spatial_frequency_preferences/derivatives/GLMdenoise/stim_class/sub-wl%s/%s/sub-wl%s_%s_task-sfp_hrf.json', subjects{jj}, sessions{jj}, subjects{jj}, sessions{jj}));
    hrfJSON = jsondecode(char(fread(fid, inf)'));
    fclose(fid);
    hrf = hrfJSON.hrf;

    R2 = MRIread(sprintf('/scratch/wfb229/spatial_frequency_preferences/derivatives/GLMdenoise/stim_class/sub-wl%s/%s/sub-wl%s_%s_task-sfp_R2.nii.gz', subjects{jj}, sessions{jj}, subjects{jj}, sessions{jj}));
    [maxR2, argmax] = max(R2.vol(:));

    boldTemplate = '/scratch/wfb229/spatial_frequency_preferences/derivatives/preprocessed/sub-wl%s/%s/sub-wl%s_%s_task-sfp_run-%02d_preproc.nii.gz';
    designTemplate = '/scratch/wfb229/spatial_frequency_preferences/derivatives/design_matrices/stim_class/sub-wl%s/%s/sub-wl%s_%s_task-sfp_run-%02d_design.tsv';
    designBlanksTemplate = '/scratch/wfb229/spatial_frequency_preferences/derivatives/design_matrices/stim_class_10_blanks/sub-wl%s/%s/sub-wl%s_%s_task-sfp_run-%02d_design.tsv';

    voxel = cell(1, 12);
    design = cell(1, 12);
    designModelBlanks = cell(1, 12);
    for ii=[1,2,3,4,5,6,7,8,9,10,11,12]
        design{ii} = dlmread(sprintf(designTemplate, subjects{jj}, sessions{jj}, subjects{jj}, sessions{jj}, ii), '\t');
        designModelBlanks{ii} = dlmread(sprintf(designBlanksTemplate, subjects{jj}, sessions{jj}, subjects{jj}, sessions{jj}, ii), '\t');
        boldTmp = MRIread(sprintf(boldTemplate, subjects{jj}, sessions{jj}, subjects{jj}, sessions{jj}, ii));
        boldTmp = reshape(boldTmp.vol, 104*104*66, size(boldTmp.vol, 4));
        voxel{ii} = boldTmp(argmax, :);
    end

    stimdur = 4;
    tr = 1;
    opt.numpcstotry = 0;
    opt.maxpolydeg = 2;

    results = GLMdenoisedata(design,voxel,stimdur,tr,'assume',hrf,opt,sprintf('/scratch/wfb229/baselineTest/%s_noblanks', subjects{jj}));
    resultsBlanks = GLMdenoisedata(designModelBlanks,voxel,stimdur,tr,'assume',hrf,opt, sprintf('/scratch/wfb229/baselineTest/%s_blanks', subjects{jj}));
    h = figure;
    plot(results.modelmd{2});
    hold on;
    plot(resultsBlanks.modelmd{2}(1:end-1) - resultsBlanks.modelmd{2}(end));
    legend('regular' , 'modeling blanks');
    title(sprintf('%s, baseline is %.04f', subjects{jj}, resultsBlanks.modelmd{2}(end)));
    saveas(h, sprintf('/scratch/wfb229/baselineTest/%s_baseline.svg', subjects{jj}));
    save(sprintf('/scratch/wfb229/baselineTest/%s_BaselineTest.mat', subjects{jj}), 'maxR2', 'argmax', 'voxel', 'design', 'hrf', 'designModelBlanks', 'stimdur', 'tr', 'opt');

end