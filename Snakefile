import os
import itertools
import warnings
import numpy as np
from glob import glob
import neuropythy as ny
import re

configfile:
    "config.yml"
if not os.path.isdir(config["DATA_DIR"]):
    raise Exception("Cannot find the dataset at %s" % config["DATA_DIR"])
if os.system("module list") == 0:
    # then we're on the cluster
    ON_CLUSTER = True
    shell.prefix("module load fsl/5.0.10; module load freesurfer/6.0.0; module load matlab/2020a; "
                 "export FSLOUTPUTTYPE=NIFTI_GZ; "
                 "export SUBJECTS_DIR=%s/derivatives/freesurfer; " % config["DATA_DIR"])
else:
    ON_CLUSTER = False
    shell.prefix("export SUBJECTS_DIR=%s/derivatives/freesurfer; " % config["DATA_DIR"])
    # need to set SUBJECTS_DIR here so that the run steps also use the right
    # SUBJECTS_DIR (setting shell.prefix above makes sure the shell steps do)
    os.environ['SUBJECTS_DIR'] = os.path.join(config['DATA_DIR'], 'derivatives', 'freesurfer')


SUBJECTS = ['sub-wlsubj001', 'sub-wlsubj004', 'sub-wlsubj042', 'sub-wlsubj045', 'sub-wlsubj014',
            'sub-wlsubj064', 'sub-wlsubj081', 'sub-wlsubj095', 'sub-wlsubj007', 'sub-wlsubj062',
            'sub-wlsubj046', 'sub-wlsubj006', 'sub-wlsubj121', 'sub-wlsubj115', 'sub-wlsubj114']
SESSIONS = {'sub-wlsubj001': ['ses-pilot01', 'ses-01', 'ses-02', 'ses-04'],
            'sub-wlsubj004': ['ses-03'],
            'sub-wlsubj042': ['ses-pilot00', 'ses-pilot01', 'ses-01', 'ses-02'],
            'sub-wlsubj045': ['ses-pilot01', 'ses-01', 'ses-02', 'ses-04', 'ses-03'],
            'sub-wlsubj014': ['ses-03'], 'sub-wlsubj064': ['ses-04'], 'sub-wlsubj081': ['ses-04'],
            'sub-wlsubj095': ['ses-04'], 'sub-wlsubj007': ['ses-04'], 'sub-wlsubj062': ['ses-04'],
            'sub-wlsubj046': ['ses-04'], 'sub-wlsubj105': ['ses-04'], 'sub-wlsubj006': ['ses-04'],
            'sub-wlsubj121': ['ses-04'], 'sub-wlsubj115': ['ses-04'], 'sub-wlsubj114': ['ses-04']}
TASKS = {('sub-wlsubj001', 'ses-pilot01'): 'task-sfp', ('sub-wlsubj001', 'ses-01'): 'task-sfp',
         ('sub-wlsubj001', 'ses-02'): 'task-sfpconstant',
         ('sub-wlsubj042', 'ses-pilot00'): 'task-sfp', ('sub-wlsubj042', 'ses-pilot01'): 'task-sfp',
         ('sub-wlsubj042', 'ses-01'): 'task-sfpconstant', ('sub-wlsubj042', 'ses-02'): 'task-sfp',
         ('sub-wlsubj045', 'ses-pilot01'): 'task-sfp',
         ('sub-wlsubj045', 'ses-01'): 'task-sfpconstant',  ('sub-wlsubj045', 'ses-02'): 'task-sfp',
         ('sub-wlsubj014', 'ses-03'): 'task-sfp', ('sub-wlsubj004', 'ses-03'): 'task-sfp',
         ('sub-wlsubj045', 'ses-04'): 'task-sfprescaled', ('sub-wlsubj045', 'ses-03'): 'task-sfp',
         ('sub-wlsubj064', 'ses-04'): 'task-sfprescaled', ('sub-wlsubj081', 'ses-04'): 'task-sfprescaled',
         ('sub-wlsubj095', 'ses-04'): 'task-sfprescaled', ('sub-wlsubj007', 'ses-04'): 'task-sfprescaled',
         ('sub-wlsubj062', 'ses-04'): 'task-sfprescaled', ('sub-wlsubj046', 'ses-04'): 'task-sfprescaled',
         ('sub-wlsubj105', 'ses-04'): 'task-sfprescaled', ('sub-wlsubj006', 'ses-04'): 'task-sfprescaled',
         ('sub-wlsubj121', 'ses-04'): 'task-sfprescaled', ('sub-wlsubj115', 'ses-04'): 'task-sfprescaled',
         ('sub-wlsubj114', 'ses-04'): 'task-sfprescaled', ('sub-wlsubj001', 'ses-04'): 'task-sfprescaled',}
# these are the subject, session pairs where I didn't add the task to the protocol name and so some
# extra work is necessary.
WRONG_TASKS = {('sub-wlsubj001', 'ses-pilot01'): 'task-TASK',
               ('sub-wlsubj042', 'ses-01'): 'task-TASK', ('sub-wlsubj014', 'ses-03'): 'task-TASK',
               ('sub-wlsubj042', 'ses-pilot00'): 'task-TASK',
               ('sub-wlsubj042', 'ses-pilot01'): 'task-spatialfrequency',
               ('sub-wlsubj045', 'ses-pilot01'): 'task-spatialfrequency',
               ('sub-wlsubj064', 'ses-04'): 'task-sfprescaledcmrr',
               ('sub-wlsubj001', 'ses-04'): 'task-TASK'}
# every sub/ses pair that's not in here has the full number of runs, 12
NRUNS = {('sub-wlsubj001', 'ses-pilot01'): 9, ('sub-wlsubj042', 'ses-pilot00'): 8,
         ('sub-wlsubj045', 'ses-04'): 7}
N_CLASSES = {'ses-pilot00': 52, 'ses-pilot01': 52, 'ses-01': 48, 'ses-02': 48, 'ses-03': 48,
             'ses-04': 48}
VAREAS = [1]
MODEL_TYPES = ['iso_constant_iso', 'iso_scaling_iso', 'iso_full_iso',
               'absolute_full_iso', 'relative_full_iso', 'full_full_iso',
               'iso_full_absolute', 'iso_full_relative', 'iso_full_full',
               'full_full_absolute', 'full_full_relative',
               'absolute_full_absolute', 'relative_full_relative', 'full_full_full']
def get_n_classes(session, mat_type):
    if mat_type == 'all_visual':
        return 1
    else:
        try:
            n = N_CLASSES[session]
        except KeyError:
            # then this is probably the groupaverage session, which is
            # slightly differnetly formatted
            ses = re.findall('(ses-[0-9]+)_v[0-9]+_s[0-9]+',session)[0]
            n = N_CLASSES[ses]
        if 'blanks' in mat_type:
            n += 1
        return n
def get_stim_files(wildcards):
    if 'pilot' in wildcards.session:
        session_prefix = wildcards.session + "_"
    else:
        session_prefix = ""
    task_prefix = wildcards.task
    file_stem = os.path.join(config['DATA_DIR'], 'stimuli', task_prefix+"_"+session_prefix+"{rest}")
    return {'stim': file_stem.format(rest='stimuli.npy'),
            'desc_csv': file_stem.format(rest='stim_description.csv')}
# the goal of these two dictionaries is to always have a unique integer
# when summed together. to that end the subjects one should increase
# first along the ones digit and then, when you've run out of digits,
# along the hundreds
SUB_SEEDS = {'sub-wlsubj001': 1, 'sub-wlsubj042': 2, 'sub-wlsubj045': 3, 'sub-wlsubj004': 4,
             'sub-wlsubj014': 5, 'sub-wlsubj004': 6, 'sub-wlsubj064': 7, 'sub-wlsubj081': 8,
             'sub-wlsubj095': 9, 'sub-wlsubj062': 0, 'sub-wlsubj007': 100,
             'sub-wlsubj046': 101, 'sub-wlsubj105': 102, 'sub-wlsubj006': 103,
             'sub-wlsubj121': 104, 'sub-wlsubj115': 105, 'sub-wlsubj114': 106}
# while session should increment along the tens digit
SES_SEEDS = {'ses-pilot00': 10, 'ses-pilot01': 20, 'ses-01': 30, 'ses-02': 40, 'ses-03': 50,
             'ses-04': 60}
wildcard_constraints:
    subject="sub-[a-z0-9-]+|sub-groupaverage_i-[a-z]+",
    fs_subject="[a-z0-9]+",
    subjects="(sub-[a-z0-9]+,?)+",
    session="ses-[a-z0-9]+|ses-[0-9]+_v[0-9]+_s[0-9]+",
    sessions="(ses-[a-z0-9]+,?)+",
    run="run-[0-9]+",
    filename_ext='[a-zA-Z0-9_]+\.[a-z.]+',
    filename='[a-zA-Z0-9_]+',
    task="task-[a-z0-9]+",
    tasks="(task-[a-z0-9]+,?)+",
    vareas="[0-9-]+",
    plot_varea="[0-9-]+",
    eccen="[0-9]+-[0-9]+",
    eccen_range="[0-9]+-[0-9]+",
    df_mode="summary|full",
    atlas_type="bayesian_posterior|atlas|data",
    plot_func="[a-z]+",
    col="[a-z-]+",
    row="[a-z-]+",
    hue="[a-z-]+",
    y="[a-z-]+",
    binning="[a-z_]+bin",
    stimulus_class="([0-9,]+|None)",
    bootstrap_num="([0-9,]+|None)",
    period_orientation_type="[a-z-]+",
    eccentricity_type="[a-z-]+",
    amplitude_orientation_type="[a-z-]+",
    model_type="[a-z-_]+",
    crossval_seed="[0-9]+",
    gpus="[0-9]+",
    y_val="period|frequency",
    groupaverage="individual|sub-groupaverage",
    summary_func="mean|median",
    df_filter="filter-any|filter-mean|no-filter",
    orient="h|v",
    sort="sort_|",
    doubleup="doubleup_|",
    schematic_type="(2d|2d-inputs|background)",

#  there's a bit of (intentional) ambiguity in the output folders of GLMdenoise_fixed_hrf and
#  GLMdenoise (GLMdenoise_fixed_hrf's output folder is "{mat_type}_fixed_hrf_{input_mat}", while
#  GLMdenoise's is "{mat_type}"; if {mat_type} is unconstrained, obviously GLMdenoise could also
#  match that folder). if something could be interpreted as GLMdenoise_fixed_hrf, we want it to be
#  interpreted that way (because we'll never have a mat_type that includes "fixed_hrf"). However,
#  we don't want to constrain what mat_type matches because we want to be able to treat the output
#  folder created by GLMdenoise_fixed_hrf the same as the output folder created by GLMdenoise for
#  the purpose of later calls. Similarly, create_GLMdenoise_fixed_hrf_json needs to happen before
#  create_GLMdenoise_json. It doesn't matter where the create_*_json rules happen relative to the
#  GLMdenoise* ones, but they all need to be in a single chain, so this works.
ruleorder:
    compute_groupaverage > GLMdenoise_fixed_hrf > GLMdenoise > create_GLMdenoise_fixed_hrf_json > create_GLMdenoise_json


def create_crossval_idx(leave_n_out, session, mat_type, seed=None):
    if seed is not None:
        np.random.seed(seed)
    total_n = get_n_classes(session, mat_type)
    idx = np.arange(total_n)
    np.random.shuffle(idx)
    # if total_n/leave_n_out isn't an integer, then some of these will be longer than the other
    splits = np.array_split(idx, total_n / leave_n_out)
    # this returns a list of strings, each of which looks like e.g., #,#,#,# (if leave_n_out=4)
    return [','.join(["%02d"%j for j in i]) for i in splits]


rule model_learning_hyperparams_full:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_bayesian_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12",
                     "learning_hyperparams_full", "g0_all_models.csv"),


rule model_recovery_initial:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_bayesian_posterior_sub-wlsubj045_ses-04_task-sfprescaled_v1_e1-12",
                     "model_recovery", "g0_all_models.csv"),


rule model_recovery_cv_initial:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_bayesian_posterior_sub-wlsubj045_ses-04_task-sfprescaled_v1_e1-12",
                     "model_recovery_cv", "b10_r.001_g0_s0_all_models.csv"),


def get_model_subj_outputs(model_type, subject, session, task, batch_size=10, learning_rate=1e-3,
                           crossval_seed=None, bootstrap_num=None, vareas=1, eccen='1-12', df_mode='summary', gpus=0,
                           mat_type='stim_class', atlas_type='bayesian_posterior', modeling_goal='initial',
                           df_filter='filter-mean'):
    output_path = os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}",
                               "{atlas_type}", "{df_filter}", "{modeling_goal}", "{subject}", "{session}",
                               "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch}_"
                               "r{lr}_g{gpus}_c{{crossval}}_n{bootstrap_num}_{model_type}_loss.csv")
    output_path = output_path.format(subject=subject, session=session, task=task, batch=batch_size,
                                     lr=learning_rate, model_type=model_type, vareas=vareas,
                                     eccen=eccen, df_mode=df_mode, gpus=gpus, atlas_type=atlas_type,
                                     mat_type=mat_type, modeling_goal=modeling_goal,
                                     bootstrap_num=bootstrap_num, df_filter=df_filter)
    if crossval_seed is None:
        return output_path.format(crossval=None)
    else:
        return [output_path.format(crossval=n) for n in create_crossval_idx(4, session, mat_type, int(crossval_seed))]


def get_simulated_model_outputs(model_type, sim_model_type, noise_level, num_voxels,
                                batch_size, learning_rate, sigma, sf_ecc_slope, sf_ecc_intercept,
                                rel_mode_cardinals, rel_mode_obliques, rel_amplitude_cardinals,
                                rel_amplitude_obliques, abs_mode_cardinals, abs_mode_obliques,
                                abs_amplitude_cardinals, abs_amplitude_obliques, noise_source,
                                crossval_seed=None, gpus=0, modeling_goal='model_recovery'):
    output_path = os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated",
                               "{noise_source}", "{modeling_goal}",
                               "n{num_voxels}_{sim_model_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_"
                               "rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_"
                               "rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_"
                               "amc{abs_mode_cardinals}_amo{abs_mode_obliques}_"
                               "aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_"
                               "l{noise_level}_b{batch_size}_r{learning_rate}_g{gpus}_"
                               "c{{crossval}}_{model_type}_loss.csv")
    output_path = output_path.format(batch_size=batch_size, learning_rate=learning_rate,
                                     gpus=gpus, modeling_goal=modeling_goal, num_voxels=num_voxels,
                                     sim_model_type=sim_model_type, noise_level=noise_level, 
                                     sf_ecc_slope=sf_ecc_slope, sf_ecc_intercept=sf_ecc_intercept,
                                     rel_mode_cardinals=rel_mode_cardinals, sigma=sigma,
                                     rel_mode_obliques=rel_mode_obliques, model_type=model_type,
                                     rel_amplitude_cardinals=rel_amplitude_cardinals,
                                     rel_amplitude_obliques=rel_amplitude_obliques,
                                     abs_mode_cardinals=abs_mode_cardinals,
                                     abs_mode_obliques=abs_mode_obliques,
                                     abs_amplitude_cardinals=abs_amplitude_cardinals,
                                     abs_amplitude_obliques=abs_amplitude_obliques,
                                     noise_source=noise_source).replace('0.', '.')
    if crossval_seed is None:
        return output_path.format(crossval=None)
    else:
        ses = re.findall(r'(ses-[0-9]+)', noise_source)[0]
        return [output_path.format(crossval=n) for n in create_crossval_idx(4, ses, 'stim_class', int(crossval_seed))]


rule model_all_subj_bootstrap:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class",
                     "bayesian_posterior", "filter-mean", "bootstrap",
                     "individual_task-sfprescaled_v1_e1-12_full_b10_r0.001_g0_full_full_full_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class",
                     "bayesian_posterior", "filter-mean", "bootstrap",
                     "individual_task-sfprescaled_v1_e1-12_full_b10_r0.001_g0_full_full_absolute_all_models.csv"),
    


rule model_all_subj_visual_field:
    input:
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class",
                      "bayesian_posterior", "filter-mean", "visual_field_%s" % p,
                      "individual_task-sfprescaled_v1_e1-12_summary_b10_r0.001_g0_full_full_full_all_models.csv") for p in
         ['upper', 'lower', 'left', 'right', 'inner', 'outer']],


rule model_all_subj:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class",
                     "bayesian_posterior", "filter-mean", "initial",
                     "individual_task-sfprescaled_v1_e1-12_summary_b10_r0.001_g0_full_full_full_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class",
                     "bayesian_posterior", "filter-mean", "initial",
                     "individual_task-sfprescaled_v1_e1-12_summary_b10_r0.001_g0_full_full_absolute_all_models.csv"),


rule model_all_subj_cv:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class",
                     "bayesian_posterior", "filter-mean", "initial_cv",
                     "individual_task-sfprescaled_v1_e1-12_summary_b10_r0.001_g0_s0_all_models.csv"),


def get_groupaverage_all(tuning_type='2d', interp='linear', session='ses-04', task='task-sfprescaled',
                         model_type='full_full_absolute', vareas='1', eccen='1-12', batch_size=10,
                         learning_rate=0.001, gpus=0, df_mode='summary', mat_type='stim_class',
                         atlas_type='bayesian_posterior', modeling_goal='initial',
                         df_filter='filter-mean'):
    if modeling_goal != 'initial':
        return []
    if tuning_type == '2d':
        path = os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", mat_type,
                            atlas_type, df_filter, "initial", f"sub-groupaverage_i-{interp}",
                            f"{session}_v{vareas}_s{{n:02d}}", f"sub-groupaverage_i-{interp}_{session}"
                            f"_v{vareas}_s{{n:02d}}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_"
                            f"r{learning_rate}_g{gpus}_cNone_nNone_{model_type}_loss.csv")
    elif tuning_type == 'curve':
        path = os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', mat_type,
                            atlas_type, f'sub-groupaverage_i-{interp}', f'{session}_v{vareas}_s{{n:02d}}',
                            f'sub-groupaverage_i-{interp}_{session}_v{vareas}_s{{n:02d}}_{task}_'
                            f'v{vareas}_e{eccen}_eccen_bin_{df_mode}.csv')
        
    seeds = list(range(104))
    # there are 4 seeds that won't work, so we remove them. there are
    # some voxels where some subjects have NaNs after
    # interpolation. when doing our weighted average across subjects, we
    # ignore those NaNs, but for these seeds, they managed to pick
    # subjects that *all* have NaNs in at least one voxel, so there's
    # nothing we can do.
    for i in [17, 31, 51, 65]:
        seeds.remove(i)
    return [path.format(n=n) for n in seeds]


rule groupaverage_all:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class",
                     "bayesian_posterior", "filter-mean", "initial",
                     "sub-groupaverage_task-sfprescaled_v1_e1-12_summary_b10_r0.001_g0_full_full_full_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class",
                     "bayesian_posterior", "filter-mean", "initial",
                     "sub-groupaverage_task-sfprescaled_v1_e1-12_summary_b10_r0.001_g0_full_full_absolute_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "stim_class",
                     "bayesian_posterior", "sub-groupaverage_ses-04_v1_e1-12_eccen_bin_tuning_curves_summary.csv"),
        


rule all_check_plots:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', "{subject}", "{atlas_type}", 'varea_plot.png').format(subject=s, atlas_type=a)
         for s in SUBJECTS for a in ['bayesian_posterior', 'atlas'] if 'ses-04' in SESSIONS[s]],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', "{subject}", "{atlas_type}", '{prf_prop}_plot.png').format(subject=s, atlas_type=a, prf_prop=p)
         for s in SUBJECTS for a in ['bayesian_posterior', 'atlas', 'data'] for p in ['angle', 'eccen'] if 'ses-04' in SESSIONS[s]],
        [os.path.join(config["DATA_DIR"], "derivatives", "GLMdenoise", "stim_class", "bayesian_posterior", "{subject}", "{session}", "figures_{task}", "FinalModel_maps.png").format(
            subject=sub, session=ses, task=TASKS[(sub, ses)]) for sub in SUBJECTS for ses in SESSIONS[sub] if ses=='ses-04'],
        [os.path.join(config["DATA_DIR"], "derivatives", "first_level_binned", "stim_class", "bayesian_posterior", "{subject}", "{session}", "{subject}_{session}_{task}_v1_e1-12_eccen_bin_full_data.svg").format(
            subject=sub, session=ses, task=TASKS[(sub, ses)]) for sub in SUBJECTS for ses in SESSIONS[sub] if ses=='ses-04'],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', 'stim_class', 'bayesian_posterior', '{subject}', '{session}', '{subject}_{session}_{task}_v1_e1-12_summary'
                      '_{df_filter}_precision_check.png').format(subject=sub, session=ses, task=TASKS[(sub, ses)], df_filter=filt) for sub in SUBJECTS for ses in SESSIONS[sub]
         for filt in ['filter-mean', 'no-filter'] if ses=='ses-04'],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "bayesian_posterior", "filter-mean", "initial_cv", "{subject}", "{session}", "{subject}_{session}_{task}_"
                      "v1_e1-12_summary_b10_r0.001_g0_s0_{model_type}_cv_loss_normed_loss.png").format(subject=sub, session=ses, task=TASKS[(sub, ses)], model_type=m) for sub in SUBJECTS
         for ses in SESSIONS[sub] for m in ['iso_constant_iso', 'iso_scaling_iso', 'iso_full_iso'] if ses=='ses-04'],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "bayesian_posterior", "filter-mean", "initial_cv", "{subject}", "{session}", "{subject}_{session}_{task}_"
                      "v1_e1-12_summary_b10_r0.001_g0_s0_cv_loss_comp_normed_loss.png").format(subject=sub, session=ses, task=TASKS[(sub, ses)]) for sub in SUBJECTS
         for ses in SESSIONS[sub] if ses=='ses-04'],
        os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'paper', 'voxel_exclusion_filter-mean_task-sfprescaled.svg'),
        [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'paper', 'individual_filter-mean_{}_training-loss-check_task-sfprescaled.svg').format(t)
             for t in ['initial_cv', 'bootstrap']],


rule GLMdenoise_all_visual:
    input:
        [os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "all_visual", "bayesian_posterior",  "{subject}", "{session}", "{subject}_{session}_{task}_results.mat").format(subject=sub, session=ses, task=TASKS[(sub, ses)]) for sub in SUBJECTS for ses in SESSIONS[sub]],


rule plots_modeling_blanks:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_localsf.svg').format(mat_type="stim_class_10_blanks_fixed_hrf_stim_class", atlas_type='bayesian_posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary'] for v in VAREAS],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_stim_prop.svg').format(mat_type="stim_class_10_blanks_fixed_hrf_stim_class", atlas_type='bayesian_posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary'] for v in VAREAS],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_data.svg').format(mat_type="stim_class_10_blanks_fixed_hrf_stim_class", atlas_type='bayesian_posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary'] for v in VAREAS],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_tuning_params.svg').format(mat_type="stim_class_10_blanks_fixed_hrf_stim_class", atlas_type='bayesian_posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary'] for v in VAREAS],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_summary_tuning_curves_check_varea={v}.svg').format(mat_type="stim_class_10_blanks_fixed_hrf_stim_class", atlas_type='bayesian_posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], vareas=v, eccen='1-12', binning='eccen_bin', v=v) for sub in SUBJECTS for ses in SESSIONS[sub] for v in VAREAS],


rule plots_all:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_localsf.svg').format(mat_type="stim_class", atlas_type='bayesian_posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary'] for v in VAREAS],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_stim_prop.svg').format(mat_type="stim_class", atlas_type='bayesian_posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary'] for v in VAREAS],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_data.svg').format(mat_type="stim_class", atlas_type='bayesian_posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary'] for v in VAREAS],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_tuning_params.svg').format(mat_type="stim_class", atlas_type='bayesian_posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary'] for v in VAREAS],
        # [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_full_tuning_curves_check_varea={v}_bootstrap={b:02d}.svg').format(mat_type="stim_class", atlas_type='bayesian_posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], vareas=v, eccen='1-12', binning='eccen_bin', v=v, b=b) for sub in SUBJECTS for ses in SESSIONS[sub] for v in VAREAS for b in range(100)],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_summary_tuning_curves_check_varea={v}.svg').format(mat_type="stim_class", atlas_type='bayesian_posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], vareas=v, eccen='1-12', binning='eccen_bin', v=v) for sub in SUBJECTS for ses in SESSIONS[sub] for v in VAREAS],


rule tuning_curves_all:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}.csv').format(mat_type="stim_class", atlas_type='bayesian_posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary', 'full'] for v in VAREAS],


rule first_level_all:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}.csv').format(mat_type="stim_class", atlas_type='bayesian_posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary', 'full'] for v in VAREAS],


rule plots_VSS_abstract:
    # these recreate the data examined for the first year talk and the VSS abstract, when I was
    # using the "prior" atlas
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_localsf.svg').format(mat_type="stim_class", atlas_type='atlas', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='eccen_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_stim_prop.svg').format(mat_type="stim_class", atlas_type='atlas', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='eccen_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_data.svg').format(mat_type="stim_class", atlas_type='atlas', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='eccen_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_tuning_params.svg').format(mat_type="stim_class", atlas_type='atlas', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='eccen_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary']],
        # [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_full_tuning_curves_check_varea=1_bootstrap={b:02d}.svg').format(mat_type="stim_class", atlas_type='atlas', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], vareas='1', eccen='2-8', binning='eccen_bin', b=b) for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for b in range(100)],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_summary_tuning_curves_check_varea=1.svg').format(mat_type="stim_class", atlas_type='atlas', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], vareas='1', eccen='2-8', binning='eccen_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045']]


rule GLMdenoise_all:
    input:
        [os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "stim_class", "bayesian_posterior",  "{subject}", "{session}", "figures_{task}", "FinalModel_maps.png").format(subject=sub, session=ses, task=TASKS[(sub, ses)]) for sub in SUBJECTS for ses in SESSIONS[sub]],


rule preprocess_all:
    input:
        [os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "{subject}_{session}_{task}_{run}_preproc.nii.gz").format(subject=sub, session=ses, task=TASKS[(sub, ses)], run="run-%02d"%i) for sub in SUBJECTS for ses in SESSIONS[sub] for i in range(1, NRUNS.get((sub, ses), 12)+1)],


rule move_all:
    input:
        [os.path.join(config['DATA_DIR'], '{subject}', '{session}').format(subject=subj, session=ses) for subj in SUBJECTS
         for ses in SESSIONS[subj]]


rule create_tsv_all:
    input:
        [os.path.join(config['DATA_DIR'], 'sourcedata', '{subject}', '{session}', '{subject}_{session}_{task}_notes.md').format(subject=subj, session=ses, task=TASKS[(subj, ses)])
         for subj in SUBJECTS for ses in SESSIONS[subj]]


rule stimuli:
    output:
        "data/stimuli/task-sfp_stimuli.npy",
        "data/stimuli/task-sfp_stim_description.csv",
        "data/stimuli/task-sfpconstant_stimuli.npy",
        "data/stimuli/task-sfpconstant_stim_description.csv",
        "data/stimuli/antialiasing_mask.npy",
    shell:
        "python -m sfp.stimuli -c"


rule rescaled_stimuli:
    input:
        "data/stimuli/mtf_func.pkl"
    output:
        "data/stimuli/task-sfprescaled_stimuli.npy",
        "data/stimuli/task-sfprescaled_stim_description.csv",
        "data/stimuli/task-sfpconstantrescaled_stimuli.npy",
        "data/stimuli/task-sfpconstantrescaled_stim_description.csv",
        "data/stimuli/antialiasing_mask.npy",
    params:
        stim_name = lambda wildcards: os.path.split(wildcards.output[0])[-1],
        csv_name = lambda wildcards: os.path.split(wildcards.output[1])[-1],
    shell:
        "python -m sfp.stimuli -c --mtf {input} -n {params.stim_name} -d {params.csv_name}"


# old way of generating stimuli, only used subject name
rule stimuli_idx_old:
    output:
        ["data/stimuli/{subject}_run%02d_idx.npy" % i for i in range(12)]
    params:
        seed = lambda wildcards: SUB_SEEDS[wildcards.subject]
    shell:
        "python -m sfp.stimuli --subject_name {wildcards.subject} -i -s {params.seed}"


def get_seed(wildcards):
    try:
        seed = SUB_SEEDS[wildcards.subject] + SES_SEEDS[wildcards.session]
    except KeyError:
        warnings.warn(f"Subject {wildcards.subject} with session {wildcards.session} "
                      "not in list of included subjects / sessions; setting seed=0")
        seed = 0
    return seed


# current way of generating stimuli, uses both subject and session name
rule stimuli_idx:
    output:
        ["data/stimuli/{subject}_{session}_run%02d_idx.npy" % i for i in range(12)]
    params:
        seed = get_seed,
    shell:
        "python -m sfp.stimuli --subject_name {wildcards.subject}_{wildcards.session}"
        " -i -s {params.seed}"


rule presented_spatial_frequency:
    input:
        os.path.join(config['DATA_DIR'], 'stimuli', 'antialiasing_mask.npy'),
        os.path.join(config['DATA_DIR'], 'stimuli', '{task}_stim_description.csv')
    output:
        os.path.join(config['DATA_DIR'], 'stimuli', '{task}_presented_frequencies.csv'),
    log:
        os.path.join(config['DATA_DIR'], 'code', 'stimuli', '{task}_presented_frequencies-%j.log'),
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'stimuli', '{task}_presented_frequencies_benchmark.txt'),
    run:
        import sfp
        import numpy as np
        import pandas as pd
        df = sfp.stimuli.find_all_presented_sfs(pd.read_csv(input[1]),
                                                stimulus_mask=np.load(input[0]))
        df.to_csv(output[0], index=False)

# we assume the prf solutions and freesurfer have been computed
# separately and copy them in here.
rule copy_prf_solutions:
    input:
        fs = os.path.join(config['SUBJECTS_DIR'], '{fs_subject}'),
        data = [os.path.join(config['RETINOTOPY_DIR'], 'derivatives', 'vistasoft', 'sub-{{fs_subject}}', 'ses-nyu3t01', 'Outputs', '{}.full-{}.mgz').format(h, d)
                for h in ['rh', 'lh'] for d in ['angle', 'eccen', 'sigma', 'xcrds', 'ycrds', 'vexpl']],
        posterior = [os.path.join(config['RETINOTOPY_DIR'], 'derivatives', 'vistasoft', 'sub-{{fs_subject}}', 'ses-nyu3t01', 'Outputs', '{}.inferred_{}.mgz').format(h, d)
                     for h in ['rh', 'lh'] for d in ['angle', 'eccen', 'sigma', 'varea']],
    output:
        fs = directory(os.path.join(config['DATA_DIR'], 'derivatives', 'freesurfer', '{fs_subject}')),
        atlas = [os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', 'sub-{{fs_subject}}', 'atlas', '{}.benson14_{}.mgz').format(h, d)
                 for h in ['rh', 'lh'] for d in ['angle', 'eccen', 'sigma', 'varea']],
        data = [os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', 'sub-{{fs_subject}}', 'data', '{}.full-{}.mgz').format(h, d)
                for h in ['rh', 'lh'] for d in ['angle', 'eccen', 'sigma', 'xcrds', 'ycrds', 'vexpl']],
        posterior = [os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', 'sub-{{fs_subject}}', 'bayesian_posterior', '{}.inferred_{}.mgz').format(h, d)
                     for h in ['rh', 'lh'] for d in ['angle', 'eccen', 'sigma', 'varea']],
    log:
        os.path.join(config['DATA_DIR'], 'code', 'copy_prf_solutions', 'sub-{fs_subject}-%j.log')
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'copy_prf_solutions', 'sub-{fs_subject}_benchmark.txt')
    run:
        import shutil
        shell(f"rsync -avPLuz {input.fs}/ {output.fs}")
        atlas_input = [os.path.join(config['DATA_DIR'], 'derivatives', 'freesurfer', '{}', 'surf', '{}.benson14_{}.mgz').format(wildcards.fs_subject, h, d)
                       for h in ['rh', 'lh'] for d in ['angle', 'eccen', 'sigma', 'varea']]
        for i, f in enumerate(atlas_input):
            shutil.copy(f, output.atlas[i])
        for i, f in enumerate(input.data):
            shutil.copy(f, output.data[i])
        for i, f in enumerate(input.posterior):
            shutil.copy(f, output.posterior[i])


rule move_off_tesla:
    input:
        os.path.join(config["TESLA_DIR"], "{subject}", "{session}"),
        os.path.join(config["TESLA_DIR"], "sourcedata", "{subject}", "{session}"),
    output:
        directory(os.path.join(config["DATA_DIR"], "{subject}", "{session}")),
        directory(os.path.join(config["DATA_DIR"], "sourcedata", "{subject}", "{session}")),
        directory(os.path.join(config["DATA_DIR"], "derivatives", "mriqc_reports", "{subject}", "{session}")),
    log:
        os.path.join(config["DATA_DIR"], "code", "move_off_tesla", "{subject}_{session}-%j.log")
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "move_off_tesla", "{subject}_{session}_benchmark.txt")
    params:
        wrong_task = lambda wildcards: WRONG_TASKS.get((wildcards.subject, wildcards.session), None),
        right_task = lambda wildcards: TASKS[(wildcards.subject, wildcards.session)]
    run:
        import glob
        import shutil
        import os
        os.makedirs(output[2])
        reports_path = os.path.join(config["TESLA_DIR"], "derivatives", "mriqc_reports", "{subject}_{session}_*").format(**wildcards)
        for f in glob.glob(reports_path):
            shutil.copy(f, output[2])
        shell("rsync --exclude=*events.tsv -avPLuz %s/ %s" % (input[0], output[0]))
        shell("rsync -avPLuz %s/ %s" % (input[1], output[1]))
        # for some scanning sessions, I put the wrong task name in the scanning protocol, so the
        # automatic BIDS extractor put the wrong name in...
        if params.wrong_task is not None:
            # we rename all the files that contain the wrong_task
            for f in glob.glob(os.path.join(output[0], '*', '*'+params.wrong_task+'*')):
                shutil.move(f, f.replace(params.wrong_task, params.right_task))
            # we rename all the files that contain the wrong_task
            for f in glob.glob(os.path.join(output[1], '*', '*'+params.wrong_task+'*')):
                shutil.move(f, f.replace(params.wrong_task, params.right_task))
            for f in glob.glob(os.path.join(output[2], '*'+params.wrong_task+'*')):
                shutil.move(f, f.replace(params.wrong_task, params.right_task))
            # and go through and edit all the text as well
            wrong_task = params.wrong_task.replace('task-', '')
            right_task = params.right_task.replace('task-', '')
            shell('grep -rl --exclude \*nii.gz "{wrong_task}" {output[0]} | xargs sed -i "s/{wrong_task}/{right_task}/g"')
            shell('grep -rl --exclude \*nii.gz "{wrong_task}" {output[2]} | xargs sed -i "s/{wrong_task}/{right_task}/g"')


def get_raw_behavioral_results(wildcards):
    behavioral_results = {}
    hdf5_name_dict = {
        ('sub-wlsubj042', 'ses-pilot00'): ["2017-Aug-23_wl_subj042_sess1.hdf5"],
        ('sub-wlsubj001', 'ses-pilot01'): ["2017-Oct-09_wl_subj001_sess1.hdf5"],
        ('sub-wlsubj042', 'ses-pilot01'): ["2017-Nov-07_wl_subj042_sess0.hdf5"],
        ('sub-wlsubj045', 'ses-pilot01'): ["2017-Nov-07_wl_subj045_sess0.hdf5"],
        ('sub-wlsubj001', 'ses-01'): ["2018-Jan-31_sub-wlsubj001_sess0.hdf5"],
        ('sub-wlsubj042', 'ses-01'): ["2018-Feb-01_sub-wlsubj042_sess0.hdf5"],
        ('sub-wlsubj001', 'ses-02'): ["2018-Feb-07_sub-wlsubj001_sess0.hdf5"],
        ('sub-wlsubj042', 'ses-02'): ["2018-Feb-09_sub-wlsubj042_sess0.hdf5"],
        ('sub-wlsubj045', 'ses-01'): ["2018-Feb-16_sub-wlsubj045_sess0.hdf5",
                                      "2018-Feb-16_sub-wlsubj045_sess1.hdf5"],
        ('sub-wlsubj045', 'ses-02'): ["2018-Feb-27_sub-wlsubj045_sess0.hdf5"],
        ('sub-wlsubj014', 'ses-03'): ["2018-Mar-20_sub-wlsubj014_sess0.hdf5"],
        ('sub-wlsubj004', 'ses-03'): ["2018-Mar-22_sub-wlsubj004_sess0.hdf5"],
        ('sub-wlsubj045', 'ses-04'): ["2019-Mar-22_sub-wlsubj045_ses-04_sess00.hdf5"],
        ('sub-wlsubj045', 'ses-03'): ["2019-Mar-29_sub-wlsubj045_ses-03_sess00.hdf5"],
        ('sub-wlsubj064', 'ses-04'): ["2019-Apr-05_sub-wlsubj064_ses-04_sess00.hdf5"],
        ('sub-wlsubj081', 'ses-04'): ["2019-Apr-09_sub-wlsubj081_ses-04_sess00.hdf5"],
        ('sub-wlsubj007', 'ses-04'): ["2019-May-01_sub-wlsubj007_ses-04_sess00.hdf5"],
        ('sub-wlsubj062', 'ses-04'): ["2019-May-01_sub-wlsubj062_ses-04_sess00.hdf5"],
        ('sub-wlsubj095', 'ses-04'): ["2019-May-03_sub-wlsubj095_ses-04_sess00.hdf5",
                                      "2019-May-03_sub-wlsubj095_ses-04_sess01.hdf5"],
        ('sub-wlsubj006', 'ses-04'): ["2020-Jan-28_sub-wlsubj006_ses-04_sess00.hdf5"],
        ('sub-wlsubj046', 'ses-04'): ["2020-Jan-28_sub-wlsubj046_ses-04_sess00.hdf5"],
        ('sub-wlsubj121', 'ses-04'): ["2020-Jan-29_sub-wlsubj121_ses-04_sess00.hdf5"],
        ('sub-wlsubj114', 'ses-04'): ["2020-Jan-30_sub-wlsubj114_ses-04_sess00.hdf5",
                                      "2020-Jan-30_sub-wlsubj114_ses-04_sess01.hdf5"],
        ('sub-wlsubj115', 'ses-04'): ["2020-Jan-30_sub-wlsubj115_ses-04_sess00.hdf5"],
        ('sub-wlsubj001', 'ses-04'): ["2020-Feb-10_sub-wlsubj001_ses-04_sess00.hdf5"],
    }
    behavioral_results['hdf5_file'] = [os.path.join(config['EXTRA_FILES_DIR'], p) for p in
                                       hdf5_name_dict[(wildcards.subject, wildcards.session)]]
    behavioral_results['notes_file'] = behavioral_results['hdf5_file'][0].replace('_sess0.hdf5', '.md').replace('_sess1.hdf5', '.md')
    return behavioral_results


rule create_BIDS_tsv:
    input:
        unpack(get_raw_behavioral_results),
        stim_df_path = lambda wildcards: get_stim_files(wildcards)['desc_csv'],
    output:
        os.path.join(config["DATA_DIR"], "{subject}", "{session}", "func", "{subject}_{session}_{task}_acq-PA_run-01_events.tsv"),
        os.path.join(config["DATA_DIR"], "sourcedata", "{subject}", "{session}", "{subject}_{session}_{task}_behavioral_results_sess00.hdf5"),
        os.path.join(config["DATA_DIR"], "sourcedata", "{subject}", "{session}", "{subject}_{session}_{task}_notes.md")
    log:
        os.path.join(config["DATA_DIR"], 'code', 'BIDS_tsv', '{subject}_{session}_{task}-%j.log')
    benchmark:
        os.path.join(config["DATA_DIR"], 'code', 'BIDS_tsv', '{subject}_{session}_{task}_benchmark.txt')
    params:
        save_path = lambda wildcards, output: output[0].replace('run-01', 'run-%02d'),
        full_TRs = lambda wildcards: {'ses-pilot00': 256, 'ses-pilot01': 256, 'ses-01': 240, 'ses-02': 240, 'ses-03': 264, 'ses-04': 264}[wildcards.session]
    run:
        import sfp
        import shutil
        sfp.create_BIDS_tsv.main(input.hdf5_file, input.stim_df_path, save_path=params.save_path,
                                 full_TRs=params.full_TRs)
        for i, f in enumerate(input.hdf5_file):
            shutil.copy(f, output[1].replace('sess00', 'sess%02d' % i))
        shutil.copy(input.notes_file, output[2])


def get_permuted(wildcards):
    if "permuted" in wildcards.mat_type:
        return "-p"
    else:
        return ""


def get_design_inputs(wildcards):
    tsv_files = os.path.join(config["DATA_DIR"], wildcards.subject, wildcards.session, "func",
                             wildcards.subject+"_"+wildcards.session+"_"+wildcards.task+"_acq-PA_run-{n:02d}_events.tsv")
    func_files = os.path.join(config["DATA_DIR"], wildcards.subject, wildcards.session, "func",
                              wildcards.subject+"_"+wildcards.session+"_"+wildcards.task+"_acq-PA_run-{n:02d}_bold.nii.gz")
    return {'tsv_files': expand(tsv_files, n=range(1, NRUNS.get((wildcards.subject, wildcards.session), 12)+1)),
            'func_files': expand(func_files, n=range(1, NRUNS.get((wildcards.subject, wildcards.session), 12)+1))}


rule create_design_matrices:
    input:
        unpack(get_design_inputs),
    output:
        os.path.join(config["DATA_DIR"], "derivatives", "design_matrices", "{mat_type}", "{subject}", "{session}", "{subject}_{session}_{task}_params.json")
    log:
        os.path.join(config["DATA_DIR"], "code", "design_matrices", "{subject}_{session}_{task}_{mat_type}-%j.log")
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "design_matrices", "{subject}_{session}_{task}_{mat_type}_benchmark.txt")
    params:
        save_path = lambda wildcards, output: output[0].replace('params.json', 'run-%02d_design.tsv'),
        permuted_flag = get_permuted,
        mat_type = lambda wildcards: wildcards.mat_type.replace("_permuted", ""),
        BIDS_dir = config["DATA_DIR"],
    shell:
        "python sfp/design_matrices.py {params.BIDS_dir} {wildcards.subject} {wildcards.session}"
        " --mat_type {params.mat_type} --save_path {params.save_path} {params.permuted_flag}"


def get_preprocess_inputs(wildcards):
    input_dict = {}
    input_dict['freesurfer_files'] = os.path.join(config["DATA_DIR"], "derivatives", "freesurfer",
                                                  wildcards.subject.replace('sub-', ''))
    input_dict['func_files'] = os.path.join(config["DATA_DIR"], "{subject}", "{session}", "func",
                                            "{subject}_{session}_{task}_acq-PA_{run}_bold.nii.gz").format(**wildcards)
    return input_dict


rule preprocess:
    input:
        unpack(get_preprocess_inputs)
    output:
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed_{run}_{task}", "{subject}", "{session}", "{subject}_{session}_{task}_acq-PA_{run}_preproc.nii.gz"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed_{run}_{task}", "{subject}", "{session}", "session.json"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed_{run}_{task}", "{subject}", "{session}", "sbref_reg_corrected.nii.gz"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed_{run}_{task}", "{subject}", "{session}", "distort2anat_tkreg.dat"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed_{run}_{task}", "{subject}", "{session}", "distortion_merged_corrected.nii.gz"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed_{run}_{task}", "{subject}", "{session}", "distortion_merged_corrected_mean.nii.gz"),
    resources:
        cpus_per_task = 10,
        mem = 48
    params:
        plugin = "MultiProc",
        data_dir = lambda wildcards: os.path.join(config['DATA_DIR'], wildcards.subject, wildcards.session),
        working_dir = lambda wildcards: os.path.join(config['WORKING_DIR'], "%s_%s_%s" % (wildcards.subject, wildcards.session, wildcards.run)),
        plugin_args = lambda wildcards, resources: ",".join("%s:%s" % (k,v) for k,v in {'n_procs': resources.cpus_per_task, 'memory_gb': resources.mem}.items()),
        epi_num = lambda wildcards: int(wildcards.run.replace('run-', '')),
        script_location = os.path.join(config["MRI_TOOLS"], "preprocessing", "prisma_preproc.py")
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "preprocessed", "{subject}_{session}_{task}_{run}_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "preprocessed", "{subject}_{session}_{task}_{run}-%j.log")
    shell:
        # we want to remove the working directory afterwards because it's big and contains many
        # files. it means that re-runs will take slightly longer, but since I was starting to run
        # into the number of files quota on the cluster, it's worth it.
        "python {params.script_location} -datadir {params.data_dir} -working_dir "
        "{params.working_dir} -plugin {params.plugin} -dir_structure bids -plugin_args "
        "{params.plugin_args} -epis {params.epi_num} -bids_derivative_name "
        "preprocessed_{wildcards.run}_{wildcards.task}; rm -rf {params.working_dir};"


rule rearrange_preprocess_extras:
    input:
        lambda wildcards: expand(os.path.join(config["DATA_DIR"], "derivatives", "preprocessed_run-{n:02d}_{task}", wildcards.subject, wildcards.session, wildcards.filename_ext), task=TASKS.get((wildcards.subject, wildcards.session), None), n=range(1, NRUNS.get((wildcards.subject, wildcards.session), 12)+1))
    output:
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "{filename_ext}")
    log:
        os.path.join(config["DATA_DIR"], "code", "preprocessed", "{subject}_{session}_rearrange_extras_{filename_ext}-%j.log")
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "preprocessed", "{subject}_{session}_rearrange_extras_{filename_ext}_benchmark.txt")
    run:
        import subprocess
        import os
        import shutil
        import json
        if os.path.split(input[0])[-1] == 'session.json':
            # we handle this differently, because we want to merge the jsons instead
            master_json = {}
            for filename in input:
                run_name = os.path.abspath(filename).split(os.sep)[-2]
                with open(filename) as f:
                    master_json[run_name] = json.load(f)
                os.remove(filename)
            with open(output[0], 'w') as f:
                json.dump(master_json, f)
        else:
            file1 = input[0]
            for file2 in input[1:]:
                if subprocess.call(['cmp', '-s', file1, file2]) == 1:
                    raise Exception("%s and %s are different, they should be the same!" % (file1, file2))
                else:
                    os.remove(file2)
            shutil.move(file1, output[0])

rule rearrange_preprocess:
    input:
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed_{run}_{task}", "{subject}", "{session}", "{subject}_{session}_{task}_acq-PA_{run}_preproc.nii.gz"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "session.json"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "sbref_reg_corrected.nii.gz"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "distort2anat_tkreg.dat"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "distortion_merged_corrected.nii.gz"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "distortion_merged_corrected_mean.nii.gz"),
    output:
        # we drop the acq-PA bit because all of our scans have that, so it's not informative
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "{subject}_{session}_{task}_{run}_preproc.nii.gz"),
    log:
        os.path.join(config["DATA_DIR"], "code", "preprocessed", "{subject}_{session}_{task}_{run}_rearrange-%j.log")
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "preprocessed", "{subject}_{session}_{task}_{run}_rearrange_benchmark.txt")
    run:
        import shutil
        import os
        shutil.move(input[0], output[0])
        os.removedirs(os.path.dirname(input[0]))


rule to_freesurfer:
    input:
        in_file = os.path.join(config['DATA_DIR'], "derivatives", "preprocessed",  "{subject}", "{session}", "{subject}_{session}_{task}_{run}_preproc.nii.gz"),
        tkreg = os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "distort2anat_tkreg.dat"),
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "preprocessed_reoriented", "{subject}", "{session}", "lh.{subject}_{session}_{task}_{run}_preproc.mgz"),
        os.path.join(config['DATA_DIR'], "derivatives", "preprocessed_reoriented", "{subject}", "{session}", "rh.{subject}_{session}_{task}_{run}_preproc.mgz")
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "to_freesurfer", "{subject}_{session}_{task}_{run}_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "to_freesurfer", "{subject}_{session}_{task}_{run}-%j.log")
    params:
        output_dir = lambda wildcards, output: os.path.dirname(output[0]),
        script_location = os.path.join(config["MRI_TOOLS"], "preprocessing", "to_freesurfer.py"),
        # this will also produce a nifti output that we don't want to keep around.
        tmp_nifti = lambda wildcards, output: output[0].replace('lh.', '').replace('.mgz', '.nii.gz')
    shell:
        "python {params.script_location} -v -s -o {params.output_dir} {input.tkreg} {input.in_file};"
        " rm {params.tmp_nifti}"


def find_prf_mgz(wildcards, prf_prop='varea', subject=None):
    try:
        prf_prop = wildcards.prf_prop
    except AttributeError:
        prf_prop = prf_prop
    if wildcards.atlas_type == 'atlas':
        benson_prefix = 'benson14_'
    elif wildcards.atlas_type == 'bayesian_posterior':
        benson_prefix = 'inferred_'
    elif wildcards.atlas_type == 'data':
        benson_prefix = 'full-'
    try:
        subject = wildcards.subject
    except AttributeError:
        subject = subject
    benson_template = os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', subject, wildcards.atlas_type, '{hemi}.'+benson_prefix+prf_prop+'.mgz')
    return expand(benson_template, hemi=['lh', 'rh'])


rule prf_check_plot:
    input:
        prf_mgzs = find_prf_mgz,
        freesurfer_dir = lambda wildcards: os.path.join(config['DATA_DIR'], 'derivatives', 'freesurfer', '{subject}').format(subject=wildcards.subject.replace('sub-', ''))
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', "{subject}", "{atlas_type}", '{prf_prop}_plot.png')
    log:
        os.path.join(config['DATA_DIR'], 'code', 'prf_solutions', '{subject}_{atlas_type}_{prf_prop}_plot-%j.log')
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'prf_solutions', '{subject}_{atlas_type}_{prf_prop}_plot_benchmark.txt')
    run:
        import neuropythy as ny
        import sfp
        atlases = {}
        for hemi in ['lh', 'rh']:
            path = [i for i in input.prf_mgzs if hemi in i][0]
            atlases[hemi] = ny.load(path)
        if wildcards.prf_prop == 'varea':
            mask = ('plot_property', [1, 2, 3])
        else:
            mask = None
        sfp.plotting.flat_cortex_plot(input.freesurfer_dir, atlases, output[0], mask)


rule create_GLMdenoise_json:
    input:
        json_template = os.path.join(config['MRI_TOOLS'], 'BIDS', 'files', 'glmOptsOptimize.json'),
        vareas_mgzs = find_prf_mgz,
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'GLMdenoise', "{mat_type}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_glmOpts.json")
    log:
        os.path.join(config['DATA_DIR'], 'code', 'GLMdenoise_json', '{subject}_{session}_{task}_{mat_type}_{atlas_type}-%j.log')
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'GLMdenoise_json', '{subject}_{session}_{task}_{mat_type}_{atlas_type}_benchmark.txt')
    run:
        import json
        import nibabel as nib
        import numpy as np
        with open(input.json_template) as f:
            opts = json.load(f)
        vareas = []
        for v in input.vareas_mgzs:
            vareas.append(nib.load(v).get_data().squeeze())
        vareas = np.concatenate(vareas)
        vareas = np.isin(vareas, [1, 2, 3])
        opts['opt']['wantsanityfigures'] = True
        opts['opt']['seed'] = SUB_SEEDS[wildcards.subject] + SES_SEEDS[wildcards.session]
        opts['opt']['hrffitmask'] = vareas.tolist()
        with open(output[0], 'w') as f:
            json.dump(opts, f)


rule create_GLMdenoise_fixed_hrf_json:
    input:
        json_template = os.path.join(config['MRI_TOOLS'], 'BIDS', 'files', 'glmOptsAssume.json'),
        old_results = os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{input_mat}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_results.mat"),
        vareas_mgzs = find_prf_mgz,
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'GLMdenoise', "{mat_type}_fixed_hrf_{input_mat}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_glmOpts.json")
    log:
        os.path.join(config['DATA_DIR'], 'code', 'GLMdenoise_json', '{subject}_{session}_{task}_{mat_type}_fixed_hrf_{input_mat}_{atlas_type}-%j.log')
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'GLMdenoise_json', '{subject}_{session}_{task}_{mat_type}_fixed_hrf_{input_mat}_{atlas_type}_benchmark.txt')
    run:
        import json
        import h5py
        import nibabel as nib
        import numpy as np
        with open(input.json_template) as f:
            opts = json.load(f)
        mat = h5py.File(input.old_results)
        hrf_ref = mat['results']['modelmd'][0, 0]
        vareas = []
        for v in input.vareas_mgzs:
            vareas.append(nib.load(v).get_data().squeeze())
        vareas = np.concatenate(vareas)
        vareas = np.isin(vareas, [1, 2, 3])
        opts['opt']['wantsanityfigures'] = True
        opts['opt']['seed'] = SUB_SEEDS[wildcards.subject] + SES_SEEDS[wildcards.session]
        opts['hrfknobs'] = np.array(mat[hrf_ref]).flatten().tolist()
        opts['opt']['hrffitmask'] = vareas.tolist()
        with open(output[0], 'w') as f:
            json.dump(opts, f)


def GLMdenoise_runs(wildcards):
    """return the runs to use for this mat_type
    """
    total_runs = NRUNS.get((wildcards.subject, wildcards.session), 12)
    # because we're passing this to matlab, we need this to be a list of
    # ints that go from 1 to total_runs (inclusive).
    if wildcards.mat_type.endswith('noise-ceiling-1'):
        runs = np.arange(1, total_runs+1, 2)
    elif wildcards.mat_type.endswith('noise-ceiling-2'):
        runs = np.arange(2, total_runs+1, 2)
    else:
        runs = []
    runs = ','.join([str(i) for i in runs])
    runs = f"[{runs}]"
    return runs


def get_GLMdenoise_params_file(wildcards):
    # need a function for this because there's a variety of mat_type
    # "suffixes" which don't change the design_matrices mat_type folder
    template = os.path.join(config["DATA_DIR"], "derivatives", "design_matrices", "{mat_type}",
                            "{subject}", "{session}", "{subject}_{session}_{task}_params.json")
    # neither of these should affect the design_matrices mat_type we're
    # looking for. this is the same as params.mat_type, but input
    # functions can't take params as input
    wildcards.mat_type = wildcards.mat_type.replace('_noise-ceiling-1', '').replace('_noise-ceiling-2', '')
    return template.format(**wildcards)


rule GLMdenoise:
    input:
        preproc_files = lambda wildcards: expand(os.path.join(config["DATA_DIR"], "derivatives", "preprocessed_reoriented", wildcards.subject, wildcards.session, "{hemi}."+wildcards.subject+"_"+wildcards.session+"_"+wildcards.task+"_run-{n:02d}_preproc.mgz"), hemi=['lh', 'rh'], n=range(1, NRUNS.get((wildcards.subject, wildcards.session), 12)+1)),
        params_file = get_GLMdenoise_params_file,
        opts_json = os.path.join(config['DATA_DIR'], 'derivatives', 'GLMdenoise', "{mat_type}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_glmOpts.json")
    output:
        GLM_results = protected(os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_results.mat")),
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "GLMdenoise", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "GLMdenoise", "{subject}_{session}_{task}_{mat_type}_{atlas_type}-%j.log")
    params:
        vistasoft_path = os.path.join(config['VISTASOFT_PATH']),
        GLMdenoise_path = config['GLMDENOISE_PATH'],
        BIDS_dir = config['DATA_DIR'],
        GLM_dir = os.path.join(config['MRI_TOOLS'], "BIDS"),
        subject = lambda wildcards: wildcards.subject.replace('sub-', ''),
        session = lambda wildcards: wildcards.session.replace('ses-', ''),
        # the bidsGLM script drops its output here, but we want to move it to the location in
        # output
        GLM_output_dir = lambda wildcards: os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}-{atlas_type}", "{subject}", "{session}", "figures").format(**wildcards),
        GLM_tmp_parent_dir = lambda wildcards: os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}-{atlas_type}", "{subject}", "{session}").format(**wildcards),
        GLM_target_dir = lambda wildcards: os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}", "{atlas_type}", "{subject}", "{session}", "figures_{task}").format(**wildcards),
        GLM_output = lambda wildcards: os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}", "{atlas_type}", "{subject}", "{session}", "figures_{task}", "{subject}_{session}_{mat_type}-{atlas_type}_results.mat").format(**wildcards),
        runs = GLMdenoise_runs,
        mat_type = lambda wildcards: wildcards.mat_type.replace('_noise-ceiling-1', '').replace('_noise-ceiling-2', '')
    resources:
        cpus_per_task = 1,
        mem = 100
    shell:
        "cd {params.GLM_dir}; matlab -nodesktop -nodisplay -r \"addpath(genpath('{params."
        "vistasoft_path}')); addpath(genpath('{params.GLMdenoise_path}')); "
        "jsonInfo=jsondecode(fileread('{input.params_file}')); bidsGLM('{params."
        "BIDS_dir}', '{params.subject}', '{params.session}', [], {params.runs}, "
        "'preprocessed_reoriented', 'preproc', '{params.mat_type}', jsonInfo.stim_length, "
        "'{wildcards.mat_type}-{wildcards.atlas_type}', '{input.opts_json}', jsonInfo.TR_length); "
        "quit;\"; mv -v {params.GLM_output_dir} {params.GLM_target_dir}; rmdir -pv {params.GLM_tmp_parent_dir}; "
        "mv -v {params.GLM_output} {output.GLM_results}"


rule GLMdenoise_fixed_hrf:
    input:
        preproc_files = lambda wildcards: expand(os.path.join(config["DATA_DIR"], "derivatives", "preprocessed_reoriented", wildcards.subject, wildcards.session, "{hemi}."+wildcards.subject+"_"+wildcards.session+"_"+wildcards.task+"_run-{n:02d}_preproc.mgz"), hemi=['lh', 'rh'], n=range(1, NRUNS.get((wildcards.subject, wildcards.session), 12)+1)),
        params_file = os.path.join(config["DATA_DIR"], "derivatives", "design_matrices", "{mat_type}", "{subject}", "{session}", "{subject}_{session}_{task}_params.json"),
        opts_json = os.path.join(config['DATA_DIR'], 'derivatives', 'GLMdenoise', "{mat_type}_fixed_hrf_{input_mat}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_glmOpts.json")
    output:
        GLM_results = protected(os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}_fixed_hrf_{input_mat}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_results.mat")),
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "GLMdenoise", "{subject}_{session}_{task}_{mat_type}_fixed_hrf_{input_mat}_{atlas_type}_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "GLMdenoise", "{subject}_{session}_{task}_{mat_type}_fixed_hrf_{input_mat}_{atlas_type}-%j.log")
    params:
        vistasoft_path = config['VISTASOFT_PATH'],
        GLMdenoise_path = config['GLMDENOISE_PATH'],
        BIDS_dir = config['DATA_DIR'],
        GLM_dir = os.path.join(config['MRI_TOOLS'], "BIDS"),
        subject = lambda wildcards: wildcards.subject.replace('sub-', ''),
        session = lambda wildcards: wildcards.session.replace('ses-', ''),
        # the bidsGLM script drops its output here, but we want to move it to the location in
        # output
        GLM_output_dir = lambda wildcards: os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}_fixed_hrf_{input_mat}-{atlas_type}", "{subject}", "{session}", "figures").format(**wildcards),
        GLM_tmp_parent_dir = lambda wildcards: os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}_fixed_hrf_{input_mat}-{atlas_type}", "{subject}", "{session}").format(**wildcards),
        GLM_target_dir = lambda wildcards: os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}_fixed_hrf_{input_mat}", "{atlas_type}", "{subject}", "{session}", "figures_{task}").format(**wildcards),
        GLM_output = lambda wildcards: os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}_fixed_hrf_{input_mat}", "{atlas_type}", "{subject}", "{session}", "figures_{task}", "{subject}_{session}_{mat_type}_fixed_hrf_{input_mat}-{atlas_type}_results.mat").format(**wildcards),
    resources:
        cpus_per_task = 1,
        mem = 150
    shell:
        "cd {params.GLM_dir}; matlab -nodesktop -nodisplay -r \"addpath(genpath('{params."
        "vistasoft_path}')); addpath(genpath('{params.GLMdenoise_path}')); "
        "jsonInfo=jsondecode(fileread('{input.params_file}')); bidsGLM('{params."
        "BIDS_dir}', '{params.subject}', '{params.session}', [], [], "
        "'preprocessed_reoriented', 'preproc', '{wildcards.mat_type}', jsonInfo.stim_length, "
        "'{wildcards.mat_type}_fixed_hrf_{wildcards.input_mat}-{wildcards.atlas_type}', "
        "'{input.opts_json}', jsonInfo.TR_length); quit;\"; "
        "mv -v {params.GLM_output_dir} {params.GLM_target_dir}; rmdir -pv {params.GLM_tmp_parent_dir}; "
        "mv -v {params.GLM_output} {output.GLM_results}"


rule GLMdenoise_png_process:
    # this crawls through the GLMdenoise figure directory and converts all the pngs, which
    # otherwise are created incorrectly because they're surface outputs, see
    # https://github.com/kendrickkay/GLMdenoise/issues/6
    input:
        os.path.join(config["DATA_DIR"], "derivatives", "GLMdenoise", "{mat_type}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_results.mat")
    output:
        os.path.join(config["DATA_DIR"], "derivatives", "GLMdenoise", "{mat_type}", "{atlas_type}", "{subject}", "{session}", "figures_{task}", "FinalModel_maps.png")
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "GLMdenoise", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_png_process_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "GLMdenoise", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_png_process-%j.log")
    params:
        GLM_figure_dir = lambda wildcards, output: os.path.dirname(output[0]),
        script_location = os.path.join(config["MRI_TOOLS"], "BIDS", "GLMdenoisePNGprocess.py"),
    shell:
        "python {params.script_location} {params.GLM_figure_dir}"


rule interpolate_to_fsaverage:
    input:
        # this makes sure that we get a single list of strs, not a list
        # of lists of strs
        prf_mgzs = lambda wildcards: [i for p in ['varea', 'eccen', 'angle'] for i in find_prf_mgz(wildcards, p)],
        freesurfer_dir = lambda wildcards: os.path.join(config['DATA_DIR'], 'derivatives', 'freesurfer', '{subject}').format(subject=wildcards.subject.replace('sub-', '')),
        GLMdenoise_path = os.path.join(config["DATA_DIR"], "derivatives", "GLMdenoise", "{mat_type}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_results.mat")
    output:
        os.path.join(config["DATA_DIR"], "derivatives", "GLMdenoise", "{mat_type}", "{atlas_type}", "sub-groupaverage_i-{interp_method}", "{session}", "{subject}", "{subject}_{session}_{task}_v{varea}_models.hdf5"),
        os.path.join(config["DATA_DIR"], "derivatives", "GLMdenoise", "{mat_type}", "{atlas_type}", "sub-groupaverage_i-{interp_method}", "{session}", "{subject}", "{subject}_{session}_{task}_v{varea}_models_lh_b00_c00_space-subject.png"),
        os.path.join(config["DATA_DIR"], "derivatives", "GLMdenoise", "{mat_type}", "{atlas_type}", "sub-groupaverage_i-{interp_method}", "{session}", "{subject}", "{subject}_{session}_{task}_v{varea}_models_rh_b00_c00_space-subject.png"),
        os.path.join(config["DATA_DIR"], "derivatives", "GLMdenoise", "{mat_type}", "{atlas_type}", "sub-groupaverage_i-{interp_method}", "{session}", "{subject}", "{subject}_{session}_{task}_v{varea}_models_lh_b00_c00_space-prior.png"),
        os.path.join(config["DATA_DIR"], "derivatives", "GLMdenoise", "{mat_type}", "{atlas_type}", "sub-groupaverage_i-{interp_method}", "{session}", "{subject}", "{subject}_{session}_{task}_v{varea}_models_rh_b00_c00_space-prior.png"),
    resources:
        mem = 25,
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "GLMdenoise", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{varea}_i-{interp_method}_b00_c00_interpolate_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "GLMdenoise", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{varea}_i-{interp_method}_b00_c00_interpolate-%j.log")
    run:
        import neuropythy as ny
        from sfp.combine_across_subjects import interpolate_GLMdenoise_to_fsaverage_prior
        prf_props = {}
        for hemi in ['lh', 'rh']:
            prf_props[hemi] = {}
            for k, prop in zip(['varea', 'eccen', 'angle'], ['visual_area', 'eccentricity',
                                                             'polar_angle']):
                path = [i for i in input.prf_mgzs if hemi in i if k in i][0]
                prf_props[hemi][prop] = ny.load(path)
        save_stem = output[0].replace('_models.hdf5', '')
        interpolate_GLMdenoise_to_fsaverage_prior(input.freesurfer_dir, prf_props, save_stem,
                                                  input.GLMdenoise_path, 0, 0,
                                                  int(wildcards.varea), wildcards.interp_method)


rule compute_groupaverage:
    input:
        lambda wildcards: [os.path.join(config["DATA_DIR"], "derivatives", "GLMdenoise", "{mat_type}",
                                        "{atlas_type}", "sub-groupaverage_i-{interp_method}", "{session}", "{subject}",
                                        "{subject}_{session}_{task}_v{varea}_models.hdf5").format(subject=sub, **wildcards)
                           for sub in SUBJECTS if wildcards.session in SESSIONS[sub]
                           if TASKS[(sub, wildcards.session)] == wildcards.task]
    output:
        os.path.join(config["DATA_DIR"], "derivatives", "GLMdenoise", "{mat_type}", "{atlas_type}", "sub-groupaverage_i-{interp_method}", "{session}_v{varea}_s{boot_seed}", "sub-groupaverage_i-{interp_method}_{session}_v{varea}_s{boot_seed}_{task}_results.mat"),
        os.path.join(config["DATA_DIR"], "derivatives", "GLMdenoise", "{mat_type}", "{atlas_type}", "sub-groupaverage_i-{interp_method}", "{session}_v{varea}_s{boot_seed}", "sub-groupaverage_i-{interp_method}_{session}_v{varea}_s{boot_seed}_{task}_b00_c00_models.png"),
    resources:
        mem = 18,
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "GLMdenoise", "sub-groupaverage_{session}_{task}_{mat_type}_{atlas_type}_v{varea}_i-{interp_method}_s{boot_seed}_b00_c00_groupaverage_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "GLMdenoise", "sub-groupaverage_{session}_{task}_{mat_type}_{atlas_type}_v{varea}_i-{interp_method}_s{boot_seed}_b00_c00_groupaverage-%j.log")
    run:
        from sfp.combine_across_subjects import compute_groupaverage
        save_stem = output[0].replace('_results.mat', '')
        compute_groupaverage(input, save_stem, int(wildcards.boot_seed), 0, 0, int(wildcards.varea))


def get_first_level_analysis_input(wildcards):
    input_dict = {}
    input_dict['GLM_results'] = os.path.join(config["DATA_DIR"], "derivatives", "GLMdenoise", "{mat_type}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_results.mat").format(**wildcards)
    benson_names = ['angle', 'eccen', 'varea']
    if wildcards.subject.startswith('sub-groupaverage'):
        benson_prefix = 'benson14'
        benson_temp = os.path.join(os.path.dirname(ny.__file__), 'lib', 'data', 'fsaverage', 'surf', '{hemi}.'+benson_prefix+'_{filename}.v4_0.mgz')
        benson_names += ['sigma']
    else:
        if wildcards.atlas_type == 'atlas':
            benson_prefix = 'benson14'
        elif wildcards.atlas_type == 'bayesian_posterior':
            benson_prefix = 'inferred'
        benson_temp = os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', wildcards.subject, wildcards.atlas_type, '{hemi}.'+benson_prefix+'_{filename}.mgz')
        prf_temp = os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', wildcards.subject, 'data', '{hemi}.full-{filename}.mgz')
        input_dict['prf_sigma_path'] = expand(prf_temp, hemi=['lh', 'rh'], filename=['sigma', 'vexpl'])
    input_dict['benson_paths'] = expand(benson_temp, hemi=['lh', 'rh'], filename=benson_names)
    return input_dict


def get_stim_type(wildcards):
    if 'pilot' in wildcards.session:
        return 'pilot'
    else:
        if 'constant' in wildcards.task:
            return 'constant'
        else:
            return 'logpolar'


def get_benson_template(wildcards, input):
    # for some reason, sometimes input.benson_paths is a single str,
    # sometimes it's a list of strings (it always requires multiple
    # benson_paths as input, just sometimes the way they're stored is
    # different?). It might be related to snakemake version or something
    # else I'm having trouble controlling across machines. regardless,
    # this does it
    if isinstance(input.benson_paths, str):
        path = input.benson_paths
    else:
        path = input.benson_paths[0]
    return path.replace('lh', '%s').replace('angle', '%s').replace('benson14_', '').replace('inferred_', '').replace(wildcards.atlas_type, '%s').replace('surf', '%s'),


rule first_level_analysis:
    input:
        unpack(get_first_level_analysis_input),
        unpack(get_stim_files),
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}.csv')
    resources:
        cpus_per_task = 1,
        mem = lambda wildcards: {'full': 40, 'summary': 10}[wildcards.df_mode]
    params:
        save_stem = lambda wildcards: "{subject}_{session}_{task}_".format(**wildcards),
        save_dir = lambda wildcards, output: os.path.dirname(output[0]),
        vareas = lambda wildcards: wildcards.vareas.split('-'),
        eccen = lambda wildcards: wildcards.eccen.split('-'),
        benson_template = get_benson_template,
        benson_names = lambda wildcards, input: [os.path.split(i)[-1].split('.')[1] for i in input if wildcards.atlas_type+'/lh' in i or 'surf/lh' in i],
        prf_names = lambda wildcards, input: [i.split('.')[-2] for i in input if 'data/lh' in i],
        class_num = lambda wildcards: get_n_classes(wildcards.session, wildcards.mat_type),
        stim_type = get_stim_type,
        mid_val = lambda wildcards: {'ses-pilot01': 127, 'ses-pilot00': 127}.get(wildcards.session, 128),
        atlas_type = lambda wildcards: 'surf' if wildcards.subject.startswith('sub-groupaverage') else wildcards.atlas_type,
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "first_level_analysis", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}_{df_mode}_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "first_level_analysis", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}_{df_mode}-%j.log")
    shell:
        "python -m sfp.first_level_analysis --save_dir {params.save_dir} --vareas {params.vareas} "
        "--df_mode {wildcards.df_mode} --eccen_range {params.eccen} "
        "--unshuffled_stim_descriptions_path {input.desc_csv} --unshuffled_stim_path {input.stim} "
        "--save_stem {params.save_stem} --class_nums {params.class_num} --stim_type "
        "{params.stim_type} --mid_val {params.mid_val} --benson_template_names "
        "{params.benson_names} --results_path {input.GLM_results} "
        "--benson_template_path {params.benson_template} --benson_atlas_type {params.atlas_type}"
        " --prf_data_names {params.prf_names}"


def get_binning(wildcards):
    bin_str = ""
    if "eccen_bin" in wildcards.binning:
        bin_str += "--eccen "
    if "angle_bin" in wildcards.binning:
        bin_str += "--angle"
    if not bin_str:
        raise Exception("You must bin by something!")
    return bin_str


rule binning:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}.csv')
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "first_level_binned", "{mat_type}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}.csv")
    resources:
        cpus_per_task = 1,
        mem = lambda wildcards: {'full': 30, 'summary': 10}[wildcards.df_mode]
    params:
        bin_str = get_binning
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "binning", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}_{binning}_{df_mode}_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "binning", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}_{binning}_{df_mode}-%j.log")
    shell:
        "python -m sfp.binning {params.bin_str} {input} {output}"


rule tuning_curves:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "first_level_binned", "{mat_type}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}.csv")
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "{mat_type}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}.csv")
    resources:
        cpus_per_task = 1,
        mem = lambda wildcards: {'full': 30, 'summary': 10}[wildcards.df_mode]
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_curves", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}_{binning}_{df_mode}_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_curves", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}_{binning}_{df_mode}-%j.log")
    shell:
        "python -m sfp.tuning_curves {input} {output}"


rule plots:
    input:
        dataframe=os.path.join(config['DATA_DIR'], "derivatives", "{step}", "{mat_type}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}.csv")
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "{step}", "{mat_type}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_{plot_name}.svg")
    params:
        stim_dir = os.path.join(config['DATA_DIR'], 'stimuli')
    resources:
        mem = 2
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "plots", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}_{binning}_{df_mode}_{step}_{plot_name}_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "plots", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}_{binning}_{df_mode}_{step}_{plot_name}-%j.log")
    shell:
        "python -m sfp.plotting {input.dataframe} {params.stim_dir} --plot_to_make "
        "{wildcards.plot_name}"


def get_tuning_curves(wildcards):
    if wildcards.groupaverage == 'individual':
        if wildcards.atlas_type == 'atlas':
            subjects = ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045']
            sessions = {'sub-wlsubj001': ['ses-pilot01'], 'sub-wlsubj042': ['ses-pilot01'],
                        'sub-wlsubj045': ['ses-pilot01']}
        else:
            sessions = {k: [wildcards.session] for k, v in SESSIONS.items() if wildcards.session in v}
            subjects = [s for s in SUBJECTS if s in sessions.keys()]
        vareas = wildcards.vareas.split('-')
        return [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}',
                             '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_'
                             'v{vareas}_e{eccen}_{binning}_{df_mode}.csv').format(mat_type=wildcards.mat_type,
                                                                                 atlas_type=wildcards.atlas_type,
                                                                                 subject=sub, session=ses,
                                                                                 task=TASKS[(sub, ses)],
                                                                                 vareas=v,
                                                                                 eccen=wildcards.eccen,
                                                                                 binning=wildcards.binning,
                                                                                 df_mode=wildcards.df_mode)
                for sub in subjects for ses in sessions[sub] for v in vareas]
    elif wildcards.groupaverage == 'sub-groupaverage':
        return get_groupaverage_all('curve')


rule tuning_curves_summary:
    input:
        get_tuning_curves
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "{mat_type}", "{atlas_type}", "{groupaverage}_{session}_v{vareas}_e{eccen}_{binning}_tuning_curves_{df_mode}.csv")
    params:
        input_dir = os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "{mat_type}", "{atlas_type}"),
        groupaverage = lambda wildcards: {'sub-groupaverage': '-g', 'individual': ''}[wildcards.groupaverage]
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_curves_summary", "{mat_type}_{atlas_type}_{groupaverage}_{session}_v{vareas}_e{eccen}_{binning}_{df_mode}_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_curves_summary", "{mat_type}_{atlas_type}_{groupaverage}_{session}_v{vareas}_e{eccen}_{binning}_{df_mode}-%j.log")
    shell:
        "python sfp/summarize_tuning_curves.py {params.input_dir} {output} {wildcards.df_mode} {params.groupaverage} -s {wildcards.session}"


rule tuning_curves_summary_plot:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "{mat_type}", "{atlas_type}", "individual_{session}_v{vareas}_e{eccen}_{binning}_tuning_curves_summary.csv")
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}',
                     "v{vareas}_e{eccen}_{binning}_tuning_curves_summary_plot_{subjects}_{session}_"
                     "{tasks}_v{plot_varea}_e{eccen_range}_row={row}_col={col}_hue={hue}_{plot_func}"
                     "_{y}.svg")
    params:
        col = lambda wildcards: wildcards.col.replace("-", '_'),
        row = lambda wildcards: wildcards.row.replace("-", '_'),
        hue = lambda wildcards: wildcards.hue.replace("-", '_'),
        y = lambda wildcards: wildcards.y.replace("-", '_'),
        plot_varea = lambda wildcards: wildcards.plot_varea.split('-'),
        eccen_range = lambda wildcards: wildcards.eccen_range.split('-'),
        subjects = lambda wildcards: wildcards.subjects.split(','),
        tasks = lambda wildcards: wildcards.tasks.split(','),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_curves_summary_plots", "{mat_type}_"
                     "{atlas_type}_v{vareas}_e{eccen}_{binning}_{subjects}_{session}_{tasks}_v"
                     "{plot_varea}_e{eccen_range}_row={row}_col={col}_hue={hue}_{plot_func}_{y}_"
                     "benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_curves_summary_plots", "{mat_type}_"
                     "{atlas_type}_v{vareas}_e{eccen}_{binning}_{subjects}_{session}_{tasks}_v"
                     "{plot_varea}_e{eccen_range}_row={row}_col={col}_hue={hue}_{plot_func}_{y}-%j.log")
    shell:
        "python -m sfp.summary_plots {input} --col {params.col} --row {params.row} --hue"
        " {params.hue} --y {params.y} --varea {params.plot_varea} --eccen_range {params.eccen_range}"
        " --subject {params.subjects} --task {params.tasks} --session {wildcards.session}"


def to_log_or_not(wildcards):
    """we only log directly if we're not on the cluster, otherwise we trust the cluster to handle it
    """
    if ON_CLUSTER:
        return "; echo"
    else:
        return "&> "


def visual_field_part(wildcards):
    """if modeling_goal specifies it, add the string to reduce part of visual field
    """
    vis_field = ""
    if wildcards.modeling_goal.startswith("visual_field"):
        vis_field += ",restrict_to_part_of_visual_field:" + wildcards.modeling_goal.split('_')[-1]
    return vis_field


def get_df_filter_str(wildcards):
    if wildcards.df_filter == 'filter-any':
        df_filter_str = 'drop_voxels_with_any_negative_amplitudes,drop_voxels_near_border'
    elif wildcards.df_filter == 'filter-mean':
        df_filter_str = 'drop_voxels_with_mean_negative_amplitudes,drop_voxels_near_border'
    elif wildcards.df_filter == 'no-filter':
        df_filter_str = None
    return df_filter_str


rule model:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}.csv')
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}", "{df_filter}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_n{bootstrap_num}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}", "{df_filter}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_n{bootstrap_num}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_model_history.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}", "{df_filter}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_n{bootstrap_num}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_model.pt"),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_{df_filter}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_n{bootstrap_num}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_{df_filter}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_n{bootstrap_num}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}-%j.log")
    resources:
        # need the same number of cpus and gpus
        cpus_per_task = lambda wildcards: max(int(wildcards.gpus), 1),
        mem = lambda wildcards: {'full': 60, 'summary': 2}[wildcards.df_mode],
        gpus = lambda wildcards: int(wildcards.gpus)
    params:
        save_stem = lambda wildcards, output: output[0].replace("_loss.csv", ''),
        stimulus_class = lambda wildcards: wildcards.stimulus_class.split(','),
        bootstrap_num = lambda wildcards: wildcards.bootstrap_num.split(','),
        logging = to_log_or_not,
        vis_field = visual_field_part,
        df_filter_str = get_df_filter_str,
    shell:
        "python -m sfp.model {wildcards.period_orientation_type} {wildcards.eccentricity_type} "
        "{wildcards.amplitude_orientation_type} {input} {params.save_stem} -b "
        "{wildcards.batch_size} -r {wildcards.learning_rate} -d "
        "{params.df_filter_str}{params.vis_field} -t 1e-6 -e"
        " 1000 -c {params.stimulus_class} -n {params.bootstrap_num} {params.logging} {log}"


# this correctly calculates the CV error, in a way we don't get otherwise
rule calc_cv_error:
    input:
        loss_files = lambda wildcards: get_model_subj_outputs(**wildcards),
        dataset_path = os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis',
                                    '{mat_type}', '{atlas_type}', '{subject}', '{session}',
                                    '{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}.csv')
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}",
                     "{atlas_type}", "{df_filter}", "{modeling_goal}", "{subject}", "{session}",
                     "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_"
                     "r{learning_rate}_g{gpus}_s{crossval_seed}_{model_type}_cv_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}",
                     "{atlas_type}", "{df_filter}", "{modeling_goal}", "{subject}", "{session}",
                     "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_"
                     "r{learning_rate}_g{gpus}_s{crossval_seed}_{model_type}_cv_preds.pt"),
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model_cv_loss", "{subject}_{session}_"
                     "{task}_{mat_type}_{atlas_type}_{df_filter}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_{model_type}-%j.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model_cv_loss", "{subject}_{session}_"
                     "{task}_{mat_type}_{atlas_type}_{df_filter}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_{model_type}_"
                     "benchmark.txt")
    resources:
        mem = 10
    run:
        import sfp
        df_filter_str = get_df_filter_str(wildcards)
        sfp.analyze_model.calc_cv_error(input.loss_files, input.dataset_path, wildcards, output,
                                        df_filter_str)


rule summarize_model_cv:
    input:
        # this will return a list of lists of strings, so we need to flatten it
        loss_files = lambda wildcards: np.array([get_model_subj_outputs(m, **wildcards) for m in MODEL_TYPES]).flatten(),
        cv_loss = lambda wildcards: [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}",
                                                  "{atlas_type}", "{df_filter}", "{modeling_goal}", "{subject}", "{session}",
                                                  "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_"
                                                  "r{learning_rate}_g{gpus}_s{crossval_seed}_{model_type}_cv_loss.csv").format(model_type=m, **wildcards)
                                     for m in MODEL_TYPES]

    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_"
                     "s{crossval_seed}_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_"
                     "s{crossval_seed}_all_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_"
                     "s{crossval_seed}_all_timing.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_"
                     "s{crossval_seed}_all_diff.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_"
                     "s{crossval_seed}_all_model_history.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_"
                     "s{crossval_seed}_all_cv_loss.csv"),
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model_summarize", "{subject}_{session}_"
                     "{task}_{mat_type}_{atlas_type}_{df_filter}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}-%j.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model_summarize", "{subject}_{session}_"
                     "{task}_{mat_type}_{atlas_type}_{df_filter}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_benchmark.txt")
    params:
        base_path = lambda wildcards, input: os.path.join(os.path.dirname(input.loss_files[0]),
                                                          '*c*model.pt'),
        metadata = ["mat_type", 'atlas_type', 'modeling_goal', 'subject', 'session', 'task',
                    'fit_model_type', 'test_subset'],
    run:
        import sfp
        sfp.analyze_model.gather_results(params.base_path, output, params.metadata, input.cv_loss)


def get_cv_summary(crossval_seed=0, batch_size=10, learning_rate=1e-3, vareas=1, eccen='1-12',
                   df_mode='summary', gpus=0, mat_type='stim_class', atlas_type='bayesian_posterior',
                   modeling_goal='initial_cv', task='task-sfprescaled', groupaverage='individual',
                   df_filter='filter-mean'):
    # for now, groupaverage does nothing because we don't do
    # cross-validation on sub-groupaverage
    output_path = os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}",
                               "{atlas_type}", "{df_filter}", "{modeling_goal}", "{{subject}}", "{{session}}",
                               "{{subject}}_{{session}}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch"
                               "_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all_cv_loss.csv")
    output_path = output_path.format(vareas=vareas, mat_type=mat_type, batch_size=batch_size,
                                     eccen=eccen, atlas_type=atlas_type, df_mode=df_mode,
                                     modeling_goal=modeling_goal, gpus=gpus, task=task,
                                     crossval_seed=crossval_seed, learning_rate=learning_rate,
                                     df_filter=df_filter)
    return [output_path.format(subject=sub, session=ses)
            for sub in SUBJECTS for ses in SESSIONS[sub] if TASKS[(sub, ses)] == task]


rule combine_model_cv_summaries:
    input:
        lambda wildcards: get_cv_summary(**wildcards)
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                      "{df_filter}", "{modeling_goal}", "{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_s{crossval_seed}_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}", "{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_s{crossval_seed}_all_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}", "{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_s{crossval_seed}_all_timing.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}", "{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_s{crossval_seed}_all_cv_loss.csv"),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_{df_filter}_{modeling_goal}_{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_{df_filter}_{modeling_goal}_{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all-%j.log")
    params:
        base_template = lambda wildcards, input: [i.replace('_all_cv_loss.csv', '') for i in input]
    run:
        import sfp
        sfp.analyze_model.combine_summarized_results(params.base_template, output)


def gather_model_results_input(wildcards):
    inputs = {}
    # wildcards is not an actual dictionary (it just functions like one
    # in some regards) and so we need to make one if we want to do the
    # pop() call below
    format_kwargs = dict(wildcards)
    groupaverage = format_kwargs.pop('groupaverage', 'individual')
    if wildcards.modeling_goal == 'bootstrap':
        loss_files = [get_model_subj_outputs(bootstrap_num=n, **format_kwargs) for n in range(100)]
    else:
        if groupaverage == 'individual':
            loss_files = [get_model_subj_outputs(subject=subj, session=ses, **format_kwargs)
                          for subj in SUBJECTS for ses in SESSIONS[subj]
                          if TASKS[(subj, ses)] == wildcards.task]
        elif groupaverage == 'sub-groupaverage':
            loss_files = get_groupaverage_all(**format_kwargs)
    # this will return a list of lists of strings, so we need to flatten it
    inputs['loss_files'] = np.array(loss_files).flatten()
    return inputs


rule gather_model_results_preliminary:
    input:
        unpack(gather_model_results_input)
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}",  "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_timing.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}",  "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_diff.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}",  "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_model_history.csv")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_{df_filter}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_{df_filter}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all-%j.log")
    resources:
        mem = 100,
    params:
        base_path = lambda wildcards, output: os.path.join(os.path.dirname(output[0]), f'*{wildcards.model_type}*'),
        metadata = ["mat_type", 'atlas_type', 'modeling_goal', 'subject', 'session', 'task',
                    'fit_model_type', 'indicator', 'bootstrap_num', 'test_subset']
    run:
        import sfp
        sfp.analyze_model.gather_results(params.base_path, output, params.metadata)


rule summarize_gathered_results:
    input:
        lambda wildcards: [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                                       "{df_filter}", "bootstrap", "{subject}", "{session}", "{subject}_{session}_{task}_"
                                       "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_"
                                        "{model_type}_all_models.csv").format(subject=subj, session=ses, **wildcards)
                           for subj in SUBJECTS for ses in SESSIONS[subj]
                           if TASKS[(subj, ses)] == wildcards.task]
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "bootstrap", "{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}"
                     "_r{learning_rate}_g{gpus}_{model_type}_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "bootstrap", "{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}"
                     "_r{learning_rate}_g{gpus}_{model_type}_all_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "bootstrap", "{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}"
                     "_r{learning_rate}_g{gpus}_{model_type}_all_timing.csv"),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_{df_filter}_bootstrap_{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_{df_filter}_bootstrap_{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all-%j.log")
    params:
        base_template = lambda wildcards, input: [i.replace('_all_models.csv', '') for i in input]
    run:
        import sfp
        sfp.analyze_model.combine_summarized_results(params.base_template, output, False)


rule gather_model_results:
    input:
        unpack(gather_model_results_input)
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}", "{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}", "{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}", "{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_timing.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}", "{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_diff.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{df_filter}", "{modeling_goal}", "{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_model_history.csv"),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_{df_filter}_{modeling_goal}_{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_{df_filter}_{modeling_goal}_{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all-%j.log")
    resources:
        mem = 100,
    params:
        base_path = lambda wildcards, output: os.path.join(os.path.dirname(output[0]), '*', '*',
                                                           '*'+wildcards.model_type+'*'),
        metadata = ["mat_type", 'atlas_type', 'modeling_goal', 'subject', 'session', 'task',
                    'fit_model_type', 'indicator', 'bootstrap_num', 'test_subset'],
        groupaverage = lambda wildcards: {'sub-groupaverage': True, 'individual': False}[wildcards.groupaverage]
    run:
        import sfp
        sfp.analyze_model.gather_results(params.base_path, output, params.metadata, groupaverage=params.groupaverage)


rule simulate_data_uniform_noise:
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'simulated_data', 'noise-uniform', 'n{num_voxels}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_simulated.csv')
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'simulated_data', 'noise-uniform_n{num_voxels}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_benchmark.txt')
    log:
        os.path.join(config['DATA_DIR'], 'code', 'simulated_data', 'noise-uniform_n{num_voxels}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}-%j.log')
    resources:
        mem=10
    shell:
        "python -m sfp.simulate_data {output} -p {wildcards.period_orientation_type} -e {wildcards.eccentricity_type} "
        "-o {wildcards.amplitude_orientation_type} -n {wildcards.num_voxels} -s {wildcards.sigma} "
        "-a {wildcards.sf_ecc_slope} -rmc {wildcards.rel_mode_cardinals} -rmo "
        "{wildcards.rel_mode_obliques} -rac {wildcards.rel_amplitude_cardinals} -rao "
        "{wildcards.rel_amplitude_obliques} -amc {wildcards.abs_mode_cardinals} -amo "
        "{wildcards.abs_mode_obliques} -aac {wildcards.abs_amplitude_cardinals} -aao "
        "{wildcards.abs_amplitude_obliques} -b {wildcards.sf_ecc_intercept} -l {wildcards.noise_level}"


rule simulate_data_voxel_noise:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary.csv')
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'simulated_data', 'noise-{mat_type}_{atlas_type}_{subject}_{session}_{task}_v{vareas}_e{eccen}', 'n{num_voxels}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_simulated.csv'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'simulated_data', 'noise-{mat_type}_{atlas_type}_{subject}_{session}_{task}_v{vareas}_e{eccen}', 'n{num_voxels}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_simulated_full.csv')
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'simulated_data', 'noise-{mat_type}_{atlas_type}_{subject}_{session}_{task}_v{vareas}_e{eccen}_n{num_voxels}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_benchmark.txt')
    log:
        os.path.join(config['DATA_DIR'], 'code', 'simulated_data', 'noise-{mat_type}_{atlas_type}_{subject}_{session}_{task}_v{vareas}_e{eccen}_n{num_voxels}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}-%j.log')
    resources:
        mem=10
    shell:
        "python -m sfp.simulate_data {output} -p {wildcards.period_orientation_type} -e {wildcards.eccentricity_type} "
        "-o {wildcards.amplitude_orientation_type} -n {wildcards.num_voxels} -s {wildcards.sigma} "
        "-a {wildcards.sf_ecc_slope} -rmc {wildcards.rel_mode_cardinals} -rmo "
        "{wildcards.rel_mode_obliques} -rac {wildcards.rel_amplitude_cardinals} -rao "
        "{wildcards.rel_amplitude_obliques} -amc {wildcards.abs_mode_cardinals} -amo "
        "{wildcards.abs_mode_obliques} -aac {wildcards.abs_amplitude_cardinals} -aao "
        "{wildcards.abs_amplitude_obliques} -b {wildcards.sf_ecc_intercept} -l {wildcards.noise_level} "
        "--noise_source_path {input} --num_bootstraps 100"


rule model_simulated_data:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'simulated_data', 'noise-{noise_source}', 'n{num_voxels}_{sim_period_orientation_type}_{sim_eccentricity_type}_{sim_amplitude_orientation_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_simulated.csv')
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-{noise_source}", "{modeling_goal}", "n{num_voxels}_{sim_period_orientation_type}_{sim_eccentricity_type}_{sim_amplitude_orientation_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-{noise_source}", "{modeling_goal}", "n{num_voxels}_{sim_period_orientation_type}_{sim_eccentricity_type}_{sim_amplitude_orientation_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_model_history.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-{noise_source}", "{modeling_goal}", "n{num_voxels}_{sim_period_orientation_type}_{sim_eccentricity_type}_{sim_amplitude_orientation_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_model.pt"),
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'tuning_2d_simulated', 'noise-{noise_source}_{modeling_goal}_n{num_voxels}_{sim_period_orientation_type}_{sim_eccentricity_type}_{sim_amplitude_orientation_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_benchmark.txt')
    log:
        os.path.join(config['DATA_DIR'], 'code', 'tuning_2d_simulated', 'noise-{noise_source}_{modeling_goal}_n{num_voxels}_{sim_period_orientation_type}_{sim_eccentricity_type}_{sim_amplitude_orientation_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}-%j.log')
    resources:
        # need the same number of cpus and gpus
        cpus_per_task = lambda wildcards: max(int(wildcards.gpus), 1),
        mem = 10,
        gpus = lambda wildcards: int(wildcards.gpus),
    params:
        save_stem = lambda wildcards, output: output[0].replace("_loss.csv", ''),
        stimulus_class = lambda wildcards: wildcards.stimulus_class.split(','),
        logging = to_log_or_not,
    shell:
        "python -m sfp.model {wildcards.period_orientation_type} {wildcards.eccentricity_type} "
        "{wildcards.amplitude_orientation_type} {input} {params.save_stem} -b {wildcards.batch_size} "
        "-r {wildcards.learning_rate} -d None -t 1e-6 -e 1000 -c {params.stimulus_class} "
        "{params.logging} {log}"


def gather_simulated_model_results_input(wildcards):
    inputs = {}
    if wildcards.modeling_goal == 'learning_hyperparams_full':
        batch = [1, 10, 100]
        lr = [1e-2, 1e-3, 1e-4]
        models = ['iso_full_iso', 'full_full_full']
        loss_files = []
        for b, l, m  in itertools.product(batch, lr, models):
            loss_files.append(get_simulated_model_outputs(
                m, 'iso_full_iso', 1, 4000, b, l, 1, .75, .25, 0, 0, 0, 0, 0, 0, 0, 0,
                **wildcards))
            loss_files.append(get_simulated_model_outputs(
                m, 'full_full_full', 1, 4000, b, l, 1, .75, .25, .1, .05, .03, .1, .2, .05, .04,
                .3, **wildcards))
    elif wildcards.modeling_goal == 'model_recovery':
        loss_files = []
        for m  in MODEL_TYPES:
            loss_files.append(get_simulated_model_outputs(
                m, 'iso_full_iso', 1, 4000, 10, 1e-3, 1, .75, .25, 0, 0, 0, 0, 0, 0, 0, 0,
                **wildcards))
            loss_files.append(get_simulated_model_outputs(
                m, 'full_full_full', 1, 4000, 10, 1e-3, 1, .75, .25, .1, .05, .03, .1, .2, .05,
                .04, .3, **wildcards))
    # this will return a list of lists of strings, so we need to flatten it
    inputs['loss_files'] = np.array(loss_files).flatten()
    return inputs


# this correctly calculates the CV error, in a way we don't get otherwise
rule calc_simulated_cv_error:
    input:
        loss_files = lambda wildcards: get_simulated_model_outputs(**wildcards),
        dataset_path = os.path.join(config['DATA_DIR'], 'derivatives', 'simulated_data',
                                    '{noise_source}', 'n{num_voxels}_{sim_model_type}_s{sigma}_'
                                    'a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_'
                                    'rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_'
                                    'rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_'
                                    'amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_'
                                    'aao{abs_amplitude_obliques}_l{noise_level}_simulated.csv')
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                     "{modeling_goal}", "n{num_voxels}_{sim_model_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_"
                     "rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_"
                     "rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_"
                     "aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_{model_type}_cv_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                     "{modeling_goal}", "n{num_voxels}_{sim_model_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_"
                     "rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_"
                     "rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_"
                     "aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_{model_type}_cv_preds.pt"),
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_simulated_cv_loss",
                     "noise-{noise_source}_{modeling_goal}_n{num_voxels}_{sim_model_type}_s{sigma}_"
                     "a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_"
                     "rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}"
                     "_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}"
                     "_aao{abs_amplitude_obliques}_l{noise_level}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_s{crossval_seed}_{model_type}-%j.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_simulated_cv_loss",
                     "noise-{noise_source}_{modeling_goal}_n{num_voxels}_{sim_model_type}_s{sigma}_"
                     "a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_"
                     "rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}"
                     "_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}"
                     "_aao{abs_amplitude_obliques}_l{noise_level}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_s{crossval_seed}_{model_type}_benchmark.txt")
    resources:
        mem = 10
    run:
        import sfp
        # that None is the df_filter argument: no df_filter is used for
        # training the model and thus none should be used when
        # calculating the cv error
        sfp.analyze_model.calc_cv_error(input.loss_files, input.dataset_path, wildcards, output,
                                        None)


rule summarize_simulated_cv:
    input:
        # this will return a list of lists of strings, so we need to flatten it
        loss_files = lambda wildcards: np.array([get_simulated_model_outputs(m, **wildcards) for m in MODEL_TYPES]).flatten(),
        cv_loss = lambda wildcards: [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                                                  "{modeling_goal}", "n{num_voxels}_{sim_model_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_"
                                                  "rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_"
                                                  "rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_"
                                                  "aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_"
                                                  "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_{model_type}_cv_loss.csv").format(model_type=m, **wildcards)
                                     for m in MODEL_TYPES],

    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                     "{modeling_goal}", "n{num_voxels}_{sim_model_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_"
                     "rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_"
                     "rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_"
                     "aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                     "{modeling_goal}", "n{num_voxels}_{sim_model_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_"
                     "rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_"
                     "rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_"
                     "aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                     "{modeling_goal}", "n{num_voxels}_{sim_model_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_"
                     "rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_"
                     "rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_"
                     "aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all_timing.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                     "{modeling_goal}", "n{num_voxels}_{sim_model_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_"
                     "rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_"
                     "rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_"
                     "aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all_diff.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                     "{modeling_goal}", "n{num_voxels}_{sim_model_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_"
                     "rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_"
                     "rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_"
                     "aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all_model_history.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                     "{modeling_goal}", "n{num_voxels}_{sim_model_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_"
                     "rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_"
                     "rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_"
                     "aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all_cv_loss.csv")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_simulated_summarize",
                     "noise-{noise_source}_{modeling_goal}_n{num_voxels}_{sim_model_type}_s{sigma}_"
                     "a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_"
                     "rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}"
                     "_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}"
                     "_aao{abs_amplitude_obliques}_l{noise_level}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_s{crossval_seed}-%j.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_simulated_summarize",
                     "noise-{noise_source}_{modeling_goal}_n{num_voxels}_{sim_model_type}_s{sigma}_"
                     "a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_"
                     "rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}"
                     "_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}"
                     "_aao{abs_amplitude_obliques}_l{noise_level}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_s{crossval_seed}_benchmark.txt")
    params:
        base_path = lambda wildcards, input: os.path.join(os.path.dirname(input.loss_files[0]),
                                                          "*c*model.pt"),
        metadata = ['modeling_goal', 'batch_size', 'learning_rate', 'fit_model_type',
                    'true_model_type', 'test_subset']
    run:
        import sfp
        sfp.analyze_model.gather_results(params.base_path, output, params.metadata, input.cv_loss)


def get_simulated_cv_summary(batch_size, learning_rate, noise_source, crossval_seed=0, gpus=0,
                             modeling_goal='model_recovery_cv'):
    output_path = os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                               "{modeling_goal}", "n4000_{{sim_model_type}}_s{{sigma}}_"
                               "a{{sf_ecc_slope}}_b{{sf_ecc_intercept}}_"
                               "rmc{{rel_mode_cardinals}}_rmo{{rel_mode_obliques}}_rac{{rel_amplitude_cardinals}}_"
                               "rao{{rel_amplitude_obliques}}_amc{{abs_mode_cardinals}}_amo{{abs_mode_obliques}}_"
                               "aac{{abs_amplitude_cardinals}}_aao{{abs_amplitude_obliques}}_l1_"
                               "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all_cv_loss.csv")
    output_path = output_path.format(batch_size=batch_size, learning_rate=learning_rate,
                                     gpus=gpus, modeling_goal=modeling_goal,
                                     noise_source=noise_source, crossval_seed=crossval_seed)
    models = [['iso_full_iso', 1, .75, .25, 0, 0, 0, 0, 0, 0, 0, 0],
              ['full_full_full', 1, .75, .25, .1, .05, .03, .1, .2, .05, .04, .03]]
    return [output_path.format(sim_model_type=m, sigma=s, sf_ecc_slope=a, sf_ecc_intercept=b,
                               rel_mode_cardinals=rmc, rel_mode_obliques=rmo,
                               rel_amplitude_cardinals=rac, rel_amplitude_obliques=rao,
                               abs_mode_cardinals=amc, abs_mode_obliques=amo,
                               abs_amplitude_cardinals=aac, abs_amplitude_obliques=aao).replace('0.', '.')
            for m, s, a, b, rmc, rmo, rac, rao, amc, amo, aac, aao in models]


rule combine_simulated_cv_summaries:
    input:
        lambda wildcards: get_simulated_cv_summary(**wildcards)
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                     "{modeling_goal}", "b{batch_size}_r{learning_rate}_g{gpus}_"
                     "s{crossval_seed}_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                     "{modeling_goal}", "b{batch_size}_r{learning_rate}_g{gpus}_"
                     "s{crossval_seed}_all_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                     "{modeling_goal}", "b{batch_size}_r{learning_rate}_g{gpus}_"
                     "s{crossval_seed}_all_timing.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                     "{modeling_goal}", "b{batch_size}_r{learning_rate}_g{gpus}_"
                     "s{crossval_seed}_all_cv_loss.csv"),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_simulated_summarize", "{noise_source}_{modeling_goal}_b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_simulated_summarize", "{noise_source}_{modeling_goal}_b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all-%j.log")
    params:
        base_template = lambda wildcards, input: [i.replace('_all_cv_loss.csv', '') for i in input]
    run:
        import sfp
        sfp.analyze_model.combine_summarized_results(params.base_template, output)


rule gather_simulated_model_results:
    input:
        unpack(gather_simulated_model_results_input)
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                     "{modeling_goal}", "g{gpus}_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                     "{modeling_goal}", "g{gpus}_all_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                     "{modeling_goal}", "g{gpus}_all_timing.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                     "{modeling_goal}", "g{gpus}_all_diff.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "{noise_source}",
                     "{modeling_goal}", "g{gpus}_all_model_history.csv"),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_simulated_summarize", "{noise_source}_{modeling_goal}_g{gpus}_all_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_simulated_summarize", "{noise_source}_{modeling_goal}_g{gpus}_all-%j.log")
    resources:
        mem = 100,
    params:
        base_path = lambda wildcards, output: os.path.join(os.path.dirname(output[0]),
                                                           '*g%s*' % wildcards.gpus),
        metadata = ['modeling_goal', 'batch_size', 'learning_rate', 'fit_model_type',
                    'true_model_type']
    run:
        import sfp
        sfp.analyze_model.gather_results(params.base_path, output, params.metadata)


rule move_full_simulated_data_noise_ceiling:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'simulated_data', 'noise-{mat_type}_{atlas_type}_sub-wlsubj062_{session}_{task}_v{vareas}_e{eccen}', 'n1000_full_full_full_s2.2_a.15_b.3_rmc-.05_rmo-.05_rac0_rao0_amc.1_amo-.1_aac.05_aao-.05_l1_simulated_full.csv'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'simulated_data', 'noise-{mat_type}_{atlas_type}_sub-wlsubj062_{session}_{task}_v{vareas}_e{eccen}', 'n1000_full_full_full_s2.2_a.15_b.3_rmc-.05_rmo-.05_rac0_rao0_amc.1_amo-.1_aac.05_aao-.05_l.2_simulated_full.csv'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'simulated_data', 'noise-{mat_type}_{atlas_type}_sub-wlsubj062_{session}_{task}_v{vareas}_e{eccen}', 'n1000_full_full_full_s2.2_a.15_b.3_rmc-.05_rmo-.05_rac0_rao0_amc.1_amo-.1_aac.05_aao-.05_l0_simulated_full.csv'),
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', 'sub-wlsubj062-simulated', '{session}', 'sub-wlsubj062-simulated_{session}_{task}_v{vareas}_e{eccen}_full.csv'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', 'sub-wlsubj062-simulated-lownoise', '{session}', 'sub-wlsubj062-simulated-lownoise_{session}_{task}_v{vareas}_e{eccen}_full.csv'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', 'sub-wlsubj062-simulated-noiseless', '{session}', 'sub-wlsubj062-simulated-noiseless_{session}_{task}_v{vareas}_e{eccen}_full.csv'),
    shell:
        "mv {input[0]} {output[0]}; mv {input[1]} {output[1]}; mv {input[2]} {output[2]}"


rule noise_ceiling_monte_carlo:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_full.csv'),
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'noise_ceiling', 'monte_carlo', '{mat_type}', '{atlas_type}', "{df_filter}", '{subject}', '{session}', 's{seed}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_full_{mode}_loss.csv'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'noise_ceiling', 'monte_carlo', '{mat_type}', '{atlas_type}', "{df_filter}", '{subject}', '{session}', 's{seed}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_full_{mode}_predictions.png')
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "noise_ceiling", 'monte_carlo', "loss_s{seed}_{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}_full_{df_filter}_{mode}_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "noise_ceiling", 'monte_carlo', "loss_s{seed}_{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}_full_{df_filter}_{mode}-%j.log")
    run:
        import sfp
        import pandas as pd
        save_stem = output[0].replace('_loss.csv', '')
        df = pd.read_csv(input[0])
        if 'simulated' in wildcards.subject:
            # don't filter the simulated subject, because the filtering is just
            # to get rid of voxels whose responses we don't trust, and there
            # are none of those in a simulation
            df_filter_str = None
            is_simulated = True
        else:
            df_filter_str = get_df_filter_str(wildcards)
            is_simulated = False
        df = sfp.noise_ceiling.sample_df(df, int(wildcards.seed), df_filter_str, is_simulated,
                                         wildcards.mode)
        sfp.noise_ceiling.monte_carlo(df, save_stem, df_mode='full', **wildcards)


rule noise_ceiling_monte_carlo_overall:
    input:
        # for now, we're only going to want to look at ses-04,
        # task-sfprescaled. this will work for other session/task pairs,
        # but we'd have to merge them ourselves afterwards.
        lambda wildcards: [os.path.join(config['DATA_DIR'], 'derivatives', 'noise_ceiling', 'monte_carlo',
                                        '{{mat_type}}', '{{atlas_type}}', '{{df_filter}}', '{subject}', '{{session}}', 's{seed}',
                                        '{subject}_{{session}}_{{task}}_v{{vareas}}_e{{eccen}}_full_{{mode}}_loss.csv').format(
                                            subject=sub, seed=seed)
                           for seed in range(100) for sub in SUBJECTS if wildcards.session in SESSIONS[sub]
                           if TASKS[(sub, wildcards.session)] == wildcards.task]
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'noise_ceiling', 'monte_carlo', '{mat_type}', '{atlas_type}', '{df_filter}', 'monte_carlo_{session}_{task}_v{vareas}_e{eccen}_{mode}.csv')
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "noise_ceiling", 'monte_carlo', "loss_{mat_type}_{atlas_type}_{session}_{task}_v{vareas}_e{eccen}_{df_filter}_{mode}_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "noise_ceiling", 'monte_carlo', "loss_{mat_type}_{atlas_type}_{session}_{task}_v{vareas}_e{eccen}_{df_filter}_{mode}-%j.log")
    run:
        import pandas as pd
        df = []
        for p in input:
            df.append(pd.read_csv(p))
        df = pd.concat(df).reset_index(drop=True)
        df.to_csv(output[0], index=False)


rule noise_ceiling_split_half_df:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}_noise-ceiling-1', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary.csv'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}_noise-ceiling-2', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary.csv'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary.csv'),
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'noise_ceiling', 'split_half', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary.csv')
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "noise_ceiling", 'split_half', "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}_summary_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "noise_ceiling", 'split_half', "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}_summary-%j.log")
    run:
        import sfp
        import pandas as pd
        dfs = []
        for p in input:
            dfs.append(pd.read_csv(p))
        df = sfp.noise_ceiling.combine_dfs(*dfs)
        df.to_csv(output[0], index=False)


rule noise_ceiling_split_half:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'noise_ceiling', 'split_half', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary.csv')
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'noise_ceiling', 'split_half', '{mat_type}', '{atlas_type}', '{subject}', '{session}', 'b{batch_size}_r{lr}_g{gpus}_s{seed}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary_loss.csv'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'noise_ceiling', 'split_half', '{mat_type}', '{atlas_type}', '{subject}', '{session}', 'b{batch_size}_r{lr}_g{gpus}_s{seed}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary_results_df.csv'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'noise_ceiling', 'split_half', '{mat_type}', '{atlas_type}', '{subject}', '{session}', 'b{batch_size}_r{lr}_g{gpus}_s{seed}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary_model.pt'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'noise_ceiling', 'split_half', '{mat_type}', '{atlas_type}', '{subject}', '{session}', 'b{batch_size}_r{lr}_g{gpus}_s{seed}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary_model_history.csv'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'noise_ceiling', 'split_half', '{mat_type}', '{atlas_type}', '{subject}', '{session}', 'b{batch_size}_r{lr}_g{gpus}_s{seed}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary_predictions.png'),
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "noise_ceiling", 'split_half', "model_b{batch_size}_r{lr}_g{gpus}_s{seed}_{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}_summary_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "noise_ceiling", 'split_half', "model_b{batch_size}_r{lr}_g{gpus}_s{seed}_{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}_summary-%j.log")
    run:
        import sfp
        save_stem = output[0].replace('_loss.csv', '')
        sfp.noise_ceiling.split_half(input[0], save_stem, int(wildcards.seed),
                                     int(wildcards.batch_size), float(wildcards.lr), 100,
                                     int(wildcards.gpus))


def get_voxel_exclusion_inputs(wildcards):
    first_level_template = os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', 'bayesian_posterior', '{{subject}}',
                                        '{{session}}', '{{subject}}_{{session}}_{task}_v{vareas}_e{eccen}_summary.csv').format(**wildcards)
    varea_template = os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', '{subject}', 'bayesian_posterior', '{hemi}.inferred_varea.mgz')
    subjects = set([sub for sub in SUBJECTS for ses in SESSIONS[sub] if TASKS[(sub, ses)] == wildcards.task])
    return {'first_level': [first_level_template.format(subject=sub, session=ses) for sub in
                            subjects for ses in SESSIONS[sub] if TASKS[(sub, ses)] == wildcards.task],
            'vareas_left': [varea_template.format(subject=sub, hemi='lh') for sub in subjects],
            'vareas_right': [varea_template.format(subject=sub, hemi='rh') for sub in subjects]}


rule voxel_exclusion_df:
    input:
        unpack(get_voxel_exclusion_inputs),
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{task}_v{vareas}_e{eccen}_{df_filter}_voxel_exclusion.csv'),
    log:
        os.path.join(config['DATA_DIR'], 'code', 'voxel_exclusion', '{mat_type}_{atlas_type}_{task}_v{vareas}_e{eccen}_{df_filter}-%j.log'),
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'voxel_exclusion', '{mat_type}_{atlas_type}_{task}_v{vareas}_e{eccen}_{df_filter}_benchmark.txt'),
    run:
        import re
        import pandas as pd
        import sfp
        df = []
        if wildcards.df_filter == 'filter-any':
            df_filter_str = ['drop_voxels_with_any_negative_amplitudes', 'drop_voxels_near_border']
        elif wildcards.df_filter == 'filter-mean':
            df_filter_str = ['drop_voxels_with_mean_negative_amplitudes', 'drop_voxels_near_border']
        df_filter_str += [','.join(df_filter_str)]
        df_filter = [sfp.model.construct_df_filter(f) for f in df_filter_str]
        vareas = [int(i) for i in wildcards.vareas.split('-')]
        if len(vareas) > 1:
            raise Exception("Wrote this assuming there was only one varea!")
        ecc_str = f'ecc in {wildcards.eccen}'
        # for each subject, find out how many voxels we remove with each filter
        for i, (first_level_path, hemis) in enumerate(zip(input.first_level,
                                                          zip(input.vareas_left,
                                                              input.vareas_right))):
            subject_name = re.findall('(sub-[a-z0-9]+)_', first_level_path)[0]
            session_name = re.findall('(ses-[0-9]+)_', first_level_path)[0]
            first_level_df = pd.read_csv(first_level_path)
            mgzs = [sfp.first_level_analysis._load_mgz(p) for p in hemis]
            tmp = {'subject': subject_name, 'session': session_name}
            tmp['total_voxels'] = np.array([(m==vareas[0]).sum() for m in mgzs]).sum()
            tmp[ecc_str] = first_level_df.voxel.nunique()
            for name, filt in zip(df_filter_str, df_filter):
                tmp[ecc_str+','+name] = filt(first_level_df).voxel.nunique()
            df.append(pd.DataFrame(tmp, index=[i]))
        df = pd.concat(df)
        # add extra metadata
        for k, v in dict(wildcards).items():
            df[k] = v
        df.to_csv(output[0], index=False)


rule voxel_exclusion_figure:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', 'stim_class', 'bayesian_posterior', '{task}_v1_e1-12_{df_filter}_voxel_exclusion.csv'),
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'figures', '{context}', 'voxel_exclusion_{df_filter}_{task}.{ext}'),
    log:
        os.path.join(config['DATA_DIR'], 'code', 'figures', '{context}', 'voxel_exclusion_{df_filter}_{task}_{ext}-%j.log'),
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'figures', '{context}', 'voxel_exclusion_{df_filter}_{task}_{ext}_benchmark.txt'),
    run:
        import sfp
        import pandas as pd
        df = pd.read_csv(input[0])
        g = sfp.figures.voxel_exclusion(df, wildcards.context)
        g.fig.savefig(output[0], bbox_inches='tight')


rule prepare_image_computable:
    input:
        stim = os.path.join(config['DATA_DIR'], 'stimuli', '{task}_stimuli.npy'),
        stim_df = os.path.join(config['DATA_DIR'], 'stimuli', '{task}_stim_description.csv')
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'stimuli_energy', '{task}_n{ori}_energy.npy'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'stimuli_energy', '{task}_n{ori}_filters.npy')
    params:
        save_path_template = lambda wildcards, output: output[1].replace('filters', '%s')
    shell:
        "python -m sfp.image_computable {input.stim} {input.stim_df} {params.save_path_template} "
        "-n {wildcards.ori}"


rule figure_summarize_1d:
    input:
        lambda wildcards: os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "stim_class",
                                       "bayesian_posterior", "{{groupaverage}}_ses-04_v1_e1-12_eccen_bin_tuning_curves_{}.csv").format(
                                           {'sub-groupaverage': 'summary', 'individual': 'full'}[wildcards.groupaverage]
                                       )
    output:
        os.path.join(config['DATA_DIR'], "derivatives", 'figures', '{context}', "{groupaverage}_1d_{tuning_param}_s-{seed}_{task}.{ext}")
    log:
        os.path.join(config['DATA_DIR'], 'code', 'figures', '{context}', "{groupaverage}_1d_{tuning_param}_s-{seed}_{task}_{ext}-%j.log")
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'figures', '{context}',
                     "{groupaverage}_1d_{tuning_param}_s-{seed}_{task}_{ext}_benchmark.txt")
    run:
        import pandas as pd
        import seaborn as sns
        import sfp
        df = sfp.figures.prep_df(pd.read_csv(input[0]), wildcards.task)
        ref_frame = {'task-sfpconstant': 'absolute', 'task-sfprescaled': 'relative'}
        if wildcards.tuning_param.endswith('overall'):
            if wildcards.groupaverage == 'sub-groupaverage':
                raise Exception(f"Can't use sub-groupaverage with {wildcards.tuning_param}! Drop "
                                "the '-overall'")
            if 'degrees' in wildcards.tuning_param:
                df['tuning_curve_bandwidth_degrees'] = df.apply(sfp.utils._octave_to_degrees, 1)
            col = {'pref-period-overall': 'preferred_period',
                   'bandwidth-overall': 'tuning_curve_bandwidth',
                   'bandwidth-degrees-overall': 'tuning_curve_bandwidth_degrees'}[wildcards.tuning_param]
            df = sfp.figures.append_precision_col(df, col)
            df = sfp.figures.precision_weighted_bootstrap(df, int(wildcards.seed), col=col,
                                                          precision_col=f"{col}_precision")
            col_wrap = None
        else:
            col_wrap = 3
        kwargs = {}
        if wildcards.tuning_param.startswith('pref-period'):
            function = sfp.figures.pref_period_1d
        elif wildcards.tuning_param.startswith('bandwidth'):
            function = sfp.figures.bandwidth_1d
            if 'degrees' in wildcards.tuning_param:
                kwargs['units'] = 'degrees'
                kwargs['ylim'] = (0, 20)
        g = function(df, wildcards.context, ref_frame[wildcards.task], row=None,
                     col_wrap=col_wrap, **kwargs)
        g.fig.savefig(output[0], bbox_inches='tight')


def get_loss_files(wildcards):
    # wildcards is not an actual dictionary (it just functions like one
    # in some regards) and so we need to make one if we want to do the
    # pop() call below
    format_kwargs = dict(wildcards)
    groupaverage = format_kwargs.pop('groupaverage')
    sub = format_kwargs.pop('subject')
    # this will return a list of lists of strings, so we need to flatten it
    if groupaverage == 'individual':
        if wildcards.modeling_goal == 'initial_cv':
            return np.array([get_model_subj_outputs(m, sub, ses, crossval_seed=0, **format_kwargs)
                             for m in MODEL_TYPES for ses in SESSIONS[sub]
                             if TASKS[(sub, ses)] == wildcards.task]).flatten()
        elif wildcards.modeling_goal == 'bootstrap':
            return np.array([get_model_subj_outputs(m, sub, ses, bootstrap_num=n, **format_kwargs)
                             for n in range(100) for m in ['full_full_absolute']
                             for ses in SESSIONS[sub]
                             if TASKS[(sub, ses)] == wildcards.task]).flatten()
    elif groupaverage == 'sub-groupaverage':
        if wildcards.modeling_goal == 'initial':
            return np.array([get_groupaverage_all(model_type=m, **format_kwargs)
                             for m in ['full_full_full', 'full_full_absolute']]).flatten()


rule combine_final_loss:
    input:
        get_loss_files,
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model', '{mat_type}', '{atlas_type}', "{df_filter}", '{modeling_goal}', '{subject}', '{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_final_epoch_loss.csv')
    log:
        os.path.join(config['DATA_DIR'], 'code', 'tuning_2d_model', '{mat_type}_{atlas_type}_{df_filter}_{modeling_goal}_{subject}_{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_final_epoch_loss-%j.log')
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'tuning_2d_model', '{mat_type}_{atlas_type}_{df_filter}_{modeling_goal}_{subject}_{groupaverage}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_final_epoch_loss_benchmark.txt')
    run:
        import sfp
        df = sfp.analyze_model.collect_final_loss(input)
        df.to_csv(output[0], index=False)


rule figure_loss_check:
    input:
        lambda wildcards: [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model', 'stim_class', 'bayesian_posterior', "{df_filter}", '{modeling_goal}', '{subject}',
                                       '{groupaverage}_{task}_v1_e1-12_{df_mode}_b10_r0.001_g0_final_epoch_loss.csv').format(df_mode={'initial_cv': 'summary', 'bootstrap': 'full', 'initial': 'summary'}[wildcards.modeling_goal],
                                                                                                                             modeling_goal=wildcards.modeling_goal, task=wildcards.task, groupaverage=wildcards.groupaverage,
                                                                                                                             df_filter=wildcards.df_filter,
                                                                                                                             subject=s)
                           for s in SUBJECTS for ses in SESSIONS[s] if TASKS[(s, ses)]==wildcards.task]
    output:
        os.path.join(config['DATA_DIR'], "derivatives", 'figures', '{context}', "{groupaverage}_{df_filter}_{modeling_goal}_training-loss-check_{task}.{ext}")
    log:
        os.path.join(config['DATA_DIR'], "code", 'figures', '{context}', "{groupaverage}_{df_filter}_{modeling_goal}_training-loss-check_{task}_{ext}-%j.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", 'figures', '{context}', "{groupaverage}_{df_filter}_{modeling_goal}_training-loss-check_{task}_{ext}_benchmark.txt")
    run:
        import sfp
        import seaborn as sns
        import pandas as pd
        font_scale = {'poster': 1.2}.get(wildcards.context, 1)
        df = pd.concat([pd.read_csv(f) for f in input])
        if wildcards.modeling_goal == 'initial_cv':
            hue = 'test_subset'
        elif wildcards.modeling_goal == 'bootstrap':
            hue = 'bootstrap_num'
        elif wildcards.modeling_goal == 'initial':
            hue = 'groupaverage_seed'
        sns.set_context(wildcards.context, font_scale=font_scale)
        with sns.axes_style('white', {'axes.spines.right': False, 'axes.spines.top': False}):
            g = sfp.figures.training_loss_check(df, hue)
            g.fig.savefig(output[0], bbox_inches='tight')


def get_noise_ceiling_df(wildcards):
    template = os.path.join(config['DATA_DIR'], 'derivatives', 'noise_ceiling', 'monte_carlo',
                            'stim_class', 'bayesian_posterior', '{df_filter}', 'monte_carlo_ses-04_{task}_v1_e1-12_{mode}.csv')
    if wildcards.cv_type.endswith('-nc'):
        return template.format(task=wildcards.task, mode='individual', df_filter=wildcards.df_filter)
    elif wildcards.cv_type.endswith('-nc-all'):
        return template.format(task=wildcards.task, mode='all', df_filter=wildcards.df_filter)
    else:
        return []


rule figure_crossvalidation:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model', 'stim_class',
                     'bayesian_posterior', "{df_filter}", 'initial_cv',
                     '{groupaverage}_{task}_v1_e1-12_summary_b10_r0.001_g0_s0_all_cv_loss.csv'),
        get_noise_ceiling_df,
    output:
        os.path.join(config['DATA_DIR'], "derivatives", 'figures', '{context}', "{groupaverage}_{df_filter}_cv_{cv_type}_{orient}_{sort}{doubleup}s-{seed}_{task}.{ext}")
    log:
        os.path.join(config['DATA_DIR'], "code", 'figures', '{context}', "{groupaverage}_{df_filter}_cv_{cv_type}_{orient}_{sort}{doubleup}s-{seed}_{task}_{ext}-%j.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", 'figures', '{context}', "{groupaverage}_{df_filter}_cv_{cv_type}_{orient}_{sort}{doubleup}s-{seed}_{task}_{ext}_benchmark.txt")
    run:
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import sfp
        df = sfp.figures.prep_df(pd.read_csv(input[0]), wildcards.task)
        if '-nc' in wildcards.cv_type:
            noise_ceiling = sfp.figures.prep_df(pd.read_csv(input[1]), wildcards.task)
        else:
            noise_ceiling = None
        if 'remeaned' in wildcards.cv_type:
            remeaned = True
        else:
            remeaned = False
        if df.loss_func.nunique() > 1:
            warnings.warn("This will only show the cross-validated loss for weighted_normed_loss loss_func")
        df = df.query('loss_func == "weighted_normed_loss"')
        if wildcards.sort == 'sort_':
            sort = True
            if not wildcards.cv_type.startswith('model_point'):
                raise Exception("Can only sort model_point plot!")
        else:
            sort = False
        if wildcards.doubleup == 'doubleup_':
            doubleup = True
            if not wildcards.cv_type.startswith('model_point'):
                raise Exception("Can only doubleup model_point plot!")
            if sort:
                raise Exception("Can only doubleup cv loss plot if we're not sorting it!")
        else:
            doubleup = False
        if wildcards.cv_type.startswith('demeaned'):
            g = sfp.figures.cross_validation_demeaned(df, int(wildcards.seed), remeaned,
                                                      context=wildcards.context,
                                                      orient=wildcards.orient)
        elif wildcards.cv_type.startswith('raw'):
            g = sfp.figures.cross_validation_raw(df, int(wildcards.seed), noise_ceiling,
                                                 context=wildcards.context,
                                                 orient=wildcards.orient)
        elif wildcards.cv_type.startswith('model_point'):
            g = sfp.figures.cross_validation_model(df, int(wildcards.seed), 'point', remeaned,
                                                   noise_ceiling, context=wildcards.context,
                                                   orient=wildcards.orient, sort=sort,
                                                   doubleup=doubleup)
        elif wildcards.cv_type.startswith('model'):
            g = sfp.figures.cross_validation_model(df, int(wildcards.seed), remeaned=remeaned,
                                                   orient=wildcards.orient,
                                                   context=wildcards.context)
        if wildcards.context == 'paper':
            g.axes[0, 0].set_title('')
        g.fig.savefig(output[0], bbox_inches='tight')


rule crossval_comparison_figures:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis',
                     '{mat_type}', '{atlas_type}', '{subject}', '{session}',
                     '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary.csv'),
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}",
                     "{atlas_type}", "{df_filter}", "{modeling_goal}", "{subject}", "{session}",
                     "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_"
                      "r{learning_rate}_g{gpus}_s{crossval_seed}_%s_cv_preds.pt") % model for
         model in ['iso_constant_iso', 'iso_scaling_iso', 'iso_full_iso']],
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}",
                     "{atlas_type}", "{df_filter}", "{modeling_goal}", "{subject}", "{session}",
                     "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_"
                     "r{learning_rate}_g{gpus}_s{crossval_seed}_cv_loss_comp_"
                     "{loss_func}.png"),
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}",
                     "{atlas_type}", "{df_filter}", "{modeling_goal}", "{subject}", "{session}",
                     "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_"
                     "r{learning_rate}_g{gpus}_s{crossval_seed}_cv_loss_comp_"
                      "{loss_func}_voxels-%s.png") % i for i in range(6)],
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model_voxel_comp", "{mat_type}",
                     "{atlas_type}", "{df_filter}", "{modeling_goal}", "{subject}", "{session}",
                     "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_"
                     "r{learning_rate}_g{gpus}_s{crossval_seed}_cv_loss_comp_"
                     "{loss_func}-%j.log"),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model_voxel_comp", "{mat_type}",
                     "{atlas_type}", "{df_filter}", "{modeling_goal}", "{subject}", "{session}",
                     "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_"
                     "r{learning_rate}_g{gpus}_s{crossval_seed}_cv_loss_comp_"
                     "{loss_func}_benchmark.txt"),
    run:
        import sfp
        import torch
        import pandas as pd
        df_filter_string = get_df_filter_str(wildcards)
        first_level_df = pd.read_csv(input[0])
        preds = []
        for i in input[1:]:
            tmp = torch.load(i)
            preds.append(tmp['predictions'])
        figs = sfp.figures.compare_cv_models(first_level_df, tmp['targets'], preds,
                                             ['constant', 'scaling', 'full'],
                                             wildcards.loss_func, df_filter_string)
        for f, o in zip(figs, output):
            f.savefig(o, bbox_inches='tight')




def get_first_level_files(wildcards):
    path_template = os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}',
                                 '{atlas_type}', '{subject}', '{session}',
                                 '{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}.csv')
    return [path_template.format(subject=sub, session=ses, **wildcards)
            for sub in SUBJECTS for ses in SESSIONS[sub] if TASKS[(sub, ses)] == wildcards.task]


rule create_precision_df:
    input:
        get_first_level_files,
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{task}_v{vareas}_e{eccen}_{df_mode}_{summary_func}_{df_filter}_precision.csv')
    log:
        os.path.join(config['DATA_DIR'], 'code', 'first_level_analysis', 'precision_{mat_type}_{atlas_type}_{task}_v{vareas}_e{eccen}_{df_mode}_{summary_func}_{df_filter}-%j.log')
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'first_level_analysis', 'precision_{mat_type}_{atlas_type}_{task}_v{vareas}_e{eccen}_{df_mode}_{summary_func}_{df_filter}_benchmark.txt')
    run:
        import sfp
        if wildcards.summary_func == 'mean':
            summary_func = np.mean
        elif wildcards.summary_func == 'median':
            summary_func = np.median
        df_filter_string = get_df_filter_str(wildcards)
        df = sfp.figures.create_precision_df(input, summary_func, df_filter_string)
        df.to_csv(output[0], index=False)


rule precision_check_figure:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary.csv'),
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary_{df_filter}_precision_check.png'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary_{df_filter}_precision_joint.png'),
    log:
        os.path.join(config['DATA_DIR'], 'code', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary_{df_filter}_precision_check-%j.log'),
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary_{df_filter}_precision_check_benchmark.txt'),
    run:
        import sfp
        import pandas as pd
        df_filter_string = get_df_filter_str(wildcards)
        fig = sfp.plotting.voxel_property_plot(pd.read_csv(input[0]), 'precision', df_filter_string=df_filter_string)
        fig.savefig(output[0])
        g = sfp.plotting.voxel_property_joint(pd.read_csv(input[0]), 'reg', ['eccen', 'precision'], df_filter_string,
                                              x_bins=20, x_estimator=np.median, marginal_kws={'kde': False})
        g.fig.savefig(output[1])


rule understand_loss_figure:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}",
                     "{atlas_type}", "{df_filter}", "{modeling_goal}", "{subject}", "{session}",
                     "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_"
                     "r{learning_rate}_g{gpus}_s{crossval_seed}_{model_type}_cv_preds.pt"),
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis',
                     '{mat_type}', '{atlas_type}', '{subject}', '{session}',
                     '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary.csv'),
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}",
                     "{atlas_type}", "{df_filter}", "{modeling_goal}", "{subject}", "{session}",
                     "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_"
                     "r{learning_rate}_g{gpus}_s{crossval_seed}_{model_type}_cv_loss_{loss_func}.png"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}",
                     "{atlas_type}", "{df_filter}", "{modeling_goal}", "{subject}", "{session}",
                     "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_"
                     "r{learning_rate}_g{gpus}_s{crossval_seed}_{model_type}_cv_loss_{loss_func}_joint.png"),
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model_cv_loss", "{subject}_{session}_"
                     "{task}_{mat_type}_{atlas_type}_{df_filter}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_{model_type}_loss_"
                     "{loss_func}_plot-%j.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model_cv_loss", "{subject}_{session}_"
                     "{task}_{mat_type}_{atlas_type}_{df_filter}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_{model_type}_loss_"
                     "{loss_func}_plot-%j_benchmark.txt")
    run:
        import sfp
        import torch
        import pandas as pd
        df_filter_string = get_df_filter_str(wildcards)
        first_level_df = pd.read_csv(input[1])
        if df_filter_string is not None:
            df_filter = sfp.model.construct_df_filter(df_filter_string)
            first_level_df = df_filter(first_level_df).reset_index()
        voxels = first_level_df.drop_duplicates('voxel')
        preds = torch.load(input[0])
        loss = sfp.analyze_model._calc_loss(preds['predictions'], preds['targets'], wildcards.loss_func,
                                            False)
        voxels[wildcards.loss_func] = loss
        fig = sfp.plotting.voxel_property_plot(voxels, wildcards.loss_func, df_filter_string=None)
        fig.savefig(output[0])
        g = sfp.plotting.voxel_property_joint(voxels, 'reg', ['eccen', wildcards.loss_func], None, x_bins=20,
                                              x_estimator=np.median, marginal_kws={'kde': False})
        g.fig.savefig(output[1])


def get_params_csv(wildcards):
    path_template = os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model',
                                 'stim_class', 'bayesian_posterior', f"{wildcards.df_filter}", '%s',
                                 f'{wildcards.groupaverage}_{wildcards.task}_v1_e1-12_%s_b10_r0.001_g0_{wildcards.model_type}_all_models.csv')
    precision = os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis',
                             'stim_class', 'bayesian_posterior', f'{wildcards.task}_v1_e1-12_'
                             f'summary_mean_no-filter_precision.csv'),
    paths = {}
    try:
        ps = []
        if wildcards.groupaverage == 'sub-groupaverage':
            if wildcards.plot_kind in ['dist', 'strip', 'bootstraps']:
                ps.append(path_template % ('initial', 'summary'))
            else:
                raise Exception(f"Can't do sub-groupaverage with plot_kind {wildcards.plot_kind}!"
                                " Only 'dist', 'strip', 'bootstraps' are allowed")
        else:
            if wildcards.plot_kind in ['dist', 'pair', 'pair-drop', 'compare', 'bootstraps',
                                       'dist-overall', 'bootstraps-overall']:
                ps.append(path_template % ('bootstrap', 'full'))
            if wildcards.plot_kind in ['point', 'strip', 'compare', 'median']:
                if wildcards.vf == 'vertical':
                    vf = ['upper', 'lower']
                elif wildcards.vf == 'horizontal':
                    vf = ['left', 'right']
                elif wildcards.vf == 'eccen':
                    vf = ['inner', 'outer']
                else:
                    vf = [wildcards.vf]
                for v in vf:
                    if v == 'all':
                        folder = 'initial'
                    else:
                        folder = 'visual_field_%s' % v
                    ps.append(path_template % (folder, 'summary'))
            if wildcards.plot_kind.endswith('overall'):
                paths['precision'] = precision
    except AttributeError:
        # this is the figure_background_with_current or sigma_interpretation
        # rules (neither of which have wildcards.plot_kind)
        if wildcards.groupaverage == 'sub-groupaverage':
            ps = path_template % ('initial', 'summary')
        else:
            ps = path_template % ('bootstrap', 'full')
            paths['precision'] = precision
    paths['params'] = ps
    return paths


rule figure_params:
    input:
        unpack(get_params_csv),
    output:
        os.path.join(config['DATA_DIR'], "derivatives", 'figures', '{context}', "{groupaverage}_{df_filter}_{model_type}_params_visualfield-{vf}_{plot_kind}_s-{seed}_{task}.{ext}")
    log:
        os.path.join(config['DATA_DIR'], "code", 'figures', '{context}', "{groupaverage}_{df_filter}_{model_type}_params_visualfield-{vf}_{plot_kind}_s-{seed}_{task}_{ext}-%j.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", 'figures', '{context}', "{groupaverage}_{df_filter}_{model_type}_params_visualfield-{vf}_{plot_kind}_s-{seed}_{task}_{ext}_benchmark.txt")
    run:
        import pandas as pd
        import sfp
        import matplotlib as mpl
        df = []
        for p in input.params:
            tmp = sfp.figures.prep_df(pd.read_csv(p), wildcards.task)
            if wildcards.plot_kind.endswith('overall'):
                # get the median parameter value per subject and model type
                tmp = tmp.groupby(['subject', 'model_parameter', 'fit_model_type']).median().reset_index()
                precision = sfp.figures.prep_df(pd.read_csv(input.precision[0]), wildcards.task)
                tmp = tmp.merge(precision, on=['subject'])
                tmp = sfp.figures.precision_weighted_bootstrap(tmp, int(wildcards.seed), 100, 'fit_value',
                                                               ['model_parameter',
                                                                'fit_model_type'], 'precision')
            df.append(sfp.figures.prep_model_df(tmp))
        if wildcards.plot_kind.startswith('pair'):
            if wildcards.plot_kind.endswith('drop'):
                drop_outlier = True
            else:
                drop_outlier = False
            # this returns the PairPlot, so we need to do .fig to
            # grab the underlying Figure
            fig = sfp.figures.model_parameters_pairplot(df[0], drop_outlier).fig
        elif wildcards.plot_kind == 'compare':
            if wildcards.vf == 'all':
                # this returns the FacetGrid, so we need to do .fig to
                # grab the underlying Figure. bootstrap_df comes before
                # regular one
                fig = sfp.figures.model_parameters_compare_plot(df[1], df[0]).fig
            else:
                # first draw the distribution of model parameters
                # for model fit to whole visual field
                fig = sfp.figures.model_parameters(df[0], 'dist', wildcards.vf, size=7)
                # this sets the markers and labels we'll use to
                # distinguish the different parts of the visual
                # field
                if wildcards.vf == 'vertical':
                    kwargs = [{'marker': '^', 'size': 7}, {'marker': 'v', 'size': 7}]
                    labels = ['Upper visual field', 'Lower visual field']
                elif wildcards.vf == 'horizontal':
                    kwargs = [{'marker': '<', 'size': 7}, {'marker': '>', 'size': 7}]
                    labels = ['Left visual field', 'Right visual field']
                elif wildcards.vf == 'eccen':
                    kwargs = [{'size': 5, 'marker': 'o'}, {'marker': "o", 'size': 10}]
                    labels = ['Inner visual field', 'Outer visual field']
                kwargs.append({'marker': 'o', 'size': 7})
                labels.append('Full visual field')
                # add the two estimates from parts of the visual
                # field onto the existing figure, as strip plots
                # (because we only have a single estimate per model,
                # not the full distribution). we don't update the
                # legend within the function...
                fig = sfp.figures.model_parameters(df[1], 'strip', wildcards.vf, fig, False,
                                                   **kwargs[0])
                fig = sfp.figures.model_parameters(df[2], 'strip', wildcards.vf, fig, False,
                                                   **kwargs[1])
                # instead doing it manually with some dummy markers
                dummy_markers = []
                for m, l in zip(kwargs[::-1], labels[::-1]):
                    m['markersize'] = m.pop('size')
                    dummy_markers.append(mpl.lines.Line2D([], [], linewidth=0, color='gray',
                                                          label=l, **m))
                fig.axes[-1].legend(handles=dummy_markers, loc=(1.01, .5), frameon=False)
        else:
            if wildcards.model_type == 'full_full_absolute':
                # since this doesn't fit those
                df[0] = df[0].query("model_parameter not in ['$A_3$', '$A_4$']")
            # don't add a legend if the plot_kind is point or dist-overall
            add_legend = {'point': False, 'dist-overall': False}.get(wildcards.plot_kind, True)
            # or if we're plotting the sub-groupaverage
            if wildcards.groupaverage == 'sub-groupaverage':
                add_legend = False
            plot_kind = wildcards.plot_kind.replace('-overall', '')
            fig = sfp.figures.model_parameters(df[0], plot_kind, wildcards.vf,
                                               add_legend=add_legend)
        fig.savefig(output[0], bbox_inches='tight')


rule figure_feature_df:
    input:
        unpack(get_params_csv),
    output:
        os.path.join(config['DATA_DIR'], "derivatives", 'figures', '{context}', "{groupaverage}_{df_filter}_{model_type}_feature_visualfield-{vf}_{feature_type}_{plot_kind}_angles-{angles}_s-{seed}_{task}_{ref_frame}.{ext}")
    log:
        os.path.join(config['DATA_DIR'], "code", 'figures', '{context}', "{groupaverage}_{df_filter}_{model_type}_feature_visualfield-{vf}_{feature_type}_{plot_kind}_angles-{angles}_s-{seed}_{task}_{ref_frame}_{ext}-%j.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", 'figures', '{context}', "{groupaverage}_{df_filter}_{model_type}_feature_visualfield-{vf}_{feature_type}_{plot_kind}_angles-{angles}_s-{seed}_{task}_{ref_frame}_{ext}_benchmark.txt")
    run:
        import pandas as pd
        import seaborn as sns
        import sfp
        df = sfp.figures.prep_df(pd.read_csv(input.params[0]), wildcards.task)
        if wildcards.plot_kind.endswith('overall'):
            # get the median parameter value per subject and model type
            df = df.groupby(['subject', 'model_parameter', 'fit_model_type']).median().reset_index()
            precision = sfp.figures.prep_df(pd.read_csv(input.precision[0]), wildcards.task)
            df = df.merge(precision, on=['subject'])
            df = sfp.figures.precision_weighted_bootstrap(df, int(wildcards.seed), 100, 'fit_value',
                                                          ['model_parameter', 'fit_model_type'],
                                                          'precision')
            col_wrap = None
        else:
            col_wrap = 3
        if wildcards.angles == 'avg':
            angles = True
        elif wildcards.angles == 'all':
            angles = False
        g = sfp.figures.feature_df_plot(df, angles, wildcards.ref_frame, wildcards.feature_type,
                                        wildcards.vf, wildcards.context, col_wrap=col_wrap)
        g.fig.savefig(output[0], bbox_inches='tight')


rule figure_schematic:
    output:
        os.path.join(config["DATA_DIR"], 'derivatives', 'figures', '{context}', 'schematic_{schematic_type}.{ext}')
    log:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'schematic_{schematic_type}_{ext}-%j.log')
    benchmark:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'schematic_{schematic_type}_{ext}_benchmark.txt')
    run:
        import sfp
        if wildcards.schematic_type == '2d':
            fig = sfp.figures.model_schematic(wildcards.context)
        elif wildcards.schematic_type == '2d-inputs':
            fig = sfp.figures.input_schematic(wildcards.context)
        elif wildcards.schematic_type == 'background':
            fig = sfp.figures.theory_background_figure(wildcards.context)
        fig.savefig(output[0], bbox_inches='tight')


rule figure_stimulus_schematic:
    input:
        os.path.join(config['DATA_DIR'], 'stimuli', '{task}_stimuli.npy'),
        os.path.join(config['DATA_DIR'], 'stimuli', '{task}_stim_description.csv'),
    output:
        os.path.join(config["DATA_DIR"], 'derivatives', 'figures', '{context}', 'schematic_stimulus_{task}.{ext}')
    log:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'schematic_stimulus_{task}_{ext}-%j.log')
    benchmark:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'schematic_stimulus_{task}_{ext}_benchmark.txt')
    run:
        import sfp
        import numpy as np
        import pandas as pd
        stim = np.load(input[0])
        stim_df = pd.read_csv(input[1])
        fig = sfp.figures.stimulus_schematic(stim, stim_df, wildcards.context)
        fig.savefig(output[0], bbox_inches='tight')


rule figure_model_schematic:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model', 'stim_class',
                     'bayesian_posterior', '{df_filter}', 'initial_cv',
                     'individual_task-sfprescaled_v1_e1-12_summary_b10_r0.001_g0_s0_all_cv_loss.csv'),
    output:
        os.path.join(config["DATA_DIR"], 'derivatives', 'figures', '{context}', 'schematic_models{extra}_{df_filter}.{ext}')
    log:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'schematic_models{extra}_{df_filter}_{ext}-%j.log')
    benchmark:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'schematic_models{extra}_{df_filter}_{ext}_benchmark.txt')
    run:
        import sfp
        import pandas as pd
        if 'annot' in wildcards.extra:
            annotate = True
        else:
            annotate = False
        if 'sort' in wildcards.extra:
            warnings.warn("Sorting by remeaned cv loss, using weighted_normed_loss, task-sfprescaled runs")
            df = sfp.figures.prep_df(pd.read_csv(input[0]), 'task-sfprescaled')
            df = sfp.figures._demean_df(df.query('loss_func == "weighted_normed_loss"'))
            gb = df.query("loss_func == 'weighted_normed_loss'").groupby('fit_model_type')
            order = gb['remeaned_cv_loss'].median().sort_values(ascending=False).index
        else:
            order = None
        if 'doubleup' in wildcards.extra:
            doubleup = True
            if order is not None:
                raise Exception("Can only doubleup the schematic if we're not sorting it!")
        else:
            doubleup = False
        fig = sfp.figures.model_types(wildcards.context, annotate=annotate, order=order,
                                      doubleup=doubleup)
        fig.savefig(output[0], bbox_inches='tight')


rule figure_example_voxel:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis',
                     'stim_class', 'bayesian_posterior', 'sub-wlsubj001', 'ses-04',
                     'sub-wlsubj001_ses-04_task-sfprescaled_v1_e1-12_summary.csv'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model', 'stim_class',
                     'bayesian_posterior', "{df_filter}", 'initial', 'sub-wlsubj001', 'ses-04',
                     'sub-wlsubj001_ses-04_task-sfprescaled_v1_e1-12_summary_b10_r0.001_'
                     'g0_cNone_nNone_{model_type}_model.pt')
    output:
        os.path.join(config["DATA_DIR"], 'derivatives', 'figures', '{context}', 'example_voxels_{df_filter}_{model_type}.{ext}')
    log:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'example_voxels_{df_filter}_{model_type}_{ext}-%j.log')
    benchmark:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'example_voxels_{df_filter}_{model_type}_{ext}_benchmark.txt')
    run:
        import pandas as pd
        import sfp
        df = pd.read_csv(input[0])
        model = sfp.analyze_model.load_LogGaussianDonut(input[1].replace('_model.pt', ''))
        g = sfp.figures.example_voxels(df, model, context=wildcards.context,
                                       extend_sf=True)
        g.fig.savefig(output[0], bbox_inches='tight')


rule figure_example_ecc_bins:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves',
                     'stim_class', 'bayesian_posterior', 'sub-wlsubj001', 'ses-04',
                     'sub-wlsubj001_ses-04_task-sfprescaled_v1_e1-12_eccen_bin_summary.csv'),
    output:
        os.path.join(config["DATA_DIR"], 'derivatives', 'figures', '{context}', 'example_ecc_bins.{ext}')
    log:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'example_ecc_bins_{ext}-%j.log')
    benchmark:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'example_ecc_bins_{ext}_benchmark.txt')
    run:
        import pandas as pd
        import sfp
        df = pd.read_csv(input[0])
        fig = sfp.figures.example_eccentricity_bins(df, context=wildcards.context)
        fig.savefig(output[0], bbox_inches='tight')


rule figure_peakiness_check:
    input:
        first_levels = [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis',
                                     'stim_class', 'bayesian_posterior', '{subject}', 'ses-04',
                                     '{subject}_ses-04_task-sfprescaled_v1_e1-12_summary.csv').format(subject=subj)
                        for subj in SUBJECTS if TASKS.get((subj, 'ses-04'), None) == 'task-sfprescaled'],
        models = [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model', 'stim_class',
                               'bayesian_posterior', "{{df_filter}}", 'initial', '{subject}', 'ses-04',
                               '{subject}_ses-04_task-sfprescaled_v1_e1-12_summary_b10_r0.001_'
                               'g0_cNone_nNone_{{model_type}}_model.pt').format(subject=subj)
                        for subj in SUBJECTS if TASKS.get((subj, 'ses-04'), None) == 'task-sfprescaled'],
    output:
        os.path.join(config["DATA_DIR"], 'derivatives', 'figures', '{context}', 'peakiness_{df_filter}_{model_type}_{col}.{ext}')
    log:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'peakiness_{df_filter}_{model_type}_{col}_{ext}-%j.log')
    benchmark:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'peakiness_{df_filter}_{model_type}_{col}_{ext}_benchmark.txt')
    run:
        import pandas as pd
        import sfp
        models = [sfp.analyze_model.load_LogGaussianDonut(p.replace('_model.pt', '')) for p in input.models]
        dfs = []
        for p in input.first_levels:
            dfs.append(pd.read_csv(p))
            subj = re.findall('(sub-wlsubj[0-9]+)_', p)[0]
            dfs[-1]['subject'] = subj
        if wildcards.col == 'all':
            col = None
        elif wildcards.col == 'individual':
            col = 'subject'
        df_filter_str = get_df_filter_str(wildcards)
        g = sfp.figures.peakiness_check(dfs, models, col=col,
                                        df_filter_string=df_filter_str,
                                        context=wildcards.context)
        # we set the dpi here because we rasterize the 2d histogram (in order
        # to reduce the size) and so we increase the dpi so it looks better
        g.fig.savefig(output[0], bbox_inches='tight', dpi=400)


rule figure_compare_sigma:
    input:
        first_levels = [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis',
                                     'stim_class', 'bayesian_posterior', '{subject}', 'ses-04',
                                     '{subject}_ses-04_task-sfprescaled_v1_e1-12_summary.csv').format(subject=subj)
                        for subj in SUBJECTS if TASKS.get((subj, 'ses-04'), None) == 'task-sfprescaled'],
        models = [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model', 'stim_class',
                               'bayesian_posterior', "{{df_filter}}", 'initial', '{subject}', 'ses-04',
                               '{subject}_ses-04_task-sfprescaled_v1_e1-12_summary_b10_r0.001_'
                               'g0_cNone_nNone_{{model_type}}_model.pt').format(subject=subj)
                        for subj in SUBJECTS if TASKS.get((subj, 'ses-04'), None) == 'task-sfprescaled'],
    output:
        os.path.join(config["DATA_DIR"], 'derivatives', 'figures', '{context}', 'sigma_vs_ecc_{df_filter}_{model_type}.{ext}'),
        os.path.join(config["DATA_DIR"], 'derivatives', 'figures', '{context}', 'sigma_vs_period_{df_filter}_{model_type}.{ext}'),
    log:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'sigma_compare_{df_filter}_{model_type}_{ext}.log'),
    benchmark:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'sigma_compare_{df_filter}_{model_type}_{ext}_benchmark.txt'),
    run:
        import pandas as pd
        import sfp
        models = [sfp.analyze_model.load_LogGaussianDonut(p.replace('_model.pt', '')) for p in input.models]
        dfs = []
        for p in input.first_levels:
            dfs.append(pd.read_csv(p))
            subj = re.findall('(sub-wlsubj[0-9]+)_', p)[0]
            dfs[-1]['subject'] = subj
        df_filter_str = get_df_filter_str(wildcards).replace(',drop_voxels_near_border', '')
        g, fig = sfp.figures.compare_sigma_and_pref_period(dfs, models, df_filter_str, wildcards.context)
        g.fig.savefig(output[0], bbox_inches='tight')
        fig.savefig(output[1], bbox_inches='tight')


rule figure_compare_surface_area:
    input:
        unpack(get_params_csv),
        # these each return 2 mgzs, so need that m for m in find_prf_mgz to
        # flatten this into a list of strings (instead of a list of lists of
        # strings)
        freesurfer_dir = lambda wildcards: [os.path.join(config['DATA_DIR'], 'derivatives', 'freesurfer',
                                                        '{subject}').format(subject=subj.replace('sub-', ''))
                                            for subj in SUBJECTS if TASKS.get((subj, 'ses-04'), None) == 'task-sfprescaled'],
        vareas = lambda wildcards: [m for subj in SUBJECTS for m in find_prf_mgz(wildcards, 'varea', subj)
                                    if TASKS.get((subj, 'ses-04'), None) == 'task-sfprescaled'],
        eccens = lambda wildcards: [m for subj in SUBJECTS for m in find_prf_mgz(wildcards, 'eccen', subj)
                                    if TASKS.get((subj, 'ses-04'), None) == 'task-sfprescaled'],
    output:
        os.path.join(config["DATA_DIR"], 'derivatives', 'figures', '{context}', '{groupaverage}_v1_area_vs_period_{task}_{df_filter}_{model_type}_{atlas_type}.{ext}'),
        # unfortunately, need the {ext} and {context} in the path, because Snakemake wants every output in a rule have the same wildcards
        os.path.join(config["DATA_DIR"], 'derivatives', 'figures', '{context}', '{groupaverage}_v1_area_vs_period_linreg_{task}_{df_filter}_{model_type}_{atlas_type}_{ext}.txt'),
    log:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', '{groupaverage}_v1_area_vs_period_{task}_{df_filter}_{model_type}_{atlas_type}_{ext}.log'),
    benchmark:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', '{groupaverage}_v1_area_vs_period_{task}_{df_filter}_{model_type}_{atlas_type}_{ext}_benchmark.txt'),
    run:
        import sfp
        import pandas as pd
        subjects = [subj for subj in SUBJECTS if TASKS.get((subj, 'ses-04'), None) == wildcards.task]
        template = input.vareas[0].replace('sub-wlsubj001', '{subject}').replace('lh', '{hemi}').replace('rh', '{hemi}').replace('varea', '{prop}')
        df = sfp.figures.prep_df(pd.read_csv(input.params), wildcards.task)
        target_ecc = 6
        g, linreg = sfp.figures.compare_surface_area_and_pref_period(df, subjects, template, target_ecc=target_ecc,
                                                                     context=wildcards.context)
        g.fig.savefig(output[0], bbox_inches='tight')
        result = f"Linear regression predicting preferred period at {target_ecc} degrees using full V1 surface area, bootstrapped 1000x across subjects:\n"
        cis = linreg.apply(np.percentile, q=[16, 50, 84])
        for col in ['coeff', 'intercept', 'R^2']:
            result += f'{col}: {cis[f"{col}"][1]:.02e}, [{cis[f"{col}"][0]:.02e}, {cis[f"{col}"][2]:.02e}]\n'
        result += 'All values are medians with 68% CI'
        with open(output[1], 'w') as f:
            f.writelines(result)


rule figure_compare_visual_field:
    input:
        models = [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model',
                               'stim_class', 'bayesian_posterior', '{{df_filter}}',
                               '{vf}', 'individual_{{task}}_v1_e1-12_summary_b10_r0.001_g0_{{model_type}}_all_models.csv').format(vf=vf)
                  for vf in ['initial', 'visual_field_vertical-meridia', 'visual_field_horizontal-meridia']],
        precision = os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis',
                                 'stim_class', 'bayesian_posterior', '{task}_v1_e1-12_'
                                 'summary_mean_no-filter_precision.csv'),
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'figures', '{context}', 'visual-field-diff_subjects_{task}_{df_filter}_{model_type}_s-{seed}.{ext}'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'figures', '{context}', 'visual-field-diff_diff_{task}_{df_filter}_{model_type}_s-{seed}.{ext}'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'figures', '{context}', 'visual-field-diff_comparison_{task}_{df_filter}_{model_type}_s-{seed}.{ext}'),
    log:
        os.path.join(config['DATA_DIR'], 'code', 'figures', '{context}', 'visual-field-diff_{task}_{df_filter}_{model_type}_s-{seed}_{ext}-%j.log'),
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'figures', '{context}', 'visual-field-diff_{task}_{df_filter}_{model_type}_s-{seed}_{ext}-%j_benchmark.txt'),
    run:
        import pandas as pd
        import sfp
        precision  = sfp.figures.prep_df(pd.read_csv(input.precision), wildcards.task)
        df = []
        for p in input.models:
            tmp = sfp.figures.prep_df(pd.read_csv(p), wildcards.task)
            if 'vertical' in p:
                tmp['Visual field'] = 'Vertical meridians'
            elif 'horizontal' in p:
                tmp['Visual field'] = 'Horizontal meridians'
            else:
                tmp['Visual field'] = 'Full visual field'
            df.append(tmp)
        df = pd.concat(df)
        pal = {'Vertical meridians': 'C0', 'Horizontal meridians': 'C1',
               'Full visual field': 'k'}
        g = sfp.figures.feature_df_plot(df, True, col_wrap=3,
                                        hue='Visual field', pal=pal,
                                        hue_kws={'linestyle': ['--', '-', '-']})
        g.add_legend()
        g.fig.savefig(output[0], bbox_inches='tight')
        df = df[df['Visual field'] != "Full visual field"]
        g = sfp.figures.feature_difference_plot(df, precision, height=3,
                                                feature_kwargs={'orientation': [0],
                                                                'retinotopic_angle': [0]})
        g.fig.savefig(output[1], bbox_inches='tight')
        df = df.groupby(['subject', 'model_parameter', 'fit_model_type', 'Visual field']).median().reset_index()
        df = df.merge(precision, on=['subject'])
        df = sfp.figures.precision_weighted_bootstrap(df, int(wildcards.seed), 100, 'fit_value',
                                                      ['model_parameter', 'fit_model_type', 'Visual field'], 'precision')
        g = sfp.figures.feature_df_plot(df, True, hue='Visual field', pal=pal, height=3)
        g.ax.legend(loc='lower right', title='Visual field')
        g.fig.savefig(output[2], bbox_inches='tight')


rule figure_background:
    output:
        os.path.join(config["DATA_DIR"], 'derivatives', 'figures', '{context}', 'background_{y_val}.{ext}')
    log:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'background_{y_val}_{ext}-%j.log')
    benchmark:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'background_{y_val}_{ext}_benchmark.txt')
    run:
        import sfp
        import seaborn as sns
        df = sfp.figures.existing_studies_df()
        y = {'period': 'Preferred period (deg)',
             'frequency': 'Preferred spatial frequency (cpd)'}[wildcards.y_val]
        g = sfp.figures.existing_studies_figure(df, y, wildcards.context)
        g.fig.savefig(output[0], bbox_inches='tight')


rule figure_background_with_current:
    input:
        unpack(get_params_csv),
    output:
        os.path.join(config["DATA_DIR"], 'derivatives', 'figures', '{context}', '{groupaverage}_{df_filter}_{task}_background_{y_val}_{model_type}_s-{seed}.{ext}')
    log:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', '{groupaverage}_{df_filter}_{task}_background_{y_val}_{model_type}_s-{seed}_{ext}-%j.log')
    benchmark:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', '{groupaverage}_{df_filter}_{task}_background_{y_val}_{model_type}_s-{seed}_{ext}_benchmark.txt')
    run:
        import sfp
        import seaborn as sns
        import pandas as pd
        df = sfp.figures.prep_df(pd.read_csv(input.params), wildcards.task)
        try:
            precision = sfp.figures.prep_df(pd.read_csv(input.precision[0]), wildcards.task)
        except AttributeError:
            # then there was no precision in input
            precision = None
        y = {'period': 'Preferred period (deg)',
             'frequency': 'Preferred spatial frequency (cpd)'}[wildcards.y_val]
        g = sfp.figures.existing_studies_with_current_figure(df, int(wildcards.seed), precision,
                                                             y, wildcards.context)
        g.fig.savefig(output[0], bbox_inches='tight')


def get_compose_input(wildcards):
    path_template = os.path.join(config['DATA_DIR'], 'derivatives', "figures", wildcards.context,
                                 "%s.svg")
    if "crossvalidation" in wildcards.figure_name:
        seed, df_filter = re.findall("crossvalidation_s-([0-9]+)_(filter-any|filter-mean|no-filter)", wildcards.figure_name)[0]
        fig_names = ["schematic_models-annot",
                     f'individual_{df_filter}_cv_model_point-remeaned_h_s-{seed}_task-sfprescaled']
        if 'sort' in wildcards.figure_name:
            fig_names[0] += '-sort'
            fig_names[1] = fig_names[1].replace('_h_s', '_h_sort_s')
        if 'doubleup' in wildcards.figure_name:
            fig_names[0] += '-doubleup'
            fig_names[1] = fig_names[1].replace('_h_s', '_h_doubleup_s')
        if '-nc' in wildcards.figure_name:
            fig_names[1] = fig_names[1].replace('point-remeaned', 'point-remeaned-nc')
        fig_names[0] += '_' + df_filter
        paths = [path_template % n for n in fig_names]
    elif "with_legend" in wildcards.figure_name:
        paths = [path_template % wildcards.figure_name.replace('_with_legend', '')]
    elif '2d_summary' in wildcards.figure_name:
        groupaverage, df_filter, model, seed = re.findall("([a-z-]+)_([a-z-]+)_([a-z_]+)_2d_summary_s-([0-9]+)", wildcards.figure_name)[0]
        template_name = (f'{groupaverage}_{df_filter}_{model}_feature_visualfield-all_{{}}_bootstraps-overall_'
                         f'angles-{{}}_s-{seed}_task-sfprescaled_{{}}')
        angles = {'pref-period': 'avg', 'pref-period-contour': 'all', 'max-amp': 'all'}
        paths = [path_template % template_name.format(feature, angles[feature], frame) for
                 frame, feature in itertools.product(['relative', 'absolute'],
                                                     ['pref-period', 'pref-period-contour', 'max-amp'])]
    elif '1d_summary' in wildcards.figure_name:
        groupaverage, seed = re.findall("([a-z-]+)_1d_summary_s-([0-9]+)", wildcards.figure_name)[0]
        period_name = f'{groupaverage}_1d_pref-period-overall_s-{seed}_task-sfprescaled_with_legend'
        bw_name = f'{groupaverage}_1d_bandwidth-overall_s-{seed}_task-sfprescaled'
        paths = [path_template.replace('figures', 'compose_figures') % period_name,
                 path_template % bw_name]
    elif 'stimulus' in wildcards.figure_name:
        task = re.findall("_(task-[a-z]+)", wildcards.figure_name)[0]
        paths = [path_template % f'{task}_base_frequencies',
                 path_template % f'schematic_stimulus_{task}',
                 path_template % f'{task}_presented_frequencies']
    elif 'background' in wildcards.figure_name:
        paths = [path_template % 'schematic_background']
    elif 'example_voxel' in wildcards.figure_name:
        df_filter, model = re.findall("([a-z-]+)_([a-z_]+)_example_voxels", wildcards.figure_name)[0]
        paths = [path_template % f"peakiness_{df_filter}_{model}_all",
                 path_template % f"example_voxels_{df_filter}_{model}"]
    elif 'parameters' in wildcards.figure_name:
        groupaverage, df_filter, model, seed = re.findall("([a-z-]+)_([a-z-]+)_([a-z_]+)_parameters_s-([0-9]+)", wildcards.figure_name)[0]
        paths = [path_template % f"{groupaverage}_{df_filter}_{model}_params_visualfield-all_dist_s-None_task-sfprescaled",
                 path_template % f"{groupaverage}_{df_filter}_{model}_params_visualfield-all_dist-overall_s-{seed}_task-sfprescaled"]
    elif 'visual-field-diff' in wildcards.figure_name:
        df_filter, model, seed = re.findall("visual-field-diff_([a-z-]+)_([a-z_]+)_s-([0-9]+)", wildcards.figure_name)[0]
        paths = [path_template % f"visual-field-diff_comparison_task-sfprescaled_{df_filter}_{model}_s-{seed}",
                 path_template % f"visual-field-diff_diff_task-sfprescaled_{df_filter}_{model}_s-{seed}"]
    elif 'example_ecc_bins_with_stim' in wildcards.figure_name:
        paths = [path_template % 'example_ecc_bins']
    elif 'schematic_model_2d' in wildcards.figure_name:
        paths = [path_template % 'schematic_2d-inputs',
                 path_template.replace('figures', 'compose_figures') % 'schematic_2d_with_legend']
    return paths


rule compose_figures:
    input:
        get_compose_input,
    output:
        os.path.join(config["DATA_DIR"], 'derivatives', 'compose_figures', '{context}', '{figure_name}.svg')
    log:
        os.path.join(config["DATA_DIR"], 'code', 'compose_figures', '{context}', '{figure_name}_svg-%j.log')
    benchmark:
        os.path.join(config["DATA_DIR"], 'code', 'compose_figures', '{context}', '{figure_name}_svg_benchmark.txt')
    run:
        from svgutils import compose
        import sfp
        if 'crossvalidation' in wildcards.figure_name:
            sfp.compose_figures.crossvalidation(*input, output[0], wildcards.context)
        elif 'with_legend' in wildcards.figure_name:
            if '1d_pref-period-overall' in wildcards.figure_name:
                sfp.compose_figures.add_legend(input[0], 'half', (143, 136),
                                               output[0], (0, 0), 1, 'rel',
                                               wildcards.context)
            if 'schematic_2d' in wildcards.figure_name:
                sfp.compose_figures.add_legend(input[0], 'half', (70, 0),
                                               output[0], (0, 70), .8, 'rel',
                                               wildcards.context)
        elif '2d_summary' in wildcards.figure_name:
            sfp.compose_figures.feature_df_summary(input[:3], input[3:],
                                                   output[0], wildcards.context)
        elif '1d_summary' in wildcards.figure_name:
            sfp.compose_figures.summary_1d(input[0], input[1], output[0], wildcards.context)
        elif 'stimulus' in wildcards.figure_name:
            sfp.compose_figures.stimulus_figure(input[0], input[1], input[2],
                                                output[0], wildcards.context)
        elif 'background' in wildcards.figure_name:
            sfp.compose_figures.background_figure(input[0], output[0],
                                                  wildcards.context)
        elif 'example_voxel' in wildcards.figure_name:
            sfp.compose_figures.example_voxels(input[0], input[1], output[0],
                                               wildcards.context)
        elif 'parameters' in wildcards.figure_name:
            sfp.compose_figures.parameters(input[0], input[1], output[0],
                                           wildcards.context)
        elif 'visual-field-diff' in wildcards.figure_name:
            sfp.compose_figures.visual_field_differences(input[0], input[1],
                                                         output[0], wildcards.context)
        elif 'example_ecc_bins' in wildcards.figure_name:
            sfp.compose_figures.example_ecc_bins(input[0], output[0], wildcards.context)
        elif 'schematic_model_2d' in wildcards.figure_name:
            sfp.compose_figures.schematic_model_2d(*input, output[0], wildcards.context)


rule presented_spatial_frequency_plot:
    input:
        os.path.join(config['DATA_DIR'], 'stimuli', '{task}_presented_frequencies.csv'),
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'figures', '{context}', '{task}_presented_frequencies.svg'),
    log:
        os.path.join(config['DATA_DIR'], 'code', 'figures', '{context}', '{task}_presented_frequencies_svg-%j.log'),
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'figures', '{context}', '{task}_presented_frequencies_svg_benchmark.txt'),
    run:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import sfp
        df = pd.read_csv(input[0])
        plot_params, fig_width = sfp.style.plotting_style(wildcards.context, figsize='half')
        plot_params['font.size'] = '8'
        plot_params['axes.titlesize'] = '8'
        plot_params['axes.labelsize'] = '8'
        plot_params['legend.fontsize'] = '8'
        plt.style.use(plot_params)
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width*.65))
        pal = sfp.plotting.get_palette('freq_space_distance', None,
                                       df.freq_space_distance.unique())
        sns.lineplot(x='eccentricity', y='spatial_frequency',
                     hue='freq_space_distance', data=df, palette=pal)
        ax.set(yscale='log', xlabel='Eccentricity (deg)',
               ylabel='Presented spatial\nfrequency (cpd)')
        # turn the legend off
        ax.legend_.set_visible(False)
        fig.tight_layout()
        fig.savefig(output[0], bbox_inches='tight')


rule stimulus_base_frequency_plot:
    input:
        os.path.join(config['DATA_DIR'], 'stimuli', '{task}_stim_description.csv'),
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'figures', '{context}', '{task}_base_frequencies.svg'),
    log:
        os.path.join(config['DATA_DIR'], 'code', 'figures', '{context}', '{task}_base_frequencies_svg-%j.log'),
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'figures', '{context}', '{task}_base_frequencies_svg_benchmark.txt'),
    run:
        import pandas as pd
        import sfp
        df = pd.read_csv(input[0])
        g = sfp.figures.stimulus_frequency(df, wildcards.context)
        g.savefig(output[0])


rule figure_mtf:
    input:
        os.path.join(config['DATA_DIR'], 'stimuli', 'mtf_func.pkl'),
        os.path.join(config['DATA_DIR'], 'stimuli', 'mtf_func_data.csv'),
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'figures', '{context}', 'mtf.{ext}')
    log:
        os.path.join(config['DATA_DIR'], 'code', 'figures', '{context}', 'mtf_{ext}-%j.log')
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'figures', '{context}', 'mtf_{ext}_benchmark.txt')
    run:
        import sfp
        import pickle
        import pandas as pd
        with open(input[0], 'rb') as f:
            mtf_func = pickle.load(f)
        df = pd.read_csv(input[1])
        fig = sfp.figures.mtf(mtf_func, df, wildcards.context)
        fig.savefig(output[0], bbox_inches='tight')


rule sigma_interpretation:
    input:
        unpack(get_params_csv),
    output:
        os.path.join(config['DATA_DIR'], "derivatives", 'figures', "{groupaverage}_{df_filter}_{model_type}_sigma-interp_visualfield-{vf}_s-{seed}_{task}.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", 'figures', "{groupaverage}_{df_filter}_{model_type}_sigma-interp_visualfield-{vf}_s-{seed}_{task}-%j.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", 'figures', "{groupaverage}_{df_filter}_{model_type}_sigma-interp_visualfield-{vf}_s-{seed}_{task}_benchmark.txt")
    run:
        import pandas as pd
        import sfp
        df = sfp.figures.prep_df(pd.read_csv(input.params), wildcards.task)
        # get the median parameter value per subject and model type
        df = df.groupby(['subject', 'model_parameter', 'fit_model_type']).median().reset_index()
        precision = sfp.figures.prep_df(pd.read_csv(input.precision[0]), wildcards.task)
        df = df.merge(precision, on=['subject'])
        df = sfp.figures.precision_weighted_bootstrap(df, int(wildcards.seed), 100, 'fit_value',
                                                       ['model_parameter',
                                                        'fit_model_type'], 'precision')
        df = sfp.figures.prep_model_df(df)
        result = sfp.figures.sigma_interpretation(df)
        with open(output[0], 'w') as f:
            f.writelines(result)


rule predicted_bold:
    input:
        os.path.join(config['DATA_DIR'], 'stimuli', '{task}_stim_description.csv'),
    output:
        # NOTE: This doesn't get placed in the DATA_DIR, because it's currently
        # a temporary thing that doesn't end up in the paper.
        os.path.join('data', 'tuning_2d_model', '{task}_params-{param_set}_predicted-bold.pkl'),
    log:
        os.path.join(config['DATA_DIR'], "code", 'figures', "{task}_params-{param_set}_bold.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", 'figures', "{task}_params-{param_set}_bold_benchmark.txt")
    run:
        import sfp
        import pickle
        import numpy as np
        import pandas as pd
        stim_df = pd.read_csv(input[0]).drop_duplicates('class_idx')
        stim_shape = stim_df.res.unique()[0]
        stims = []
        mags = []
        oris = []
        eccen, angle = sfp.utils.create_prf_loc_map(stim_shape, 24)
        bolds = []
        w_r = []
        w_a = []
        phase = []
        param_remap = dict(zip(sfp.plotting.ORIG_PARAM_ORDER, sfp.plotting.PLOT_PARAM_ORDER))
        if wildcards.param_set == 'paper':
            model_params = {'sigma': 2,
                            'sf_ecc_slope': .1,
                            'sf_ecc_intercept': .35,
                            'abs_mode_cardinals': .06,
                            'abs_mode_obliques': -.03,
                            'rel_mode_cardinals': .06,
                            'rel_mode_obliques': 0,
                            'abs_amplitude_cardinals': .04,
                            'abs_amplitude_obliques': -.01,
                            'rel_amplitude_cardinals': 0,
                            'rel_amplitude_obliques': 0}
        elif wildcards.param_set == 'simple':
            model_params = {'sigma': 2,
                            'sf_ecc_slope': .1,
                            'sf_ecc_intercept': .35,
                            'abs_mode_cardinals': 0,
                            'abs_mode_obliques': 0,
                            'rel_mode_cardinals': 0,
                            'rel_mode_obliques': 0,
                            'abs_amplitude_cardinals': 0,
                            'abs_amplitude_obliques': 0,
                            'rel_amplitude_cardinals': 0,
                            'rel_amplitude_obliques': 0}
        elif wildcards.param_set == 'flat':
            model_params = {'sigma': 2,
                            'sf_ecc_slope': 0,
                            'sf_ecc_intercept': .5,
                            'abs_mode_cardinals': 0,
                            'abs_mode_obliques': 0,
                            'rel_mode_cardinals': 0,
                            'rel_mode_obliques': 0,
                            'abs_amplitude_cardinals': 0,
                            'abs_amplitude_obliques': 0,
                            'rel_amplitude_cardinals': 0,
                            'rel_amplitude_obliques': 0}
        elif wildcards.param_set == 'scaling':
            model_params = {'sigma': 2,
                            'sf_ecc_slope': .15,
                            'sf_ecc_intercept': 0,
                            'abs_mode_cardinals': 0,
                            'abs_mode_obliques': 0,
                            'rel_mode_cardinals': 0,
                            'rel_mode_obliques': 0,
                            'abs_amplitude_cardinals': 0,
                            'abs_amplitude_obliques': 0,
                            'rel_amplitude_cardinals': 0,
                            'rel_amplitude_obliques': 0}
        model = sfp.model.LogGaussianDonut('full', 'full', 'full', **model_params)
        # this gets every other frequency for the radial and tangential stimuli
        for i, d in stim_df[:20:2].iterrows():
            _, _, mag, ori = sfp.stimuli.create_sf_maps_cpd(stim_shape, 24, w_r=d.w_r, w_a=d.w_a)
            # we create it here instead of loading in our stimuli since these
            # don't have the mask at the center or edge
            stim = sfp.stimuli.log_polar_grating(stim_shape, d.w_r, d.w_a)
            stims.append(stim)
            mags.append(mag)
            oris.append(ori)
            w_a.append(d.w_a)
            w_r.append(d.w_r)
            phase.append(d.phi)
            bolds.append(model.evaluate(mag, ori, eccen, angle).detach().numpy())
        # see
        # https://stackoverflow.com/questions/13906623/using-pickle-dump-typeerror-must-be-str-not-bytes
        # for why the b is necessary
        with open(output[0], 'wb') as f:
            pickle.dump({
                'eccentricity': eccen,
                'retinotopic_angle': angle,
                'stimuli': np.stack(stims),
                'stimuli_spatial_frequency': np.stack(mags),
                'stimuli_orientation': np.stack(oris),
                'predicted_bold': np.stack(bolds),
                'model_parameters': {param_remap[k].replace('$', '').replace('\\', ''): v
                                     for k, v in model_params.items()},
                'w_r': np.array(w_r),
                'w_a': np.array(w_a),
                'stimuli_phase': np.array(phase),
            }, f)


rule combine_params_csv:
    # bootstraps and combines the params csv for use in later analyses
    input:
        parameters = os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model',
                                  'stim_class', 'bayesian_posterior', "filter-mean", 'bootstrap',
                                  'individual_{task}_v1_e1-12_full_b10_r0.001_g0_full_full_absolute_all_models.csv'),
        precision = os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis',
                                 'stim_class', 'bayesian_posterior', '{task}_v1_e1-12_'
                                 'summary_mean_no-filter_precision.csv'),
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model', '{task}_final_bootstrapped_combined_parameters_s-{seed}.csv'),
    log:
        os.path.join(config['DATA_DIR'], 'code', 'tuning_2d_model', '{task}_final_bootstrapped_combined_parameters_s-{seed}.log'),
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'tuning_2d_model', '{task}_final_bootstrapped_combined_parameters_s-{seed}_benchmark.txt'),
    run:
        import sfp
        import pandas as pd
        df = sfp.figures.prep_df(pd.read_csv(input.parameters), wildcards.task)
        df = df.groupby(['subject', 'model_parameter', 'fit_model_type']).median().reset_index()
        precision = sfp.figures.prep_df(pd.read_csv(input.precision), wildcards.task)
        df = df.merge(precision, on=['subject'])
        df = sfp.figures.precision_weighted_bootstrap(df, int(wildcards.seed), 100, 'fit_value',
                                                      ['model_parameter', 'fit_model_type'],
                                                      'precision')
        df.to_csv(output[0], index=False)



rule report:
    input:
        benchmarks = lambda wildcards: glob(os.path.join(config['DATA_DIR'], 'code', wildcards.step, '*_benchmark.txt')),
        logs = lambda wildcards: glob(os.path.join(config['DATA_DIR'], 'code', wildcards.step, '*.log'))
    output:
        os.path.join(config['DATA_DIR'], 'code', "{step}", "{step}_report.html")
    log:
        os.path.join(config["DATA_DIR"], "code", "{step}", "report-%j.log")
    run:
        from snakemake.utils import report
        import pandas as pd
        step = wildcards.step
        benchmarks = []
        for f in input.benchmarks:
            tmp = pd.read_csv(f, sep='\t')
            tmp['file'] = os.path.split(f)[-1].replace('_benchmark.txt', '')
            benchmarks.append(tmp)
        try:
            benchmarks = pd.concat(benchmarks)
            benchmarks = benchmarks.set_index('file').sort_index().style.render()
        except ValueError:
            # then benchmarks was empty
            benchmarks = (u'\n        <style text="text/css">\n        </style>\n\n'
                          '        <h3>No benchmark files!</h3>\n\n\n')
        report("""
        Benchmark report for {step}
        =============================================

        The following benchmark reports were generated:

        .. raw:: html
           {benchmarks}

        """, output[0], **input)


rule all:
    input:
        rules.model_learning_hyperparams_full.input,
        rules.model_recovery_initial.input,
        rules.model_recovery_cv_initial.input,
        rules.model_all_subj_bootstrap.input,
        rules.model_all_subj_visual_field.input,
        rules.model_all_subj.input,
        rules.model_all_subj_cv.input,
        rules.all_check_plots.input,
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "stim_class",
                     "bayesian_posterior", "individual_ses-04_v1_e1-12_eccen_bin_tuning_curves_summary.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "stim_class",
                     "bayesian_posterior", "individual_ses-04_v1_e1-12_eccen_bin_tuning_curves_full.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "noise_ceiling", "monte_carlo",
                     "stim_class", "bayesian_posterior", 'filter-mean', "monte_carlo_ses-04_task-sfprescaled_v1_e1-12_individual.csv"),
        rules.groupaverage_all.input,


def get_figures_all(context='paper', visual_field_analyses=False):
    # now we're only going to use svg
    ext = 'svg'
    figs = []
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'individual_1d_{{}}_s-8_{{}}.{ext}').format(param, task)
             for param in ['bandwidth', 'pref-period', 'bandwidth-overall', 'pref-period-overall'] for task in ['task-sfprescaled', 'task-sfpconstant']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'sub-groupaverage_1d_{{}}_s-7_{{}}.{ext}').format(param, task)
             for param in ['bandwidth', 'pref-period'] for task in ['task-sfprescaled']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'individual_filter-mean_cv_{{}}_v_s-3_task-sfprescaled.{ext}').format(cv)
             for cv in ['raw', 'demeaned', 'model', 'model_point', 'demeaned-remeaned',
                        'model-remeaned', 'model_point-remeaned', 'raw-nc', 'model_point-nc',
                        'model_point-remeaned-nc']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'individual_filter-mean_cv_{{}}_h_s-3_task-sfprescaled.{ext}').format(cv)
             for cv in ['model_point', 'model_point-remeaned']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'individual_filter-mean_{{}}_params_visualfield-all_dist-overall_s-5_task-sfprescaled.{ext}').format(model)
             for model in ['full_full_full', 'full_full_absolute']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'individual_filter-mean_{{}}_params_visualfield-all_{{}}_s-None_task-sfprescaled.{ext}').format(model, kind)
             for kind  in ['point', 'strip', 'dist', 'compare', 'pair', 'pair-drop'] for model in ['full_full_full', 'full_full_absolute']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'sub-groupaverage_filter-mean_{{}}_params_visualfield-all_{{}}_s-7_task-sfprescaled.{ext}').format(model, kind)
             for kind  in ['dist', 'strip'] for model in ['full_full_full', 'full_full_absolute']]
    if visual_field_analyses:
        figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'individual_filter-mean_{{}}_params_visualfield-{{}}_{{}}_s-None_task-sfprescaled.{ext}').format(model, vf, kind)
                 for vf in ['all', 'inner', 'outer', 'left', 'right', 'upper', 'lower'] for kind  in ['point', 'strip'] for model in ['full_full_full', 'full_full_absolute']]
        figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'individual_filter-mean_full_full_full_params_visualfield-{{}}_compare_s-None_task-sfprescaled.{ext}').format(vf)
                 for vf in ['vertical', 'horizontal', 'eccen']]
        figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'individual_filter-mean_full_full_absolute_params_visualfield-{{}}_compare_s-None_task-sfprescaled.{ext}').format(vf)
                 for vf in ['vertical', 'horizontal', 'eccen']]
        figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'individual_filter-mean_{{}}_feature_visualfield-{{}}_pref-period_median_angles-{{}}_s-None_task-sfprescaled_{{}}.{ext}').format(model, vf, angles, frame)
                 for vf in ['inner', 'outer', 'left', 'right', 'upper', 'lower'] for angles in ['all', 'avg'] for frame in ['relative', 'absolute']
                 for model in ['full_full_full', 'full_full_absolute']],
        figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'individual_filter-mean_{{}}_feature_visualfield-{{}}_{{}}_median_angles-all_s-None_task-sfprescaled_{{}}.{ext}').format(model, vf, feature, frame)
                 for vf in ['inner', 'outer', 'left', 'right', 'upper', 'lower'] for feature in ['pref-period-contour', 'iso-pref-period', 'max-amp']
                 for frame in ['relative', 'absolute'] for model in ['full_full_full', 'full_full_absolute']],
        figs += [os.path.join(config['DATA_DIR'], "derivatives", 'figures', "individual_{}_sigma-interp_visualfield-{}_s-5_task-sfprescaled.txt").format(model, vf)
                 for vf in ['inner', 'outer', 'left', 'right', 'upper', 'lower'] for model in ['full_full_full', 'full_full_absolute']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'individual_filter-mean_{{}}_feature_visualfield-all_pref-period_{{}}_angles-{{}}_s-{{}}_task-sfprescaled_{{}}.{ext}').format(model, kind, angles, seed, frame)
             for seed, kind in zip([None, None, 5], ['median', 'bootstraps', 'bootstraps-overall'])
             for angles in ['all', 'avg'] for frame in ['relative', 'absolute']
             for model in ['full_full_full', 'full_full_absolute']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'sub-groupaverage_filter-mean_{{}}_feature_visualfield-all_pref-period_bootstraps_angles-{{}}_s-5_task-sfprescaled_{{}}.{ext}').format(model, angles, frame)
             for angles in ['all', 'avg'] for frame in ['relative', 'absolute'] for model in ['full_full_full', 'full_full_absolute']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'individual_filter-mean_{{}}_feature_visualfield-all_{{}}_{{}}_angles-all_s-{{}}_task-sfprescaled_{{}}.{ext}').format(model, feature, kind, seed, frame)
             for feature in ['pref-period-contour', 'iso-pref-period', 'max-amp']
             for seed, kind in zip([None, None, 5], ['median', 'bootstraps', 'bootstraps-overall'])
             for frame in ['relative', 'absolute']
             for model in ['full_full_full', 'full_full_absolute']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'sub-groupaverage_filter-mean_{{}}_feature_visualfield-all_{{}}_bootstraps_angles-all_s-5_task-sfprescaled_{{}}.{ext}').format(model, feature, frame)
             for feature in ['pref-period-contour', 'iso-pref-period', 'max-amp']
             for frame in ['relative', 'absolute'] for model in ['full_full_full', 'full_full_absolute']]
    figs +=[os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'schematic_{{}}.{ext}').format(kind)
            for kind in ['2d', '2d-inputs', 'models-annot-doubleup']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'background_period.{ext}')]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'{{}}_task-sfprescaled_background_period_{{}}_s-{{}}.{ext}').format(group, model, seed)
             for model in ['full_full_full', 'full_full_absolute'] for group, seed in zip(['sub-groupaverage', 'individual'], [7, 5])]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'individual_filter-mean_{{}}_training-loss-check_task-sfprescaled.{ext}').format(t)
             for t in ['initial_cv', 'bootstrap']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'sub-groupaverage_filter-mean_initial_training-loss-check_task-sfprescaled.{ext}')]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'mtf.{ext}')]
    figs += [os.path.join(config['DATA_DIR'], "derivatives", 'figures', "{}_{}_sigma-interp_visualfield-all_s-{}_task-sfprescaled.txt").format(group, model, seed)
             for model in ['full_full_full', 'full_full_absolute'] for group, seed in zip(['sub-groupaverage', 'individual'], [7, 5])]
    return figs


rule figures:
    input:
        lambda wildcards: get_figures_all(),


rule figures_poster:
    input:
        lambda wildcards: get_figures_all('poster'),


def figure_paper_input(wildcards):
    inputs = [
        # main figures
        os.path.join(config['DATA_DIR'], 'derivatives', 'compose_figures', 'paper', "background.svg"),
        os.path.join(config['DATA_DIR'], 'derivatives', 'compose_figures', 'paper', "stimulus_task-sfprescaled.svg"),
        os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'paper', 'mtf.svg'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'compose_figures', 'paper', 'schematic_model_2d.svg'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'compose_figures', 'paper', 'example_ecc_bins_with_stim.svg'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'compose_figures', 'paper', 'individual_1d_summary_s-8.svg'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'compose_figures', 'paper',
                     'crossvalidation_s-3_filter-mean_doubleup.svg'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'compose_figures', 'paper', "filter-mean_full_full_absolute_example_voxels.svg"),
        os.path.join(config['DATA_DIR'], 'derivatives', 'compose_figures', 'paper',
                     'individual_filter-mean_full_full_absolute_parameters_s-5.svg'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'compose_figures', 'paper',
                     'individual_filter-mean_full_full_absolute_2d_summary_s-5.svg'),
        os.path.join(config['DATA_DIR'], "derivatives", 'figures', 'paper',
                     "individual_filter-mean_task-sfprescaled_background_period_full_full_absolute_s-5.svg"),
        # appendix figures
        os.path.join(config['DATA_DIR'], "derivatives", 'figures', 'paper',
                     "individual_v1_area_vs_period_task-sfprescaled_filter-mean_full_full_absolute_bayesian_posterior.svg"),
        os.path.join(config['DATA_DIR'], 'derivatives', 'compose_figures', 'paper', 'visual-field-diff_filter-mean_iso_full_iso_s-5.svg'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'paper', "individual_1d_pref-period_s-None_task-sfprescaled.svg"),
        os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'paper',
                     'individual_filter-mean_full_full_absolute_feature_visualfield-all_pref-period_bootstraps_angles-avg_s-None_task-sfprescaled_relative.svg'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'paper',
                     'individual_filter-mean_full_full_absolute_feature_visualfield-all_pref-period_bootstraps_angles-avg_s-None_task-sfprescaled_absolute.svg'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'paper',
                     'individual_filter-mean_full_full_absolute_feature_visualfield-all_pref-period-contour_bootstraps_angles-all_s-None_task-sfprescaled_relative.svg'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'paper',
                     'individual_filter-mean_full_full_absolute_feature_visualfield-all_pref-period-contour_bootstraps_angles-all_s-None_task-sfprescaled_absolute.svg'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'paper',
                     'individual_filter-mean_full_full_absolute_feature_visualfield-all_max-amp_bootstraps_angles-all_s-None_task-sfprescaled_relative.svg'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'paper',
                     'individual_filter-mean_full_full_absolute_feature_visualfield-all_max-amp_bootstraps_angles-all_s-None_task-sfprescaled_absolute.svg'),
        # txt files used to get numbers in the paper
        os.path.join(config['DATA_DIR'], "derivatives", 'figures',
                     "individual_filter-mean_full_full_absolute_sigma-interp_visualfield-all_s-5_task-sfprescaled.txt"),
        os.path.join(config['DATA_DIR'], "derivatives", 'figures', 'paper',
                     "individual_v1_area_vs_period_linreg_task-sfprescaled_filter-mean_full_full_absolute_bayesian_posterior_svg.txt"),
    ]
    outputs = (['fig-{:02d}.svg'.format(i) for i in range(1, 12)] + ['fig-S{:02d}.svg'.format(i) for i in range(1, 10)] +
               ['sigma_interpretation.txt', 'v1_size_interpretation.txt'])
    mapping = dict(zip(outputs, inputs))
    return mapping[wildcards.fig_name]


rule figure_paper:
    input:
        figure_paper_input,
    output:
        os.path.join('reports', 'paper_figures', '{fig_name}'),
    run:
        import shutil
        shutil.copy(input[0], output[0])


def output_csv_input(wildcards):
    inputs = [
        # csv that gives the parameters (combined across participants) used in
        # figures throughout the paper, this isn't copied out anywhere, but is
        # a good output to have
        os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model', 'task-sfprescaled_final_bootstrapped_combined_parameters_s-5.csv'),
        # similarly, this csv givers the parameters for each individual subjects'
        # model fits
        os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model', 'stim_class', 'bayesian_posterior', 'filter-mean', 'bootstrap',
                     'individual_task-sfprescaled_v1_e1-12_full_b10_r0.001_g0_full_full_absolute_all_models.csv')
    ]
    mapping = dict(zip(['combined', 'individual'], inputs))
    return mapping[wildcards.csv_type]


# this rule doesn't get considered part of the "main_figure_paper" outputs,
# because we include these two csvs in the github repo itself (they're small).
# this is here in case I want to easily generate them again
rule output_csv:
    input:
        output_csv_input,
    output:
        os.path.join('data', 'tuning_2d_model', '{csv_type}_subject_params.csv')
    run:
        import shutil
        shutil.copy(input[0], output[0])


rule main_figure_paper:
    input:
        [os.path.join('reports', 'paper_figures', 'fig-{:02d}.svg').format(i) for i in range(1, 12)],
        os.path.join('reports', 'paper_figures', 'sigma_interpretation.txt'),


rule supplement_figure_paper:
    input:
        [os.path.join('reports', 'paper_figures', 'fig-S{:02d}.svg').format(i) for i in range(1, 10)],
        os.path.join('reports', 'paper_figures', 'v1_size_interpretation.txt'),
