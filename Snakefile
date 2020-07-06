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
    shell.prefix(". /share/apps/anaconda3/5.3.1/etc/profile.d/conda.sh; conda activate sfp; "
                 "module load fsl/5.0.10; module load freesurfer/6.0.0; module load matlab/2017a; "
                 "export SUBJECTS_DIR=%s/derivatives/freesurfer; " % config["DATA_DIR"])
else:
    ON_CLUSTER = False
    shell.prefix("export SUBJECTS_DIR=%s/derivatives/freesurfer; " % config["DATA_DIR"])


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
    subject="sub-[a-z0-9]+|sub-groupaverage_i-[a-z]+",
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
    y_val="period|frequency"

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
                           mat_type='stim_class', atlas_type='bayesian_posterior', modeling_goal='initial'):
    output_path = os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}",
                               "{atlas_type}", "{modeling_goal}", "{subject}", "{session}",
                               "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch}_"
                               "r{lr}_g{gpus}_c{{crossval}}_n{bootstrap_num}_{model_type}_loss.csv")
    output_path = output_path.format(subject=subject, session=session, task=task, batch=batch_size,
                                     lr=learning_rate, model_type=model_type, vareas=vareas,
                                     eccen=eccen, df_mode=df_mode, gpus=gpus, atlas_type=atlas_type,
                                     mat_type=mat_type, modeling_goal=modeling_goal,
                                     bootstrap_num=bootstrap_num)
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
                     "bayesian_posterior", "bootstrap",
                     "task-sfprescaled_v1_e1-12_full_b10_r0.001_g0_full_full_full_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class",
                     "bayesian_posterior", "bootstrap",
                     "task-sfprescaled_v1_e1-12_full_b10_r0.001_g0_full_full_absolute_all_models.csv"),
    


rule model_all_subj_visual_field:
    input:
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class",
                      "bayesian_posterior", "visual_field_%s" % p,
                      "task-sfprescaled_v1_e1-12_summary_b10_r0.001_g0_full_full_full_all_models.csv") for p in
         ['upper', 'lower', 'left', 'right', 'inner', 'outer']],


rule model_all_subj:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class",
                     "bayesian_posterior", "initial",
                     "task-sfprescaled_v1_e1-12_summary_b10_r0.001_g0_full_full_full_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class",
                     "bayesian_posterior", "initial",
                     "task-sfprescaled_v1_e1-12_summary_b10_r0.001_g0_full_full_absolute_all_models.csv"),


rule model_all_subj_cv:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class",
                     "bayesian_posterior", "initial_cv",
                     "task-sfprescaled_v1_e1-12_summary_b10_r0.001_g0_s0_all_models.csv"),


def get_groupaverage_all(tuning_type='2d', interp='linear', session='ses-04', task='task-sfprescaled',
                         model_type='full_full_absolute', vareas='1', eccen='1-12', batch_size=10,
                         learning_rate=0.001, gpus=0, df_mode='summary', mat_type='stim_class',
                         atlas_type='bayesian_posterior', modeling_goal='initial'):
    if modeling_goal != 'initial':
        return []
    if tuning_type == '2d':
        path = os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", mat_type,
                            atlas_type, "initial", f"sub-groupaverage_i-{interp}",
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
        lambda wildcards: get_groupaverage_all(model_type='full_full_full'),
        lambda wildcards: get_groupaverage_all(model_type='full_full_absolute'),
        lambda wildcards: get_groupaverage_all('curve'),


rule all_check_plots:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', "{subject}", "{atlas_type}", 'varea_plot.png').format(subject=s, atlas_type=a)
         for s in SUBJECTS for a in ['bayesian_posterior', 'atlas']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', "{subject}", "{atlas_type}", '{prf_prop}_plot.png').format(subject=s, atlas_type=a, prf_prop=p)
         for s in SUBJECTS for a in ['bayesian_posterior', 'atlas', 'data'] for p in ['angle', 'eccen']],
        [os.path.join(config["DATA_DIR"], "derivatives", "GLMdenoise", "stim_class", "bayesian_posterior", "{subject}", "{session}", "figures_{task}", "FinalModel_maps.png").format(
            subject=sub, session=ses, task=TASKS[(sub, ses)]) for sub in SUBJECTS for ses in SESSIONS[sub]],
        [os.path.join(config["DATA_DIR"], "derivatives", "first_level_binned", "stim_class", "bayesian_posterior", "{subject}", "{session}", "{subject}_{session}_{task}_v1_e1-12_eccen_bin_full_data.svg").format(
            subject=sub, session=ses, task=TASKS[(sub, ses)]) for sub in SUBJECTS for ses in SESSIONS[sub]],


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
        "data/stimuli/task-sfpconstant_stim_description.csv"
    shell:
        "python -m sfp.stimuli -c"


rule rescaled_stimuli:
    input:
        "data/stimuli/mtf_func.pkl"
    output:
        "data/stimuli/task-sfprescaled_stimuli.npy",
        "data/stimuli/task-sfprescaled_stim_description.csv",
        "data/stimuli/task-sfpconstantrescaled_stimuli.npy",
        "data/stimuli/task-sfpconstantrescaled_stim_description.csv"
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


# current way of generating stimuli, uses both subject and session name
rule stimuli_idx:
    output:
        ["data/stimuli/{subject}_{session}_run%02d_idx.npy" % i for i in range(12)]
    params:
        seed = lambda wildcards: SUB_SEEDS[wildcards.subject] + SES_SEEDS[wildcards.session]
    shell:
        "python -m sfp.stimuli --subject_name {wildcards.subject}_{wildcards.session}"
        " -i -s {params.seed}"


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


def find_prf_mgz(wildcards, prf_prop='varea'):
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
    benson_template = os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', wildcards.subject, wildcards.atlas_type, '{hemi}.'+benson_prefix+prf_prop+'.mgz')
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
            subjects = SUBJECTS
            sessions = SESSIONS
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
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "{mat_type}", "{atlas_type}", "{groupaverage}_v{vareas}_e{eccen}_{binning}_tuning_curves_{df_mode}.csv")
    params:
        input_dir = os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "{mat_type}", "{atlas_type}"),
        groupaverage = lambda wildcards: {'sub-groupaverage': '-g', 'individual': ''}[wildcards.groupaverage]
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_curves_summary", "{mat_type}_{atlas_type}_{groupaverage}_v{vareas}_e{eccen}_{binning}_{df_mode}_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_curves_summary", "{mat_type}_{atlas_type}_{groupaverage}_v{vareas}_e{eccen}_{binning}_{df_mode}-%j.log")
    shell:
        "python sfp/summarize_tuning_curves.py {params.input_dir} {output} {wildcards.df_mode} {params.groupaverage}"


rule tuning_curves_summary_plot:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "{mat_type}", "{atlas_type}", "individual_v{vareas}_e{eccen}_{binning}_tuning_curves_summary.csv")
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}',
                     "v{vareas}_e{eccen}_{binning}_tuning_curves_summary_plot_{subjects}_{sessions}_"
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
        sessions = lambda wildcards: wildcards.sessions.split(','),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_curves_summary_plots", "{mat_type}_"
                     "{atlas_type}_v{vareas}_e{eccen}_{binning}_{subjects}_{sessions}_{tasks}_v"
                     "{plot_varea}_e{eccen_range}_row={row}_col={col}_hue={hue}_{plot_func}_{y}_"
                     "benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_curves_summary_plots", "{mat_type}_"
                     "{atlas_type}_v{vareas}_e{eccen}_{binning}_{subjects}_{sessions}_{tasks}_v"
                     "{plot_varea}_e{eccen_range}_row={row}_col={col}_hue={hue}_{plot_func}_{y}-%j.log")
    shell:
        "python -m sfp.summary_plots {input} --col {params.col} --row {params.row} --hue"
        " {params.hue} --y {params.y} --varea {params.plot_varea} --eccen_range {params.eccen_range}"
        " --subject {params.subjects} --task {params.tasks} --session {params.sessions}"


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


rule model:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}.csv')
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_n{bootstrap_num}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_n{bootstrap_num}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_model_history.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_n{bootstrap_num}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_model.pt"),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_n{bootstrap_num}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_n{bootstrap_num}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}-%j.log")
    resources:
        # need the same number of cpus and gpus
        cpus_per_task = lambda wildcards: max(int(wildcards.gpus), 1),
        mem = lambda wildcards: {'full': 40, 'summary': 1}[wildcards.df_mode],
        gpus = lambda wildcards: int(wildcards.gpus)
    params:
        save_stem = lambda wildcards, output: output[0].replace("_loss.csv", ''),
        stimulus_class = lambda wildcards: wildcards.stimulus_class.split(','),
        bootstrap_num = lambda wildcards: wildcards.bootstrap_num.split(','),
        logging = to_log_or_not,
        vis_field = visual_field_part,
    shell:
        "python -m sfp.model {wildcards.period_orientation_type} {wildcards.eccentricity_type} "
        "{wildcards.amplitude_orientation_type} {input} {params.save_stem} -b "
        "{wildcards.batch_size} -r {wildcards.learning_rate} -d "
        "drop_voxels_with_negative_amplitudes,drop_voxels_near_border{params.vis_field} -t 1e-6 -e"
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
                     "{atlas_type}", "{modeling_goal}", "{subject}", "{session}",
                     "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_"
                     "r{learning_rate}_g{gpus}_s{crossval_seed}_{model_type}_cv_loss.csv")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model_cv_loss", "{subject}_{session}_"
                     "{task}_{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_{model_type}-%j.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model_cv_loss", "{subject}_{session}_"
                     "{task}_{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_{model_type}_"
                     "benchmark.txt")
    resources:
        mem = 10
    run:
        import sfp
        sfp.analyze_model.calc_cv_error(input.loss_files, input.dataset_path, wildcards, output)


rule summarize_model_cv:
    input:
        # this will return a list of lists of strings, so we need to flatten it
        loss_files = lambda wildcards: np.array([get_model_subj_outputs(m, **wildcards) for m in MODEL_TYPES]).flatten(),
        cv_loss = lambda wildcards: [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}",
                                                  "{atlas_type}", "{modeling_goal}", "{subject}", "{session}",
                                                  "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_"
                                                  "r{learning_rate}_g{gpus}_s{crossval_seed}_{model_type}_cv_loss.csv").format(model_type=m, **wildcards)
                                     for m in MODEL_TYPES]

    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_"
                     "s{crossval_seed}_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_"
                     "s{crossval_seed}_all_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_"
                     "s{crossval_seed}_all_timing.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_"
                     "s{crossval_seed}_all_diff.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_"
                     "s{crossval_seed}_all_model_history.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_"
                     "s{crossval_seed}_all_cv_loss.csv"),
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model_summarize", "{subject}_{session}_"
                     "{task}_{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}-%j.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model_summarize", "{subject}_{session}_"
                     "{task}_{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_"
                     "b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_benchmark.txt")
    params:
        base_path = lambda wildcards, input: os.path.join(os.path.dirname(input.loss_files[0]),
                                                          '*c*.pt'),
        metadata = ["mat_type", 'atlas_type', 'modeling_goal', 'subject', 'session', 'task',
                    'fit_model_type', 'test_subset'],
    run:
        import sfp
        sfp.analyze_model.gather_results(params.base_path, output, params.metadata, input.cv_loss)


def get_cv_summary(crossval_seed=0, batch_size=10, learning_rate=1e-3, vareas=1, eccen='1-12',
                   df_mode='summary', gpus=0, mat_type='stim_class', atlas_type='bayesian_posterior',
                   modeling_goal='initial_cv', task='task-sfprescaled'):
        output_path = os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}",
                                   "{atlas_type}", "{modeling_goal}", "{{subject}}", "{{session}}",
                                   "{{subject}}_{{session}}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch"
                                   "_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all_cv_loss.csv")
        output_path = output_path.format(vareas=vareas, mat_type=mat_type, batch_size=batch_size,
                                         eccen=eccen, atlas_type=atlas_type, df_mode=df_mode,
                                         modeling_goal=modeling_goal, gpus=gpus, task=task,
                                         crossval_seed=crossval_seed, learning_rate=learning_rate)
        return [output_path.format(subject=sub, session=ses)
                for sub in SUBJECTS for ses in SESSIONS[sub] if TASKS[(sub, ses)] == task]


rule combine_model_cv_summaries:
    input:
        lambda wildcards: get_cv_summary(**wildcards)
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_s{crossval_seed}_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_s{crossval_seed}_all_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_s{crossval_seed}_all_timing.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_s{crossval_seed}_all_cv_loss.csv"),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_{modeling_goal}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_{modeling_goal}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all-%j.log")
    params:
        base_template = lambda wildcards, input: [i.replace('_all_cv_loss.csv', '') for i in input]
    run:
        import sfp
        sfp.analyze_model.combine_summarized_results(params.base_template, output)


def gather_model_results_input(wildcards):
    inputs = {}
    if wildcards.modeling_goal == 'bootstrap':
        loss_files = [get_model_subj_outputs(bootstrap_num=n, **wildcards) for n in range(100)]
    else:
        loss_files = [get_model_subj_outputs(subject=subj, session=ses, **wildcards)
                      for subj in SUBJECTS for ses in SESSIONS[subj]
                      if TASKS[(subj, ses)] == wildcards.task]
        loss_files += get_groupaverage_all(**wildcards)
    # this will return a list of lists of strings, so we need to flatten it
    inputs['loss_files'] = np.array(loss_files).flatten()
    return inputs


rule gather_model_results_preliminary:
    input:
        unpack(gather_model_results_input)
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}",  "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_timing.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}",  "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_diff.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}",  "{subject}", "{session}", "{subject}_{session}_{task}_"
                     "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_model_history.csv")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all-%j.log")
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
                                       "bootstrap", "{subject}", "{session}", "{subject}_{session}_{task}_"
                                       "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_"
                                        "{model_type}_all_models.csv").format(subject=subj, session=ses, **wildcards)
                           for subj in SUBJECTS for ses in SESSIONS[subj]
                           if TASKS[(subj, ses)] == wildcards.task]
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "bootstrap", "{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "bootstrap", "{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "bootstrap", "{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_timing.csv"),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_bootstrap_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_bootstrap_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all-%j.log")
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
                     "{modeling_goal}", "{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_timing.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_diff.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_model_history.csv"),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_{modeling_goal}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_{modeling_goal}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all-%j.log")
    resources:
        mem = 100,
    params:
        base_path = lambda wildcards, output: os.path.join(os.path.dirname(output[0]), '*', '*',
                                                           '*'+wildcards.model_type+'*'),
        metadata = ["mat_type", 'atlas_type', 'modeling_goal', 'subject', 'session', 'task',
                    'fit_model_type', 'indicator', 'bootstrap_num', 'test_subset']
    run:
        import sfp
        sfp.analyze_model.gather_results(params.base_path, output, params.metadata)


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
        os.path.join(config['DATA_DIR'], 'derivatives', 'simulated_data', 'noise-{mat_type}_{atlas_type}_{subject}_{session}_{task}_v{vareas}_e{eccen}', 'n{num_voxels}_{period_orientation_type}_{eccentricity_type}_{amplitude_orientation_type}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_simulated.csv')
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
        "--noise_source_path {input}"


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
                                                          "*c*.pt"),
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


rule noise_ceiling_monte_carlo:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_full.csv'),
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'noise_ceiling', 'monte_carlo', '{mat_type}', '{atlas_type}', '{subject}', '{session}', 's{seed}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_full_loss.csv'),
        os.path.join(config['DATA_DIR'], 'derivatives', 'noise_ceiling', 'monte_carlo', '{mat_type}', '{atlas_type}', '{subject}', '{session}', 's{seed}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_full_predictions.png')
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "noise_ceiling", 'monte_carlo', "loss_s{seed}_{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}_full_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "noise_ceiling", 'monte_carlo', "loss_s{seed}_{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}_full-%j.log")
    run:
        import sfp
        import pandas as pd
        save_stem = output[0].replace('_loss.csv', '')
        df = pd.read_csv(input[0])
        df = sfp.noise_ceiling.sample_df(df, int(wildcards.seed))
        sfp.noise_ceiling.monte_carlo(df, save_stem, df_mode='full', **wildcards)


rule noise_ceiling_monte_carlo_overall:
    input:
        # for now, we're only going to want to look at ses-04,
        # task-sfprescaled. this will work for other session/task pairs,
        # but we'd have to merge them ourselves afterwards.
        lambda wildcards: [os.path.join(config['DATA_DIR'], 'derivatives', 'noise_ceiling', 'monte_carlo',
                                        '{{mat_type}}', '{{atlas_type}}', '{subject}', '{{session}}', 's{seed}',
                                        '{subject}_{{session}}_{{task}}_v{{vareas}}_e{{eccen}}_full_loss.csv').format(
                                            subject=sub, seed=seed)
                           for seed in range(100) for sub in SUBJECTS if wildcards.session in SESSIONS[sub]
                           if TASKS[(sub, wildcards.session)] == wildcards.task]
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'noise_ceiling', 'monte_carlo', '{mat_type}', '{atlas_type}', 'monte_carlo_{session}_{task}_v{vareas}_e{eccen}.csv')
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "noise_ceiling", 'monte_carlo', "loss_{mat_type}_{atlas_type}_{session}_{task}_v{vareas}_e{eccen}_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "noise_ceiling", 'monte_carlo', "loss_{mat_type}_{atlas_type}_{session}_{task}_v{vareas}_e{eccen}-%j.log")
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
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "stim_class",
                     "bayesian_posterior", "v1_e1-12_eccen_bin_tuning_curves_full.csv")
    output:
        os.path.join(config['DATA_DIR'], "derivatives", 'figures', '{context}', "1d_{tuning_param}_{task}.{ext}")
    log:
        os.path.join(config['DATA_DIR'], 'code', 'figures', '{context}', "1d_{tuning_param}_{task}_{ext}.log")
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'figures', '{context}',
                     "1d_{tuning_param}_{task}_{ext}_benchmark.txt")
    run:
        import pandas as pd
        import seaborn as sns
        import sfp
        font_scale = {'poster': 1.5}.get(wildcards.context, 1)
        height_scale = {'poster': 2}.get(wildcards.context, 1)
        df = sfp.figures.prep_df(pd.read_csv(input[0]), wildcards.task)
        ref_frame = {'task-sfpconstant': 'absolute', 'task-sfprescaled': 'relative'}
        if wildcards.tuning_param.endswith('overall'):
            col = {'pref-period-overall': 'preferred_period',
                   'bandwidth-overall': 'tuning_curve_bandwidth'}[wildcards.tuning_param]
            df = sfp.figures.append_precision_col(df, col)
            df = sfp.figures.precision_weighted_bootstrap(df, col=col)
        sns.set_context(wildcards.context, font_scale=font_scale)
        kwargs = {'size_plot': [None, 5*height_scale], 'linewidth': 2*(height_scale+1)}
        with sns.axes_style('white'):
            if wildcards.tuning_param == 'pref-period':
                g = sfp.figures.pref_period_1d(df, ref_frame[wildcards.task], row=None,
                                               height=4*height_scale, **kwargs)
            elif wildcards.tuning_param == 'bandwidth':
                g = sfp.figures.bandwidth_1d(df, ref_frame[wildcards.task], row=None,
                                             height=4*height_scale, **kwargs)
            elif wildcards.tuning_param == 'pref-period-overall':
                g = sfp.figures.pref_period_1d(df, ref_frame[wildcards.task], row=None,
                                               height=5*height_scale, ylim=(0, 2), **kwargs)
            elif wildcards.tuning_param == 'bandwidth-overall':
                g = sfp.figures.bandwidth_1d(df, ref_frame[wildcards.task], row=None,
                                             height=5*height_scale, **kwargs)
            g.fig.savefig(output[0], bbox_inches='tight')


def get_loss_files(wildcards):
    # this will return a list of lists of strings, so we need to flatten it
    if wildcards.modeling_goal == 'initial_cv':
        return np.array([get_model_subj_outputs(m, sub, ses, crossval_seed=0, **wildcards)
                         for m in MODEL_TYPES for sub in SUBJECTS for ses in SESSIONS[sub]
                         if TASKS[(sub, ses)] == wildcards.task]).flatten()
    elif wildcards.modeling_goal == 'bootstrap':
        return np.array([get_model_subj_outputs(m, sub, ses, bootstrap_num=n, **wildcards)
                         for n in range(100) for m in ['full_full_full', 'full_full_absolute']
                         for sub in SUBJECTS for ses in SESSIONS[sub]
                         if TASKS[(sub, ses)] == wildcards.task]).flatten()


rule combine_final_loss:
    input:
        get_loss_files,
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model', '{mat_type}', '{atlas_type}', '{modeling_goal}', '{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_final_epoch_loss.csv')
    log:
        os.path.join(config['DATA_DIR'], 'code', 'tuning_2d_model', '{mat_type}_{atlas_type}_{modeling_goal}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_final_epoch_loss.log')
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'tuning_2d_model', '{mat_type}_{atlas_type}_{modeling_goal}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_final_epoch_loss_benchmark.txt')
    run:
        import sfp
        df = sfp.analyze_model.collect_final_loss(input)
        df.to_csv(output[0])


rule figure_loss_check:
    input:
        lambda wildcards: os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model', 'stim_class', 'bayesian_posterior', '{modeling_goal}',
                                       '{task}_v1_e1-12_{df_mode}_b10_r0.001_g0_final_epoch_loss.csv').format(df_mode={'initial_cv': 'summary', 'bootstrap': 'full'}[wildcards.modeling_goal],
                                                                                                              modeling_goal=wildcards.modeling_goal, task=wildcards.task)
    output:
        os.path.join(config['DATA_DIR'], "derivatives", 'figures', '{context}', "{modeling_goal}_training-loss-check_{task}.{ext}")
    log:
        os.path.join(config['DATA_DIR'], "code", 'figures', '{context}', "{modeling_goal}_training-loss-check_{task}_{ext}.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", 'figures', '{context}', "{modeling_goal}_training-loss-check_{task}_{ext}_benchmark.txt")
    run:
        import sfp
        import seaborn as sns
        import pandas as pd
        font_scale = {'poster': 1.2}.get(wildcards.context, 1)
        df = pd.read_csv(input[0])
        if wildcards.modeling_goal == 'initial_cv':
            hue = 'test_subset'
        elif wildcards.modeling_goal == 'bootstrap':
            hue = 'bootstrap_num'
        sns.set_context(wildcards.context, font_scale=font_scale)
        with sns.axes_style('white', {'axes.spines.right': False, 'axes.spines.top': False}):
            g = sfp.figures.training_loss_check(df, hue)
            g.fig.savefig(output[0], bbox_inches='tight')


def get_noise_ceiling_df(wildcards):
    template = os.path.join(config['DATA_DIR'], 'derivatives', 'noise_ceiling', 'monte_carlo',
                            'stim_class', 'bayesian_posterior', 'monte_carlo_ses-04_{task}_v1_e1-12.csv')
    if wildcards.cv_type.endswith('-nc'):
        return template.format(task=wildcards.task)
    else:
        return []


rule figure_crossvalidation:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model', 'stim_class',
                     'bayesian_posterior', 'initial_cv',
                     '{task}_v1_e1-12_summary_b10_r0.001_g0_s0_all_cv_loss.csv'),
        get_noise_ceiling_df,
    output:
        os.path.join(config['DATA_DIR'], "derivatives", 'figures', '{context}', "cv_{cv_type}_{orient}_{task}.{ext}")
    log:
        os.path.join(config['DATA_DIR'], "code", 'figures', '{context}', "cv_{cv_type}_{orient}_{task}_{ext}.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", 'figures', '{context}', "cv_{cv_type}_{orient}_{task}_{ext}_benchmark.txt")
    run:
        import pandas as pd
        import seaborn as sns
        import sfp
        font_scale = {'poster': 1.2}.get(wildcards.context, 1)
        df = sfp.figures.prep_df(pd.read_csv(input[0]), wildcards.task)
        if wildcards.cv_type.endswith('-nc'):
            noise_ceiling = sfp.figures.prep_df(pd.read_csv(input[1]), wildcards.task)
        else:
            noise_ceiling = None
        sns.set_context(wildcards.context, font_scale=font_scale)
        with sns.axes_style('white'):
            if 'remeaned' in wildcards.cv_type:
                remeaned = True
            else:
                remeaned = False
            if wildcards.cv_type.startswith('demeaned'):
                g = sfp.figures.cross_validation_demeaned(df, remeaned, context=wildcards.context,
                                                          orient=wildcards.orient)
            elif wildcards.cv_type.startswith('raw'):
                g = sfp.figures.cross_validation_raw(df, noise_ceiling, context=wildcards.context,
                                                     orient=wildcards.orient)
            elif wildcards.cv_type.startswith('model_point'):
                g = sfp.figures.cross_validation_model(df, 'point', remeaned, noise_ceiling,
                                                       context=wildcards.context,
                                                       orient=wildcards.orient)
            elif wildcards.cv_type.startswith('model'):
                g = sfp.figures.cross_validation_model(df, remeaned=remeaned, orient=wildcards.orient,
                                                       context=wildcards.context)
            g.fig.savefig(output[0], bbox_inches='tight')


def get_params_csv(wildcards):
    path_template = os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model',
                                 'stim_class', 'bayesian_posterior', '%s',
                                 f'{wildcards.task}_v1_e1-12_%s_b10_r0.001_g0_{wildcards.model_type}_all_models.csv')
    paths = []
    try:
        if wildcards.plot_kind in ['dist', 'pair', 'pair-drop', 'compare', 'bootstraps',
                                   'dist-overall', 'bootstraps-overall']:
            paths.append(path_template % ('bootstrap', 'full'))
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
                paths.append(path_template % (folder, 'summary'))
    except AttributeError:
        # this is the figure_background_with_current rule
        paths = path_template % ('bootstrap', 'full')
    return paths


rule figure_params:
    input:
        get_params_csv,
    output:
        os.path.join(config['DATA_DIR'], "derivatives", 'figures', '{context}', "{model_type}_params_visualfield-{vf}_{plot_kind}_{task}.{ext}")
    log:
        os.path.join(config['DATA_DIR'], "code", 'figures', '{context}', "{model_type}_params_visualfield-{vf}_{plot_kind}_{task}_{ext}.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", 'figures', '{context}', "{model_type}_params_visualfield-{vf}_{plot_kind}_{task}_{ext}_benchmark.txt")
    run:
        import pandas as pd
        import seaborn as sns
        import sfp
        import matplotlib as mpl
        font_scale = {'poster': 1.2}.get(wildcards.context, 1)
        df = []
        for p in input:
            tmp = sfp.figures.prep_df(pd.read_csv(p), wildcards.task)
            if wildcards.plot_kind.endswith('overall'):
                tmp = sfp.figures.append_precision_col(tmp, 'fit_value',
                                                       ['subject', 'model_parameter',
                                                        'fit_model_type'])
                tmp = sfp.figures.precision_weighted_bootstrap(tmp, 100, 'fit_value',
                                                               ['model_parameter',
                                                                'fit_model_type'])
            df.append(sfp.figures.prep_model_df(tmp))
        sns.set_context(wildcards.context, font_scale=font_scale)
        with sns.axes_style('white', {'axes.spines.right': False, 'axes.spines.top': False}):
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
                # don't add a legend if the plot_kind is point or dist-overall
                add_legend = {'point': False, 'dist-overall': False}.get(wildcards.plot_kind, True)
                plot_kind = wildcards.plot_kind.replace('-overall', '')
                fig = sfp.figures.model_parameters(df[0], plot_kind, wildcards.vf,
                                                   add_legend=add_legend)
            fig.savefig(output[0], bbox_inches='tight')


rule figure_feature_df:
    input:
        get_params_csv,
    output:
        os.path.join(config['DATA_DIR'], "derivatives", 'figures', '{context}', "{model_type}_feature_visualfield-{vf}_{feature_type}_{plot_kind}_angles-{angles}_{task}_{ref_frame}.{ext}")
    log:
        os.path.join(config['DATA_DIR'], "code", 'figures', '{context}', "{model_type}_feature_visualfield-{vf}_{feature_type}_{plot_kind}_angles-{angles}_{task}_{ref_frame}_{ext}.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", 'figures', '{context}', "{model_type}_feature_visualfield-{vf}_{feature_type}_{plot_kind}_angles-{angles}_{task}_{ref_frame}_{ext}_benchmark.txt")
    run:
        import pandas as pd
        import seaborn as sns
        import sfp
        font_scale = {'poster': 1.2}.get(wildcards.context, 1)
        df = sfp.figures.prep_df(pd.read_csv(input[0]), wildcards.task)
        if wildcards.plot_kind.endswith('overall'):
            df = sfp.figures.append_precision_col(df, 'fit_value', ['subject', 'model_parameter',
                                                                    'fit_model_type'])
            df = sfp.figures.precision_weighted_bootstrap(df, 100, 'fit_value',
                                                          ['model_parameter', 'fit_model_type'])
        sns.set_context(wildcards.context, font_scale=font_scale)
        with sns.axes_style('white', {'axes.spines.right': False, 'axes.spines.top': False}):
            if wildcards.angles == 'avg':
                angles = True
            elif wildcards.angles == 'all':
                angles = False
            g = sfp.figures.feature_df_plot(df, angles, wildcards.ref_frame, wildcards.feature_type,
                                            wildcards.vf, wildcards.context)
            g.fig.savefig(output[0], bbox_inches='tight')


rule figure_schematic:
    output:
        os.path.join(config["DATA_DIR"], 'derivatives', 'figures', '{context}', 'schematic_{schematic_type}.{ext}')
    log:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'schematic_{schematic_type}_{ext}.log')
    benchmark:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'schematic_{schematic_type}_{ext}_benchmark.txt')
    run:
        import sfp
        import seaborn as sns
        font_scale = {'poster': 1.2}.get(wildcards.context, 1)
        sns.set_context(wildcards.context, font_scale=font_scale)
        with sns.axes_style('white', {'axes.spines.right': False, 'axes.spines.top': False}):
            if wildcards.schematic_type == '2d':
                fig = sfp.figures.model_schematic(wildcards.context)
            elif wildcards.schematic_type == '2d-inputs':
                fig = sfp.figures.input_schematic()
            elif wildcards.schematic_type.startswith('models'):
                if 'annot' in wildcards.schematic_type:
                    annotate = True
                else:
                    annotate = False
                fig = sfp.figures.model_types(wildcards.context, annotate=annotate)
            fig.savefig(output[0], bbox_inches='tight')


rule figure_background:
    output:
        os.path.join(config["DATA_DIR"], 'derivatives', 'figures', '{context}', 'background_{y_val}.{ext}')
    log:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'background_{y_val}_{ext}.log')
    benchmark:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', 'background_{y_val}_{ext}_benchmark.txt')
    run:
        import sfp
        import seaborn as sns
        font_scale = {'poster': 1.2}.get(wildcards.context, 1)
        sns.set_context(wildcards.context, font_scale=font_scale)
        with (sns.axes_style('white', {'axes.spines.right': False, 'axes.spines.top': False})):
            df = sfp.figures.existing_studies_df()
            y = {'period': 'Preferred period (dpc)',
                 'frequency': 'Preferred spatial frequency (cpd)'}[wildcards.y_val]
            g = sfp.figures.existing_studies_figure(df, y, wildcards.context)
            g.fig.savefig(output[0], bbox_inches='tight')


rule figure_background_with_current:
    input:
        get_params_csv,
    output:
        os.path.join(config["DATA_DIR"], 'derivatives', 'figures', '{context}', '{task}_background_{y_val}_{model_type}.{ext}')
    log:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', '{task}_background_{y_val}_{model_type}_{ext}.log')
    benchmark:
        os.path.join(config["DATA_DIR"], 'code', 'figures', '{context}', '{task}_background_{y_val}_{model_type}_{ext}_benchmark.txt')
    run:
        import sfp
        import seaborn as sns
        import pandas as pd
        font_scale = {'poster': 1.2}.get(wildcards.context, 1)
        sns.set_context(wildcards.context, font_scale=font_scale)
        df = sfp.figures.prep_df(pd.read_csv(input[0]), wildcards.task)
        with (sns.axes_style('white', {'axes.spines.right': False, 'axes.spines.top': False})):
            y = {'period': 'Preferred period (dpc)',
                 'frequency': 'Preferred spatial frequency (cpd)'}[wildcards.y_val]
            g = sfp.figures.existing_studies_with_current_figure(df, y, wildcards.context)
            g.fig.savefig(output[0], bbox_inches='tight')


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
                     "bayesian_posterior", "individual_v1_e1-12_eccen_bin_tuning_curves_summary.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "stim_class",
                     "bayesian_posterior", "individual_v1_e1-12_eccen_bin_tuning_curves_full.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "noise_ceiling", "monte_carlo",
                     "stim_class", "bayesian_posterior", "monte_carlo_ses-04_task-sfprescaled_v1_e1-12.csv"),
        # this would ideally be just rules.groupaverage_all.input but
        # can't do that because groupaverage_all.input consists of three
        # functions (it doesn't evaluate them first), so need
        # this. eventually we'll have an output that combines across
        # seeds, and use that instead
        lambda wildcards: get_groupaverage_all(model_type='full_full_full'),
        lambda wildcards: get_groupaverage_all(model_type='full_full_absolute'),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "stim_class",
                     "bayesian_posterior", "sub-groupaverage_v1_e1-12_eccen_bin_tuning_curves_summary.csv"),


def get_figures_all(context='paper', visual_field_analyses=False):
    if context == 'paper':
        ext = 'pdf'
    else:
        ext = 'svg'
    figs = []
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'1d_{{}}_{{}}.{ext}').format(param, task)
             for param in ['bandwidth', 'pref-period', 'bandwidth-overall', 'pref-period-overall'] for task in ['task-sfprescaled', 'task-sfpconstant']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'cv_{{}}_v_task-sfprescaled.{ext}').format(cv)
             for cv in ['raw', 'demeaned', 'model', 'model_point', 'demeaned-remeaned',
                        'model-remeaned', 'model_point-remeaned', 'raw-nc', 'model_point-nc',
                        'model_point-remeaned-nc']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'cv_{{}}_h_task-sfprescaled.{ext}').format(cv)
             for cv in ['model_point', 'model_point-remeaned']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'{{}}_params_visualfield-all_{{}}_task-sfprescaled.{ext}').format(model, kind)
             for kind  in ['point', 'strip', 'dist', 'compare', 'pair', 'pair-drop', 'dist-overall'] for model in ['full_full_full', 'full_full_absolute']]
    if visual_field_analyses:
        figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'{{}}_params_visualfield-{{}}_{{}}_task-sfprescaled.{ext}').format(model, vf, kind)
                 for vf in ['all', 'inner', 'outer', 'left', 'right', 'upper', 'lower'] for kind  in ['point', 'strip'] for model in ['full_full_full', 'full_full_absolute']]
        figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'full_full_full_params_visualfield-{{}}_compare_task-sfprescaled.{ext}').format(vf)
                 for vf in ['vertical', 'horizontal', 'eccen']]
        figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'full_full_absolute_params_visualfield-{{}}_compare_task-sfprescaled.{ext}').format(vf)
                 for vf in ['vertical', 'horizontal', 'eccen']]
        figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'{{}}_feature_visualfield-{{}}_pref-period_median_angles-{{}}_task-sfprescaled_{{}}.{ext}').format(model, vf, angles, frame)
                 for vf in ['inner', 'outer', 'left', 'right', 'upper', 'lower'] for angles in ['all', 'avg'] for frame in ['relative', 'absolute']
                 for model in ['full_full_full', 'full_full_absolute']],
        figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'{{}}_feature_visualfield-{{}}_{{}}_median_angles-all_task-sfprescaled_{{}}.{ext}').format(model, vf, feature, frame)
                 for vf in ['inner', 'outer', 'left', 'right', 'upper', 'lower'] for feature in ['pref-period-contour', 'iso-pref-period', 'max-amp']
                 for frame in ['relative', 'absolute'] for model in ['full_full_full', 'full_full_absolute']],
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'{{}}_feature_visualfield-all_pref-period_{{}}_angles-{{}}_task-sfprescaled_{{}}.{ext}').format(model, kind, angles, frame)
             for kind  in ['median', 'bootstraps', 'bootstraps-overall'] for angles in ['all', 'avg'] for frame in ['relative', 'absolute']
             for model in ['full_full_full', 'full_full_absolute']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'{{}}_feature_visualfield-all_{{}}_{{}}_angles-all_task-sfprescaled_{{}}.{ext}').format(model, feature, kind, frame)
             for feature in ['pref-period-contour', 'iso-pref-period', 'max-amp']
             for kind  in ['median', 'bootstraps', 'bootstraps-overall'] for frame in ['relative', 'absolute']
             for model in ['full_full_full', 'full_full_absolute']]
    figs +=[os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'schematic_{{}}.{ext}').format(kind)
            for kind in ['2d', 'models', '2d-inputs']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'background_period.{ext}')]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'task-sfprescaled_background_period_{{}}.{ext}').format(model)
             for model in ['full_full_full', 'full_full_absolute']]
    figs += [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', f'{context}', f'{{}}_training-loss-check_task-sfprescaled.{ext}').format(t)
             for t in ['initial_cv', 'bootstrap']]
    return figs


rule figures:
    input:
        lambda wildcards: get_figures_all(),


rule figures_poster:
    input:
        lambda wildcards: get_figures_all('poster'),
