import os
import itertools
import warnings
import numpy as np
from glob import glob

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
            'sub-wlsubj064', 'sub-wlsubj081', 'sub-wlsubj095', 'sub-wlsubj007', 'sub-wlsubj062']
SESSIONS = {'sub-wlsubj001': ['ses-pilot01', 'ses-01', 'ses-02'],
            'sub-wlsubj004': ['ses-03'],
            'sub-wlsubj042': ['ses-pilot00', 'ses-pilot01', 'ses-01', 'ses-02'],
            'sub-wlsubj045': ['ses-pilot01', 'ses-01', 'ses-02', 'ses-04', 'ses-03'],
            'sub-wlsubj014': ['ses-03'], 'sub-wlsubj064': ['ses-04'], 'sub-wlsubj081': ['ses-04'],
            'sub-wlsubj095': ['ses-04'], 'sub-wlsubj007': ['ses-04'], 'sub-wlsubj062': ['ses-04']}
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
         ('sub-wlsubj062', 'ses-04'): 'task-sfprescaled'}
# these are the subject, session pairs where I didn't add the task to the protocol name and so some
# extra work is necessary.
WRONG_TASKS = {('sub-wlsubj001', 'ses-pilot01'): 'task-TASK',
               ('sub-wlsubj042', 'ses-01'): 'task-TASK', ('sub-wlsubj014', 'ses-03'): 'task-TASK',
               ('sub-wlsubj042', 'ses-pilot00'): 'task-TASK',
               ('sub-wlsubj042', 'ses-pilot01'): 'task-spatialfrequency',
               ('sub-wlsubj045', 'ses-pilot01'): 'task-spatialfrequency',
               ('sub-wlsubj064', 'ses-04'): 'task-sfprescaledcmrr'}
# every sub/ses pair that's not in here has the full number of runs, 12
NRUNS = {('sub-wlsubj001', 'ses-pilot01'): 9, ('sub-wlsubj042', 'ses-pilot00'): 8,
         ('sub-wlsubj045', 'ses-04'): 7}
VAREAS = [1]
MODEL_TYPES = ['iso_constant_constant', 'iso_scaling_constant', 'iso_full_constant',
               'absolute_full_constant', 'relative_full_constant', 'full_full_constant',
               'absolute_full_vary', 'relative_full_vary', 'full_full_vary']
def get_n_classes(session, mat_type):
    if mat_type == 'all_visual':
        return 1
    else:
        n = {'ses-pilot00': 52, 'ses-pilot01': 52, 'ses-01': 48, 'ses-02': 48,
             'ses-03': 48, 'ses-04': 48}[session]
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
             'sub-wlsubj095': 9, 'sub-wlsubj062': 0, 'sub-wlsubj007': 100}
# while session should increment along the tens digit
SES_SEEDS = {'ses-pilot00': 10, 'ses-pilot01': 20, 'ses-01': 30, 'ses-02': 40, 'ses-03': 50,
             'ses-04': 60}
wildcard_constraints:
    subject="sub-[a-z0-9]+",
    subjects="(sub-[a-z0-9]+,?)+",
    session="ses-[a-z0-9]+",
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
    atlas_type="bayesian_posterior|atlas",
    plot_func="[a-z]+",
    col="[a-z-]+",
    row="[a-z-]+",
    hue="[a-z-]+",
    y="[a-z-]+",
    binning="[a-z_]+bin",
    stimulus_class="([0-9,]+|None)",
    bootstrap_num="([0-9,]+|None)",
    orientation_type="[a-z-]+",
    eccentricity_type="[a-z-]+",
    train_amps="[a-z-]+",
    model_type="[a-z-_]+",
    crossval_seed="[0-9]+",
    gpus="[0-9]+"

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
    GLMdenoise_fixed_hrf > GLMdenoise > create_GLMdenoise_fixed_hrf_json > create_GLMdenoise_json


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
                     "v1_e1-12_full_b10_r0.001_g0_full_full_vary_all_models.csv"),
    


rule model_all_subj_visual_field:
    input:
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class",
                      "bayesian_posterior", "visual_field_%s" % p,
                      "v1_e1-12_summary_b10_r0.001_g0_full_full_vary_all_models.csv") for p in
         ['upper', 'lower', 'left', 'right', 'inner', 'outer']],


rule model_all_subj:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class",
                     "bayesian_posterior", "initial",
                     "v1_e1-12_summary_b10_r0.001_g0_full_full_vary_all_models.csv"),


rule model_all_subj_cv:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class",
                     "bayesian_posterior", "initial_cv", "v1_e1-12_summary_b10_r0.001_g0_s0_"
                     "all_models.csv"),


rule all_flat_plots:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', "{subject}", "{atlas_type}", 'varea_plot.png').format(subject=s, atlas_type=a)
         for s in SUBJECTS for a in ['bayesian_posterior', 'atlas']],
        [os.path.join(config["DATA_DIR"], "derivatives", "GLMdenoise", "stim_class", "bayesian_posterior", "{subject}", "{session}", "figures_{task}", "FinalModel_maps.png").format(
            subject=sub, session=ses, task=TASKS[(sub, ses)]) for sub in SUBJECTS for ses in SESSIONS[sub]]


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
        lambda wildcards: expand(os.path.join(config["DATA_DIR"], "derivatives", "preprocessed_run-{n:02d}_{task}", wildcards.subject, wildcards.session, wildcards.filename_ext), task=TASKS[(wildcards.subject, wildcards.session)], n=range(1, NRUNS.get((wildcards.subject, wildcards.session), 12)+1))
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


def find_benson_varea(wildcards):
    if wildcards.atlas_type == 'atlas':
        benson_prefix = 'benson14'
    elif wildcards.atlas_type == 'bayesian_posterior':
        benson_prefix = 'inferred'
    benson_template = os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', wildcards.subject, wildcards.atlas_type, '{hemi}.'+benson_prefix+'_varea.mgz')
    return expand(benson_template, hemi=['lh', 'rh'])


rule varea_check_plot:
    input:
        vareas_mgzs = find_benson_varea,
        freesurfer_dir = lambda wildcards: os.path.join(config['DATA_DIR'], 'derivatives', 'freesurfer', '{subject}').format(subject=wildcards.subject.replace('sub-', ''))
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', "{subject}", "{atlas_type}", 'varea_plot.png')
    log:
        os.path.join(config['DATA_DIR'], 'code', 'prf_solutions', '{subject}_{atlas_type}_varea_plot-%j.log')
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'prf_solutions', '{subject}_{atlas_type}_varea_plot_benchmark.txt')
    run:
        import neuropythy as ny
        import sfp
        atlases = {}
        for hemi in ['lh', 'rh']:
            path = [i for i in input.vareas_mgzs if hemi in i][0]
            atlases[hemi] = ny.load(path)
        sfp.plotting.flat_cortex_plot(input.freesurfer_dir, atlases, output[0],
                                      ('plot_property', [1, 2, 3]))


rule create_GLMdenoise_json:
    input:
        json_template = os.path.join(config['MRI_TOOLS'], 'BIDS', 'files', 'glmOptsOptimize.json'),
        vareas_mgzs = find_benson_varea
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
        vareas_mgzs = find_benson_varea
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


rule GLMdenoise:
    input:
        preproc_files = lambda wildcards: expand(os.path.join(config["DATA_DIR"], "derivatives", "preprocessed_reoriented", wildcards.subject, wildcards.session, "{hemi}."+wildcards.subject+"_"+wildcards.session+"_"+wildcards.task+"_run-{n:02d}_preproc.mgz"), hemi=['lh', 'rh'], n=range(1, NRUNS.get((wildcards.subject, wildcards.session), 12)+1)),
        params_file = os.path.join(config["DATA_DIR"], "derivatives", "design_matrices", "{mat_type}", "{subject}", "{session}", "{subject}_{session}_{task}_params.json"),
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
    resources:
        cpus_per_task = 1,
        mem = 100
    shell:
        "cd {params.GLM_dir}; matlab -nodesktop -nodisplay -r \"addpath(genpath('{params."
        "vistasoft_path}')); addpath(genpath('{params.GLMdenoise_path}')); "
        "jsonInfo=jsondecode(fileread('{input.params_file}')); bidsGLM('{params."
        "BIDS_dir}', '{params.subject}', '{params.session}', [], [], "
        "'preprocessed_reoriented', 'preproc', '{wildcards.mat_type}', jsonInfo.stim_length, "
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
        vistasoft_path = os.path.join(config['VISTASOFT_PATH']),
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


def get_first_level_analysis_input(wildcards):
    input_dict = {}
    input_dict['GLM_results'] = os.path.join(config["DATA_DIR"], "derivatives", "GLMdenoise", "{mat_type}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_results.mat").format(**wildcards)
    benson_names = ['angle', 'eccen', 'varea']
    if wildcards.atlas_type == 'atlas':
        benson_prefix = 'benson14'
    elif wildcards.atlas_type == 'bayesian_posterior':
        benson_prefix = 'inferred'
    if wildcards.subject in ['sub-wlsubj064', 'sub-wlsubj007', 'sub-wlsubj095', 'sub-wlsubj081',
                             'sub-wlsubj062']:
        prf_prefix = 'full'
    else:
        prf_prefix = 'all00'
    benson_temp = os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', wildcards.subject, wildcards.atlas_type, '{hemi}.'+benson_prefix+'_{filename}.mgz')
    input_dict['benson_paths'] = expand(benson_temp, hemi=['lh', 'rh'], filename=benson_names)
    prf_temp = os.path.join(config['DATA_DIR'], 'derivatives', 'prf_solutions', wildcards.subject, 'data', '{hemi}.'+prf_prefix+'-{filename}.mgz')
    input_dict['prf_sigma_path'] = expand(prf_temp, hemi=['lh', 'rh'], filename=['sigma', 'vexpl'])
    return input_dict


def get_stim_type(wildcards):
    if 'pilot' in wildcards.session:
        return 'pilot'
    else:
        if 'constant' in wildcards.task:
            return 'constant'
        else:
            return 'logpolar'


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
        benson_template = lambda wildcards, input: input.benson_paths[0].replace('lh', '%s').replace('angle', '%s').replace('benson14_', '').replace('inferred_', '').replace(wildcards.atlas_type, '%s'),
        benson_names = lambda wildcards, input: [i.split('.')[-2] for i in input if wildcards.atlas_type+'/lh' in i],
        prf_names = lambda wildcards, input: [i.split('.')[-2] for i in input if 'data/lh' in i],
        class_num = lambda wildcards: get_n_classes(wildcards.session, wildcards.mat_type),
        stim_type = get_stim_type,
        mid_val = lambda wildcards: {'ses-pilot01': 127, 'ses-pilot00': 127}.get(wildcards.session, 128)
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
        "--benson_template_path {params.benson_template} --benson_atlas_type {wildcards.atlas_type}"
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


rule tuning_curves_summary:
    input:
        get_tuning_curves
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "{mat_type}", "{atlas_type}", "v{vareas}_e{eccen}_{binning}_tuning_curves_{df_mode}.csv")
    params:
        input_dir = os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "{mat_type}", "{atlas_type}")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_curves_summary", "{mat_type}_{atlas_type}_v{vareas}_e{eccen}_{binning}_{df_mode}_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_curves_summary", "{mat_type}_{atlas_type}_v{vareas}_e{eccen}_{binning}_{df_mode}-%j.log")
    shell:
        "python sfp/summarize_tuning_curves.py {params.input_dir} {output} {wildcards.df_mode}"


rule tuning_curves_summary_plot:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "{mat_type}", "{atlas_type}", "v{vareas}_e{eccen}_{binning}_tuning_curves_summary.csv")
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


def parse_train_amps(wildcards):
    if wildcards.train_amps == 'vary':
        return '-v'
    elif wildcards.train_amps == 'constant':
        return ''
    else:
        raise Exception("train_amps must be either 'vary' or 'constant'!")


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
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_n{bootstrap_num}_{orientation_type}_{eccentricity_type}_{train_amps}_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_n{bootstrap_num}_{orientation_type}_{eccentricity_type}_{train_amps}_model_history.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_n{bootstrap_num}_{orientation_type}_{eccentricity_type}_{train_amps}_model.pt"),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_n{bootstrap_num}_{orientation_type}_{eccentricity_type}_{train_amps}_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_n{bootstrap_num}_{orientation_type}_{eccentricity_type}_{train_amps}-%j.log")
    resources:
        # need the same number of cpus and gpus
        cpus_per_task = lambda wildcards: max(int(wildcards.gpus), 1),
        mem = lambda wildcards: {'full': 40, 'summary': 10}[wildcards.df_mode],
        gpus = lambda wildcards: int(wildcards.gpus)
    params:
        save_stem = lambda wildcards, output: output[0].replace("_loss.csv", ''),
        stimulus_class = lambda wildcards: wildcards.stimulus_class.split(','),
        bootstrap_num = lambda wildcards: wildcards.bootstrap_num.split(','),
        train_amps = parse_train_amps,
        logging = to_log_or_not,
        vis_field = visual_field_part,
    shell:
        "python -m sfp.model {wildcards.orientation_type} {wildcards.eccentricity_type} "
        "{params.train_amps} {input} {params.save_stem} -b "
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
                   modeling_goal='initial_cv'):
        output_path = os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}",
                                   "{atlas_type}", "{modeling_goal}", "{{subject}}", "{{session}}",
                                   "{{subject}}_{{session}}_{{task}}_v{vareas}_e{eccen}_{df_mode}_b{batch"
                                   "_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all_cv_loss.csv")
        output_path = output_path.format(vareas=vareas, mat_type=mat_type, batch_size=batch_size,
                                         eccen=eccen, atlas_type=atlas_type, df_mode=df_mode,
                                         modeling_goal=modeling_goal, gpus=gpus,
                                         crossval_seed=crossval_seed, learning_rate=learning_rate)
        return [output_path.format(subject=subj, session=ses, task=TASKS[(subj, ses)]) for subj in SUBJECTS for ses in SESSIONS[subj]]


rule combine_model_cv_summaries:
    input:
        lambda wildcards: get_cv_summary(**wildcards)
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_s{crossval_seed}_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_s{crossval_seed}_all_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_s{crossval_seed}_all_timing.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_s{crossval_seed}_all_cv_loss.csv"),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_s{crossval_seed}_all-%j.log")
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
        loss_files = [get_model_subj_outputs(subject=subj, session=ses, task=TASKS[(subj, ses)],
                                             **wildcards)
                      for subj in SUBJECTS for ses in SESSIONS[subj]]
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
        base_path = lambda wildcards, output: os.path.join(os.path.dirname(output[0]), '*'),
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
                                        "{model_type}_all_models.csv").format(subject=subj, session=ses, task=TASKS[(subj, ses)], **wildcards)
                           for subj in SUBJECTS for ses in SESSIONS[subj]]
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "bootstrap", "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "bootstrap", "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "bootstrap", "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_timing.csv"),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_bootstrap_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_bootstrap_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all-%j.log")
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
                     "{modeling_goal}", "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_models.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_timing.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_diff.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}",
                     "{modeling_goal}", "v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_"
                     "g{gpus}_{model_type}_all_model_history.csv"),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_{model_type}_all-%j.log")
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
        os.path.join(config['DATA_DIR'], 'derivatives', 'simulated_data', 'noise-uniform', 'n{num_voxels}_{orientation_type}_{eccentricity_type}_{train_amps}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_simulated.csv')
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'simulated_data', 'noise-uniform_n{num_voxels}_{orientation_type}_{eccentricity_type}_{train_amps}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_benchmark.txt')
    log:
        os.path.join(config['DATA_DIR'], 'code', 'simulated_data', 'noise-uniform_n{num_voxels}_{orientation_type}_{eccentricity_type}_{train_amps}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}-%j.log')
    resources:
        mem=10
    params:
        train_amps = parse_train_amps
    shell:
        "python -m sfp.simulate_data {output} -o {wildcards.orientation_type} -e {wildcards.eccentricity_type} "
        "{params.train_amps} -n {wildcards.num_voxels} -s {wildcards.sigma} "
        "-a {wildcards.sf_ecc_slope} -rmc {wildcards.rel_mode_cardinals} -rmo "
        "{wildcards.rel_mode_obliques} -rac {wildcards.rel_amplitude_cardinals} -rao "
        "{wildcards.rel_amplitude_obliques} -amc {wildcards.abs_mode_cardinals} -amo "
        "{wildcards.abs_mode_obliques} -aac {wildcards.abs_amplitude_cardinals} -aao "
        "{wildcards.abs_amplitude_obliques} -b {wildcards.sf_ecc_intercept} -l {wildcards.noise_level}"


rule simulate_data_voxel_noise:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_summary.csv')
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'simulated_data', 'noise-{mat_type}_{atlas_type}_{subject}_{session}_{task}_v{vareas}_e{eccen}', 'n{num_voxels}_{orientation_type}_{eccentricity_type}_{train_amps}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_simulated.csv')
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'simulated_data', 'noise-{mat_type}_{atlas_type}_{subject}_{session}_{task}_v{vareas}_e{eccen}_n{num_voxels}_{orientation_type}_{eccentricity_type}_{train_amps}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_benchmark.txt')
    log:
        os.path.join(config['DATA_DIR'], 'code', 'simulated_data', 'noise-{mat_type}_{atlas_type}_{subject}_{session}_{task}_v{vareas}_e{eccen}_n{num_voxels}_{orientation_type}_{eccentricity_type}_{train_amps}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}-%j.log')
    resources:
        mem=10
    params:
        train_amps = parse_train_amps
    shell:
        "python -m sfp.simulate_data {output} -o {wildcards.orientation_type} -e {wildcards.eccentricity_type} "
        "{params.train_amps} -n {wildcards.num_voxels} -s {wildcards.sigma} "
        "-a {wildcards.sf_ecc_slope} -rmc {wildcards.rel_mode_cardinals} -rmo "
        "{wildcards.rel_mode_obliques} -rac {wildcards.rel_amplitude_cardinals} -rao "
        "{wildcards.rel_amplitude_obliques} -amc {wildcards.abs_mode_cardinals} -amo "
        "{wildcards.abs_mode_obliques} -aac {wildcards.abs_amplitude_cardinals} -aao "
        "{wildcards.abs_amplitude_obliques} -b {wildcards.sf_ecc_intercept} -l {wildcards.noise_level} "
        "--noise_source_path {input}"


rule model_simulated_data:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'simulated_data', 'noise-{noise_source}', 'n{num_voxels}_{sim_orientation_type}_{sim_eccentricity_type}_{sim_train_amps}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_simulated.csv')
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-{noise_source}", "{modeling_goal}", "n{num_voxels}_{sim_orientation_type}_{sim_eccentricity_type}_{sim_train_amps}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_{orientation_type}_{eccentricity_type}_{train_amps}_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-{noise_source}", "{modeling_goal}", "n{num_voxels}_{sim_orientation_type}_{sim_eccentricity_type}_{sim_train_amps}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_{orientation_type}_{eccentricity_type}_{train_amps}_model_history.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-{noise_source}", "{modeling_goal}", "n{num_voxels}_{sim_orientation_type}_{sim_eccentricity_type}_{sim_train_amps}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_{orientation_type}_{eccentricity_type}_{train_amps}_model.pt"),
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'tuning_2d_simulated', 'noise-{noise_source}_{modeling_goal}_n{num_voxels}_{sim_orientation_type}_{sim_eccentricity_type}_{sim_train_amps}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_{orientation_type}_{eccentricity_type}_{train_amps}_benchmark.txt')
    log:
        os.path.join(config['DATA_DIR'], 'code', 'tuning_2d_simulated', 'noise-{noise_source}_{modeling_goal}_n{num_voxels}_{sim_orientation_type}_{sim_eccentricity_type}_{sim_train_amps}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_{orientation_type}_{eccentricity_type}_{train_amps}-%j.log')
    resources:
        # need the same number of cpus and gpus
        cpus_per_task = lambda wildcards: max(int(wildcards.gpus), 1),
        mem = 10,
        gpus = lambda wildcards: int(wildcards.gpus),
    params:
        save_stem = lambda wildcards, output: output[0].replace("_loss.csv", ''),
        stimulus_class = lambda wildcards: wildcards.stimulus_class.split(','),
        train_amps = parse_train_amps,
        logging = to_log_or_not,
    shell:
        "python -m sfp.model {wildcards.orientation_type} {wildcards.eccentricity_type} "
        "{params.train_amps} {input} {params.save_stem} -b {wildcards.batch_size} "
        "-r {wildcards.learning_rate} -d None -t 1e-6 -e 1000 -c {params.stimulus_class} "
        "{params.logging} {log}"


def gather_simulated_model_results_input(wildcards):
    inputs = {}
    if wildcards.modeling_goal == 'learning_hyperparams_full':
        batch = [1, 10, 100]
        lr = [1e-2, 1e-3, 1e-4]
        models = ['iso_full_constant', 'full_full_vary']
        loss_files = []
        for b, l, m  in itertools.product(batch, lr, models):
            loss_files.append(get_simulated_model_outputs(
                m, 'iso_full_constant', 1, 4000, b, l, 1, .75, .25, 0, 0, 0, 0, 0, 0, 0, 0,
                **wildcards))
            loss_files.append(get_simulated_model_outputs(
                m, 'full_full_vary', 1, 4000, b, l, 1, .75, .25, .1, .05, .03, .1, .2, .05, .04,
                .3, **wildcards))
    elif wildcards.modeling_goal == 'model_recovery':
        loss_files = []
        for m  in MODEL_TYPES:
            loss_files.append(get_simulated_model_outputs(
                m, 'iso_full_constant', 1, 4000, 10, 1e-3, 1, .75, .25, 0, 0, 0, 0, 0, 0, 0, 0,
                **wildcards))
            loss_files.append(get_simulated_model_outputs(
                m, 'full_full_vary', 1, 4000, 10, 1e-3, 1, .75, .25, .1, .05, .03, .1, .2, .05,
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
        sfp.analyze_model.calc_cv_error(input.loss_files, input.dataset_path, wildcards, output)


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
    models = [['iso_full_constant', 1, .75, .25, 0, 0, 0, 0, 0, 0, 0, 0],
              ['full_full_vary', 1, .75, .25, .1, .05, .03, .1, .2, .05, .04, .03]]
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
        os.path.join(config['DATA_DIR'], "derivatives", "figures", "1d_{tuning_param}_{task}.{ext}")
    log:
        os.path.join(config['DATA_DIR'], 'code', "figures", "1d_{tuning_param}_{task}_{ext}.log")
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', "figures",
                     "1d_{tuning_param}_{task}_{ext}_benchmark.txt")
    run:
        import pandas as pd
        import seaborn as sns
        import sfp
        df = sfp.figures.prep_df(pd.read_csv(input[0]), wildcards.task)
        ref_frame = {'task-sfpconstant': 'absolute', 'task-sfprescaled': 'relative'}
        with sns.axes_style('white'):
            if wildcards.tuning_param == 'pref-period':
                g = sfp.figures.pref_period_1d(df, ref_frame[wildcards.task], row=None)
            elif wildcards.tuning_param == 'bandwidth':
                g = sfp.figures.bandwidth_1d(df, ref_frame[wildcards.task], row=None)
            g.fig.savefig(output[0], bbox_inches='tight')


rule figure_crossvalidation:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model', 'stim_class',
                     'bayesian_posterior', 'initial_cv',
                     'v1_e1-12_summary_b10_r0.001_g0_s0_all_cv_loss.csv')
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "figures", "cv_{cv_type}_{task}.{ext}")
    log:
        os.path.join(config['DATA_DIR'], "code", "figures", "cv_{cv_type}_{task}_{ext}.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "figures", "cv_{cv_type}_{task}_{ext}_benchmark.txt")
    run:
        import pandas as pd
        import seaborn as sns
        import sfp
        df = sfp.figures.prep_df(pd.read_csv(input[0]), wildcards.task)
        with sns.axes_style('white'):
            if len(wildcards.cv_type.split('-')) > 1:
                assert wildcards.cv_type.split('-')[1] == 'remeaned'
                remeaned = True
            else:
                remeaned = False
            if wildcards.cv_type.startswith('demeaned'):
                g = sfp.figures.cross_validation_demeaned(df, remeaned)
            elif wildcards.cv_type == 'raw':
                g = sfp.figures.cross_validation_raw(df)
            elif wildcards.cv_type.startswith('model_point'):
                g = sfp.figures.cross_validation_model(df, 'point', remeaned)
            elif wildcards.cv_type.startswith('model'):
                g = sfp.figures.cross_validation_model(df, remeaned=remeaned)
            g.fig.savefig(output[0], bbox_inches='tight')


def get_params_csv(wildcards):
    path_template = os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_2d_model',
                                 'stim_class', 'bayesian_posterior', '%s',
                                 'v1_e1-12_%s_b10_r0.001_g0_full_full_vary_all_models.csv')
    paths = []
    if wildcards.plot_kind in ['dist', 'pair', 'pair-drop', 'compare', 'bootstraps']:
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
    return paths


rule figure_params:
    input:
        get_params_csv,
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "figures", "params_visualfield-{vf}_{plot_kind}_{task}.{ext}")
    log:
        os.path.join(config['DATA_DIR'], "code", "figures", "params_visualfield-{vf}_{plot_kind}_{task}_{ext}.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "figures", "params_visualfield-{vf}_{plot_kind}_{task}_{ext}_benchmark.txt")
    run:
        import pandas as pd
        import seaborn as sns
        import sfp
        import matplotlib as mpl
        df = []
        for p in input:
            tmp = sfp.figures.prep_df(pd.read_csv(p), wildcards.task)
            df.append(sfp.figures.prep_model_df(tmp))
        with sns.axes_style('white', {'axes.spines.right': False, 'axes.spines.top': False}):
            if wildcards.plot_kind.startswith('pair'):
                if len(wildcards.plot_kind.split('-')) > 1:
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
                # don't add a legend if the plot_kind is point
                add_legend = {'point': False}.get(wildcards.plot_kind, True)
                fig = sfp.figures.model_parameters(df[0], wildcards.plot_kind, wildcards.vf,
                                                   add_legend=add_legend)
            fig.savefig(output[0], bbox_inches='tight')


rule figure_feature_df:
    input:
        get_params_csv,
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "figures", "feature_visualfield-{vf}_{feature_type}_{plot_kind}_angles-{angles}_{task}_{ref_frame}.{ext}")
    log:
        os.path.join(config['DATA_DIR'], "code", "figures", "feature_visualfield-{vf}_{feature_type}_{plot_kind}_angles-{angles}_{task}_{ref_frame}_{ext}.log")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "figures", "feature_visualfield-{vf}_{feature_type}_{plot_kind}_angles-{angles}_{task}_{ref_frame}_{ext}_benchmark.txt")
    run:
        import pandas as pd
        import seaborn as sns
        import sfp
        df = sfp.figures.prep_df(pd.read_csv(input[0]), wildcards.task)
        with sns.axes_style('white', {'axes.spines.right': False, 'axes.spines.top': False}):
            if wildcards.angles == 'avg':
                angles = True
            elif wildcards.angles == 'all':
                angles = False
            g = sfp.figures.feature_df_plot(df, angles, wildcards.ref_frame, wildcards.feature_type,
                                            wildcards.vf)
            g.fig.savefig(output[0], bbox_inches='tight')


rule figure_schematic:
    output:
        os.path.join(config["DATA_DIR"], 'derivatives', 'figures', 'schematic_{schematic_type}.{ext}')
    log:
        os.path.join(config["DATA_DIR"], 'code', 'figures', 'schematic_{schematic_type}_{ext}.log')
    benchmark:
        os.path.join(config["DATA_DIR"], 'code', 'figures', 'schematic_{schematic_type}_{ext}_benchmark.txt')
    run:
        import sfp
        import seaborn as sns
        with sns.axes_style('white', {'axes.spines.right': False, 'axes.spines.top': False}):
            if wildcards.schematic_type == '2d':
                fig = sfp.figures.model_schematic()
            elif wildcards.schematic_type == '2d-inputs':
                fig = sfp.figures.input_schematic()
            elif wildcards.schematic_type == 'models':
                fig = sfp.figures.model_types()
            fig.savefig(output[0], bbox_inches='tight')


rule figure_background:
    output:
        os.path.join(config["DATA_DIR"], 'derivatives', 'figures', 'background_{y_val}.{ext}')
    log:
        os.path.join(config["DATA_DIR"], 'code', 'figures', 'background_{y_val}_{ext}.log')
    benchmark:
        os.path.join(config["DATA_DIR"], 'code', 'figures', 'background_{y_val}_{ext}_benchmark.txt')
    run:
        import sfp
        import seaborn as sns
        with sns.axes_style('white', {'axes.spines.right': False, 'axes.spines.top': False}):
            df = sfp.figures.existing_studies_df()
            y = {'period': 'Preferred period (dpc)',
                 'frequency': 'Preferred spatial frequency (cpd)'}[wildcards.y_val]
            g = sfp.figures.existing_studies_figure(df, y)
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
        rules.all_flat_plots.input,
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "stim_class",
                     "bayesian_posterior", "v1_e1-12_eccen_bin_tuning_curves_summary.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "stim_class",
                     "bayesian_posterior", "v1_e1-12_eccen_bin_tuning_curves_full.csv")


rule figures:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', '1d_{}_{}.pdf').format(param, task)
         for param in ['bandwidth', 'pref-period'] for task in ['task-sfprescaled', 'task-sfpconstant']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'cv_{}_task-sfprescaled.pdf').format(cv)
         for cv in ['raw', 'demeaned', 'model', 'model_point', 'demeaned-remeaned',
                    'model-remeaned', 'model_point-remeaned']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'params_visualfield-all_{}_task-sfprescaled.pdf').format(kind)
         for kind  in ['point', 'strip', 'dist', 'compare', 'pair', 'pair-drop']],
        # [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'params_visualfield-{}_{}_task-sfprescaled.pdf').format(vf, kind)
        #  for vf in ['all', 'inner', 'outer', 'left', 'right', 'upper', 'lower'] for kind  in ['point', 'strip']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'params_visualfield-{}_compare_task-sfprescaled.pdf').format(vf)
         for vf in ['vertical', 'horizontal', 'eccen']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'feature_visualfield-all_pref-period_{}_angles-{}_task-sfprescaled_{}.pdf').format(kind, angles, frame)
         for kind  in ['median', 'bootstraps'] for angles in ['all', 'avg'] for frame in ['relative', 'absolute']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'feature_visualfield-all_{}_{}_angles-all_task-sfprescaled_{}.pdf').format(feature, kind, frame)
         for feature in ['pref-period-contour', 'iso-pref-period', 'max-amp']
         for kind  in ['median', 'bootstraps'] for frame in ['relative', 'absolute']],
        # [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'feature_visualfield-{}_pref-period_median_angles-{}_task-sfprescaled_{}.pdf').format(vf, angles, frame)
        #  for vf in ['inner', 'outer', 'left', 'right', 'upper', 'lower'] for angles in ['all', 'avg'] for frame in ['relative', 'absolute']],
        # [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'feature_visualfield-{}_{}_median_angles-all_task-sfprescaled_{}.pdf').format(vf, feature, frame)
        #  for vf in ['inner', 'outer', 'left', 'right', 'upper', 'lower'] for feature in ['pref-period-contour', 'iso-pref-period', 'max-amp']
        #  for frame in ['relative', 'absolute']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'schematic_{}.pdf').format(kind)
         for kind in ['2d', 'models', '2d-inputs']],
        os.path.join(config['DATA_DIR'], 'derivatives', 'figures', 'background_period.pdf'),
