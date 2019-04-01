import os
import warnings
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


SUBJECTS = ['sub-wlsubj001', 'sub-wlsubj004', 'sub-wlsubj042', 'sub-wlsubj045', 'sub-wlsubj014']
SESSIONS = {'sub-wlsubj001': ['ses-pilot01', 'ses-01', 'ses-02'],
            'sub-wlsubj004': ['ses-03'],
            'sub-wlsubj042': ['ses-pilot00', 'ses-pilot01', 'ses-01', 'ses-02'],
            'sub-wlsubj045': ['ses-pilot01', 'ses-01', 'ses-02', 'ses-04', 'ses-03'],
            'sub-wlsubj014': ['ses-03']}
TASKS = {('sub-wlsubj001', 'ses-pilot01'): 'task-sfp', ('sub-wlsubj001', 'ses-01'): 'task-sfp',
         ('sub-wlsubj001', 'ses-02'): 'task-sfpconstant', 
         ('sub-wlsubj042', 'ses-pilot00'): 'task-sfp', ('sub-wlsubj042', 'ses-pilot01'): 'task-sfp',
         ('sub-wlsubj042', 'ses-01'): 'task-sfpconstant', ('sub-wlsubj042', 'ses-02'): 'task-sfp',
         ('sub-wlsubj045', 'ses-pilot01'): 'task-sfp',
         ('sub-wlsubj045', 'ses-01'): 'task-sfpconstant',  ('sub-wlsubj045', 'ses-02'): 'task-sfp',
         ('sub-wlsubj014', 'ses-03'): 'task-sfp', ('sub-wlsubj004', 'ses-03'): 'task-sfp',
         ('sub-wlsubj045', 'ses-04'): 'task-sfprescaled', ('sub-wlsubj045', 'ses-03'): 'task-sfp'}
# every sub/ses pair that's not in here has the full number of runs, 12
NRUNS = {('sub-wlsubj001', 'ses-pilot01'): 9, ('sub-wlsubj042', 'ses-pilot00'): 8,
         ('sub-wlsubj045', 'ses-04'): 7}
VAREAS = [1]
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
SUB_SEEDS = {'sub-wlsubj001': 1, 'sub-wlsubj042': 2, 'sub-wlsubj045': 3, 'sub-wlsubj004': 4,
             'sub-wlsubj014': 5, 'sub-wlsubj004': 6}
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
    plot_func="[a-z]+",
    col="[a-z-]+",
    row="[a-z-]+",
    hue="[a-z-]+",
    y="[a-z-]+",
    binning="[a-z_]+bin",
    stimulus_class="([0-9,]+|None)",
    orientation_type="[a-z-]+",
    eccentricity_type="[a-z-]+",
    train_amps="[a-z-]+",
    model_type="[a-z-]+"

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
    

# all: plots_all plots_modeling_blanks plots_VSS_abstract summary_plots_all summary_plots_VSS_abstract

rule model_learning_hyperparams:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-3_g0_cNone_iso_full_constant_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b10_r1e-3_g0_cNone_iso_full_constant_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b100_r1e-3_g0_cNone_iso_full_constant_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_cNone_iso_full_constant_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b10_r1e-2_g0_cNone_iso_full_constant_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b100_r1e-2_g0_cNone_iso_full_constant_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-4_g0_cNone_iso_full_constant_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b10_r1e-4_g0_cNone_iso_full_constant_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b100_r1e-4_g0_cNone_iso_full_constant_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-3_g0_cNone_full_full_vary_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b10_r1e-3_g0_cNone_full_full_vary_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b100_r1e-3_g0_cNone_full_full_vary_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_cNone_full_full_vary_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b10_r1e-2_g0_cNone_full_full_vary_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b100_r1e-2_g0_cNone_full_full_vary_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-4_g0_cNone_full_full_vary_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b10_r1e-4_g0_cNone_full_full_vary_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "learning_hyperparams", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b100_r1e-4_g0_cNone_full_full_vary_loss.csv"),


rule model_recovery_initial:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_cNone_iso_constant_constant_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_cNone_iso_scaling_constant_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_cNone_iso_full_constant_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_cNone_absolute_full_constant_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_cNone_relative_full_constant_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_cNone_full_full_constant_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_cNone_absolute_full_vary_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_cNone_relative_full_vary_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_cNone_full_full_vary_loss.csv"),


rule model_recovery_cv_initial:
    input:
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery_cv", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_c{:02d}_iso_constant_constant_loss.csv").format(n) for n in range(get_n_classes('ses-02', 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery_cv", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_c{:02d}_iso_scaling_constant_loss.csv").format(n) for n in range(get_n_classes('ses-02', 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery_cv", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_c{:02d}_iso_full_constant_loss.csv").format(n) for n in range(get_n_classes('ses-02', 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery_cv", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_c{:02d}_absolute_full_constant_loss.csv").format(n) for n in range(get_n_classes('ses-02', 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery_cv", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_c{:02d}_relative_full_constant_loss.csv").format(n) for n in range(get_n_classes('ses-02', 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery_cv", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_c{:02d}_full_full_constant_loss.csv").format(n) for n in range(get_n_classes('ses-02', 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery_cv", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_c{:02d}_absolute_full_vary_loss.csv").format(n) for n in range(get_n_classes('ses-02', 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery_cv", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_c{:02d}_relative_full_vary_loss.csv").format(n) for n in range(get_n_classes('ses-02', 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-stim_class_posterior_sub-wlsubj045_ses-02_task-sfp_v1_e1-12", "model_recovery_cv", "n100_iso_full_constant_s1_a.75_b.25_rmc0_rmo0_rac0_rao0_amc0_amo0_aac0_aao0_l1_b1_r1e-2_g0_c{:02d}_full_full_vary_loss.csv").format(n) for n in range(get_n_classes('ses-02', 'stim_class'))],


rule model_subj_initial:
    input:
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "initial", "{subject}", "{session}", "{subject}_{session}_{task}_v1_e1-12_summary_b1_r1e-2_g0_cNone_iso_constant_constant_loss.csv").format(subject=subj, session=ses, task=task) for subj, ses, task in zip(['sub-wlsubj001', 'sub-wlsubj045', 'sub-wlsubj045'], ['ses-01', 'ses-02', 'ses-04'], ['task-sfp', 'task-sfp', 'task-sfprescaled'])],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "initial", "{subject}", "{session}", "{subject}_{session}_{task}_v1_e1-12_summary_b1_r1e-2_g0_cNone_iso_scaling_constant_loss.csv").format(subject=subj, session=ses, task=task) for subj, ses, task in zip(['sub-wlsubj001', 'sub-wlsubj045', 'sub-wlsubj045'], ['ses-01', 'ses-02', 'ses-04'], ['task-sfp', 'task-sfp', 'task-sfprescaled'])],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "initial", "{subject}", "{session}", "{subject}_{session}_{task}_v1_e1-12_summary_b1_r1e-2_g0_cNone_iso_full_constant_loss.csv").format(subject=subj, session=ses, task=task) for subj, ses, task in zip(['sub-wlsubj001', 'sub-wlsubj045', 'sub-wlsubj045'], ['ses-01', 'ses-02', 'ses-04'], ['task-sfp', 'task-sfp', 'task-sfprescaled'])],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "initial", "{subject}", "{session}", "{subject}_{session}_{task}_v1_e1-12_summary_b1_r1e-2_g0_cNone_absolute_full_constant_loss.csv").format(subject=subj, session=ses, task=task) for subj, ses, task in zip(['sub-wlsubj001', 'sub-wlsubj045', 'sub-wlsubj045'], ['ses-01', 'ses-02', 'ses-04'], ['task-sfp', 'task-sfp', 'task-sfprescaled'])],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "initial", "{subject}", "{session}", "{subject}_{session}_{task}_v1_e1-12_summary_b1_r1e-2_g0_cNone_relative_full_constant_loss.csv").format(subject=subj, session=ses, task=task) for subj, ses, task in zip(['sub-wlsubj001', 'sub-wlsubj045', 'sub-wlsubj045'], ['ses-01', 'ses-02', 'ses-04'], ['task-sfp', 'task-sfp', 'task-sfprescaled'])],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "initial", "{subject}", "{session}", "{subject}_{session}_{task}_v1_e1-12_summary_b1_r1e-2_g0_cNone_full_full_constant_loss.csv").format(subject=subj, session=ses, task=task) for subj, ses, task in zip(['sub-wlsubj001', 'sub-wlsubj045', 'sub-wlsubj045'], ['ses-01', 'ses-02', 'ses-04'], ['task-sfp', 'task-sfp', 'task-sfprescaled'])],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "initial", "{subject}", "{session}", "{subject}_{session}_{task}_v1_e1-12_summary_b1_r1e-2_g0_cNone_absolute_full_vary_loss.csv").format(subject=subj, session=ses, task=task) for subj, ses, task in zip(['sub-wlsubj001', 'sub-wlsubj045', 'sub-wlsubj045'], ['ses-01', 'ses-02', 'ses-04'], ['task-sfp', 'task-sfp', 'task-sfprescaled'])],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "initial", "{subject}", "{session}", "{subject}_{session}_{task}_v1_e1-12_summary_b1_r1e-2_g0_cNone_relative_full_vary_loss.csv").format(subject=subj, session=ses, task=task) for subj, ses, task in zip(['sub-wlsubj001', 'sub-wlsubj045', 'sub-wlsubj045'], ['ses-01', 'ses-02', 'ses-04'], ['task-sfp', 'task-sfp', 'task-sfprescaled'])],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "initial", "{subject}", "{session}", "{subject}_{session}_{task}_v1_e1-12_summary_b1_r1e-2_g0_cNone_full_full_vary_loss.csv").format(subject=subj, session=ses, task=task) for subj, ses, task in zip(['sub-wlsubj001', 'sub-wlsubj045', 'sub-wlsubj045'], ['ses-01', 'ses-02', 'ses-04'], ['task-sfp', 'task-sfp', 'task-sfprescaled'])],


rule model_all_data:
    input:
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj001", "ses-01", "sub-wlsubj001_ses-01_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_iso_full_constant_loss.csv").format(num=n) for n in range(get_n_classes("ses-01", 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj001", "ses-01", "sub-wlsubj001_ses-01_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_iso_scaling_constant_loss.csv").format(num=n) for n in range(get_n_classes("ses-01", 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj001", "ses-01", "sub-wlsubj001_ses-01_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_iso_constant_constant_loss.csv").format(num=n) for n in range(get_n_classes("ses-01", 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj001", "ses-01", "sub-wlsubj001_ses-01_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_absolute_full_constant_loss.csv").format(num=n) for n in range(get_n_classes("ses-01", 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj001", "ses-01", "sub-wlsubj001_ses-01_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_relative_full_constant_loss.csv").format(num=n) for n in range(get_n_classes("ses-01", 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj001", "ses-01", "sub-wlsubj001_ses-01_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_full_full_constant_loss.csv").format(num=n) for n in range(get_n_classes("ses-01", 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj001", "ses-01", "sub-wlsubj001_ses-01_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_absolute_full_vary_loss.csv").format(num=n) for n in range(get_n_classes("ses-01", 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj001", "ses-01", "sub-wlsubj001_ses-01_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_relative_full_vary_loss.csv").format(num=n) for n in range(get_n_classes("ses-01", 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj001", "ses-01", "sub-wlsubj001_ses-01_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_full_full_vary_loss.csv").format(num=n) for n in range(get_n_classes("ses-01", 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj045", "ses-02", "sub-wlsubj045_ses-02_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_iso_full_constant_loss.csv").format(num=n) for n in range(get_n_classes("ses-02", 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj045", "ses-02", "sub-wlsubj045_ses-02_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_iso_scaling_constant_loss.csv").format(num=n) for n in range(get_n_classes("ses-02", 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj045", "ses-02", "sub-wlsubj045_ses-02_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_iso_constant_constant_loss.csv").format(num=n) for n in range(get_n_classes("ses-02", 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj045", "ses-02", "sub-wlsubj045_ses-02_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_absolute_full_constant_loss.csv").format(num=n) for n in range(get_n_classes("ses-02", 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj045", "ses-02", "sub-wlsubj045_ses-02_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_relative_full_constant_loss.csv").format(num=n) for n in range(get_n_classes("ses-02", 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj045", "ses-02", "sub-wlsubj045_ses-02_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_full_full_constant_loss.csv").format(num=n) for n in range(get_n_classes("ses-02", 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj045", "ses-02", "sub-wlsubj045_ses-02_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_absolute_full_vary_loss.csv").format(num=n) for n in range(get_n_classes("ses-02", 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj045", "ses-02", "sub-wlsubj045_ses-02_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_relative_full_vary_loss.csv").format(num=n) for n in range(get_n_classes("ses-02", 'stim_class'))],
        [os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "stim_class", "posterior", "sub-wlsubj045", "ses-02", "sub-wlsubj045_ses-02_task-sfp_v1_e1-12_summary_b1_r1e-3_g1_c{num:02d}_full_full_vary_loss.csv").format(num=n) for n in range(get_n_classes("ses-02", 'stim_class'))],


rule GLMdenoise_all_visual:
    input:
        [os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "all_visual", "posterior",  "{subject}", "{session}", "{subject}_{session}_{task}_results.mat").format(subject=sub, session=ses, task=TASKS[(sub, ses)]) for sub in SUBJECTS for ses in SESSIONS[sub]],


rule plots_modeling_blanks:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_localsf.svg').format(mat_type="stim_class_10_blanks_fixed_hrf_stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary'] for v in VAREAS],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_stim_prop.svg').format(mat_type="stim_class_10_blanks_fixed_hrf_stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary'] for v in VAREAS],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_data.svg').format(mat_type="stim_class_10_blanks_fixed_hrf_stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary'] for v in VAREAS],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_tuning_params.svg').format(mat_type="stim_class_10_blanks_fixed_hrf_stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary'] for v in VAREAS],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_summary_tuning_curves_check_varea={v}.svg').format(mat_type="stim_class_10_blanks_fixed_hrf_stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], vareas=v, eccen='1-12', binning='eccen_bin', v=v) for sub in SUBJECTS for ses in SESSIONS[sub] for v in VAREAS],


rule plots_all:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_localsf.svg').format(mat_type="stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary'] for v in VAREAS],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_stim_prop.svg').format(mat_type="stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary'] for v in VAREAS],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_data.svg').format(mat_type="stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary'] for v in VAREAS],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_tuning_params.svg').format(mat_type="stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary'] for v in VAREAS],
        # [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_full_tuning_curves_check_varea={v}_bootstrap={b:02d}.svg').format(mat_type="stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], vareas=v, eccen='1-12', binning='eccen_bin', v=v, b=b) for sub in SUBJECTS for ses in SESSIONS[sub] for v in VAREAS for b in range(100)],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_summary_tuning_curves_check_varea={v}.svg').format(mat_type="stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], vareas=v, eccen='1-12', binning='eccen_bin', v=v) for sub in SUBJECTS for ses in SESSIONS[sub] for v in VAREAS],


rule tuning_curves_all:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}.csv').format(mat_type="stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary', 'full'] for v in VAREAS],


rule first_level_all:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}.csv').format(mat_type="stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas=v, eccen='1-12', binning='eccen_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary', 'full'] for v in VAREAS],


rule plots_VSS_abstract:
    # these recreate the data examined for the first year talk and the VSS abstract, when I was
    # using the "prior" atlas
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_localsf.svg').format(mat_type="stim_class", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='eccen_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_stim_prop.svg').format(mat_type="stim_class", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='eccen_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_binned', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_data.svg').format(mat_type="stim_class", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='eccen_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_{df_mode}_tuning_params.svg').format(mat_type="stim_class", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='eccen_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary']],
        # [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_full_tuning_curves_check_varea=1_bootstrap={b:02d}.svg').format(mat_type="stim_class", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], vareas='1', eccen='2-8', binning='eccen_bin', b=b) for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for b in range(100)],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{binning}_summary_tuning_curves_check_varea=1.svg').format(mat_type="stim_class", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], vareas='1', eccen='2-8', binning='eccen_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045']]


rule GLMdenoise_all:
    input:
        [os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "stim_class", "posterior",  "{subject}", "{session}", "figures_{task}", "FinalModel_maps.png").format(subject=sub, session=ses, task=TASKS[(sub, ses)]) for sub in SUBJECTS for ses in SESSIONS[sub]],


rule preprocess_all:
    input:
        [os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "{subject}_{session}_{task}_{run}_preproc.nii.gz").format(subject=sub, session=ses, task=TASKS[(sub, ses)], run="run-%02d"%i) for sub in SUBJECTS for ses in SESSIONS[sub] for i in range(1, NRUNS.get((sub, ses), 12)+1)],


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


def get_permuted(wildcards):
    if "permuted" in wildcards.mat_type:
        return "-p"
    else:
        return ""


def get_design_inputs(wildcards):
    if (wildcards.session in ['ses-pilot00', 'ses-pilot01', 'ses-01', 'ses-02'] or
        (wildcards.subject, wildcards.session) in [('sub-wlsubj004', 'ses-03'), ('sub-wlsubj014', 'ses-03')]):
        ext = 'nii'
    else:
        ext = 'nii.gz'
    tsv_files = os.path.join(config["DATA_DIR"], wildcards.subject, wildcards.session, "func",
                             wildcards.subject+"_"+wildcards.session+"_"+wildcards.task+"_run-{n:02d}_events.tsv")
    func_files = os.path.join(config["DATA_DIR"], wildcards.subject, wildcards.session, "func",
                              wildcards.subject+"_"+wildcards.session+"_"+wildcards.task+"_run-{n:02d}_bold."+ext)
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
    if (wildcards.session in ['ses-pilot00', 'ses-pilot01', 'ses-01', 'ses-02'] or
        (wildcards.subject, wildcards.session) in [('sub-wlsubj004', 'ses-03'), ('sub-wlsubj014', 'ses-03')]):
        ext = 'nii'
    else:
        ext = 'nii.gz'
    input_dict['func_files'] = os.path.join(config["DATA_DIR"], "{subject}", "{session}", "func",
                                            "{subject}_{session}_{task}_{run}_bold.{ext}").format(ext=ext, **wildcards)
    return input_dict


rule preprocess:
    input:
        unpack(get_preprocess_inputs)
    output:
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed_{run}_{task}", "{subject}", "{session}", "{subject}_{session}_{task}_{run}_preproc.nii.gz"),
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
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed_{run}_{task}", "{subject}", "{session}", "{subject}_{session}_{task}_{run}_preproc.nii.gz"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "session.json"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "sbref_reg_corrected.nii.gz"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "distort2anat_tkreg.dat"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "distortion_merged_corrected.nii.gz"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "distortion_merged_corrected_mean.nii.gz"),
    output:
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
    if wildcards.atlas_type == 'prior':
        benson_prefix = 'benson14'
    elif wildcards.atlas_type == 'posterior':
        benson_prefix = 'inferred'
    benson_template = os.path.join(config['DATA_DIR'], 'derivatives', 'freesurfer', wildcards.subject.replace('sub-', ''), 'surf', '{hemi}.'+benson_prefix+'_varea.mgz')
    return expand(benson_template, hemi=['lh', 'rh'])


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
        "'preprocessed_reoriented', '{wildcards.mat_type}', jsonInfo.stim_length, "
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
        "'preprocessed_reoriented', '{wildcards.mat_type}', jsonInfo.stim_length, "
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
    if wildcards.atlas_type == 'prior':
        benson_prefix = 'benson14'
    elif wildcards.atlas_type == 'posterior':
        benson_prefix = 'inferred'
        benson_names.append('sigma')
    benson_temp = os.path.join(config['DATA_DIR'], 'derivatives', 'freesurfer', wildcards.subject.replace('sub-', ''), 'surf', '{hemi}.'+benson_prefix+'_{filename}.mgz')
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
        # for some reason, input.benson_paths only includes the first of the benson_paths (but all
        # of them are included in input, so snakemake checks for them correctly)
        benson_template = lambda wildcards, input: input.benson_paths.replace('lh', '%s').replace('angle', '%s'),
        # ... and fortunately the benson_paths are now the only mgz files we use as input, so this
        # check will only catch them.
        benson_names = lambda wildcards, input: [i.split('_')[-1].replace('.mgz', '') for i in input if 'lh' in i],
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
        "--benson_template_path {params.benson_template}"


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
    if wildcards.atlas_type == 'prior':
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
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves_summary", "{mat_type}", "{atlas_type}", "v{vareas}_e{eccen}_{binning}_tuning_curves_{df_mode}.csv")
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
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves_summary", "{mat_type}", "{atlas_type}", "v{vareas}_e{eccen}_{binning}_tuning_curves_summary.csv")
    output:
        os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves_summary', '{mat_type}', '{atlas_type}',
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


rule model:
    input:
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}.csv')
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_{orientation_type}_{eccentricity_type}_{train_amps}_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_{orientation_type}_{eccentricity_type}_{train_amps}_model.pt"),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_{orientation_type}_{eccentricity_type}_{train_amps}_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_{orientation_type}_{eccentricity_type}_{train_amps}-%j.log")
    resources:
        # need the same number of cpus and gpus
        cpus_per_task = lambda wildcards: int(wildcards.gpus),
        mem = 10,
        gpus = lambda wildcards: int(wildcards.gpus)
    params:
        save_stem = lambda wildcards, output: output[0].replace("_loss.csv", ''),
        stimulus_class = lambda wildcards: wildcards.stimulus_class.split(','),
        train_amps = parse_train_amps,
        logging = to_log_or_not,
    shell:
        "python -m sfp.model {wildcards.orientation_type} {wildcards.eccentricity_type} "
        "{params.train_amps} {input} {params.save_stem} -b "
        "{wildcards.batch_size} -r {wildcards.learning_rate} -d "
        "drop_voxels_with_negative_amplitudes,drop_voxels_near_border -t 1e-12 -e 1000 "
        "-c {params.stimulus_class} {params.logging} {log}"


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
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_simulated", "noise-{noise_source}", "{modeling_goal}", "n{num_voxels}_{sim_orientation_type}_{sim_eccentricity_type}_{sim_train_amps}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_{orientation_type}_{eccentricity_type}_{train_amps}_model.pt"),
    benchmark:
        os.path.join(config['DATA_DIR'], 'code', 'tuning_2d_simulated', 'noise-{noise_source}_{modeling_goal}_n{num_voxels}_{sim_orientation_type}_{sim_eccentricity_type}_{sim_train_amps}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_{orientation_type}_{eccentricity_type}_{train_amps}_benchmark.txt')
    log:
        os.path.join(config['DATA_DIR'], 'code', 'tuning_2d_simulated', 'noise-{noise_source}_{modeling_goal}_n{num_voxels}_{sim_orientation_type}_{sim_eccentricity_type}_{sim_train_amps}_s{sigma}_a{sf_ecc_slope}_b{sf_ecc_intercept}_rmc{rel_mode_cardinals}_rmo{rel_mode_obliques}_rac{rel_amplitude_cardinals}_rao{rel_amplitude_obliques}_amc{abs_mode_cardinals}_amo{abs_mode_obliques}_aac{abs_amplitude_cardinals}_aao{abs_amplitude_obliques}_l{noise_level}_b{batch_size}_r{learning_rate}_g{gpus}_c{stimulus_class}_{orientation_type}_{eccentricity_type}_{train_amps}-%j.log')
    resources:
        # need the same number of cpus and gpus
        cpus_per_task = lambda wildcards: int(wildcards.gpus),
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
        "-r {wildcards.learning_rate} -d None -t 1e-12 -e 1000 -c {params.stimulus_class} "
        "{params.logging} {log}"


rule gather_model_results:
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_all_loss.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_all_model.csv"),
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_2d_model", "{mat_type}", "{atlas_type}", "{modeling_goal}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_all_features.csv"),
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_all_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_2d_model", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_{modeling_goal}_v{vareas}_e{eccen}_{df_mode}_b{batch_size}_r{learning_rate}_g{gpus}_all-%j.log")
    resources:
        mem = 100,
    params:
        base_path_template = lambda wildcards, output: output[0].replace("_all_loss.csv", '\*'),
        save_stem = lambda wildcards, output: output[0].replace("_loss.csv", ''),
    shell:
        "python -m sfp.analyze_model {params.base_path_template} {params.save_stem}"


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
