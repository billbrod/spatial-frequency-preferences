import os
import warnings
from glob import glob

configfile:
    "config.yml"
if not os.path.isdir(config["DATA_DIR"]):
    raise Exception("Cannot find the dataset at %s" % config["DATA_DIR"])
if os.system("module list") == 0:
    # then we're on the cluster
    shell.prefix("module purge; module load anaconda2/4.3.1; source activate sfp; "
                 "module load fsl/5.0.10; module load freesurfer/6.0.0; module load matlab/2017a; "
                 "export SUBJECTS_DIR=%s/derivatives/freesurfer; " % config["DATA_DIR"])
else:
    shell.prefix("export SUBJECTS_DIR=%s/derivatives/freesurfer; " % config["DATA_DIR"])


SUBJECTS = ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045']
SESSIONS = {'sub-wlsubj001': ['ses-pilot01', 'ses-01', 'ses-02'],
            'sub-wlsubj042': ['ses-pilot00', 'ses-pilot01', 'ses-01', 'ses-02'],
            'sub-wlsubj045': ['ses-pilot01', 'ses-01', 'ses-02']}
TASKS = {('sub-wlsubj001', 'ses-pilot01'): 'task-sfp', ('sub-wlsubj001', 'ses-01'): 'task-sfp',
         ('sub-wlsubj001', 'ses-02'): 'task-sfpconstant', 
         ('sub-wlsubj042', 'ses-pilot00'): 'task-sfp', ('sub-wlsubj042', 'ses-pilot01'): 'task-sfp',
         ('sub-wlsubj042', 'ses-01'): 'task-sfpconstant', ('sub-wlsubj042', 'ses-02'): 'task-sfp',
         ('sub-wlsubj045', 'ses-pilot01'): 'task-sfp',
         ('sub-wlsubj045', 'ses-01'): 'task-sfpconstant',  ('sub-wlsubj045', 'ses-02'): 'task-sfp'}
# every sub/ses pair that's not in here has the full number of runs, 12
NRUNS = {('sub-wlsubj001', 'ses-pilot01'): 9, ('sub-wlsubj042', 'ses-pilot00'): 8}
def get_n_classes(session, mat_type):
    if mat_type == 'all_visual':
        return 1
    else:
        n = {'ses-pilot00': 52, 'ses-pilot01': 52, 'ses-01': 48, 'ses-02': 48}[session]
        if 'blanks' in mat_type:
            n += 1
        return n
def get_stim_files(wildcards):
    if 'pilot00' in wildcards.session:
        stim_prefix = 'pilot00_'
    elif 'pilot01' in wildcards.session:
        stim_prefix = 'pilot01_'
    else:
        if 'constant' in wildcards.task:
            stim_prefix = 'constant_'
        else:
            stim_prefix = ''
    file_stem = os.path.join(config['DATA_DIR'], 'stimuli', stim_prefix+"unshuffled{rest}")
    return {'stim': file_stem.format(rest='.npy'),
            'desc_csv': file_stem.format(rest='_stim_description.csv')}
SUB_SEEDS = {'sub-wlsubj001': 1, 'sub-wlsubj042': 2, 'sub-wlsubj045': 3}
SES_SEEDS = {'ses-pilot00': 10, 'ses-pilot01': 20, 'ses-01': 30, 'ses-02': 40}
wildcard_constraints:
    subject="sub-[a-z0-9]+",
    session="ses-[a-z0-9]+",
    run="run-[0-9]+",
    filename_ext='[a-zA-Z0-9_]+\.[a-z.]+',
    filename='[a-zA-Z0-9_]+',
    task="task-[a-z0-9]+",
    varea="[0-9-]+",
    eccen="[0-9]+-[0-9]+",
    df_mode="summary|full"


rule plots_modeling_blanks:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}_localsf.svg').format(mat_type="stim_class_5_blanks", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas='1-2-3', eccen='1-12', binning='_eccen_bin_hemi_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}_stim_prop.svg').format(mat_type="stim_class_5_blanks", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas='1-2-3', eccen='1-12', binning='_eccen_bin_hemi_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}_data.svg').format(mat_type="stim_class_5_blanks", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas='1-2-3', eccen='1-12', binning='_eccen_bin_hemi_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}_tuning_params.svg').format(mat_type="stim_class_5_blanks", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas='1-2-3', eccen='1-12', binning='_eccen_bin_hemi_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_summary_tuning_curves_check_varea={v}.svg').format(mat_type="stim_class_5_blanks", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], vareas='1-2-3', eccen='1-12', binning='_eccen_bin_hemi_bin', v=v) for sub in SUBJECTS for ses in SESSIONS[sub] for v in [1, 2, 3]],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}_localsf.svg').format(mat_type="stim_class_5_blanks", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='_eccen_bin_hemi_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}_stim_prop.svg').format(mat_type="stim_class_5_blanks", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='_eccen_bin_hemi_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}_data.svg').format(mat_type="stim_class_5_blanks", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='_eccen_bin_hemi_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}_tuning_params.svg').format(mat_type="stim_class_5_blanks", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='_eccen_bin_hemi_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_summary_tuning_curves_check_varea=1.svg').format(mat_type="stim_class_5_blanks", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], vareas='1', eccen='2-8', binning='_eccen_bin_hemi_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045']]


rule plots_all:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}_localsf.svg').format(mat_type="stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas='1-2-3', eccen='1-12', binning='_eccen_bin_hemi_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}_stim_prop.svg').format(mat_type="stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas='1-2-3', eccen='1-12', binning='_eccen_bin_hemi_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}_data.svg').format(mat_type="stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas='1-2-3', eccen='1-12', binning='_eccen_bin_hemi_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}_tuning_params.svg').format(mat_type="stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas='1-2-3', eccen='1-12', binning='_eccen_bin_hemi_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary']],
        # [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_full_tuning_curves_check_varea={v}_bootstrap={b:02d}.svg').format(mat_type="stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], vareas='1-2-3', eccen='1-12', binning='_eccen_bin_hemi_bin', v=v, b=b) for sub in SUBJECTS for ses in SESSIONS[sub] for v in [1, 2, 3] for b in range(100)],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_summary_tuning_curves_check_varea={v}.svg').format(mat_type="stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], vareas='1-2-3', eccen='1-12', binning='_eccen_bin_hemi_bin', v=v) for sub in SUBJECTS for ses in SESSIONS[sub] for v in [1, 2, 3]],


rule tuning_curves_all:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}.csv').format(mat_type="stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas='1-2-3', eccen='1-12', binning='_eccen_bin_hemi_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary', 'full']],


rule first_level_all:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}.csv').format(mat_type="stim_class", atlas_type='posterior', subject=sub, session=ses, task=TASKS[(sub, ses)], df_mode=dfm, vareas='1-2-3', eccen='1-12', binning='_eccen_bin_hemi_bin') for sub in SUBJECTS for ses in SESSIONS[sub] for dfm in ['summary', 'full']],


rule plots_VSS_abstract:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}_localsf.svg').format(mat_type="stim_class", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='_eccen_bin_hemi_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}_stim_prop.svg').format(mat_type="stim_class", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='_eccen_bin_hemi_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}_data.svg').format(mat_type="stim_class", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='_eccen_bin_hemi_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary']],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}_tuning_params.svg').format(mat_type="stim_class", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='_eccen_bin_hemi_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary']],
        # [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_full_tuning_curves_check_varea=1_bootstrap={b:02d}.svg').format(mat_type="stim_class", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], vareas='1', eccen='2-8', binning='_eccen_bin_hemi_bin', b=b) for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for b in range(100)],
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_summary_tuning_curves_check_varea=1.svg').format(mat_type="stim_class", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], vareas='1', eccen='2-8', binning='_eccen_bin_hemi_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045']]


rule tuning_curves_VSS_abstract:
    input:
        [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}.csv').format(mat_type="stim_class", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='_eccen_bin_hemi_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary', 'full']]


rule VSS_abstract_data:
    input:
        # these recreate the data we looked at for first year talk and VSS abstract
        [os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}.csv').format(mat_type="stim_class", atlas_type='prior', subject=sub, session='ses-pilot01', task=TASKS[(sub, 'ses-pilot01')], df_mode=dfm, vareas='1', eccen='2-8', binning='_eccen_bin_hemi_bin') for sub in ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045'] for dfm in ['summary', 'full']]


rule GLMdenoise_all:
    input:
        [os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise_reoriented", "{mat_type}",  "{subject}", "{session}", "{subject}_{session}_{task}_modelmd.nii.gz").format(subject=sub, session=ses, task=TASKS[(sub, ses)], mat_type='stim_class') for sub in SUBJECTS for ses in SESSIONS[sub]],
        [os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise_reoriented", "{mat_type}",  "{subject}", "{session}", "{subject}_{session}_{task}_modelse.nii.gz").format(subject=sub, session=ses, task=TASKS[(sub, ses)], mat_type='stim_class') for sub in SUBJECTS for ses in SESSIONS[sub]],
        [os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise_reoriented", "{mat_type}",  "{subject}", "{session}", "{subject}_{session}_{task}_R2.nii.gz").format(subject=sub, session=ses, task=TASKS[(sub, ses)], mat_type='stim_class') for sub in SUBJECTS for ses in SESSIONS[sub]],
        [os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise_reoriented", "{mat_type}",  "{subject}", "{session}", "{subject}_{session}_{task}_R2run.nii.gz").format(subject=sub, session=ses, task=TASKS[(sub, ses)], mat_type='stim_class') for sub in SUBJECTS for ses in SESSIONS[sub]],
        [os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise_reoriented", "{mat_type}",  "{subject}", "{session}", "{subject}_{session}_{task}_models_class_{n}.nii.gz").format(subject=sub, session=ses, task=TASKS[(sub, ses)], mat_type='stim_class', n=n) for sub in SUBJECTS for ses in SESSIONS[sub] for n in range(get_n_classes(ses, 'stim_class'))],


rule preprocess_all:
    input:
        [os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "{subject}_{session}_{task}_{run}_preproc.nii.gz").format(subject=sub, session=ses, task=TASKS[(sub, ses)], run="run-%02d"%i) for sub in SUBJECTS for ses in SESSIONS[sub] for i in range(1, NRUNS.get((sub, ses), 12)+1)],


rule stimuli:
    output:
        "data/stimuli/unshuffled.npy",
        "data/stimuli/unshuffled_stim_description.csv",
        "data/stimuli/constant_unshuffled.npy",
        "data/stimuli/constant_unshuffled_stim_description.csv"
    shell:
        "python sfp/stimuli.py -c"


rule stimuli_idx:
    output:
        ["data/stimuli/{subject}_run%02d_idx.npy" % i for i in range(12)]
    params:
        seed = lambda wildcards: SUB_SEEDS[wildcards.subject]
    shell:
        "python sfp/stimuli.py --subject_name {wildcards.subject} -i -s {params.seed}"


rule preprocess:
    input:
        os.path.join(config["DATA_DIR"], "derivatives", "freesurfer", "{subject}"),
        func_files = os.path.join(config["DATA_DIR"], "{subject}", "{session}", "func", "{subject}_{session}_{task}_{run}_bold.nii"),
        freesurfer_data = os.path.join(config["DATA_DIR"], "derivatives", "freesurfer"),
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
        working_dir = lambda wildcards: "/scratch/wfb229/preprocess/%s_%s_%s" % (wildcards.subject, wildcards.session, wildcards.run),
        plugin_args = lambda wildcards, resources: ",".join("%s:%s" % (k,v) for k,v in {'n_procs': resources.cpus_per_task, 'memory_gb': resources.mem}.items()),
        epi_num = lambda wildcards: int(wildcards.run.replace('run-', '')),
        script_location = os.path.join(config["MRI_TOOLS"], "preprocessing", "prisma_preproc.py")
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "preprocessed", "{subject}_{session}_{task}_{run}_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "preprocessed", "{subject}_{session}_{task}_{run}.log")
    shell:
        "python {params.script_location} -datadir {params.data_dir} -working_dir "
        "{params.working_dir} -plugin {params.plugin} -dir_structure bids -plugin_args "
        "{params.plugin_args} -epis {params.epi_num} -bids_derivative_name "
        "preprocessed_{wildcards.run}_{wildcards.task}"


rule rearrange_preprocess_extras:
    input:
        lambda wildcards: expand(os.path.join(config["DATA_DIR"], "derivatives", "preprocessed_run-{n:02d}_{task}", wildcards.subject, wildcards.session, wildcards.filename_ext), task=TASKS[(wildcards.subject, wildcards.session)], n=range(1, NRUNS.get((wildcards.subject, wildcards.session), 12)+1))
    output:
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "{filename_ext}")
    log:
        os.path.join(config["DATA_DIR"], "code", "preprocessed", "{subject}_{session}_rearrange_extras_{filename_ext}.log")
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
        os.path.join(config["DATA_DIR"], "code", "preprocessed", "{subject}_{session}_{run}_rearrange.log")
    run:
        import shutil
        import os
        shutil.move(input[0], output[0])
        os.removedirs(os.path.dirname(input[0]))


def get_permuted(wildcards):
    if "permuted" in wildcards.mat_type:
        return "-p"
    else:
        return ""


def get_design_inputs(wildcards):
    tsv_files = os.path.join(config["DATA_DIR"], wildcards.subject, wildcards.session, "func", wildcards.subject+"_"+wildcards.session+"_"+wildcards.task+"_run-{n:02d}_events.tsv")
    func_files = os.path.join(config["DATA_DIR"], wildcards.subject, wildcards.session, "func", wildcards.subject+"_"+wildcards.session+"_"+wildcards.task+"_run-{n:02d}_bold.nii")
    return {'tsv_files': expand(tsv_files, n=range(1, NRUNS.get((wildcards.subject, wildcards.session), 12)+1)),
            'func_files': expand(func_files, n=range(1, NRUNS.get((wildcards.subject, wildcards.session), 12)+1))}


rule create_design_matrices:
    input:
        unpack(get_design_inputs),
        data_dir = os.path.join(config["DATA_DIR"], "{subject}", "{session}"),
    output:
        os.path.join(config["DATA_DIR"], "derivatives", "design_matrices", "{mat_type}", "{subject}", "{session}", "{subject}_{session}_{task}_params.json")
    log:
        os.path.join(config["DATA_DIR"], "code", "design_matrices", "{subject}_{session}_{mat_type}.log")
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "design_matrices", "{subject}_{session}_{mat_type}_benchmark.txt")
    params:
        save_path = lambda wildcards, output: output[0].replace('params.json', 'run-%s_design_matrix.tsv'),
        permuted_flag = get_permuted,
        mat_type = lambda wildcards: wildcards.mat_type.replace("_permuted", "")
    shell:
        "python sfp/design_matrices.py {input.data_dir} --mat_type {params.mat_type} --save_path "
        "{params.save_path} {params.permuted_flag}"


rule GLMdenoise:
    input:
        preproc_files = lambda wildcards: expand(os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", wildcards.subject, wildcards.session, wildcards.subject+"_"+wildcards.session+"_"+wildcards.task+"_run-{n:02d}_preproc.nii.gz"), n=range(1, NRUNS.get((wildcards.subject, wildcards.session), 12)+1)),
        params_file = os.path.join(config["DATA_DIR"], "derivatives", "design_matrices", "{mat_type}", "{subject}", "{session}", "{subject}_{session}_{task}_params.json"),
        GLMdenoise_path = os.path.join(os.path.expanduser('~'), 'matlab-toolboxes', 'GLMdenoise')
    output:
        GLM_results_md = os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}",  "{subject}", "{session}", "{subject}_{session}_{task}_modelmd.nii.gz"),
        GLM_results_se = os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}",  "{subject}", "{session}", "{subject}_{session}_{task}_modelse.nii.gz"),
        GLM_results_r2 = os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}",  "{subject}", "{session}", "{subject}_{session}_{task}_R2.nii.gz"),
        GLM_results_r2run = os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}",  "{subject}", "{session}", "{subject}_{session}_{task}_R2run.nii.gz"),
        GLM_results = protected(os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}",  "{subject}", "{session}", "{subject}_{session}_{task}_results.mat")),
        GLM_results_detrended = protected(os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}",  "{subject}", "{session}", "{subject}_{session}_{task}_denoiseddata.mat"))
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "GLMdenoise", "{subject}_{session}_{task}_{mat_type}_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "GLMdenoise", "{subject}_{session}_{task}_{mat_type}.log")
    params:
        output_dir = lambda wildcards, output: os.path.dirname(output.GLM_results_md),
        save_stem = lambda wildcards: "{subject}_{session}_{task}_".format(**wildcards),
        design_matrix_template = lambda wildcards, input: input.params_file.replace('params.json', 'run-%02d_design_matrix.tsv'),
        preproc_file_template = lambda wildcards, input: input.preproc_files[0].replace('run-01', 'run-%02d'),
        runs = lambda wildcards: ",".join(str(i) for i in range(1, NRUNS.get((wildcards.subject, wildcards.session), 12)+1)),
        seed = lambda wildcards: SUB_SEEDS[wildcards.subject] + SES_SEEDS[wildcards.session],
        freesurfer_matlab_dir = os.path.join(config['FREESURFER_DIR'], 'matlab')
    resources:
        cpus_per_task = 1,
        mem = 100
    shell:
        "cd matlab; matlab -nodesktop -nodisplay -r \"runGLM('{params.design_matrix_template}', "
        "'{params.preproc_file_template}', [{params.runs}], [{params.runs}], '{input.params_file}',"
        "'{params.freesurfer_matlab_dir}', '{input.GLMdenoise_path}', {params.seed}, "
        "'{params.output_dir}', '{params.save_stem}'); quit;\""


rule save_results_niftis:
    input:
        GLM_results = os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}",  "{subject}", "{session}", "{subject}_{session}_{task}_results.mat"),
        preproc_example_file = os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "{subject}_{session}_{task}_run-01_preproc.nii.gz")
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}",  "{subject}", "{session}", "{subject}_{session}_{task}_models_class_{n}.nii.gz")
    params:
        freesurfer_matlab_dir = os.path.join(config['FREESURFER_DIR'], 'matlab'),
        output_dir = lambda wildcards, output: os.path.dirname(output[0]),
        save_stem = lambda wildcards: "{subject}_{session}_{task}_".format(**wildcards),
        saveN = lambda wildcards: int(wildcards.n)+1
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "save_results_niftis", "{subject}_{session}_{task}_{mat_type}_models_class_{n}_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "save_results_niftis", "{subject}_{session}_{task}_{mat_type}_models_class_{n}.log")
    resources:
        mem = 100,
        cpus_per_task = 1
    shell:
        "cd matlab; matlab -nodesktop -nodisplay -r \"saveout({params.saveN}, '{input.GLM_results}'"
        ", '{input.preproc_example_file}', '{params.output_dir}', '{params.save_stem}', "
        "'{params.freesurfer_matlab_dir}'); quit;\""


rule to_freesurfer:
    input:
        in_file = os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise", "{mat_type}",  "{subject}", "{session}", "{subject}_{session}_{task}_{filename}.nii.gz"),
        tkreg = os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "distort2anat_tkreg.dat"),
        freesurfer_data = os.path.join(config["DATA_DIR"], "derivatives", "freesurfer"),
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise_reoriented", "{mat_type}",  "{subject}", "{session}", "{subject}_{session}_{task}_{filename}.nii.gz"),
        os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise_reoriented", "{mat_type}",  "{subject}", "{session}", "lh.{subject}_{session}_{task}_{filename}.mgz"),
        os.path.join(config['DATA_DIR'], "derivatives", "GLMdenoise_reoriented", "{mat_type}",  "{subject}", "{session}", "rh.{subject}_{session}_{task}_{filename}.mgz")
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "to_freesurfer", "{subject}_{session}_{task}_{mat_type}_{filename}_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "to_freesurfer", "{subject}_{session}_{task}_{mat_type}_{filename}.log")
    params:
        output_dir = lambda wildcards, output: os.path.dirname(output[0]),
        script_location = os.path.join(config["MRI_TOOLS"], "preprocessing", "to_freesurfer.py")
    shell:
        "python {params.script_location} -v -s -o {params.output_dir} {input.tkreg} {input.in_file}"
        

def get_first_level_analysis_input(wildcards):
    files = os.path.join(config["DATA_DIR"], "derivatives", "GLMdenoise_reoriented", wildcards.mat_type, wildcards.subject, wildcards.session, "{hemi}."+wildcards.subject+"_"+wildcards.session+"_"+wildcards.task+"_{filename}.mgz")
    input_dict = {}
    input_dict['R2_files'] = expand(files, hemi=['lh', 'rh'], filename=['R2'])
    if wildcards.df_mode == 'summary':
        input_dict['GLM_results'] = expand(files, hemi=['lh', 'rh'], filename=['modelmd', 'modelse'])
    elif wildcards.df_mode == 'full':
        class_num = range(get_n_classes(wildcards.session, wildcards.mat_type))
        models_names = ['models_class_%02d' % i for i in class_num]
        input_dict['GLM_results'] = expand(files, hemi=['lh', 'rh'], filename=models_names)
    benson_names = ['angle', 'eccen', 'varea']
    if wildcards.atlas_type == 'prior':
        benson_prefix = 'benson14'
    elif wildcards.atlas_type == 'posterior':
        benson_prefix = 'inferred'
        benson_names.append('sigma')
    benson_temp = os.path.join(config['DATA_DIR'], 'derivatives', 'freesurfer', wildcards.subject, 'surf', '{hemi}.'+benson_prefix+'_{filename}.mgz')
    input_dict['benson_paths'] = expand(benson_temp, hemi=['lh', 'rh'], filename=benson_names)
    return input_dict


def get_binning(wildcards):
    bin_str = ""
    if "eccen_bin" in wildcards.binning:
        bin_str += "--eccen_bin "
    if "hemi_bin" in wildcards.binning:
        bin_str += "--hemi_bin"
    if not bin_str:
        warnings.warn("We are NOT binning anything! Note this can take a while.")
    return bin_str


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
        os.path.join(config['DATA_DIR'], 'derivatives', 'first_level_analysis', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}.csv')
    resources:
        cpus_per_task = 1,
        mem = 10
    params:
        save_stem = lambda wildcards: "{subject}_{session}_{task}_".format(**wildcards),
        save_dir = lambda wildcards, output: os.path.dirname(output[0]),
        vareas = lambda wildcards: wildcards.vareas.split('-'),
        eccen = lambda wildcards: wildcards.eccen.split('-'),
        bin_str = get_binning,
        results_template = lambda wildcards, input: input.R2_files[0].replace('lh', '%s').replace('R2', '%s'),
        benson_template = lambda wildcards, input: input.benson_paths[0].replace('lh', '%s').replace('angle', '%s'),
        class_num = lambda wildcards: get_n_classes(wildcards.session, wildcards.mat_type),
        stim_type = get_stim_type,
        mid_val = lambda wildcards: {'ses-pilot01': 127, 'ses-pilot00': 127}.get(wildcards.session, 128)
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "first_level_analysis", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}{binning}_{df_mode}_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "first_level_analysis", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}{binning}_{df_mode}.log")
    shell:
        "python sfp/first_level_analysis.py --save_dir {params.save_dir} --vareas {params.vareas} "
        "--df_mode {wildcards.df_mode} --eccen_range {params.eccen} {params.bin_str} "
        "--unshuffled_stim_descriptions_path {input.desc_csv} --unshuffled_stim_path {input.stim} "
        "--save_stem {params.save_stem} --class_nums {params.class_num} --stim_type "
        "{params.stim_type} --mid_val {params.mid_val} {params.results_template} "
        "{params.benson_template}"


rule tuning_curves:
    input:
        os.path.join(config['DATA_DIR'], "derivatives", "first_level_analysis", "{mat_type}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}.csv")
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "{mat_type}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}.csv")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_curves", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}{binning}_{df_mode}_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_curves", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}{binning}_{df_mode}.log")
    shell:
        "python sfp/tuning_curves.py {input} {output}"


rule plots:
    input:
        dataframe=os.path.join(config['DATA_DIR'], "derivatives", "{step}", "{mat_type}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}.csv")
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "{step}", "{mat_type}", "{atlas_type}", "{subject}", "{session}", "{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_{df_mode}_{plot_name}.svg")
    params:
        stim_dir = os.path.join(config['DATA_DIR'], 'stimuli')
    resources:
        mem = 1
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "plots", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}{binning}_{df_mode}_{plot_name}_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "plots", "{subject}_{session}_{task}_{mat_type}_{atlas_type}_v{vareas}_e{eccen}{binning}_{df_mode}_{plot_name}.log")
    shell:
        "python sfp/plotting.py {input.dataframe} {params.stim_dir} --plot_to_make "
        "{wildcards.plot_name}"


def get_tuning_curves(wildcards):
    if wildcards.atlas_type == 'prior':
        subjects = ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045']
        sessions = {'sub-wlsubj001': ['ses-pilot01'], 'sub-wlsubj042': ['ses-pilot01'],
                    'sub-wlsubj045': ['ses-pilot01']}
    else:
        subjects = SUBJECTS
        sessions = SESSIONS
    return [os.path.join(config['DATA_DIR'], 'derivatives', 'tuning_curves', '{mat_type}', '{atlas_type}', '{subject}', '{session}', '{subject}_{session}_{task}_v{vareas}_e{eccen}{binning}_summary.csv').format(mat_type=wildcards.mat_type, atlas_type=wildcards.atlas_type, subject=sub, session=ses, task=TASKS[(sub, ses)], vareas=wildcards.vareas, eccen=wildcards.eccen, binning=wildcards.binning) for sub in subjects for ses in sessions[sub]]


rule tuning_curves_summary:
    input:
        get_tuning_curves
    output:
        os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves_summary", "{mat_type}", "{atlas_type}", "v{vareas}_e{eccen}{binning}_tuning_curves_summary.csv")
    params:
        input_dir = os.path.join(config['DATA_DIR'], "derivatives", "tuning_curves", "{mat_type}", "{atlas_type}")
    benchmark:
        os.path.join(config['DATA_DIR'], "code", "tuning_curves_summary", "{mat_type}_{atlas_type}_summary_benchmark.txt")
    log:
        os.path.join(config['DATA_DIR'], "code", "tuning_curves_summary", "{mat_type}_{atlas_type}_summary.log")
    shell:
        "python sfp/summarize_tuning_curves.py {params.input_dir} {output}"


rule report:
    input:
        benchmarks = lambda wildcards: glob(os.path.join(config['DATA_DIR'], 'code', wildcards.step, 'sub*_benchmark.txt')),
        logs = lambda wildcards: glob(os.path.join(config['DATA_DIR'], 'code', wildcards.step, 'sub*.log'))
    output:
        os.path.join(config['DATA_DIR'], 'code', "{step}", "{step}_report.html")
    log:
        os.path.join(config["DATA_DIR"], "code", "{step}", "report.log")
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
