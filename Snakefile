import os

configfile:
    "config.yml"
if not os.path.isdir(config["DATA_DIR"]):
    raise Exception("Cannot find the dataset at %s" % config["DATA_DIR"])
if os.system("module list") == 0:
    # then we're on the cluster
    shell.prefix("module purge; module load anaconda2/4.3.1; module load fsl/5.0.10; "
                 "module load freesurfer/6.0.0; module load matlab/2017a; ")

SUBJECTS = ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045']
SESSIONS = {'sub-wlsubj001': ['ses-pilot01'], 'sub-wlsubj042': ['ses-pilot00', 'ses-pilot01'],
            'sub-wlsubj045': ['ses-pilot01']}

rule all:
    input:
        [os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}").format(subject=sub, session=ses) for sub in SUBJECTS for ses in SESSIONS[sub]]


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
        seed = lambda wildcards: {'wl_subj001': 1, 'wl_subj042': 2, 'wl_subj045': 3}.get(wildcards.subject)
    shell:
        "python sfp/stimuli.py {wildcards.subject} -i -s {params.seed}"

rule preprocess:
    input:
        os.path.join(config["DATA_DIR"], "derivatives", "freesurfer", "{subject}"),
        data_dir = os.path.join(config["DATA_DIR"], "{subject}", "{session}"),
        freesurfer_dir = os.path.join(config["DATA_DIR"], "derivatives", "freesurfer"),
    output:
        output_dir = os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}")
    resources:
        cpus_per_task = 10,
        mem = 48
    params:
        sbref = 1,
        epis = lambda wildcards:
        {('sub-wlsubj001', 'ses-pilot01'): [1, 2, 3, 4, 5, 6, 7, 8, 9],
         ('sub-wlsubj042', 'ses-pilot00'): [1, 2, 3, 4, 5, 6, 7, 8],
         ('sub-wlsubj042', 'ses-pilot01'): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
         ('sub-wlsubj045', 'ses-pilot01'): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}.get((wildcards.subject, wildcards.session)),
        distortPE = 'PA',
        distortrevPE = 'AP',
        plugin = "MultiProc",        
        working_dir = lambda wildcards: "/scratch/wfb229/preproc_%s_%s" % (wildcards.subject, wildcards.session),
        PEdim = 'y',
        plugin_args = lambda wildcards, resources: ",".join("%s:%s" % (k,v) for k,v in {'n_procs': resources.cpus_per_task, 'memory_gb': resources.mem}.items())
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "preprocessed", "{subject}_{session}_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "preprocessed", "{subject}_{session}.log")
    shell:
        "export SUBJECTS_DIR={input.freesurfer_dir};"
        "python ~/MRI_tools/preprocessing/prisma_preproc.py -subject {wildcards.subject} -datadir "
        "{input.data_dir} -outdir {output.output_dir} -epis {params.epis} -sbref {params.sbref} "
        "-distortPE {params.distortPE} -distortrevPE {params.distortrevPE} -working_dir "
        "{params.working_dir} -PEdim {params.PEdim} -plugin {params.plugin} -dir_structure bids "
        "-plugin_args {params.plugin_args}"

rule GLMdenoise:
    input:
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}"),
        GLMdenoise_path = os.path.join(os.path.expanduser('~'), 'matlab-toolboxes', 'GLMdenoise')
    resources:
        cpus_per_task = 8,
        mem = 62
