import os

configfile:
    "config.yml"
if not os.path.isdir(config["DATA_DIR"]):
    raise Exception("Cannot find the dataset at %s" % config["DATA_DIR"])
if os.system("module list") == 0:
    # then we're on the cluster
    shell.prefix("module purge; module load anaconda2/4.3.1; source activate sfp; "
                 "module load fsl/5.0.10; module load freesurfer/6.0.0; module load matlab/2017a; ")

SUBJECTS = ['sub-wlsubj001', 'sub-wlsubj042', 'sub-wlsubj045']
SESSIONS = {'sub-wlsubj001': ['ses-pilot01'], 'sub-wlsubj042': ['ses-pilot00', 'ses-pilot01'],
            'sub-wlsubj045': ['ses-pilot01']}
# every sub/ses pair that's not in here, has the full number of runs, 12
NRUNS = {('sub-wlsubj001', 'ses-pilot01'): 9, ('sub-wlsubj042', 'ses-pilot00'): 8}
wildcard_constraints:
    subject="sub-[a-z0-9]+",
    session="ses-[a-z0-9]+",
    run="run-[0-9]+",
    filename='[a-zA-z0-9_]+\.[a-z.]+'

rule preprocess_all:
    input:
        [os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "timeseries_corrected_{run}.nii.gz").format(subject=sub, session=ses, run="run-%02d"%i) for sub in SUBJECTS for ses in SESSIONS[sub] for i in range(1, NRUNS.get((sub, ses), 12)+1)],


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
        seed = lambda wildcards: {'sub-wlsubj001': 1, 'sub-wlsubj042': 2, 'sub-wlsubj045': 3}.get(wildcards.subject)
    shell:
        "python sfp/stimuli.py --subject_name {wildcards.subject} -i -s {params.seed}"

rule preprocess:
    input:
        os.path.join(config["DATA_DIR"], "derivatives", "freesurfer", "{subject}"),
        data_dir = os.path.join(config["DATA_DIR"], "{subject}", "{session}"),
        func_files = os.path.join(config["DATA_DIR"], "{subject}", "{session}", "func", "{subject}"+"_"+"{session}"+"_task-sfp_{run}_bold.nii"),
        freesurfer_dir = os.path.join(config["DATA_DIR"], "derivatives", "freesurfer"),
    output:
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "{run}", "timeseries_corrected_{run}.nii.gz"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "{run}", "session.json"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "{run}", "sbref_reg_corrected.nii.gz"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "{run}", "distort2anat_tkreg.dat"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "{run}", "distortion_merged_corrected.nii.gz"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "{run}", "distortion_merged_corrected_mean.nii.gz"),
    resources:
        cpus_per_task = 10,
        mem = 48
    params:
        plugin = "MultiProc",        
        working_dir = lambda wildcards: "/scratch/wfb229/preprocess/%s_%s_%s" % (wildcards.subject, wildcards.session, wildcards.run),
        plugin_args = lambda wildcards, resources: ",".join("%s:%s" % (k,v) for k,v in {'n_procs': resources.cpus_per_task, 'memory_gb': resources.mem}.items()),
        epi_num = lambda wildcards: int(wildcards.run.replace('run-', '')),
        output_dir = lambda wildcards, output: os.path.dirname(output[0])
    benchmark:
        os.path.join(config["DATA_DIR"], "code", "preprocessed", "{subject}_{session}_{run}_benchmark.txt")
    log:
        os.path.join(config["DATA_DIR"], "code", "preprocessed", "{subject}_{session}_{run}.log")
    shell:
        "export SUBJECTS_DIR={input.freesurfer_dir};"
        "python ~/MRI_tools/preprocessing/prisma_preproc.py -datadir {input.data_dir} -outdir "
        "{params.output_dir} -working_dir {params.working_dir} -plugin {params.plugin} "
        "-dir_structure bids -plugin_args {params.plugin_args} -epis {params.epi_num}"


rule rearrange_preprocess_extras:
    input:
        lambda wildcards: expand(os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", wildcards.subject, wildcards.session, "run-{run:02d}", wildcards.filename), run=range(1, NRUNS.get((wildcards.subject, wildcards.session), 12+1)))
    output:
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "{filename}")
    log:
        os.path.join(config["DATA_DIR"], "code", "preprocessed", "{subject}_{session}_rearrange_extras_{filename}.log")
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
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "{run}", "timeseries_corrected_{run}.nii.gz"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "session.json"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "sbref_reg_corrected.nii.gz"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "distort2anat_tkreg.dat"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "distortion_merged_corrected.nii.gz"),
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "distortion_merged_corrected_mean.nii.gz"),
    output:
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}", "timeseries_corrected_{run}.nii.gz"),
    log:
        os.path.join(config["DATA_DIR"], "code", "preprocessed", "{subject}_{session}_{run}_rearrange.log")
    run:
        import shutil
        import os
        shutil.move(input[0], output[0])
        os.removedirs(os.path.dirname(input[0]))


def find_benchmarks(wildcards):
    (subjects, sessions, runs) = glob_wildcards(os.path.join(config['DATA_DIR'], 'code', wildcards.step, '{subject}_{session}_{run}_benchmark.txt'))
    # for some reason, subjects and sessions have an entry for each run, so we use set below to make sure we only get each unique
    return expand(os.path.join(config['DATA_DIR'], 'code', wildcards.step, '{subject}_{session}_{run}_benchmark.txt'), zip, subject=subjects, session=sessions, run=runs)


def find_logs(wildcards):
    (subjects, sessions, runs) = glob_wildcards(os.path.join(config['DATA_DIR'], 'code', wildcards.step, '{subject}_{session}_{run}.log'))
    return expand(os.path.join(config['DATA_DIR'], 'code', wildcards.step, '{subject}_{session}_{run}.log'), zip, subject=subjects, session=sessions, run=runs)

rule report:
    input:
        benchmarks = find_benchmarks,
        logs = find_logs
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
        benchmarks = pd.concat(benchmarks)
        benchmarks = benchmarks.set_index('file').sort_index().style.render()
        report("""
        Benchmark report for {step}
        ===================================

        The following benchmark reports were generated:

        .. raw:: html
           {benchmarks}

        """, output[0], **input)


rule GLMdenoise:
    input:
        os.path.join(config["DATA_DIR"], "derivatives", "preprocessed", "{subject}", "{session}"),
        GLMdenoise_path = os.path.join(os.path.expanduser('~'), 'matlab-toolboxes', 'GLMdenoise')
    resources:
        cpus_per_task = 8,
        mem = 62
