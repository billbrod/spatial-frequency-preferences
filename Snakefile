import os
# the directory that contains the data (in BIDS format)
DATA_DIR="/scratch/wfb229/spatial_frequency_preferences"

rule stimuli:
    output:
        "data/stimuli/unshuffled.npy",
        "data/stimuli/unshuffled_stim_description.csv"
    shell:
        "python sfp/stimuli.py -c"

rule stimuli_idx:
    output:
        ["data/stimuli/{subject}_run%02d_idx.npy" % i for i in range(12)]
    params:
        seeds_dict = {'wl_subj001': 1, 'wl_subj042': 2, 'wl_subj045': 3}
        seed = lambda wildcards: SUBJECTS_SEEDS.get(wildcards.subject)
    shell:
        "python sfp/stimuli.py {wildcards.subject} -i -s {params.seed}"

# I was thinking I should do something like this
# https://groups.google.com/forum/#!topic/snakemake/e0XNmXqL7Bg in order to be able to run things
# both on and off cluster, but now I'm thinking maybe I make the things that should be run on the
# cluster always call `module` so they fail if run locally. that would work for me at any rate.

rule preprocess:
    input:
        data_dir = os.path.join(DATA_DIR, "{subject}", "{session}"),
        freesurfer_dir = os.path.join(DATA_DIR, "derivatives", "freesurfer"),
        os.path.join(DATA_DIR, "derivatives", "freesurfer", "{subject}"),
    output:
        output_dir = os.path.join(DATA_DIR, "derivatives", "preprocessed", "{subject}", "{session}"),
    params:
        # NEXT: get this working and then re-run preprocessing using it.
