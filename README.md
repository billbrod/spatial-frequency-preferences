Spatial frequency preferences
==============================

[![Build Status](https://travis-ci.com/billbrod/spatial-frequency-preferences.svg?branch=master)](https://travis-ci.com/billbrod/spatial-frequency-preferences)

An fMRI experiment to determine the relationship between spatial
frequency and eccentricity in the human early visual cortex.

Much of the structure is based on the cookiecutter-data-science and so
currently empty / unused.

# Requirements

Matlab:

 - Version at or more recent than 2016b (because we need `jsondecode`)
 - [GLMdenoise](https://github.com/kendrickkay/GLMdenoise/)
 
Python: 

 - use the included conda environment files:
   - in order to run the experiment, you'll need to create a psychopy
     environment: `conda env create -f environment-psychopy.yml`. By
     typing `conda activate psypy`, you'll then be able to run the
     experiment.
   - if you only want to analyze / investigate the data, you don't
     need to set up a psychopy environment, just run: `conda env
     create -f environment.yml`. Then, type `conda activate sfp` to
     activate the environment.
 - You'll also need to install pytorch, but the exact way to do so
   depends on your machine. See the [pytorch
   website](https://pytorch.org/) for details, but the command is
   probably `conda install pytorch torchvision -c pytorch` if you have
   a GPU on your machine and `conda install pytorch-cpu
   torchvision-cpu -c pytorch` if you don't (if in doubt, use the
   second command).
 - [WinawerLab's MRI_tools](https://github.com/WinawerLab/MRI_tools)
   (which requires [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/))

Other: 

 - [FreeSurfer](http://freesurfer.net/)

# Overview

The analysis for this project takes place over several step and makes
use of Python, Matlab, and command-line tools. The notebooks folder
contains several Jupyter notebooks which (hopefully) walk through the
logic of the experiment and analysis.

The Snakefile can perform all of the analysis steps (i.e., from 3 on),
making sure that all the requirements of each step are met. The
following is an overview:

1. Create the stimuli (`python -m sfp.stimuli subject_name -c
   -i`). After you run this the first time (and thus create the
   unshuffled stimuli), you probably only need the `-i` flag to create
   the index. This can also be done using the `stimuli` and
   `stimuli_idx` rules in the Snakefile.
2. Run the experiment and gather fMRI data (`python sfp/experiment.py
   data/stimuli/unshuffled.npy 12 subject_name`). Each run will last 4
   minutes 24 seconds (48 stimulus classes and 10 blank trials, each
   for 4 seconds, gives you 3 minutes 52 seconds, and then each run
   starts and ends with 16 seconds of blank screen).
3. Pre-process your fMRI data
   (using
   [WinawerLab's MRI_tools](https://github.com/WinawerLab/MRI_tools)). This
   is accomplished by the `preprocess`, `rearrange_preprocess_extras`,
   and `rearrange_preprocess` rules in the Snakefile.
4. Create design matrices for each run This is done by the
   `create_design_matrices` rule
5. Run GLMdenoise (`runGLM.m`) and save out the nifti outputs, done by
   the `GLMdenoise` and `save_results_niftis` rules.
6. Align to freesurfer anatomy and get into the mgz format, done by
   the `to_freesurfer` rule, which uses the `to_freesurfer.py` script
   found in
   the
   [WinawerLab's MRI_tools](https://github.com/WinawerLab/MRI_tools)
7. Arrange the outputs into a pandas dataframe for ease of further
   analysis. This is done using the `first_level_analysis` rule.
8. Construct tuning curves, using the `tuning_curves` rule.
    - Note that for this to work, I currently require Noah Benson's
      retinotopy templates. These can be created by running
      this
      [Docker image](https://hub.docker.com/r/nben/occipital_atlas/)
      (the website contains usage and help information). Generally, we
      just need retinotopy information (specifically, for each vertex
      on a Freesurfer flattened cortex, what are the angle and
      eccentricity of its population receptive field in the visual
      field and what is it visual area?), so we could pretty easily
      extend this to use other sources for that.
9. Collect tuning curves across subjects and scanning sessions, to
   compare. This is done using the `tuning_curves_summary` rule.

Note that several of these steps (preprocessing and running
GLMdenoise) should be run on a cluster and will take way too long or
too much memory to run on laptop.

When running python scripts from the command line, they should be run
from this directory (and so you should call `python -m sfp.foo`), in
order for the default paths to work. You can run them from somewhere
else, but then you'll need to use the optional arguments to set the
paths yourself.

`sfp/transfer_to_BIDS.py` is a file used to get the data from the
format it came off the scanner into the BIDS structure. As such, you
will not need it.

# Note on running the experiment

All the steps above except for step 2, running the experiment, should
be done using the `sfp` environment and using either `snakemake` or
`python -m sfp.foo` syntax. Running the experiment should be done
using the `psypy` environment and `python sfp/experiment.py`
syntax. The difference is that `experiment.py` is a separate script
which doesn't rely on any of the other scripts in the `sfp` module
and, because the `psypy` enviroment doesn't contain their
dependencies, will actually fail if you try to import them (which is
what the `python -m sfp.foo` syntax does). The other scripts rely on
each other and share their dependencies, so the `python -m sfp.foo`
syntax is necessary (they will fail if you try to do `python
sfp/stimuli.py`, for example).

# Changes to experiment / stimuli

Things that have been constant: stimuli are presented 300 msec on, 200
msec off, with a stream of digits at fixation to use for a 1-back
distractor task. 10 blank "classes" (4 seconds in length) are randomly
interspersed on each run. 12 runs are gathered (or attempted to, first
two sessions ran out of time because of on-the-fly bug-fixing).

1. ses-pilot00 (git commit b88434d6af8cdc92fb741c99954fae05af02f651,
   Aug 23, 2017): negative spiral stimuli had `w_r < 0`, digits for
   the distractor task were shown every stimulus (500 msec). Only
   sub-wlsubj042 scanned using this protocol. Parameters `w_r` and
   `w_a` ran from `2^2.5` to `2^7.5`, with values every half-octave
   (rounded to the nearest integer), for each of the four main
   stimulus classes. 
2. ses-pilot01 (git commit 2ab9d11c8ba077f53997dea5e525c53ef9c0dd64,
   Oct 9, 2017): negative spiral stimuli switched to have `w_a < 0`,
   digits are now shown every other stimulus (every
   second). sub-wlsubj001, sub-wlsubj042, and sub-wlsubj045 all
   scanned using this protocol.
3. ses-01, ses-02 (experiment git commit
   3f0920aee0f4f8f198c0f258b63482afbe47e3de, git commit
   aa661f8f0093a7e444fc60796a48d82006679596 for creation of local
   spatial frequency maps for all stimulus types): `w_r` and `w_a` now
   run from `2^2.5` to `2^7` in half-octave steps; this allows us to
   get an extra half-degree closer to the fovea. Also adds constant
   stimuli (spatial frequency constant across the whole image) to
   serve as a sanity check. The two sessions include one with the
   log-polar stimuli, one with the constant (these are referred to ask
   task-sfp and task-sfpconstant, respectively). The stimuli no longer
   have an alpha value and the calculation of their spatial frequency
   is correct (and is correct for the pilot stimuli as well).
4. ses-03 (git commit 69558708537c4d1a82617b05f6e39b4f2c8d7d9a): adds
   16 seconds of blank time at the beginning of each run (to improve
   estimation of baseline) and an extra 8 seconds at the end of each
   run. Still task-sfp (only log-polar stimuli).

# Snakemake

A Snakefile is included with this project to enable the use
of [snakemake](http://snakemake.readthedocs.io/en/latest/). This is
highly recommended, since it will allow you to easily to rerun the
analyses exactly as I have performed them and it enables easy use on a
cluster. To
use,
[install snakemake](http://snakemake.readthedocs.io/en/latest/getting_started/installation.html) and,
if using on a cluster, set up your
appropriate
[Snakemake profile](https://github.com/Snakemake-Profiles/doc) (see
the
[snakemake docs](http://snakemake.readthedocs.io/en/latest/executable.html#profiles)for
more info on profiles; I had to make some modifications to get this
working on NYU's HPC, see my
profile [here](https://github.com/billbrod/snakemake-slurm)). Then
simply type `snakemake {target}` to re-run the analyses.

For example, if running on NYU's HPC cluster, set up the SLURM profile
and use the following command: `snakemake --profile slurm --jobs {n}
--cluster-config cluster.json {target}`, where `{n}` is the number of
jobs you allow `snakemake` to simultaneously submit to the cluster and
`cluster.json` is an included configuration file with some reasonable
values for the cluster (feel free to change these as needed).

I also found that `snakemake>=5.4`, as now required, installs its own
`mpi4py` on the SLURM cluster. If you attempt to run any python
command from the environment that includes this `mpi4py` on an
interactive job (not on a login node), you'll get a very scary-looking
error. To deal with this, just run `unset $SLURM_NODELIST`.

# Eventual sharing goals

I would like to make this as easy for others to reproduce as
possible. To that end I'll eventually upload the (complete?) dataset
to openneuro.org and provide people with two entry points. They can
either

1. Rerun my entire analysis by downloading just the
   `derivatives/freesurfer` and subject directories, along with a
   Docker or Singularity (haven't decided) container and then use the
   Snakefile. Ideally this will be able to make use of parallelization
   because otherwise it will take forever (~20 minutes to preprocess
   each run, ~2 hours for each session's GLMdenoise). Can't make this
   the default because of how long it would take to run linearly and
   the matlab requirement (for GLMdenoise). (will also require FSL and
   Freesurfer)
2. Trust my analysis up until the outputs of
   GLMdenoise_realigned. Download the data from that point and re-run
   everything afterwards, which will be pure python and not take
   nearly as long. This I can probably do using BinderHub (though I'll
   need to double-check how long creating the various csv files
   takes), and give people the option to take the outputs of the
   `first_level_analysis` or trust mine.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── sfp/               <- Source code for use in this project.
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
