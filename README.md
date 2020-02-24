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
 - [vistasoft](https://github.com/vistalab/vistasoft)
 
Python: 

 - Python 3.6
 - use the included conda environment files:
   - in order to run the experiment, you'll need to create a psychopy
     environment: `conda env create -f environment-psychopy.yml`. By
     typing `conda activate psypy`, you'll then be able to run the
     experiment.
     - I use `pyglet` for creating the window and have consistently
       had issues on Ubuntu (on Lenovo laptops), where the following
       strange error gets raised: `AssertionError:
       XF86VidModeGetGammaRamp failed`. See [this
       issue](https://github.com/psychopy/psychopy/issues/2061) for
       solutions, but what's worked for me is adding the
       `/etc/X11/xorg.conf.d/20-intel.conf` file, updating drivers if
       necessary, and restarting the machine. I have not had these
       issues on Macs. Another potential fix is to switch to
       `winType='glfw'` (as also suggested in that issue), but that
       may break other parts of the experiment code.
   - if you only want to analyze / investigate the data, you don't
     need to set up a psychopy environment, just run: `conda env
     create -f environment.yml`. Then, type `conda activate sfp` to
     activate the environment.
 - [WinawerLab's MRI_tools](https://github.com/WinawerLab/MRI_tools)
   (which requires [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)),
   commit
   [8508652bd9e6b5d843d70be0910da413bbee432e](https://github.com/WinawerLab/MRI_tools/tree/8508652bd9e6b5d843d70be0910da413bbee432e)

Other: 

 - [FreeSurfer](http://freesurfer.net/)

# Overview

The analysis for this project takes place over several step and makes
use of Python, Matlab, and command-line tools. The notebooks folder
contains several Jupyter notebooks which (hopefully) walk through the
logic of the experiment and analysis.

The Snakefile can do everything except for running the experiment and
extracting the data from the NYU CBI WebDB.

1. Create the stimuli: `snakemake
   data/stimuli/task-sfprescaled_stimuli.npy`. This requires the
   `data/stimuli/mtf_func.pkl`, which is created by the code found in
   the
   [spatial-calibration](https://github.com/WinawerLab/spatial-calibration/)
   repo and is included in this repo. This only needs to be done once.
2. Create the presentation indices for the subject: `snakemake
   data/stimuli/sub-wlsubj###_ses-##_run00_idx.npy`. In order for this
   to work, the subject and session both need to be keys in the
   `SUB_SEEDS` and `SES_SEEDS` dictionaries found at the top of the
   Snakefile.
3. Run the experiment and gather fMRI data (`python sfp/experiment.py
   data/stimuli/task-sfprescaled_stimuli.npy 12
   sub-wlsubj###_ses-##`). Each run will last 4 minutes 24 seconds (48
   stimulus classes and 10 blank trials, each for 4 seconds, gives you
   3 minutes 52 seconds, and then each run starts and ends with 16
   seconds of blank screen).
4. Extract from NYU CBI's WebDB, which will run `heudiconv` on the
   data, getting it into the BIDS format.
5. Transfer the data for this scanning session from `Tesla` to the
   BIDS directory (`move_off_tesla` rule) and create the correct
   `events.tsv` files (`create_BIDS_tsv` rule).
6. Pre-process your fMRI data (using [WinawerLab's
   MRI_tools](https://github.com/WinawerLab/MRI_tools)). This is
   accomplished by the `preprocess`, `rearrange_preprocess_extras`,
   and `rearrange_preprocess` rules in the Snakefile.
7. Align to freesurfer anatomy and get into the mgz format, done by
   the `to_freesurfer` rule, which uses the `to_freesurfer.py` script
   found in
   the
   [WinawerLab's MRI_tools](https://github.com/WinawerLab/MRI_tools)
8. Create design matrices for each run This is done by the
   `create_design_matrices` rule
9. Run GLMdenoise (`runGLM.m`) done by the `GLMdenoise` rule.
10. Arrange the outputs into a pandas dataframe for ease of further
   analysis. This is done using the `first_level_analysis` rule.
11. Construct tuning curves, using the `tuning_curves` rule.
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
12. Collect tuning curves across subjects and scanning sessions, to
   compare. This is done using the `tuning_curves_summary` rule.

ADD STEPS FOR 2d MODEL

Note that several of these steps (preprocessing and running
GLMdenoise) should be run on a cluster and will take way too long or
too much memory to run on laptop.

When running python scripts from the command line, they should be run
from this directory (and so you should call `python -m sfp.foo`), in
order for the default paths to work. You can run them from somewhere
else, but then you'll need to use the optional arguments to set the
paths yourself.


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
5. ses-04 (git commit c7d6ea6543b368f3721ec836e7591f6c86baa438): same
   experimental design as before, but with new task,
   task-sfprescaled. We were concerned about the effect of the scanner
   projector's modulation transfer function, that we might be losing
   contrast at the higher frequencies (because of the projector's
   pointspread function, which effectively acts to blur the
   image). This could result in a lower response to those high spatial
   frequencies, simply from the reduced contrast (rather than the
   higher spatial frequency). In order to test this, we measured the
   projector's MTF (see the
   [spatial-calibration](https://github.com/WinawerLab/spatial-calibration/)
   Github repo for more details) and then constructed our stimuli so
   that their amplitude is rescaled in a spatial frequency-dependent
   manner; the amplitudes of lower spatial frequencies is reduced such
   that, when displayed by the scanner projector, both the high and
   low spatial should have (approximately) the same contrast. We then
   gather these new measurements and run the same analysis to compare
   the effect. We also add the stimuli for task-sfpconstantrescaled
   (same thing for the constant stimuli), but it's unclear if we will
   gather that data (would be ses-05).

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

the `GLMdenoise_png_process` rule (which converts the outputs of
GLMdenoise into a format we can actually look at) does not run on the
cluster; I think this is because there is no display available for
those machines (it runs fine on my local machine). If that rule fails
for you on the cluster, just run it locally. The error appears to be
related to [this
issue](https://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server#4935945),
so you could also make sure to use a different matplotlib backend
(`'svg'` will work without an X server). I have included a
matplotlibrc file in this folder, which should make svg the default
backend, but this warning is just in case it doesn't.

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

In addition to sharing the code:
1. Put something together to explore effect of changing model
   parameters on plots (ideally that can be both in notebook and a web
   app), with buttons that give random parametrizations and make clear
   the model name -- I think plotly necessary for this, altair would
   require pre-generating the dataframe (which would be too large
   because we have 11 parameters), and bokeh doesn't have native
   support for polar plots. (app would therefore be built with
   [dash](https://github.com/plotly/dash))
2. have separate GitHub repo with just code for model, some
   exploration of it, code for analyzing it, parameters csvs (up on
   pip); this should include the above
3. also take that spatial-frequency-stimuli repo, and extend that /
   make sure it's easy to generate the stimuli. potentially put
   together some web app for that as well? could try to do something
   simple to show the frequency of stimuli, etc

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
