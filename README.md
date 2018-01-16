Spatial frequency preferences
==============================

An fMRI experiment to determine the relationship between spatial
frequency and eccentricity in the human early visual cortex.

Much of the structure is based on the cookiecutter-data-science and so
currently empty / unused.

# Requirements

Matlab:

 - [GLMdenoise](https://github.com/kendrickkay/GLMdenoise/)
 
Python: 

 - see `requirements.txt` file
 - ![WinawerLab's MRI_tools](https://github.com/WinawerLab/MRI_tools))
   (which requires [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/))

Other: 

 - [FreeSurfer](http://freesurfer.net/)

# Overview of analysis

The analysis for this project takes place over several step and makes
use of Python, Matlab, and command-line tools. The notebooks folder
contains several Jupyter notebooks which (hopefully) walk through the
logic of the experiment and analysis.

Eventually, the Snakefile will be updated to include all the steps, but
for now, the following the analysis steps:

1. Create the stimuli (`python -m sfp.stimuli subject_name -c
   -i`). After you run this the first time (and thus create the
   unshuffled stimuli), you probably only need the `-i` flag to create
   the index.
2. Run the experiment and gather fMRI data (`python -m sfp.experiment
   data/stimuli/unshuffled.npy 12 subject_name`). Experiment will run
   4 minutes and 16 seconds (52 stimulus classes and 10 blank trials,
   each for 4 seconds, gives you 4 minutes 8 seconds, and then each
   run ends with 8 seconds of blank screen).
3. Pre-process your fMRI data
   (using
   [WinawerLab's MRI_tools](https://github.com/WinawerLab/MRI_tools))
4. Create design matrices for each run (`python sfp/design_matrices.py
   -s subject_name behavioral_results_path`). The `behavioral_results`
   h5py file is created when the experiment is run and the
   `unshuffled_stim_descriptions` csv file is created when the stimuli
   are created (you can probably trust the default for its path).
5. Run GLMdenoise (`runGLM.m`)
6. Align to freesurfer anatomy (`sfp.realign`)
7. Construct tuning curves
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

Note that several of these steps (preprocessing and running
GLMdenoise) should be run on a cluster and will take way too long or
too much memory to run on laptop.

When running python scripts from the command line, they should be run
from this directory (and so you should call `python sfp/foo.py` or
`python -m sfp.foo`), in order for the default paths to work. You can
run them from somewhere else, but then you'll need to use the optional
arguments to set the paths yourself.

`transfer_to_BIDS.py` is a file used to get the data from the format
it came off the scanner into the BIDS structure. As such, you will not
need it.

# Snakemake

A Snakefile is included with this project to enable the use
of ![snakemake](http://snakemake.readthedocs.io/en/latest/). This is
highly recommended, since it will allow you to easily to rerun the
analyses exactly as I have performed them and it enables easy use on a
cluster. To
use,
![install snakemake](http://snakemake.readthedocs.io/en/latest/getting_started/installation.html) and,
if using on a cluster, set up your
appropriate
![Snakemake profile](https://github.com/Snakemake-Profiles/doc) (see
the
![snakemake docs](http://snakemake.readthedocs.io/en/latest/executable.html#profiles)for
more info on profiles). Then simply type `snakemake {target}` to
re-run the analyses. 

For example, if running on NYU's HPC cluster, set up the SLURM profile
and use the following command: `snakemake --profile slurm --jobs {n}
--cluster-config cluster.json {target}`, where `{n}` is the number of
jobs you allow `snakemake` to simultaneously submit to the cluster and
`cluster.json` is an included configuration file with some reasonable
values for the cluster (feel free to change these as needed).

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
