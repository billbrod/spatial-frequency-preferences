Spatial frequency preferences
==============================

An fMRI experiment to determine the relationship between spatial
frequency and eccentricity in the human early visual cortex.

Much of the structure is based on the cookiecutter-data-science and so
currently empty / unused.

# Requirements

Matlab:

 - [GLMdenoise](https://github.com/kendrickkay/GLMdenoise/)
 
Python: see `requirements.txt` file

Other: [FreeSurfer](http://freesurfer.net/)

# Overview of analysis

The analysis for this project takes place over several step and makes
use of Python, Matlab, and command-line tools. The notebooks folder
contains several Jupyter notebooks which (hopefully) walk through the
logic of the experiment and analysis.

Eventually, the Makefile will be updated to include all the steps, but
for now, the following the analysis steps:

1. Create the stimuli (`python -m sfp.stimuli subject_name -c -i`)
2. Run the experiment and gather fMRI data (`python -m sfp.experiment
   data/stimuli/unshuffled.npy 12 subject_name`). Experiment will run
   4 minutes and 16 seconds (52 stimulus classes and 10 blank trials,
   each for 4 seconds, gives you 4 minutes 8 seconds, and then each
   run ends with 8 seconds of blank screen).
3. Pre-process your fMRI data
   (using
   [WinawerLab's MRI_tools](https://github.com/WinawerLab/MRI_tools))
4. Create design matrices for each run
   (`sfp.first_level_analysis.create_all_design_matrices`)
5. Run GLMdenoise (`runGLM.m`)
6. Align to freesurfer anatomy (`sfp.realign`)
7. Construct tuning curves

Note that several of these steps (preprocessing and running
GLMdenoise) should be run on a cluster and will take way too long or
too much memory to run on laptop

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
