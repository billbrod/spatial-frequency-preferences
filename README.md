Spatial frequency preferences
==============================
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/billbrod/spatial-frequency-preferences/HEAD?filepath=notebooks)

An fMRI experiment to determine the relationship between spatial
frequency and eccentricity in the human early visual cortex.

See the paper **LINK** for scientific details. If you re-use some component
of this project in an academic publication, see the [citing](#citation) section
for how to credit us.

# Usage

The data and code for this project are shared with the primary goal of enabling
reproduction of the results presented in the associated paper. Novel analyses
should be possible, but no guarantees.

To that end, we provide [several entrypoints into the data](#data) for
re-running the full or partial analysis, with a script to automate their
download and proper arrangement. There are also several ways to reproduce the
[computational environment](#software-requirements) of the analysis.

The following steps will walk you through downloading the fully-processed data
and recreating the figures, read further on in this README for details:
1. Clone this repo.
2. Open `config.yml` and modify the `DATA_DIR` path to wherever you wish to
   download the data.
3. Install the python environment:
   - Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) on your
     system for python 3.7.
   - Install [mamba](https://github.com/mamba-org/mamba): `conda install mamba
     -n base -c conda-forge`
   - Navigate to this directory and run `mamba env create -f environment.yml` to
     install the environment.
   - Run `conda activate sfp` to activate the python environment.
4. Run `python download_data.py fully-processed` to download the fully-processed
   data (note that you need both) (this is about 500MB).
5. Run `snakemake -k -j N reports/paper_figures/fig-XX.svg`
   (where `N` is the number of cores to use in parallel) to recreate a given
   figure from the paper (note the number must be 0-padded, i.e., `fig-01.svg`,
   *not* `fig-1.svg`). These will end up in the `reports/paper_figures/`
   directory. Note that they are svgs, a vector file format. If your default
   image viewer cannot open them, your browser can. They can be converted to
   pdfs using [inkscape](https://inkscape.org/) or Adobe Illustrator.
   - **WARNING**: while most figures take only a few minutes to create, one of
     these, `fig-08.svg`, takes much longer (up to 8 minutes on the cluster, 21
     minutes on my personal laptop).
6. If you wish to create all the figures from the main body of the text, run
   `snakemake -k -j N main_figure_paper`. If one job fails, this
   will continue to run the others (that's what the `-k` flag means).
   
If you wish to create the supplemental figures as well:
1. Download the additional data required: `python download_data.py supplemental`
   (this is about 5GB). *NOTE*: this is not required if you have already
   downloaded the full `preprocessed` data set from
   [OpenNeuro](https://openneuro.org/datasets/ds003812/).
2. Run `snakemake -k -j N reports/paper_figures/fig-SXX.svg` (where again the
   number must be 0-padded) to create a single supplemental figure or `snakemake
   -k -j N supplement_figure_paper` to create all of them.
   
If you have any trouble with the above, check the
[troubleshooting](#troubleshooting) section to see if there's a solution to your
problem.
   
## Notebooks

We also include several jupyter notebooks in the `notebooks/` directory:

- `How-to-use-model`: examples of how to use the 2d tuning model described in
  the paper.
- `Stimuli`: some exploration of how to create the stimuli and visualize linear
  approximations thereof.

If you'd like to use them, you can either view it on
[Binder](https://mybinder.org/v2/gh/billbrod/spatial-frequency-preferences/HEAD?filepath=notebooks)
or [install
jupyter](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
so you can view it locally (you can also view a static version of the notebooks
using [nbviewer](https://nbviewer.jupyter.org/)).

If you would like to install jupyter locally and are unfamiliar with it, there
are two main ways of getting it working (after setting up your [conda
environment](#conda-environment)):

1. Install jupyter in this `sfp` environment: 

``` sh
conda activate sfp
conda install -c conda-forge jupyterlab
```

   This is easy but, if you have multiple conda environments and want to use
   Jupyter notebooks in each of them, it will take up a lot of space.
   
2. Use [nb_conda_kernels](https://github.com/Anaconda-Platform/nb_conda_kernels):

``` sh
# activate your 'base' environment, the default one created by miniconda
conda activate 
# install jupyter lab and nb_conda_kernels in your base environment
conda install -c conda-forge jupyterlab
conda install nb_conda_kernels
# install ipykernel in the sfp environment
conda install -n sfp ipykernel
```

   This is a bit more complicated, but means you only have one installation of
   jupyter lab on your machine.
   
In either case, to open the notebooks, navigate to this directory on your
terminal and activate the environment you install jupyter into (`sfp`
for 1, `base` for 2), then run `jupyter` and open up the notebook. If you
followed the second method, you should be prompted to select your kernel the
first time you open a notebook: select the one named "sfp".

## Model parameters

`data/tuning_2d_model` contains two csv files containing the parameter values
presented in the paper:

- `combined_subject_params.csv` contains the parameter values from combining
  across subjects, as presented in the main body of the paper.
- `individual_subject_params.csv` contains the parameter values from each
  subject individually, as shown in Figure 9B and the appendix.

Both csvs have five columns: fit_model_type, model_parameter, subject, bootstrap_num and fit_value:

- `fit_model_type` (str): the (long) name of this submodel. Both csvs only
  contain a single model, which corresponds to model 9 in our paper, the
  best-fit one.
- `subject` (str): the subject whose parameters these are. In
  combined_subject_params, there's only one subject, 'all'.
- `bootstrap_num` (int): the bootstrap number. Each combination of
  model_parameter and subject has 100 bootstraps. For combined_subject_params,
  these are the bootstraps used to combine across subjects in a
  precision-weighted average. For individual_subject_params, these are the
  bootstraps across fMRI runs. See paper for more details.
- `fit_value` (float): the actual value of the parameter.
- `model_parameter` (str): the (long) name of the model parameter. The following
  gives the mapping between the two:
  - `'sigma'`: <img src="https://latex.codecogs.com/gif.latex?\sigma" />
  - `'sf_ecc_slope'`: <img src="https://latex.codecogs.com/gif.latex?a" />
  - `'sf_ecc_intercept'`: <img src="https://latex.codecogs.com/gif.latex?b" />
  - `'abs_mode_cardinals'`: <img src="https://latex.codecogs.com/gif.latex?p_1" />
  - `'abs_mode_obliques'`: <img src="https://latex.codecogs.com/gif.latex?p_2" />
  - `'rel_mode_cardinals'`: <img src="https://latex.codecogs.com/gif.latex?p_3" />
  - `'rel_mode_obliques'`: <img src="https://latex.codecogs.com/gif.latex?p_4" />
  - `'abs_amplitude_cardinals'`: <img src="https://latex.codecogs.com/gif.latex?A_1" />
  - `'abs_amplitude_obliques'`: <img src="https://latex.codecogs.com/gif.latex?A_2 " />
  - `'rel_amplitude_cardinals'`: <img src="https://latex.codecogs.com/gif.latex?A_3" />
  - `'rel_amplitude_obliques'`: <img src="https://latex.codecogs.com/gif.latex?A_4" />
 
See the [included How-to-use-model notebook](#notebooks) for details about how
to use these csvs with our model.
   
# Setup 

The analyses were all run on Linux (Ubuntu, Red Hat, and CentOS, several
different releases). The steps required to create the figures have also been
tested on Macs, and everything else should work on them as well. For Windows, I
would suggest looking into the [Windows Subsystem for
Linux](https://docs.microsoft.com/en-us/windows/wsl/about), as Windows is very
different from the others.

## Software requirements

If you are starting with the partially- or fully-processed data, as explained
[below](#data), then you only need the python requirements. If you're re-running
the analysis from the beginning, you will also need MATLAB, FSL, and Freesurfer.

In order to use the included download script `download_data.py`, you will also
need `rsync`, which is probably already on your machine.

In order to use `download_data.py` to download the `preprocessed` data from
OpenNeuro, you will need to install the [OpenNeuro command line
interface](https://docs.openneuro.org/packages-openneuro-cli-readme).

### Python

This code works with python 3.6 and 3.7.

We provide several ways to reproduce the python environment used in this
experiment, in order of increasing complexity and future-proof-ness:

1. [Conda](#conda-environment)
2. Docker / singularity: under development
3. Reprozip: under development

#### Conda environment

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) for your
   system with the appropriate python version.
2. Install [mamba](https://github.com/mamba-org/mamba): `conda install mamba
  -n base -c conda-forge`
3. In this directory, run `mamba env create -f environment.yml`.
4. If everything works, type `conda activate sfp` to activate this environment
   and all required packages will be available.
   
As of this writing, few of the required packages have specific version
requirements (those that do are marked in the `environment.yml` file). As this
will probably change in the future, we include a `pip_freeze.txt` file which
shows the installed versions of each python package used to run the analysis
(note this includes more packages than specified in `environment.yml`, because
it includes all of *their* dependencies as well).

If you're re-running the pre-processing, you'll need [WinawerLab's
MRI_tools](https://github.com/WinawerLab/MRI_tools), commit
[8508652bd9e6b5d843d70be0910da413bbee432e](https://github.com/WinawerLab/MRI_tools/tree/8508652bd9e6b5d843d70be0910da413bbee432e).
The results shouldn't change substantially if pre-processed using fMRIPrep. This
repo is included in the docker and reprozip options.

I recommend using `mamba` instead of `conda` to install the environment because
`conda` tends to hang while attempting to solve the environment.

### Experimental environment

We also include a separate environment containing psychopy, if you wish to run
the experiment yourself. If you just want to see what the experiment looked
like, there is a video of a single run on the [OSF](https://osf.io/cauhd/) Once
you have miniconda installed (i.e., once you've completed step 1 of the [conda
environment](#conda-environment) section above), then run `conda env create -f
environment-psychopy.yml` to install the environment, and type `conda activate
psypy` to activate it.

See [here](#running-the-experiment) for how to run the experiment.

### Matlab

 - Version at or more recent than 2016b (because we need `jsondecode`)
 - [GLMdenoise](https://github.com/kendrickkay/GLMdenoise/)
 - [vistasoft](https://github.com/vistalab/vistasoft)
 
### Other 

 - [FreeSurfer](http://freesurfer.net/) 6.0.0
 - [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) 5.0.10

## Data

We provide the script `download_data.py`, which should work with a plain python
install, in order to automate the downloading and proper arrangement of one of
the three following subsets of the data. Which one you choose depends on your
use-case:
1. `preprocessed` (40.54GB): The minimally pre-processed BIDS-compliant data is
   shared on [OpenNeuro](https://dx.doi.org/10.18112/openneuro.ds003812.v1.0.0).
   If you want to re-run our analyses from the beginning, this is the one you
   should use. In addition to the fMRI data gathered in this experiment, it also
   includes the (defaced) freesurfer data for each subject and the population
   receptive field solutions. See that dataset's README for more details.
2. `partially-processed`: Partially-processed data is shared on OSF (**LINK**).
   If you want to re-run our 1d tuning curve analysis and the 2d model fits,
   this is the one you should use. It is not fully BIDS-compliant, but tries to
   be BIDS-inspired. -- this is currently under development
3. `fully-processed` (523MB): Fully-processed data is shared on the
   [OSF](https://osf.io/djak4/). If you just want to re-create our figures, this
   is what you should use. It is also not fully BIDS-compliant, but
   BIDS-inspired. This contains:
   - `stimuli/`: The stimuli arrays, modulation transfer function of the
     display, and csvs describing them (these are also contained in the
     `preprocessed` data)
   - `derivatives/first_level_analysis/`: The outputs of the "first-level
     analysis" step, csvs which contains the median amplitude response of each
     vertex in V1 to our stimuli (as computed by GLMdenoise), and each vertex's
     population receptive field locations.
   - `derivatives/tuning_curves/stim_class/bayesian_posterior/individual_ses-04_v1_e1-12_eccen_bin_tuning_curves_full.csv`:
     A csv summarizing the outputs of our 1d analysis, containing the parameters
     for each tuning curve fit to the different stimuli classes and
     eccentricities for each subject, across all bootstraps.
   - `derivatives/tuning_curves/stim_class/bayesian_posterior/sub-wlsubj001/ses-04/sub-wlsubj001_ses-04_task-sfprescaled_v1_e1-12_eccen_bin_summary.csv`:
     A csv containing that information for a single subject, fit to their
     medians across bootstraps.
   - `derivatives/tuning_2d_model/stim_class/bayesian_posterior/filter-mean/initial/sub-*/ses-04/sub-*_ses-04_task-sfprescaled_v1_e1-12_summary_b10_r0.001_g0_cNone_nNone_full_full_absolute_model.pt`:
     For each subject, the trained model for our best submodel (as determined by
     crossvalidation), fit to each voxel's median response.
   - `derivatives/tuning_2d_model/stim_class/bayesian_posterior/filter-mean/initial/individual_task-sfprescaled_v1_e1-12_summary_b10_r0.001_g0_iso_full_iso_all_models.csv`:
     A csv summarizing those models.
   - `derivatives/tuning_2d_model/stim_class/bayesian_posterior/filter-mean/initial_cv/individual_task-sfprescaled_v1_e1-12_summary_b10_r0.001_g0_s0_all_cv_loss.csv`:
     A csv summarizing the cross-validation loss for our model comparison
     analysis.
   - `derivatives/tuning_2d_model/stim_class/bayesian_posterior/filter-mean/bootstrap/individual_task-sfprescaled_v1_e1-12_full_b10_r0.001_g0_full_full_absolute_all_models.csv`:
     A csv summarizing the models fit to each subject, bootstrapping across
     runs, for the best submodel.
   - `derivatives/tuning_2d_model/task-sfprescaled_final_bootstrapped_combined_parameters_s-5.csv`:
     A csv containing the parameters for the best submodel, bootstrapped across
     subjects.
   - `derivatives/tuning_2d_model/stim_class/bayesian_posterior/filter-mean/visual_field_{vertical,horizontal}-meridia/individual_task-sfprescaled_v1_e1-12_summary_b10_r0.001_g0_iso_full_iso_all_models.csv`:
     Two csvs, one for the quarters around the vertical meridia, one for the
     quarters around the horizontal meridia, summarizing a simple model
     (preferred period is an affine function of eccentricity, no dependency on
     orientation, no modulation of relative gain) fit to a subset of the visual
     field.
4. `supplemental` (5GB): Extra data required to create the figures in the
   appendix, along with data present in `fully-processed`. This data is also
   present in `preprocessed` and is separate of `fully-processed` because it's
   much larger.
   - `derivatives/freesurfer/`: the freesurfer directories of each subject.
   - `derivatives/prf_solutions/`: the pRF solutions for each subject, including
     the Benson 2014 anatomical atlases (Benson et al, 2014) and Bayesian
     retinotopy solutions (Benson and Winawer, 2018).
   
Note that the partially-processed data requires the OpenNeuro dataset.
Additionally, these two data sets do not contain the *entire* outputs of the
analysis at that point, just what is required to complete the following steps.

Also note that the subjects in the paper are just numbered from 1 to 12. In the
data and this code, they use their subject codes; the subject codes map to the
names in the paper in an increasing way, so that `sub-wlsubj001` is `sub-01`,
`sub-wlsubj006` is `sub-02`, etc.

To use `download_data.py`, simply call `python download_data.py TARGET_DATASET`
from the command-line where `TARGET_DATASET` is one of the three names above.

# Usage details

## Snakemake on the cluster

If you're running this analysis locally, nothing else needs to be configured. If
you're doing anything beyond just creating the figures, however, you probably
will want to use a compute cluster (GLMdenoise takes ~2 hours per subject, each
fit of the 2d model takes ~1 hour and we fit ~3000 of them in total). Snakemake
requires a bit of configuration to do so. View their
[docs](https://snakemake.readthedocs.io/en/stable/executing/cluster.html?highlight=cluster)
for details.

If you're using NYU's greene cluster, things are both easier (because I've
already gone through this process) and harder (the configuration is more
difficult because of how they want you to handle conda environments) for you. I
put together a little [blog
post](https://wfbroderick.com/2021-May-06.html#2021-May-06) detailing how I got
snakemake to work with conda environments on greene (note that during step 7,
where I say "install conda environment like normal", you should run `mamba env
create -f environment.yml` to install the necessary packages).

Once you've managed to get snakemake working with the cluster (my blog post
includes a small test), use the following command to run our analyses instead:

`snakemake --ri -k -j 60 --restart-times 2 --profile slurm --cluster-config cluster.json main_figure_paper`

This tells snakemake to use our `slurm` profile (so if you set up a different
profile for another cluster, change the name here) and the included
`cluster.json` configuration file (which tells slurm how much memory and time to
request per job), to try to restart jobs 2 times if they fail (sometimes job
just fail on submission and then rerun without a problem), to rerun any jobs
that look incomplete (`--ri`), and to keep running any independent jobs if any
jobs fail (`-k`). This should hopefully run to completion (in my experience on
greene, it takes about one to two days).

## Running the experiment

If you just wish to see what the runs of our experiment looked like, a video has
been uploaded to the [OSF](https://osf.io/cauhd/).

If you wish to use the existing stimuli to run the experiment, you can do so for a new subject `sub-test` by doing the following:
1. Activate the `sfp` environment: `conda activate sfp`.
2. Generate presentation indices: `snakemake data/stimuli/sub-test_ses-04_run00_idx.npy`
3. Activate the `psypy` environment: `conda activate pspy`.
4. Run `python sfp/experiment.py DATA_DIR/stimuli/task-sfprescaled_stimuli.npy 1
   sub-test_ses-04`, replacing `DATA_DIR` with the path to the root of the
   directory (as given in `config.yml`). If you want to run more than 1 run,
   change the `1` in that command. Each run will last 4 minutes 24 seconds (48
   stimulus classes and 10 blank trials, each for 4 seconds, with 16 seconds of
   blank screen at the beginning and end of each run).

Note that unless you use one of the subject names used in our project (i.e.,
`sub-wlsubj045`), the presentation index will be generated with `seed=0` (a
warning is raised to this effect when you create them). Therefore, all new
subjects will have the same presentation order, which is not what you want for
an actual experiment.

# Troubleshooting 


- There appears to be an issue installing torch version 1.1 on Macs (it affected
  about half the tested machines). If you try to create the figures and get an
  error message that looks like:
  
  ``` sh
  Rule Exception:
  ImportError in line 2592 of /Users/jh7685/Documents/Github/spatial-frequency-preferences/Snakefile:
  dlopen(/Users/jh7685/opt/miniconda/envs/sfp/lib/python3.7/site-packages/torch/_C.cpython-37m-darwin.so, 9): Library not loaded: ...
    Referenced from: /Users/jh7685/opt/miniconda/envs/sfp/lib/python3.7/site-packages/torch/lib/libshhm.dylib
    Reason: image not found
    File "/Users/jh7685/Documents/Github/spatial-frequency-preferences/Snakefile", line 2592, in __rule_figure_feature_df
    File "/Users/jh7685/Documents/Github/spatial-frequency-preferences/sfp/__init__.py", line 1, in <module>
    File "/Users/jh7685/Documents/Github/spatial-frequency-preferences/sfp/plotting.py", line 16, in <module>
    File "/Users/jh7685/Documents/Github/spatial-frequency-preferences/sfp/model.py", line 13, in <module>
    File "/Users/jh7685/opt/miniconda/envs/sfp/lib/python3.7/site-packages/torch/__init__.py", line 79, in <module>
    File "/Users/jh7685/opt/miniconda/envs/sfp/lib/python3.7/concurrent/futures/thread", line 57, in run
  ```
  then try uninstalling torch and installing a more recent version:
  
  ``` sh
  pip uninstall torch
  pip install torch
  ```
  
  The results were generated with torch version 1.1, but the code appears to be
  compliant with version up to at least 1.9.
  
- On a Mac, if the name of your `DATA_DIR` is uppercase, this can mess things up
  (on Macs, paths are case-insensitive, but they're case-sensitive on Linux and
  with many command-line tools). If you try to create the figures and get a
  scary-looking error message whose traceback goes through `snakemake/dag.py`
  and ends with :
  
  ```sh
     File "/Users/rania/anaconda3/envs/sfp/lib/python3.7/site-packages/snakemake/rules.py", line 756, in _apply_wildcards
       item = self.apply_path_modifier(item, property=property)
     File "/Users/rania/anaconda3/envs/sfp/lib/python3.7/site-packages/snakemake/rules.py", line 453, in _apply_path_modifier
       return {k: apply(v) for k, v in item.items()}
     File "/Users/rania/anaconda3/envs/sfp/lib/python3.7/site-packages/snakemake/rules.py", line 453, in <dictcomp>
       return {k: apply(v) for k, v in item.items()}
     File "/Users/rania/anaconda3/envs/sfp/lib/python3.7/site-packages/snakemake/path_modifier.py", line 35, in modify
       if modified_path == path:
  ValueError: the truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
  ```
  
  then that's probably what's going on. I recommend deleting this repo and your
  `DATA_DIR` from your local machine, and re-downloading everything, this time
  making sure that your `DATA_DIR` is lowercase (the whole path doesn't need to
  be lowercase; `/Users/billbrod/Desktop/sfp_data` should be fine, but
  `/Users/billbrod/Desktop/SFP_data` probably is not). 

    - Alternatively, this may actually be a `snakemake` version issue -- make
      sure your `snakemake` version is `5.x` (**not** `6.0` or higher). I think
      it's related to [this
      issue](https://github.com/snakemake/snakemake/issues/988), which is
      apparently fixed in [this
      PR](https://github.com/snakemake/snakemake/issues/1021), which hasn't been
      merged as of September 23, 2021.
      
    - If neither of the above solutions work, try the following solution. I
      think this problem arises because of some ambiguity in how to generate one
      of the inputs to some of the figures (and a change in how `snakemake`
      resolves that ambiguity between `5.x` and `6.x`), and so the following
      removes the ambiguity.
  
- If you're trying to create the figures after downloading the `fully-processed`
  and/or the `supplemental` data and `snakemake` complains about
  `MissingInputException`, try adding `cat reports/figure_rules.txt | xargs`
  before the snakemake command and `--allowed-rules` at the end (e.g., `cat
  reports/figure_rules.txt | xargs snakemake reports/paper_figures/fig-01.svg
  --allowed-rules`). What appears to be happening here is that `snakemake` is
  getting confused about timestamps or something similar and wants to rerun more
  of the analysis than necessary (or at least, wants to double-check how it
  would do that). Since `fully-processed` only contains the files at the end of
  analysis (and not the original inputs), snakemake is unable to trace the
  analysis back to the beginning and so complains. By adding the modifications
  above, we tell `snakemake` that it should **only** consider using the rules
  that produce figures, and it no longer has this problem.

- Previously, I found that `snakemake>=5.4`, as now required, installs its own
  `mpi4py` on the NYU's prince cluster. If you attempt to run any python command
  from the environment that includes this `mpi4py` on an interactive job (not on
  a login node), you'll get a very scary-looking error. If this happens, just
  run `unset $SLURM_NODELIST`. Since using singularity, as now required on
  greene, this has not been a problem.
  
- the `GLMdenoise_png_process` rule (which converts the outputs of GLMdenoise
  into a format we can actually look at) does not run on the cluster; I think
  this is because there is no display available for those machines (it runs fine
  on my local machine). If that rule fails for you on the cluster, just run it
  locally. The error appears to be related to [this
  issue](https://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server#4935945),
  so you could also make sure to use a different matplotlib backend (`'svg'`
  will work without an X server). I have included a matplotlibrc file in this
  folder, which should make svg the default backend, but this warning is just in
  case it doesn't.
  
- For the experiment: I use `pyglet` for creating the window and have
  consistently had issues on Ubuntu (on Lenovo laptops), where the following
  strange error gets raised: `AssertionError: XF86VidModeGetGammaRamp failed`.
  See [this issue](https://github.com/psychopy/psychopy/issues/2061) for
  solutions, but what's worked for me is adding the
  `/etc/X11/xorg.conf.d/20-intel.conf` file, updating drivers if necessary, and
  restarting the machine. I have not had these issues on Macs. Another potential
  fix is to switch to `winType='glfw'` (as also suggested in that issue), but
  that may break other parts of the experiment code.

## Getting help

Reproducing someone else's research code is hard and, in all likelihood, you'll
run into some problem. If that happens, double-check that your issue isn't
addressed [above](#troubleshooting), then please [open an
issue](https://github.com/billbrod/spatial-frequency-preferences/issues) on this
repo, with as much info about your machine and the steps you've taken as
possible, and I'll try to help you fix the problem.

# Related repos

If you are only interested in the stimuli, there is a separate [Github
repo](https://github.com/billbrod/spatial-frequency-stimuli) for the creation of
the stimuli, including a brief notebook examining their properties and how to
use the associated functions (note that it does not currently support the use
of an display modulation transfer function to rescale the stimuli's contrast).

If you are only interested in the 2d model, there is a separate [Github
repo](https://github.com/billbrod/spatial-frequency-model) for examining this
model in more detail. It also includes a little webapp for visualizing model
predictions for different parameter values. It does not contain the code used to
the fit the model, though it does contain the loss function we used,
`weighted_normed_loss`.
   
# Citation

If you use the data or code (including the stimuli) from this project in an
academic publication, please cite the paper (**LINK**). Additionally:
1. If you use the code in this or any of the linked repos
   [above](#related-repos) directly (for example, using it to generate the
   log-polar stimuli, modifying that code to create a similar set of stimuli,
   using the model code directly, adapting the code to extend the model, or
   re-running the analysis on a novel dataset), please cite this Github repo's
   Zenodo doi (**LINK**).
2. If you re-analyze the data without using the code here, please cite the
   dataset's [OpenNeuro
   doi](https://dx.doi.org/10.18112/openneuro.ds003812.v1.0.0).
3. If you reproduce our analysis on our data, please cite both the Github repo's
   Zenodo doi and the dataset's OpenNeuro doi.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
