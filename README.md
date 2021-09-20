Spatial frequency preferences
==============================

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
4. Run `python download_data.py preprocessed` to download the preprocessed data.
5. Run `python download_data.py fully` to download the fully-processed data
   (note that you need both).
6. Run `snakemake -j N figures_paper` (where `N` is the number of cores to use
   in parallel) to recreate the figures, properly numbered, in the
   `reports/paper_figures/` directory. Note that they are svgs, a vector file
   format. If your default image viewer cannot open them, your browser can. They
   can be converted to pdfs using [inkscape](https://inkscape.org/) or Adobe
   Illustrator.
   
# Setup 

## Software requirements

If you are starting with the partially- or fully-processed data, as explained
[above](#data), then you only need the python requirements. If you're re-running
the analysis from the beginning, you will also need MATLAB, FSL, and Freesurfer.
In order to use the included download script `download_data.py`, you will also
need `rsync`, which is probably already on your machine.

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
2. In this directory, run `conda env create -f environment.yml`.
3. If everything works, type `conda activate sfp` to activate this environment
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

##### Troubleshooting

If `conda env create -f environment.yml` seems to be taking forever, it's having
trouble solving the environment. This is a known issue and someone has put
together the drop-in replacement `mamba`, which is much faster. Once you've
installed `miniconda`, you can install `mamba` with `conda install mamba -n base
-c conda-forge`, and then replace `conda` with `mamba` in the install command:
`mamba env create -f environment.yml`; in my experience, this runs much more
quickly. You then activate the environment as normal: `conda activate sfp`.

### Experimental environment

We also include a separate environment containing psychopy, if you wish to run
the experiment yourself. If you just want to see what the experiment looked
like, there is a video of a single run on the [OSF](https://osf.io/cauhd/) Once
you have miniconda installed (i.e., once you've completed step 1 of the [conda
environment](#conda-environment) section above), then run `conda env create -f
environment-psychopy.yml` to install the environment, and type `conda activate
psypy` to activate it.

See [here](#running-the-experiment) for how to run the experiment.

Troubleshooting note: I use `pyglet` for creating the window and have
consistently had issues on Ubuntu (on Lenovo laptops), where the following
strange error gets raised: `AssertionError: XF86VidModeGetGammaRamp failed`. See
[this issue](https://github.com/psychopy/psychopy/issues/2061) for solutions,
but what's worked for me is adding the `/etc/X11/xorg.conf.d/20-intel.conf`
file, updating drivers if necessary, and restarting the machine. I have not had
these issues on Macs. Another potential fix is to switch to `winType='glfw'` (as
also suggested in that issue), but that may break other parts of the experiment
code.

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
1. `preprocessed`: The minimally pre-processed BIDS-compliant data is shared on
   OpenNeuro (**LINK**). If you want to re-run our analyses from the beginning,
   this is the one you should use. In addition to the fMRI data gathered in this
   experiment, it also includes the (defaced) freesurfer data for each subject
   and the population receptive field solutions. See that dataset's README for
   more details.
2. `partially`: Partially-processed data is shared on OSF (**LINK**). If you
   want to re-run our 1d tuning curve analysis and the 2d model fits, this is
   the one you should use. It is not fully BIDS-compliant, but tries to be
   BIDS-inspired.
3. `fully`: Fully-processed data is shared on the [OSF](https://osf.io/djak4/).
   If you just want to re-create our figures, this is what you should use. It is
   also not fully BIDS-compliant, but BIDS-inspired. This contains:
   - The outputs of the "first-level analysis" step, which contains the median
     amplitude response of each vertex in V1 to our stimuli (as computed by
     GLMdenoise), and each vertex's population receptive field locations.
   - A csv summarizing the outputs of our 1d analysis, containing the parameters
     for each tuning curve fit to the different stimuli classes and
     eccentricities for each subject, across all bootstraps.
   - A csv containing that information for a single subject.
   - For each subject, the trained model for our best submodel (as determined by
     crossvalidation), fit to each voxel's median response.
   - A csv summarizing those models.
   - A csv summarizing the cross-validation loss for our model comparison
     analysis.
   - A csv summarizing the models fit to each subject, bootstrapping across
     runs, for the best submodel.
   - A csv containing the parameters for the best submodel, bootstrapped across
     subjects.
   - Two csvs, one for the quarters around the vertical meridia, one for the
     quarters around the horizontal meridia, summarizing a simple model
     (preferred period is an affine function of eccentricity, no dependency on
     orientation, no modulation of relative gain) fit to a subset of the visual
     field.
   
Note that both the partially-processed and fully-processed data require the
OpenNeuro dataset (though the fully-processed data does not require the
partially-processed). Additionally, these two data sets do not contain the
*entire* outputs of the analysis at that point, just what is required to
complete the following steps.

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
[docs](https://snakemake.readthedocs.io/en/stable/executing/cluster.html?highlight=cluster) for details.

If you're using NYU's greene cluster, things are both easier (because I've
already gone through this process) and harder (the configuration is more
difficult because of how they want you to handle conda environments) for you. I
put together a little [blog
post](https://wfbroderick.com/2021-May-06.html#2021-May-06) detailing how I got
snakemake to work with conda environments on greene (note that during step 7,
where I say "install conda environment like normal", you should run `conda env
create -f environment.yml` to install the necessary packages).

Once you've managed to get snakemake working with the cluster, use the following
command to run our analyses instead:

`snakemake --ri -k -j 60 --restart-times 2 --profile slurm --cluster-config cluster.json figures_paper`

This tells snakemake to use our `slurm` profile (so if you set up a different
profile for another cluster, change the name here) and the included
`cluster.json` configuration file (which tells slurm how much memory and time to
request per job), to try to restart jobs 2 times if they fail (sometimes job
just fail on submission and then rerun without a problem), to rerun any jobs
that look incomplete (`--ri`), and to keep running any independent jobs if any
jobs fail (`-k`). This should hopefully run to completion (in my experience on
greene, it takes about one to two days).

### Troubleshooting notes

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

## Running the experiment

If you just wish to see what the runs of our experiment looked like, a video has been uploaded to the [OSF](https://osf.io/cauhd/).

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

# Getting help

Reproducing someone else's research code is hard and, in all likelihood, you'll
run into some problem. If that happens, please open an issue on this repo, with
as much info about your machine and the steps you've taken as possible, and I'll
try to help you fix the problem.

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

If you use the data or code from this project in an academic publication, please
cite the paper (**LINK**). Additionally:
1. If you use the code in this or any of the linked repos
   [above](#related-repos) directly (for example, using it to generate the
   log-polar stimuli, modifying that code to create a similar set of stimuli,
   using the model code directly, adapting the code to extend the model, or
   re-running the analysis on a novel dataset), please cite this Github repo's
   Zenodo doi (**LINK**).
2. If you use re-analyze the data without using the code here, please cite the
   dataset's OpenNeuro doi (*LINK*).
3. If you reproduce our analysis on our data, please cite both the Github repo's
   Zenodo doi and the dataset's OpenNeuro doi.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
