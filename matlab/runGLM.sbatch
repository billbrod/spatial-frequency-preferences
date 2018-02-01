#!/bin/bash
# See NYU HPC documentation for more details on these options
#SBATCH --job-name=runGLM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62GB
#SBATCH --time=12:00:00
 
# Make sure you have GLMdenoise downloaded to ~/matlab-toolboxes
module purge
module load matlab/2017a
module load freesurfer/6.0.0

cd ~/spatial-frequency-preferences/matlab
 
matlab -nodesktop -nodisplay -r "runGLM('/home/wfb229/spatial-frequency-preferences/data/MRI_first_level/run_%02d_design_matrix.tsv', '/scratch/wfb229/spatial_frequency_preferences/derivatives/preprocessed/sub-wlsubj042/ses-pilot01/timeseries_corrected_run-%02d.nii.gz', [1 2 3 4 5 6 7 8 9 10 11 12], [1 2 3 4 5 6 7 8 9 10 11 12], '/home/wfb229/spatial-frequency-preferences/data/MRI_first_level/run_details_stim_class.json', '/share/apps/freesurfer/6.0.0/matlab', '~/matlab-toolboxes/GLMdenoise', 0, '/scratch/wfb229/wl_subj045/MRI_first_level/results/stim_class'); quit;"
