#!/bin/bash
# See NYU HPC documentation for more details on these options
#SBATCH --job-name=runGLM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=12:00:00
 
# Make sure you have VistaSoft and GLMdenoise downloaded
module purge
module load matlab/2017a
module load spm/intel/12

cd ~/spatial-frequency-preferences/matlab
 
# niftiRead can't handle the ~
matlab -nodesktop -nodisplay -r "runGLM('~/spatial_frequency_pilot/MRI_first_level/run_%02d_design_matrix.mat', '/scratch/wfb229/preproc/wl_subj042/pilot/timeseries_corrected_run%02d.nii.gz', [0 1 2 3 4 5 6 7], [2 3 4 5 6 7 8 9], '~/spatial_frequency_pilot/MRI_first_level/run_details.mat', '~/matlab-toolboxes/vistasoft', '/share/apps/spm/12/intel', '~/matlab-toolboxes/GLMdenoise', 0, '/scratch/wfb229/MRI_first_level_results'); quit;"
