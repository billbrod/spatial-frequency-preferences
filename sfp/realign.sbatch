#!/bin/bash
# See NYU HPC documentation for more details on these options
#SBATCH --job-name=runGLM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=12:00:00
 
# Make sure you have GLMdenoise downloaded to ~/matlab-toolboxes
module purge
module load anaconda2/4.3.1
module load freesurfer/6.0.0

export SUBJECTS_DIR=/scratch/wfb229/Freesurfer_subjects

cd ~/spatial-frequency-preferences
 
python sfp/realign.py
