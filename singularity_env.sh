#!/usr/bin/env bash

# set up environment variables for other libraries
export FREESURFER_HOME=/home/sfp_user/freesurfer
export PATH=$FREESURFER_HOME/bin:$PATH

export PATH=/home/sfp_user/matlab/bin:$PATH

export FSLOUTPUTTYPE=NIFTI_GZ
export FSLDIR=/home/sfp_user/fsl
export PATH=$FSLDIR/bin:$PATH
