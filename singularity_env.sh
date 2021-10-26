#!/usr/bin/env bash

# set up environment variables for other libraries
export FREESURFER_HOME=/home/sfp_user/freesurfer
export PATH=$FREESURFER_HOME/bin:$PATH

export PATH=/home/sfp_user/matlab/bin:$PATH

export FSLOUTPUTTYPE=NIFTI_GZ
export FSLDIR=/home/sfp_user/fsl
export PATH=$FSLDIR/bin:$PATH

if [ -f /home/sfp_user/spatial-frequency-preferences/config.json ]; then
    cp /home/sfp_user/spatial-frequency-preferences/config.json /home/sfp_user/sfp_config.json
    sed -i 's|"DATA_DIR":.*|"DATA_DIR": "/home/sfp_user/sfp_data",|g' /home/sfp_user/sfp_config.json
    sed -i 's|"MRI_TOOLS":.*|"MRI_TOOLS": "/home/sfp_user/MRI_tools",|g' /home/sfp_user/sfp_config.json
    sed -i 's|"GLMDENOISE_PATH":.*|"GLMDENOISE_PATH": "/home/sfp_user/GLMdenoise",|g' /home/sfp_user/sfp_config.json
    sed -i 's|"VISTASOFT_PATH":.*|"VISTASOFT_PATH": "/home/sfp_user/vistasoft",|g' /home/sfp_user/sfp_config.json
    sed -i 's|"WORKING_DIR":.*|"WORKING_DIR": "/home/sfp_user/sfp_data/preproc_tmp_dir",|g' /home/sfp_user/sfp_config.json
    mkdir -p /home/sfp_user/sfp_data/preproc_tmp_dir
    echo "this is a temporary directory created for preprocessing, and can be deleted" > /home/sfp_user/sfp_data/preproc_tmp_dir/delete_me.txt
fi
