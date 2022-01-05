#!/usr/bin/env bash

# set up environment variables for other libraries, add them to path
export FREESURFER_HOME=/home/sfp_user/freesurfer
export PATH=$FREESURFER_HOME/bin:$PATH

export PATH=/home/sfp_user/matlab/bin:$PATH

export FSLOUTPUTTYPE=NIFTI_GZ
export FSLDIR=/home/sfp_user/fsl
export PATH=$FSLDIR/bin:$PATH

# modify the config.json file so it points to the location of MRI_tools,
# GLMDenoise, and Vistasoft within the container
if [ -f /home/sfp_user/spatial-frequency-preferences/config.json ]; then
    cp /home/sfp_user/spatial-frequency-preferences/config.json /home/sfp_user/sfp_config.json
    sed -i 's|"MRI_TOOLS":.*|"MRI_TOOLS": "/home/sfp_user/MRI_tools",|g' /home/sfp_user/sfp_config.json
    sed -i 's|"GLMDENOISE_PATH":.*|"GLMDENOISE_PATH": "/home/sfp_user/GLMdenoise",|g' /home/sfp_user/sfp_config.json
    sed -i 's|"VISTASOFT_PATH":.*|"VISTASOFT_PATH": "/home/sfp_user/vistasoft",|g' /home/sfp_user/sfp_config.json
fi
