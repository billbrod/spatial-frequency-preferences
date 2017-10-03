#!/usr/bin/python
"""realigns to freesurfer anatomy

This script takes the outputs of the GLMdenoise (as created and saved in runGLM.m) and realigns
them to the freesurfer anatomy so we can combine them with earllier retinotopy data to get the
corresponding visual area, polar angle, and eccentricity of each voxel. This is required for
further analyses.

This assumes you have freesurfer already in your path and that your SUBJECTS_DIR environmental
variable is properly set.

"""

import subprocess
import os
import argparse


def vol2vol(input_volume="MRI_first_level/stim_class/results/R2.nii.gz",
            reg_path="preprocessed/pilot/distort2anat_tkreg.dat",
            base_path=("/mnt/Acadia/Projects/spatial_frequency_preferences/wl_subj042/"
                       "20170823_prisma_pilot")):
    """run mri_vol2vol

    command is: mri_vol2vol --mov input_volume --reg reg_path --o out_file --fstarg

    where out_file is the input volume with the file extension (either .nii or .nii.gz) replaced by
    -vol.mgz
    """
    input_volume = os.path.join(base_path, input_volume)
    reg_path = os.path.join(base_path, reg_path)
    if ".gz" in input_volume:
        out_path = os.path.join(base_path, input_volume).replace(".nii.gz", "-vol.mgz")
    else:
        out_path = os.path.join(base_path, input_volume).replace(".nii", "-vol.mgz")
    print("vol2vol: %s to \n\t%s\n" % (input_volume, out_path))
    proc = subprocess.Popen(["mri_vol2vol", "--mov", input_volume, "--reg", reg_path, "--o",
                             out_path, "--fstarg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.communicate()


def vol2surf(hemisphere, projfrac_options=[0, 1, .05],
             input_volume="MRI_first_level/stim_class/results/R2.nii.gz",
             reg_path="preprocessed/pilot/distort2anat_tkreg.dat",
             base_path=("/mnt/Acadia/Projects/spatial_frequency_preferences/wl_subj042/"
                        "20170823_prisma_pilot")):
    """run mri_vol2surf

    command is: mri_vol2surf --mov input_volume --reg reg_path --projfrac-avg projrac_options \
                             --o out_file --hemi hemisphere

    where out_file is the input volume with the file extension (either .nii or .nii.gz) replaced by
    -rh.mgz / -lh.mgz depending on hemisphere

    hemisphere: rh/lh or right/left

    projfrac_options: must be a list of length 3.
    """
    input_volume = os.path.join(base_path, input_volume)
    reg_path = os.path.join(base_path, reg_path)
    hemisphere = {"right": "rh", "left": "lh"}.get(hemisphere, hemisphere)
    if ".gz" in input_volume:
        out_path = os.path.join(base_path, input_volume).replace(".nii.gz", "-%s.mgz" % hemisphere)
    else:
        out_path = os.path.join(base_path, input_volume).replace(".nii", "-%s.mgz" % hemisphere)
    print("vol2surf: %s to \n\t%s\n" % (input_volume, out_path))
    projfrac_options = [str(i) for i in projfrac_options]
    proc = subprocess.Popen(["mri_vol2surf", "--mov", input_volume, "--reg", reg_path,
                             "--projfrac-avg", projfrac_options[0], projfrac_options[1],
                             projfrac_options[2], "--o", out_path, "--hemi", hemisphere],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.communicate()


def main():
    results_dir = "MRI_first_level/stim_class/results"
    for vol in ['modelmd.nii.gz', 'modelse.nii.gz', 'R2.nii.gz', 'R2run.nii.gz']:
        # out, err = vol2vol(os.path.join(results_dir, vol))
        # print(out)
        # print(err)
        for hemi in ['lh', 'rh']:
            out, err = vol2surf(hemi, input_volume=os.path.join(results_dir, vol))
            print(out)
            print(err)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Realign GLMdenoise outputs to freesurfer anatomy (vol and surf)")
    parser.parse_args()
    main()
