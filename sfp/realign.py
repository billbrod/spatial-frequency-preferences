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
import glob


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


def main(results_dir, reg_path, base_path, vol_mode='all', projfrac_options=[0, 1, .05]):
    """run mri_vol2surf and mri_vol2vol

    vol_mode: {'summary', 'classes', 'all'}. which vols to realign to Freesurfer space. If
    'summary', then will do ['modelmd.nii.gz', 'modelse.nii.gz', 'R2.nii.gz', 'R2run.nii.gz']. If
    'classes', then will do 'models_niftis/models_class_*.nii.gz', finding all volumes that match
    that pattern. If 'all', will do both of these.
    """
    vols = []
    if vol_mode in ['summary', 'all']:
        vols += ['modelmd.nii.gz', 'modelse.nii.gz', 'R2.nii.gz', 'R2run.nii.gz']
    if vol_mode in ['classes', 'all']:
        vols += glob.glob('models_niftis/models_class_*.nii.gz')
    for vol in vols:
        out, err = vol2vol(os.path.join(results_dir, vol), reg_path, base_path)
        print(out)
        print(err)
        for hemi in ['lh', 'rh']:
            out, err = vol2surf(hemi, projfrac_options, os.path.join(results_dir, vol), reg_path,
                                base_path)
            print(out)
            print(err)


if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        description=("Realign GLMdenoise outputs to freesurfer anatomy (vol and surf). Will run "
                     "`mri_vol2vol --mov input_volume --reg reg_path --o out_file --fstarg` and "
                     "`mri_vol2surf --mov input_volume --reg reg_path --projfrac-avg "
                     "projrac_options --o out_file --hemi hemisphere` on all results files."),
        formatter_class=CustomFormatter)
    parser.add_argument("base_path",
                        help=("Path to directory that contains all of your results and source "
                              "registration files. This is mainly provided as a convenience so "
                              "your paths for results_dir and reg_path can be shorter. For example"
                              ", if your registration file lives at /this/is/a/directory/and/file."
                              "dat and your results are in /this/is/a/directory/results/files/, "
                              "then base_path=/this/is/a/directory, while reg_path=and/file.dat, "
                              "and results_dir=results/files/"))
    parser.add_argument("results_dir",
                        help=("Directory (relative to base_path) that contains your results files."
                              " The specific results files we'll look for are specified by "
                              "vol_mode. See help for base_path for an example."))
    parser.add_argument("reg_path",
                        help=("Path (relative to base_path) to source registration file. See help "
                              "for base_path for an example."))
    parser.add_argument("--vol_mode", '-v',
                        help=("{summary, classes, all}. Which vols to realign to Freesurfer space."
                              " If 'summary', then will do ['modelmd.nii.gz', 'modelse.nii.gz', "
                              "'R2.nii.gz', 'R2run.nii.gz']. If 'classes', then will do 'models_"
                              "niftis/models_class_*.nii.gz', finding all volumes that match that"
                              " pattern. If 'all', will do both of these."))
    parser.add_argument("--projfrac_options", '-p', nargs=3, default=[0, 1, .05],
                        help=("Options to pass to mri_vol2surf for the projfrac-avg flag. From "
                              "mri_vol2surf's help: Options for projecting along the surface "
                              "normal; average along normal"))
    args = vars(parser.parse_args())
    if args['vol_mode'] not in ['summary', 'classes', 'all']:
        raise Exception("Unable to align vols specified by vol_mode %s!" % args['vol_mode'])
    main(**args)
