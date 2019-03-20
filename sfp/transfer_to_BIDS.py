"""transfer_to_BIDS.py

this is a helper script that uses the Winawer lab's MRI_tools and spatial_frequency_preferences to
get the data for the Spatial frequency preferences project into the BIDS format from the way it
comes off of the prisma scanner.

However, it only does this approximately. Ideally we'll eventually have a fully-featured way of
moving it over, but this should have most of the essential details.

Before running this, you must set the path constants at the top of the file

This should be run from the BIDS code directory (by default, all paths will assume that).
"""

MRI_TOOLS_PATH = "/home/billbrod/Documents/MRI_tools"
SFP_PATH = "/home/billbrod/Documents/spatial-frequency-preferences"

import os
import sys
import argparse
import shutil
import h5py
sys.path.append(os.path.join(MRI_TOOLS_PATH, "BIDS"))
sys.path.append(SFP_PATH)
import prisma_to_BIDS
from sfp import create_BIDS_tsv
import warnings
import glob


def _BIDSify(base_dir, wl_subject_name, prisma_session, epis, sbrefs, task_label, session_label,
             PEdim, distortPE, distortrevPE, anatomy_directory, anat_nums, anat_modality_label,
             behavioral_results_path, unshuffled_stim_description_path, notes_path,
             stimuli_presentation_idx_paths, unshuffled_stimuli_path):
    BIDS_subj = "sub-" + wl_subject_name.replace('_', '')
    source_dir = os.path.join(base_dir, "sourcedata", wl_subject_name, prisma_session)
    BIDS_ses = "ses-" + session_label
    BIDS_task = "task-" + task_label
    if 'pilot' in BIDS_ses:
        full_TRs = 256
    elif BIDS_ses in ['ses-01', 'ses-02']:
        full_TRs = 240
    elif BIDS_ses in ['ses-03']:
        full_TRs = 264
    try:
        prisma_to_BIDS.copy_func(source_dir, base_dir, epis, sbrefs, task_label, PEdim,
                                 session_label)
        print("  Successfully moved over functional data")
    except IOError:
        warnings.warn("Functional data already found, skipping...")
    prisma_to_BIDS.json_check(os.path.join(base_dir, BIDS_subj, BIDS_ses, "func"))
    try:
        prisma_to_BIDS.copy_fmap(source_dir, base_dir, distortPE, distortrevPE, PEdim,
                                 session_label)
        print("  Successfully moved over field map data")
    except IOError:
        warnings.warn("Field map data already found, skipping...")
    prisma_to_BIDS.json_check(os.path.join(base_dir, BIDS_subj, BIDS_ses, "fmap"))
    try:
        prisma_to_BIDS.copy_anat(anatomy_directory, base_dir, anat_nums, anat_modality_label)
        print("  Successfully moved over anatomical data")
    except IOError:
        warnings.warn("Anatomical data already found, skipping...")
    create_BIDS_tsv.main(
        behavioral_results_path, unshuffled_stim_description_path,
        save_path = os.path.join(base_dir, BIDS_subj, BIDS_ses, "func",
                                 BIDS_subj+"_"+BIDS_ses+"_"+BIDS_task+"_run-%02d_events.tsv"),
        full_TRs=full_TRs)
    print("  Successfully moved over events tsv")
    if isinstance(behavioral_results_path, str):
        behavioral_results_path = [behavioral_results_path]
    for res in behavioral_results_path:
        if not os.path.exists(os.path.join(source_dir, os.path.split(res)[-1])):
            shutil.copy(res, os.path.join(source_dir, os.path.split(res)[-1]))
    print("  Successfully moved over behavioral data hdf5")
    if not os.path.exists(os.path.join(source_dir, os.path.split(notes_path)[-1])):
        shutil.copy(notes_path, os.path.join(source_dir, os.path.split(notes_path)[-1]))
    print("  Successfully moved over notes")
    for f in stimuli_presentation_idx_paths:
        if not os.path.exists(os.path.join(source_dir, os.path.split(f)[-1])):
            shutil.copy(f, os.path.join(source_dir, os.path.split(f)[-1]))
    print("  Successfully moved over stimuli presentation indices")
    if not os.path.isfile(os.path.join(base_dir, "stimuli", os.path.split(unshuffled_stim_description_path)[-1])):
        shutil.copy(unshuffled_stim_description_path,
                    os.path.join(base_dir, "stimuli", os.path.split(unshuffled_stim_description_path)[-1]))
        print("  Successfully moved over stimuli description csv")
    else:
        print("  Found stimuli description csv, skipping")
    if not os.path.isfile(os.path.join(base_dir, "stimuli", os.path.split(unshuffled_stimuli_path)[-1])):
        shutil.copy(unshuffled_stimuli_path,
                    os.path.join(base_dir, "stimuli", os.path.split(unshuffled_stimuli_path)[-1]))
        print("  Successfully moved over stimuli numpy array")
    else:
        print("  Found stimuli numpy array, skipping")


def wlsubj001_oct(base_dir, acadia_projects_dir):
    print("Moving wl_subj001's data from 20171007_prisma")
    anat_dir = os.path.join(acadia_projects_dir, "Anatomy", "wlsubj001", "RAS", 'raw',
                            "EK_2013_12_17_T1")
    _BIDSify(base_dir, "wl_subj001", "20171007_prisma", [10, 12, 14, 16, 18, 20, 22, 24, 26],
             [9, 11, 13, 15, 17, 19, 21, 23, 25], "sfp", "pilot01", 'j', 6, 5, anat_dir, [2, 3, 4], "T1w",
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Oct-09_wl_subj001_sess1.hdf5"),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_ses-pilot01_stim_description.csv"),
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Oct-09_wl_subj001.md"),
             glob.glob(os.path.join(SFP_PATH, "data", "stimuli", "wl_subj001_run*_idx.npy")),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_ses-pilot01_stimuli.npy"))


def wlsubj042_aug(base_dir, acadia_projects_dir):
    print("Moving wl_subj042's data from 20170823_prisma_pilot")
    anat_dir = os.path.join(acadia_projects_dir, "Retinotopy", "wlsubj042",
                            "20170713_PrismaPilot", "RAW")
    _BIDSify(base_dir, "wl_subj042", "20170823_prisma_pilot", [10, 12, 14, 16, 18, 20, 22, 24],
             [9, 11, 13, 15, 17, 19, 21, 23], "sfp", "pilot00", 'j', 6, 5, anat_dir, [15, 16], "T1w",
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Aug-23_wl_subj042_sess1.hdf5"),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_ses-pilot00_stim_description.csv"),
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Aug-23_wl_subj042.md"),
             glob.glob(os.path.join(SFP_PATH, "data", "stimuli", "wl_subj042_run*_idx.npy")),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_ses-pilot00_stimuli.npy"))


def wlsubj042_nov(base_dir, acadia_projects_dir):
    print("Moving wl_subj042's data from 20171107")
    anat_dir = os.path.join(acadia_projects_dir, "Retinotopy", "wlsubj042",
                            "20170713_PrismaPilot", "RAW")
    _BIDSify(base_dir, "wl_subj042", "20171107", [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
             [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], "sfp", "pilot01", 'j', 6, 5, anat_dir,
             [15, 16], "T1w",
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Nov-07_wl_subj042_sess0.hdf5"),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_ses-pilot01_stim_description.csv"),
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Nov-07_wl_subj042.md"),
             glob.glob(os.path.join(SFP_PATH, "data", "stimuli", "wl_subj042_run*_idx.npy")),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_ses-pilot01_stimuli.npy"))


def wlsubj045_nov(base_dir, acadia_projects_dir):
    print("Moving wl_subj045's data from 20171107")
    anat_dir = os.path.join(acadia_projects_dir, "Retinotopy", "wlsubj045",
                            "20171031_Anatomy", "RAW")
    _BIDSify(base_dir, "wl_subj045", "20171107", [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
             [9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31], "sfp", "pilot01", 'j', 6, 5, anat_dir,
             [5, 6, 7], "T1w",
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Nov-07_wl_subj045_sess0.hdf5"),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_ses-pilot01_stim_description.csv"),
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Nov-07_wl_subj045.md"),
             glob.glob(os.path.join(SFP_PATH, "data", "stimuli", "wl_subj045_run*_idx.npy")),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_ses-pilot01_stimuli.npy"))


def wlsubj001_01(base_dir, acadia_projects_dir):
    print("Moving wl_subj001's data from 20180131")
    anat_dir = os.path.join(acadia_projects_dir, "Anatomy", "wlsubj001", "RAS", 'raw',
                            "EK_2013_12_17_T1")
    _BIDSify(base_dir, "wl_subj001", "20180131", [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
             [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], "sfp", "01", 'j', 6, 5, anat_dir,
             [2, 3, 4], "T1w",
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2018-Jan-31_sub-wlsubj001_sess0.hdf5"),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_stim_description.csv"),
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2018-Jan-31_sub-wlsubj001.md"),
             glob.glob(os.path.join(SFP_PATH, "data", "stimuli", "sub-wlsubj001_run*_idx.npy")),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_stimuli.npy"))


def wlsubj042_01(base_dir, acadia_projects_dir):
    print("Moving wl_subj042's data from 20180201")
    anat_dir = os.path.join(acadia_projects_dir, "Retinotopy", "wlsubj042",
                            "20170713_PrismaPilot", "RAW")
    _BIDSify(base_dir, "wl_subj042", "20180201", [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
             [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], "sfpconstant", "01", 'j', 6, 5,
             anat_dir, [15, 16], "T1w",
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2018-Feb-01_sub-wlsubj042_sess0.hdf5"),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfpconstant_stim_description.csv"),
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2018-Feb-01_sub-wlsubj042.md"),
             glob.glob(os.path.join(SFP_PATH, "data", "stimuli", "sub-wlsubj042_run*_idx.npy")),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfpconstant_stimuli.npy"))


def wlsubj001_02(base_dir, acadia_projects_dir):
    print("Moving wl_subj001's data from 20180207")
    anat_dir = os.path.join(acadia_projects_dir, "Anatomy", "wlsubj001", "RAS", 'raw',
                            "EK_2013_12_17_T1")
    _BIDSify(base_dir, "wl_subj001", "20180207", [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
             [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], "sfpconstant", "02", 'j', 6, 5,
             anat_dir, [2, 3, 4], "T1w",
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2018-Feb-07_sub-wlsubj001_sess0.hdf5"),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfpconstant_stim_description.csv"),
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2018-Feb-07_sub-wlsubj001.md"),
             glob.glob(os.path.join(SFP_PATH, "data", "stimuli", "sub-wlsubj001_run*_idx.npy")),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfpconstant_stimuli.npy"))


def wlsubj042_02(base_dir, acadia_projects_dir):
    print("Moving wl_subj042's data from 20180209")
    anat_dir = os.path.join(acadia_projects_dir, "Retinotopy", "wlsubj042",
                            "20170713_PrismaPilot", "RAW")
    # number 24 is the incomplete run
    _BIDSify(base_dir, "wl_subj042", "20180209", [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
             [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], "sfp", "02", 'j', 6, 5,
             anat_dir, [15, 16], "T1w",
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2018-Feb-09_sub-wlsubj042_sess0.hdf5"),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_stim_description.csv"),
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2018-Feb-09_sub-wlsubj042.md"),
             glob.glob(os.path.join(SFP_PATH, "data", "stimuli", "sub-wlsubj042_run*_idx.npy")),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_stimuli.npy"))


def wlsubj045_01(base_dir, acadia_projects_dir):
    print("Moving wl_subj045's data from 20180216")
    anat_dir = os.path.join(acadia_projects_dir, "Retinotopy", "wlsubj045",
                            "20171031_Anatomy", "RAW")
    _BIDSify(base_dir, "wl_subj045", "20180216", [8, 10, 12, 14, 16, 18, 20, 22, 26, 28, 30, 32],
             [7, 9, 11, 13, 15, 17, 19, 21, 25, 27, 29, 31], "sfpconstant", "01", 'j', 6, 5,
             anat_dir, [5, 6, 7], "T1w",
             [os.path.join(SFP_PATH, "data", "raw_behavioral", "2018-Feb-16_sub-wlsubj045_sess0.hdf5"),
              os.path.join(SFP_PATH, "data", "raw_behavioral", "2018-Feb-16_sub-wlsubj045_sess1.hdf5")],
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfpconstant_stim_description.csv"),
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2018-Feb-16_sub-wlsubj045.md"),
             glob.glob(os.path.join(SFP_PATH, "data", "stimuli", "sub-wlsubj045_run*_idx.npy")),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfpconstant_stimuli.npy"))


def wlsubj045_02(base_dir, acadia_projects_dir):
    print("Moving wl_subj045's data from 20180227")
    anat_dir = os.path.join(acadia_projects_dir, "Retinotopy", "wlsubj045",
                            "20171031_Anatomy", "RAW")
    _BIDSify(base_dir, "wl_subj045", "20180227", [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
             [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], "sfp", "02", 'j', 6, 5,
             anat_dir, [5, 6, 7], "T1w",
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2018-Feb-27_sub-wlsubj045_sess0.hdf5"),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_stim_description.csv"),
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2018-Feb-27_sub-wlsubj045.md"),
             glob.glob(os.path.join(SFP_PATH, "data", "stimuli", "sub-wlsubj045_run*_idx.npy")),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_stimuli.npy"))


def wlsubj014_03(base_dir, acadia_projects_dir):
    print("Moving wl_subj014's data from 20180320")
    anat_dir = os.path.join(acadia_projects_dir, "Anatomy", "wlsubj014", "Raw")
    _BIDSify(base_dir, "wl_subj014", "20180320", [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
             [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], "sfp", "03", 'j', 6, 5,
             anat_dir, [3, 4], "T1w",
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2018-Mar-20_sub-wlsubj014_sess0.hdf5"),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_stim_description.csv"),
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2018-Mar-20_sub-wlsubj014.md"),
             glob.glob(os.path.join(SFP_PATH, "data", "stimuli", "sub-wlsubj014_run*_idx.npy")),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_stimuli.npy"))


def wlsubj004_03(base_dir, acadia_projects_dir):
    print("Moving wl_subj004's data from 20180322")
    anat_dir = os.path.join(acadia_projects_dir, "Anatomy", "wlsubj004")
    _BIDSify(base_dir, "wl_subj004", "20180322", [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
             [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], "sfp", "03", 'j', 6, 5,
             anat_dir, [1], "T1w",
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2018-Mar-22_sub-wlsubj004_sess0.hdf5"),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_stim_description.csv"),
             os.path.join(SFP_PATH, "data", "raw_behavioral", "2018-Mar-22_sub-wlsubj004.md"),
             glob.glob(os.path.join(SFP_PATH, "data", "stimuli", "sub-wlsubj004_run*_idx.npy")),
             os.path.join(SFP_PATH, "data", "stimuli", "task-sfp_stimuli.npy"))


def rename_stimuli(new_stim_name, old_stim_name="unshuffled.npy",
                   raw_behavioral_glob_str="data/raw_behavioral/2017-Aug*.hdf5"):
    """renames the stimuli in hdf5 file from old_stim_name to new_stim_name

    this is useful because 'canonical' stimuli name for this experiment was unshuffled.npy but as
    I've updated the stimuli a couple of times and so want to update the files referred to in the
    raw behavioral files (which is how we store this information) to make sure they're correct.

    NOTE that this doesn't rename the actual stimuli npy file!

    """
    # in hdf5 files, they're bytestrings, not strings.
    if type(new_stim_name) == str:
        new_stim_name = new_stim_name.encode()
    if type(old_stim_name) == str:
        old_stim_name = old_stim_name.encode()
    for f in glob.glob(raw_behavioral_glob_str):
        res = h5py.File(f)
        for k, v in res.items():
            if 'stim_path' in k and old_stim_name == os.path.split(v.value)[-1]:
                del res[k]
                res.create_dataset(k, data=v.value.replace(old_stim_name, new_stim_name))
        res.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Move some subjects from prisma to (approximate) BIDS format"))
    parser.add_argument("subject", nargs='+', type=str,
                        help=("Which subject / session to run. One or more from this list: wl_subj"
                              "001 (pilot in Oct 2017), wl_subj001-01 (Jan 31, 2018, after revising"
                              " stimuli, log-polar stimuli), wl_subj001-02 (Feb 7, 2018, with "
                              "constant stimuli) wl_subj042-0 (pilot in Aug 2017), wl_subj042-1 ("
                              "pilot in Nov 2017), wl_subj042-01 (Feb 1, 2018, after revising "
                              "stimuli, constant stimuli), wl_subj042-02 (Feb 9, 2018 with log-"
                              "polar stimuli), wl_subj045 (pilot in Nov 2017), wl_subj045-01 (Feb "
                              "16, 2018, after revising stimuli, constant stimuli), wl_subj045-02"
                              " (Feb 27, 2018, after revising stimuli, log-polar stimuli), "
                              "wl_subj014-03 (Mar 20, 2018, after revising stimuli, log-polar "
                              "stimuli, with extra time added before and after run), wl_subj004-03"
                              " (Mar 22, 2018, after revising stimuli, log-polar stimuli, with "
                              "extra time added before and after run)"))
    parser.add_argument("--base_dir", default='..',
                        help=("Base directory for the BIDS project. If unset, will assume this is"
                              "being run from the code directory within the BIDS structure."))
    parser.add_argument("--acadia_projects_dir", default="/mnt/winawerlab/Projects",
                        help=("Path to the winawerlab Projects directory. Necessary to find the "
                              "anatomical data"))
    args = vars(parser.parse_args())
    if 'wl_subj001' in args['subject']:
        wlsubj001_oct(args['base_dir'], args['acadia_projects_dir'])
    if 'wl_subj001-01' in args['subject']:
        wlsubj001_01(args['base_dir'], args['acadia_projects_dir'])
    if 'wl_subj001-02' in args['subject']:
        wlsubj001_02(args['base_dir'], args['acadia_projects_dir'])
    if 'wl_subj042-0' in args['subject']:
        wlsubj042_aug(args['base_dir'], args['acadia_projects_dir'])
    if 'wl_subj042-1' in args['subject']:
        wlsubj042_nov(args['base_dir'], args['acadia_projects_dir'])
    if 'wl_subj042-01' in args['subject']:
        wlsubj042_01(args['base_dir'], args['acadia_projects_dir'])
    if 'wl_subj042-02' in args['subject']:
        wlsubj042_02(args['base_dir'], args['acadia_projects_dir'])
    if 'wl_subj045' in args['subject']:
        wlsubj045_nov(args['base_dir'], args['acadia_projects_dir'])
    if 'wl_subj045-01' in args['subject']:
        wlsubj045_01(args['base_dir'], args['acadia_projects_dir'])
    if 'wl_subj045-02' in args['subject']:
        wlsubj045_02(args['base_dir'], args['acadia_projects_dir'])
    if 'wl_subj014-03' in args['subject']:
        wlsubj014_03(args['base_dir'], args['acadia_projects_dir'])
    if 'wl_subj004-03' in args['subject']:
        wlsubj004_03(args['base_dir'], args['acadia_projects_dir'])
