"""transfer_to_BIDS.py

this is a helper script that uses the Winawer lab's MRI_tools and spatial_frequency_preferences to
get the data for the Spatial frequency preferences project into the BIDS format from the way it
comes off of the prisma scanner.

However, it only does this approximately. Ideally we'll eventually have a fully-featured way of
moving it over, but this should have most of the essential details.

Before running this, you must set the path constants at the top of the file

This should be run from the BIDS code directory (by default, all paths will assume that).
"""

MRI_TOOLS_PATH = "/home/wfb229/MRI_tools"
SFP_PATH = "/home/wfb229/spatial-frequency-preferences"

import os
import sys
import argparse
import shutil
import h5py
sys.path.append(os.path.join(MRI_TOOLS_PATH, "BIDS"))
sys.path.append(SFP_PATH)
import prisma_to_BIDS
from sfp import design_matrices
import warnings
import glob


def wlsubj001_oct(base_dir, acadia_projects_dir):
    print("Moving wl_subj001's data from 20171007_prisma")
    try:
        prisma_to_BIDS.copy_func(os.path.join(base_dir, "sourcedata", 'wl_subj001', '20171007_prisma'),
                                 base_dir, [10, 12, 14, 16, 18, 20, 22, 24, 26],
                                 [9, 11, 13, 15, 17, 19, 21, 23, 25], "sfp",
                                 session_label="pilot01")
        print("  Successfully moved over functional data")
    except IOError:
        warnings.warn("Functional data already found, skipping...")
    prisma_to_BIDS.json_check(os.path.join(base_dir, "sub-wlsubj001", "ses-pilot01", "func"))
    try:
        prisma_to_BIDS.copy_fmap(os.path.join(base_dir, "sourcedata", 'wl_subj001', '20171007_prisma'),
                                 base_dir, 6, 5, session_label="pilot01")
        print("  Successfully moved over field map data")
    except IOError:
        warnings.warn("Field map data already found, skipping...")
    prisma_to_BIDS.json_check(os.path.join(base_dir, "sub-wlsubj001", "ses-pilot01", "fmap"))
    try:
        prisma_to_BIDS.copy_anat(os.path.join(acadia_projects_dir, "Anatomy", "wl_subj001", "RAS",
                                              'raw', "EK_2013_12_17_T1"),
                                 base_dir, [2, 3, 4], 'T1w')
        print("  Successfully moved over anatomical data")
    except IOError:
        warnings.warn("Anatomical data already found, skipping...")
    design_matrices.create_all_BIDS_events_tsv(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Oct-09_wl_subj001_sess1.hdf5"),
        os.path.join(SFP_PATH, "data", "stimuli", "pilot01_unshuffled_stim_description.csv"),
        os.path.join(base_dir, "sub-wlsubj001", "ses-pilot01", "func",
                     "sub-wlsubj001_ses-pilot01_task-sfp_run-%02d_events.tsv"))
    print("  Successfully moved over events tsv")
    if not os.path.isdir(os.path.join(base_dir, 'sub-wlsubj001', 'ses-pilot01', 'beh')):
        os.makedirs(os.path.join(base_dir, 'sub-wlsubj001', 'ses-pilot01', 'beh'))
    shutil.copy(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Oct-09_wl_subj001_sess1.hdf5"),
        os.path.join(base_dir, "sourcedata", "wl_subj001", '20171007_prisma',
                     '2017-Oct-09_wl_subj001_sess1.hdf5'))
    print("  Successfully moved over behavioral data hdf5")
    shutil.copy(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Oct-09_wl_subj001.md"),
        os.path.join(base_dir, "sourcedata", "wl_subj001", '20171007_prisma',
                     '2017-Oct-09_wl_subj001_notes.md'))
    print("  Successfully moved over notes")
    for f in glob.glob(os.path.join(SFP_PATH, "data", "stimuli", "wl_subj001_run*_idx.npy")):
        shutil.copy(f, os.path.join(base_dir, "sourcedata", "wl_subj001", '20171007_prisma',
                                    os.path.split(f)[1]))
    print("  Successfully moved over stimuli presentation indices")
    if not os.path.isfile(os.path.join(base_dir, "stimuli", "pilot01_unshuffled_stim_description.csv")):
        shutil.copy(os.path.join(SFP_PATH, "data", "stimuli", "pilot01_unshuffled_stim_description.csv"),
                    os.path.join(base_dir, "stimuli", "pilot01_unshuffled_stim_description.csv"))
        print("  Successfully moved over stimuli description csv")
    else:
        print("  Found stimuli description csv, skipping")
    if not os.path.isfile(os.path.join(base_dir, "stimuli", "pilot01_unshuffled.npy")):
        shutil.copy(os.path.join(SFP_PATH, "data", "stimuli", "pilot01_unshuffled.npy"),
                    os.path.join(base_dir, "stimuli", "pilot01_unshuffled.npy"))
        print("  Successfully moved over stimuli numpy array")
    else:
        print("  Found stimuli numpy array, skipping")


def wlsubj042_aug(base_dir, acadia_projects_dir):
    print("Moving wl_subj042's data from 20170823_prisma_pilot")
    try:
        prisma_to_BIDS.copy_func(os.path.join(base_dir, "sourcedata", 'wl_subj042', '20170823_prisma_pilot'),
                                 base_dir, [10, 12, 14, 16, 18, 20, 22, 24],
                                 [9, 11, 13, 15, 17, 19, 21, 23], "sfp", session_label="pilot00")
        print("  Successfully moved over functional data")
    except IOError:
        warnings.warn("Functional data already found, skipping...")
    prisma_to_BIDS.json_check(os.path.join(base_dir, "sub-wlsubj042", "ses-pilot00", "func"))
    try:
        prisma_to_BIDS.copy_fmap(os.path.join(base_dir, "sourcedata", 'wl_subj042', '20170823_prisma_pilot'),
                                 base_dir, 6, 5, session_label="pilot00")
        print("  Successfully moved over field map data")
    except IOError:
        warnings.warn("Field map data already found, skipping...")
    prisma_to_BIDS.json_check(os.path.join(base_dir, "sub-wlsubj042", "ses-pilot00", "fmap"))
    try:
        prisma_to_BIDS.copy_anat(os.path.join(acadia_projects_dir, "Retinotopy", "wl_subj042",
                                              "20170713_PrismaPilot", "RAW"),
                                 base_dir, [15, 16], 'T1w')
        print("  Successfully moved over anatomical data")
    except IOError:
        warnings.warn("Anatomical data already found, skipping...")
    design_matrices.create_all_BIDS_events_tsv(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Aug-23_wl_subj042_sess1.hdf5"),
        os.path.join(SFP_PATH, "data", "stimuli", "pilot00_unshuffled_stim_description.csv"),
        os.path.join(base_dir, "sub-wlsubj042", "ses-pilot00", "func",
                     "sub-wlsubj042_ses-pilot00_task-sfp_run-%02d_events.tsv"))
    print("  Successfully moved over events tsv")
    if not os.path.isdir(os.path.join(base_dir, 'sub-wlsubj042', 'ses-pilot00', 'beh')):
        os.makedirs(os.path.join(base_dir, 'sub-wlsubj042', 'ses-pilot00', 'beh'))
    shutil.copy(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Aug-23_wl_subj042_sess1.hdf5"),
        os.path.join(base_dir, "sourcedata", "wl_subj042", '20170823_prisma_pilot',
                     '2017-Aug-23_wl_subj042_sess1.hdf5'))
    print("  Successfully moved over behavioral data hdf5")
    shutil.copy(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Aug-23_wl_subj042.md"),
        os.path.join(base_dir, "sourcedata", "wl_subj042", '20170823_prisma_pilot',
                     '2017-Aug-23_wl_subj042_notes.md'))
    print("  Successfully moved over notes")
    for f in glob.glob(os.path.join(SFP_PATH, "data", "stimuli", "wl_subj042_run*_idx.npy")):
        shutil.copy(f, os.path.join(base_dir, "sourcedata", "wl_subj042", '201710823_prisma_pilot',
                                    os.path.split(f)[1]))
    print("  Successfully moved over stimuli presentation indices")
    if not os.path.isfile(os.path.join(base_dir, "stimuli", "pilot00_unshuffled_stim_description.csv")):
        shutil.copy(os.path.join(SFP_PATH, "data", "stimuli", "pilot00_unshuffled_stim_description.csv"),
                    os.path.join(base_dir, "stimuli", "pilot00_unshuffled_stim_description.csv"))
        print("  Successfully moved over stimuli description csv")
    else:
        print("  Found stimuli description csv, skipping")
    if not os.path.isfile(os.path.join(base_dir, "stimuli", "pilot00_unshuffled.npy")):
        shutil.copy(os.path.join(SFP_PATH, "data", "stimuli", "pilot00_unshuffled.npy"),
                    os.path.join(base_dir, "stimuli", "pilot00_unshuffled.npy"))
        print("  Successfully moved over stimuli numpy array")
    else:
        print("  Found stimuli numpy array, skipping")


def wlsubj042_nov(base_dir, acadia_projects_dir):
    print("Moving wl_subj042's data from 20171107")
    try:
        prisma_to_BIDS.copy_func(os.path.join(base_dir, "sourcedata", 'wl_subj042', '20171107'),
                                 base_dir, [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                                 [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], "sfp",
                                 session_label="pilot01")
        print("  Successfully moved over functional data")
    except IOError:
        warnings.warn("Functional data already found, skipping...")
    prisma_to_BIDS.json_check(os.path.join(base_dir, "sub-wlsubj042", "ses-pilot01", "func"))
    try:
        prisma_to_BIDS.copy_fmap(os.path.join(base_dir, "sourcedata", 'wl_subj042', '20171107'),
                                 base_dir, 6, 5, session_label="pilot01")
        print("  Successfully moved over field map data")
    except IOError:
        warnings.warn("Field map data already found, skipping...")
    prisma_to_BIDS.json_check(os.path.join(base_dir, "sub-wlsubj042", "ses-pilot01", "fmap"))
    try:
        prisma_to_BIDS.copy_anat(os.path.join(acadia_projects_dir, "Retinotopy", "wl_subj042",
                                              "20170713_PrismaPilot", "RAW"),
                                 base_dir, [15, 16], 'T1w')
        print("  Successfully moved over anatomical data")
    except IOError:
        warnings.warn("Anatomical data already found, skipping...")
    design_matrices.create_all_BIDS_events_tsv(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Nov-07_wl_subj042_sess0.hdf5"),
        os.path.join(SFP_PATH, "data", "stimuli", "pilot01_unshuffled_stim_description.csv"),
        os.path.join(base_dir, "sub-wlsubj042", "ses-pilot01", "func",
                     "sub-wlsubj042_ses-pilot01_task-sfp_run-%02d_events.tsv"))
    print("  Successfully moved over events tsv")
    if not os.path.isdir(os.path.join(base_dir, 'sub-wlsubj042', 'ses-pilot01', 'beh')):
        os.makedirs(os.path.join(base_dir, 'sub-wlsubj042', 'ses-pilot01', 'beh'))
    shutil.copy(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Nov-07_wl_subj042_sess0.hdf5"),
        os.path.join(base_dir, "sourcedata", "wl_subj042", '20171107',
                     '2017-Nov-07_wl_subj042_sess0.hdf5'))
    print("  Successfully moved over behavioral data hdf5")
    shutil.copy(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Nov-07_wl_subj042.md"),
        os.path.join(base_dir, "sourcedata", "wl_subj042", '20171107',
                     '2017-Nov-07_wl_subj042_notes.md'))
    print("  Successfully moved over notes")
    for f in glob.glob(os.path.join(SFP_PATH, "data", "stimuli", "wl_subj042_pilot_run*_idx.npy")):
        shutil.copy(f, os.path.join(base_dir, "sourcedata", "wl_subj042", '20171107',
                                    os.path.split(f)[1]))
    print("  Successfully moved over stimuli presentation indices")
    if not os.path.isfile(os.path.join(base_dir, "stimuli", "pilot01_unshuffled_stim_description.csv")):
        shutil.copy(os.path.join(SFP_PATH, "data", "stimuli", "pilot01_unshuffled_stim_description.csv"),
                    os.path.join(base_dir, "stimuli", "pilot01_unshuffled_stim_description.csv"))
        print("  Successfully moved over stimuli description csv")
    else:
        print("  Found stimuli description csv, skipping")
    if not os.path.isfile(os.path.join(base_dir, "stimuli", "pilot01_unshuffled.npy")):
        shutil.copy(os.path.join(SFP_PATH, "data", "stimuli", "pilot01_unshuffled.npy"),
                    os.path.join(base_dir, "stimuli", "pilot01_unshuffled.npy"))
        print("  Successfully moved over stimuli numpy array")
    else:
        print("  Found stimuli numpy array, skipping")


def wlsubj045_nov(base_dir, acadia_projects_dir):
    print("Moving wl_subj045's data from 20171107")
    try:
        prisma_to_BIDS.copy_func(os.path.join(base_dir, "sourcedata", 'wl_subj045', '20171107'),
                                 base_dir, [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
                                 [9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31], "sfp",
                                 session_label="pilot01")
        print("  Successfully moved over functional data")
    except IOError:
        warnings.warn("Functional data already found, skipping...")
    prisma_to_BIDS.json_check(os.path.join(base_dir, "sub-wlsubj045", "ses-pilot01", "func"))
    try:
        prisma_to_BIDS.copy_fmap(os.path.join(base_dir, "sourcedata", 'wl_subj045', '20171107'),
                                 base_dir, 6, 5, session_label="pilot01")
        print("  Successfully moved over field map data")
    except IOError:
        warnings.warn("Field map data already found, skipping...")
    prisma_to_BIDS.json_check(os.path.join(base_dir, "sub-wlsubj045", "ses-pilot01", "fmap"))
    try:
        prisma_to_BIDS.copy_anat(os.path.join(acadia_projects_dir, "Retinotopy", "wl_subj045",
                                              "20171031_Anatomy", "RAW"),
                                 base_dir, [5, 6, 7], 'T1w')
        print("  Successfully moved over anatomical data")
    except IOError:
        warnings.warn("Anatomical data already found, skipping...")
    design_matrices.create_all_BIDS_events_tsv(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Nov-07_wl_subj045_sess0.hdf5"),
        os.path.join(SFP_PATH, "data", "stimuli", "pilot01_unshuffled_stim_description.csv"),
        os.path.join(base_dir, "sub-wlsubj045", "ses-pilot01", "func",
                     "sub-wlsubj045_ses-pilot01_task-sfp_run-%02d_events.tsv"))
    print("  Successfully moved over events tsv")
    if not os.path.isdir(os.path.join(base_dir, 'sub-wlsubj045', 'ses-pilot01', 'beh')):
        os.makedirs(os.path.join(base_dir, 'sub-wlsubj045', 'ses-pilot01', 'beh'))
    shutil.copy(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Nov-07_wl_subj045_sess0.hdf5"),
        os.path.join(base_dir, "sourcedata", "wl_subj045", '20171107',
                     '2017-Nov-07_wl_subj045_sess0.hdf5'))
    print("  Successfully moved over behavioral data hdf5")
    shutil.copy(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Nov-07_wl_subj045.md"),
        os.path.join(base_dir, "sourcedata", "wl_subj045", '20171107',
                     '2017-Nov-07_wl_subj045_notes.md'))
    print("  Successfully moved over notes")
    for f in glob.glob(os.path.join(SFP_PATH, "data", "stimuli", "wl_subj045_run*_idx.npy")):
        shutil.copy(f, os.path.join(base_dir, "sourcedata", "wl_subj045", '20171107',
                                    os.path.split(f)[1]))
    print("  Successfully moved over stimuli presentation indices")
    if not os.path.isfile(os.path.join(base_dir, "stimuli", "pilot01_unshuffled_stim_description.csv")):
        shutil.copy(os.path.join(SFP_PATH, "data", "stimuli", "pilot01_unshuffled_stim_description.csv"),
                    os.path.join(base_dir, "stimuli", "pilot01_unshuffled_stim_description.csv"))
        print("  Successfully moved over stimuli description csv")
    else:
        print("  Found stimuli description csv, skipping")
    if not os.path.isfile(os.path.join(base_dir, "stimuli", "pilot01_unshuffled.npy")):
        shutil.copy(os.path.join(SFP_PATH, "data", "stimuli", "pilot01_unshuffled.npy"),
                    os.path.join(base_dir, "stimuli", "pilot01_unshuffled.npy"))
        print("  Successfully moved over stimuli numpy array")
    else:
        print("  Found stimuli numpy array, skipping")


def rename_stimuli(new_stim_name, old_stim_name="unshuffled.npy",
                   raw_behavioral_glob_str="data/raw_behavioral/2017-Aug*.hdf5"):
    """renames the stimuli in hdf5 file from old_stim_name to new_stim_name

    this is useful because 'canonical' stimuli name for this experiment is unshuffled.npy but as
    I've updated the stimuli a couple of times and so want to update the files referred to in the
    raw behavioral files (which is how we store this information) to make sure they're correct. I
    always want to keep the current stimuli as unshuffled.npy, and so would like to rename the old
    stimuli to pilot00_unshuffled.npy or something similar.

    NOTE that this doesn't rename the actual stimuli npy file!
    """
    for f in glob.glob(raw_behavioral_glob_str):
        res = h5py.File(f)
        for k, v in res.iteritems():
            if 'stim_path' in k and old_stim_name == os.path.split(v.value)[-1]:
                del res[k]
                res.create_dataset(k, data=v.value.replace(old_stim_name, new_stim_name))
        res.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Move some subjects from prisma to (approximate) BIDS format"))
    parser.add_argument("subject", nargs='+', type=str,
                        help=("Which subjects / session to run. Must come from this list: wl_subj"
                              "001, wl_subj042-0 (pilot in Aug 2017), wl_subj042-1 (Nov 2017), "
                              "wl_subj045"))
    parser.add_argument("--base_dir", default='..',
                        help=("Base directory for the BIDS project. If unset, will assume this is"
                              "being run from the code directory within the BIDS structure."))
    parser.add_argument("--acadia_projects_dir", default="/mnt/Acadia/Projects",
                        help=("Path to the Acadia Projects directory. Necessary to find the "
                              "anatomical data"))
    args = vars(parser.parse_args())
    if 'wl_subj001' in args['subject']:
        wlsubj001_oct(args['base_dir'], args['acadia_projects_dir'])
    if 'wl_subj042-0' in args['subject']:
        wlsubj042_aug(args['base_dir'], args['acadia_projects_dir'])
    if 'wl_subj042-1' in args['subject']:
        wlsubj042_nov(args['base_dir'], args['acadia_projects_dir'])
    if 'wl_subj045' in args['subject']:
        wlsubj045_nov(args['base_dir'], args['acadia_projects_dir'])
