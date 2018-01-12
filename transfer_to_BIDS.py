"""transfer_to_BIDS.py

this is a helper script that uses the Winawer lab's MRI_tools and spatial_frequency_preferences to
get the data for the Spatial frequency preferences project into the BIDS format from the way it
comes off of the prisma scanner.

However, it only does this approximately. Ideally we'll eventually have a fully-featured way of
moving it over, but this should have most of the essential details.

Before running this, you must set the path constants at the top of the file

This should be run from the BIDS code directory (by default, all paths will assume that).
"""

MRI_TOOLS_PATH = "/home/billbrod/Documents/Winawer_lab_MRI_tools"
SFP_PATH = "/home/billbrod/Documents/spatial-frequency-preferences"
ACADIA_PROJECTS_PATH = "/mnt/Acadia/Projects"

import os
import sys
import argparse
import shutil
sys.path.append(os.path.join(MRI_TOOLS_PATH, "BIDS"))
sys.path.append(SFP_PATH)
import prisma_to_BIDS
from sfp import design_matrices
import warnings


def wlsubj001_oct(base_dir):
    print("Moving wl_subj001's data from 20171007_prisma")
    try:
        prisma_to_BIDS.copy_func(os.path.join(base_dir, "sourcedata", 'wl_subj001', '20171007_prisma'),
                                 base_dir, [10, 12, 14, 16, 18, 20, 22, 24, 26],
                                 [9, 11, 13, 15, 17, 19, 21, 23, 25], "sfp", "01")
        print("  Successfully moved over functional data")
    except IOError:
        warnings.warn("Functional data already found, skipping...")
    try:
        prisma_to_BIDS.copy_fmap(os.path.join(base_dir, "sourcedata", 'wl_subj001', '20171007_prisma'),
                                 base_dir, 6, 5, session_label="01")
        print("  Successfully moved over field map data")
    except IOError:
        warnings.warn("Field map data already found, skipping...")
    try:
        prisma_to_BIDS.copy_anat(os.path.join(ACADIA_PROJECTS_PATH, "Anatomy", "wl_subj001", "RAS",
                                              'raw', "EK_2013_12_17_T1"),
                                 base_dir, [2, 3, 4], 'T1w')
        print("  Successfully moved over anatomical data")
    except IOError:
        warnings.warn("Anatomical data already found, skipping...")
    design_matrices.create_all_BIDS_events_tsv(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Oct-09_wl_subj001_sess1.hdf5"),
        os.path.join(base_dir, "stimuli", "unshuffled_stim_description.csv"),
        os.path.join(base_dir, "sub-wlsubj001", "ses-01", "func",
                     "sub-wlsubj001_ses-01_task-sfp_run-%02d_events.tsv"))
    print("  Successfully moved over events tsv")
    if not os.path.isdir(os.path.join(base_dir, 'sub-wlsubj001', 'ses-01', 'beh')):
        os.makedirs(os.path.join(base_dir, 'sub-wlsubj001', 'ses-01', 'beh'))
    shutil.copy(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Oct-09_wl_subj001_sess1.hdf5"),
        os.path.join(base_dir, "sub-wlsubj001", 'ses-01', 'beh',
                     'sub-wlsubj001_ses-01_task-sfp_beh.hdf5'))
    print("  Successfully moved over behavioral data hdf5")
    shutil.copy(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Oct-09_wl_subj001.md"),
        os.path.join(base_dir, "sub-wlsubj001", 'ses-01',
                     'sub-wlsubj001_ses-01_task-sfp_notes.md'))
    print("  Successfully moved over notes")


def wlsubj042_pilot(base_dir):
    print("Moving wl_subj042's data from 20170823_prisma_pilot")
    try:
        prisma_to_BIDS.copy_func(os.path.join(base_dir, "sourcedata", 'wl_subj042', '20170823_prisma_pilot'),
                                 base_dir, [10, 12, 14, 16, 18, 20, 22, 24],
                                 [9, 11, 13, 15, 17, 19, 21, 23], "sfp", "00")
        print("  Successfully moved over functional data")
    except IOError:
        warnings.warn("Functional data already found, skipping...")
    try:
        prisma_to_BIDS.copy_fmap(os.path.join(base_dir, "sourcedata", 'wl_subj042', '20170823_prisma_pilot'),
                                 base_dir, 6, 5, session_label="00")
        print("  Successfully moved over field map data")
    except IOError:
        warnings.warn("Field map data already found, skipping...")
    try:
        prisma_to_BIDS.copy_anat(os.path.join(ACADIA_PROJECTS_PATH, "Retinotopy", "wl_subj042",
                                              "20170713_PrismaPilot", "RAW"),
                                 base_dir, [15, 16], 'T1w')
        print("  Successfully moved over anatomical data")
    except IOError:
        warnings.warn("Anatomical data already found, skipping...")
    design_matrices.create_all_BIDS_events_tsv(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Aug-23_wl_subj042_sess1.hdf5"),
        os.path.join(base_dir, "stimuli", "pilot_unshuffled_stim_description.csv"),
        os.path.join(base_dir, "sub-wlsubj042", "ses-00", "func",
                     "sub-wlsubj042_ses-00_task-sfp_run-%02d_events.tsv"))
    print("  Successfully moved over events tsv")
    if not os.path.isdir(os.path.join(base_dir, 'sub-wlsubj042', 'ses-00', 'beh')):
        os.makedirs(os.path.join(base_dir, 'sub-wlsubj042', 'ses-00', 'beh'))
    shutil.copy(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Aug-23_wl_subj042_sess1.hdf5"),
        os.path.join(base_dir, "sub-wlsubj042", 'ses-00', 'beh',
                     'sub-wlsubj042_ses-00_task-sfp_beh.hdf5'))
    print("  Successfully moved over behavioral data hdf5")
    shutil.copy(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Aug-23_wl_subj042.md"),
        os.path.join(base_dir, "sub-wlsubj042", 'ses-00',
                     'sub-wlsubj042_ses-00_task-sfp_notes.md'))
    print("  Successfully moved over notes")


def wlsubj042_nov(base_dir):
    print("Moving wl_subj042's data from 20171107")
    try:
        prisma_to_BIDS.copy_func(os.path.join(base_dir, "sourcedata", 'wl_subj042', '20171107'),
                                 base_dir, [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                                 [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], "sfp", "01")
        print("  Successfully moved over functional data")
    except IOError:
        warnings.warn("Functional data already found, skipping...")
    try:
        prisma_to_BIDS.copy_fmap(os.path.join(base_dir, "sourcedata", 'wl_subj042', '20171107'),
                                 base_dir, 6, 5, session_label="01")
        print("  Successfully moved over field map data")
    except IOError:
        warnings.warn("Field map data already found, skipping...")
    try:
        prisma_to_BIDS.copy_anat(os.path.join(ACADIA_PROJECTS_PATH, "Retinotopy", "wl_subj042",
                                              "20170713_PrismaPilot", "RAW"),
                                 base_dir, [15, 16], 'T1w')
        print("  Successfully moved over anatomical data")
    except IOError:
        warnings.warn("Anatomical data already found, skipping...")
    design_matrices.create_all_BIDS_events_tsv(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Nov-07_wl_subj042_sess0.hdf5"),
        os.path.join(base_dir, "stimuli", "unshuffled_stim_description.csv"),
        os.path.join(base_dir, "sub-wlsubj042", "ses-01", "func",
                     "sub-wlsubj042_ses-01_task-sfp_run-%02d_events.tsv"))
    print("  Successfully moved over events tsv")
    if not os.path.isdir(os.path.join(base_dir, 'sub-wlsubj042', 'ses-01', 'beh')):
        os.makedirs(os.path.join(base_dir, 'sub-wlsubj042', 'ses-01', 'beh'))
    shutil.copy(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Nov-07_wl_subj042_sess0.hdf5"),
        os.path.join(base_dir, "sub-wlsubj042", 'ses-01', 'beh',
                     'sub-wlsubj042_ses-01_task-sfp_beh.hdf5'))
    print("  Successfully moved over behavioral data hdf5")
    shutil.copy(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Nov-07_wl_subj042.md"),
        os.path.join(base_dir, "sub-wlsubj042", 'ses-01',
                     'sub-wlsubj042_ses-01_task-sfp_notes.md'))
    print("  Successfully moved over notes")


def wlsubj045_nov(base_dir):
    print("Moving wl_subj045's data from 20171107")
    try:
        prisma_to_BIDS.copy_func(os.path.join(base_dir, "sourcedata", 'wl_subj045', '20171107'),
                                 base_dir, [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
                                 [9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31], "sfp", "01")
        print("  Successfully moved over functional data")
    except IOError:
        warnings.warn("Functional data already found, skipping...")
    try:
        prisma_to_BIDS.copy_fmap(os.path.join(base_dir, "sourcedata", 'wl_subj045', '20171107'),
                                 base_dir, 6, 5, session_label="01")
        print("  Successfully moved over field map data")
    except IOError:
        warnings.warn("Field map data already found, skipping...")
    try:
        prisma_to_BIDS.copy_anat(os.path.join(ACADIA_PROJECTS_PATH, "Retinotopy", "wl_subj045",
                                              "20171031_Anatomy", "RAW"),
                                 base_dir, [5, 6, 7], 'T1w')
        print("  Successfully moved over anatomical data")
    except IOError:
        warnings.warn("Anatomical data already found, skipping...")
    design_matrices.create_all_BIDS_events_tsv(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Nov-07_wl_subj045_sess0.hdf5"),
        os.path.join(base_dir, "stimuli", "unshuffled_stim_description.csv"),
        os.path.join(base_dir, "sub-wlsubj045", "ses-01", "func",
                     "sub-wlsubj045_ses-01_task-sfp_run-%02d_events.tsv"))
    print("  Successfully moved over events tsv")
    if not os.path.isdir(os.path.join(base_dir, 'sub-wlsubj045', 'ses-01', 'beh')):
        os.makedirs(os.path.join(base_dir, 'sub-wlsubj045', 'ses-01', 'beh'))
    shutil.copy(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Nov-07_wl_subj045_sess0.hdf5"),
        os.path.join(base_dir, "sub-wlsubj045", 'ses-01', 'beh',
                     'sub-wlsubj045_ses-01_task-sfp_beh.hdf5'))
    print("  Successfully moved over behavioral data hdf5")
    shutil.copy(
        os.path.join(SFP_PATH, "data", "raw_behavioral", "2017-Nov-07_wl_subj045.md"),
        os.path.join(base_dir, "sub-wlsubj045", 'ses-01',
                     'sub-wlsubj045_ses-01_task-sfp_notes.md'))
    print("  Successfully moved over notes")


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
    args = vars(parser.parse_args())
    if 'wl_subj001' in args['subject']:
        wlsubj001_oct(args['base_dir'])
    if 'wl_subj042-0' in args['subject']:
        wlsubj042_pilot(args['base_dir'])
    if 'wl_subj042-1' in args['subject']:
        wlsubj042_nov(args['base_dir'])
    if 'wl_subj045' in args['subject']:
        wlsubj045_nov(args['base_dir'])
