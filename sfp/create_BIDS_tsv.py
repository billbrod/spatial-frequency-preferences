#!/usr/bin/python
"""create BIDS tsv files from raw behavioral data
"""
import numpy as np
import argparse
import h5py
import warnings
import pandas as pd
from collections import Counter
import os


def create_tsv_df(behavioral_results, unshuffled_stim_description, run_num):
    """create and return the df with info for BIDS tsv for a run

    behavioral_results: h5py File (not the path) containing behavioral results

    unshuffled_stim_description: dataframe containing info on the stimuli

    run_num: int, which run (as used in behavioral_results) to examine

    returns the design dataframe
    """
    df = unshuffled_stim_description[['index', 'class_idx']].set_index('index')
    if len(df) != len(behavioral_results['run_%02d_shuffled_indices' % run_num]):
        raise Exception("Behavioral results and stimulus description csv have different numbers of stimuli!")
    df = df.reindex(behavioral_results['run_%02d_shuffled_indices' % run_num][()])
    timing = behavioral_results['run_%02d_timing_data' % run_num][()]
    # Because we want to skip the first one and drop the last nblanks * 2 (since for each stimuli
    # we have two entries: one for on, one for off). Finally, we only grab every other because we
    # only want the on timing
    try:
        # in this case, it's the initial way we ran the experiment, where we had no (additional)
        # blanks at the beginning of the run, only at the end, and so the field was just nblanks
        timing = timing[1:-behavioral_results['run_%02d_nblanks' % run_num][()]*2:2]
    except KeyError:
        # in this case, it's the later way we ran the experiment, where we had(additional) blanks
        # at the beginning and end of the run
        timing = timing[behavioral_results['run_%02d_init_nblanks' % run_num][()]*2+1:
                        -behavioral_results['run_%02d_final_nblanks' % run_num][()]*2:2]
    # Now we get rid of the first TR
    initial_TR_time = float(behavioral_results['run_%02d_button_presses' % run_num][()][0][1])
    timing = [float(i[2]) - initial_TR_time for i in timing]
    # and add to our dataframe
    df['Onset time (sec)'] = timing
    button_presses = behavioral_results['run_%02d_button_presses' % run_num][()].astype(float)
    button_presses = button_presses[button_presses[:, 0] != 5]
    button_presses[:, 1] -= initial_TR_time
    return df


def _find_timing_from_results(results, run_num):
    """find stimulus directly timing from results hdf5
    """
    timing = pd.DataFrame(results['run_%02d_timing_data' % run_num][()], columns=['stimulus', 'event_type', 'timing'])
    timing.timing = timing.timing.astype(float)
    timing.timing = timing.timing.apply(lambda x: x - timing.timing.iloc[0])
    # the first entry is the start, which doesn't correspond to any stimuli
    timing = timing.drop(0)
    # this way the stimulus column just contains the stimulus number
    timing.stimulus = timing.stimulus.apply(lambda x: int(x.replace(b'stimulus_', b'')))
    # this way we get the duration of time that the stimulus was on, for each stimulus (sorting by
    # stimulus shouldn't be necessary, but just ensures that the values line up correctly)
    times = (timing[timing.event_type == b'off'].sort_values('stimulus').timing.values -
             timing[timing.event_type == b'on'].sort_values('stimulus').timing.values)
    times = np.round(times, 2)
    assert times.max() - times.min() < .050, "Stimulus timing differs by more than 40 msecs!"
    warnings.warn("Stimulus timing varies by up to %.03f seconds!" % (times.max() - times.min()))
    return Counter(times).most_common(1)[0][0]


def main(behavioral_results_path, unshuffled_stim_descriptions_path, stimulus_file_name=None,
         save_path='data/MRI_first_level/run_%02d_events.tsv', full_TRs=264):
    """create and save BIDS events tsvs for all runs.

    we do this for all non-empty runs in the h5py File found at behavioral_results_path

    save_path should contain some string formatting symbol (e.g., %s, %02d) that can indicate the
    run number and should end in .tsv

    full_TRs: int. the number of TRs in a run if it went to completion. 256 for the pilot sessions,
    240 for the sessions 1 and 2, 264 for session 3.
    """
    if isinstance(behavioral_results_path, str):
        behavioral_results_path = [behavioral_results_path]
    results_files = [h5py.File(p) for p in behavioral_results_path]
    df = pd.read_csv(unshuffled_stim_descriptions_path)
    if not os.path.exists(os.path.dirname(save_path)) and os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path))
    save_num = 1
    for results in results_files:
        run_num = 0
        while "run_%02d_button_presses" % run_num in results.keys():
            # we had to store strings as byte-strings to make it hdf5 compliant (see
            # psychopy_example._convert_str for more details), so we need to use byte-strings to
            # double check things
            n_TRs = sum([b'5' in i[0] for i in results['run_%02d_button_presses' % run_num][()]])
            if n_TRs == full_TRs:
                design_df = create_tsv_df(results, df, run_num)
                design_df = design_df.reset_index().rename(
                    columns={'index': 'stim_file_index', 'class_idx': 'trial_type',
                             'Onset time (sec)': 'onset'})
                design_df.trial_type = design_df.trial_type.replace({np.nan: design_df.trial_type.max()+1})
                design_df['trial_type'] = design_df['trial_type'].astype(int)
                design_df['duration'] = _find_timing_from_results(results, run_num)
                if stimulus_file_name is None:
                    # we decode this (from bytes to string) because we want this to be a string
                    stim_path = results['run_%02d_stim_path' % run_num][()].decode()
                    stim_path = stim_path.replace('data/stimuli/', '')
                    design_df['stim_file'] = stim_path
                else:
                    design_df['stim_file'] = stimulus_file_name
                # blank fields should be n/a
                design_df['note'] = "n/a"
                design_df.loc[design_df.trial_type == design_df.trial_type.max(), 'note'] = "blank trial"
                # design_df = design_df[['onset', 'duration', 'trial_type', 'stim_file', 'stim_file_index', 'note']]
                design_df['onset'] = design_df.onset.apply(lambda x: "%.03f" % x)
                # reorder the columns so that onset is first and duration is second (otherwise it
                # won't pass the bids validator)
                design_df = design_df[['onset', 'duration', 'trial_type', 'stim_file',
                                       'stim_file_index', 'note']]
                design_df.to_csv(save_path % (save_num), '\t', index=False)
                save_num += 1
            run_num += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=("Create and save BIDS events tsv for all runs."))
    parser.add_argument("behavioral_results_path",
                        help=("The path to the hdf5 raw behavioral file created by the "
                              "psychopy_example.py script when running the experiment"))
    parser.add_argument("unshuffled_stim_descriptions_path",
                        help=("Path to the csv that describes the stimuli (we use this to determine"
                              "which stimuli are in which stimulus class)"))
    parser.add_argument("--stimulus_file_name", "-s", default=None,
                        help=("Name of the stimulus (npy) file. If None (default), will use the "
                              "value found in the behavioral results hdf5. Otherwise will use this "
                              "(as all runs use same file, don't need multiple values here). This "
                              "is useful if you called the stimulus file one thing at run-time "
                              "(which will be stored in the hdf5 file) and have renamed it in your"
                              " BIDS directory (which is what you want in your tsv)"))
    # the doubling of the % in the help string escapes the character so python doesn't think I want
    # to format the string
    parser.add_argument("--save_path", "-p", default="data/MRI_first_level/run_%02d_events.tsv",
                        help=("Template path that specifies where we should save the created tsv"
                              " file. This should contain %%02d so we can indicate the run number"
                              " and should end in .tsv"))
    args = vars(parser.parse_args())
    main(**args)
