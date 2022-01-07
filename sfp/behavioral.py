#!/usr/bin/env python3

"""Functions for behavioral analyses.
"""
from .first_level_analysis import _add_freq_metainfo


def create_outcome_df(trials_df, stim_df):
    """Create df summarizing run outcome.

    This operates on a single run.

    Parameters
    ----------
    trials_df : pd.DataFrame
        The dataframe (saved as *_events.tsv) summarizing the behavior during the scan.
    stim_df : pd.DataFrame
        The stimulus description dataframe

    Returns
    -------
    outcomes : pd.DataFrame
        The dataframe summarizing the outcomes

    """
    # add metainfo to stim_df
    stim_df = _add_freq_metainfo(stim_df)
    # drop all unnecessary rows (those that have no behavioral info)
    trials_df = trials_df.dropna(subset=['outcome'])
    # get stimulus info into trials_df
    trials_df = trials_df.merge(stim_df, 'left', left_on='stim_file_index', right_on='index')
    # remove unnecessary columns
    trials_df = trials_df.drop(columns=['stim_file', 'stim_file_index', 'duration', 'onset', 'note',
                                        'phi', 'res', 'index', 'trial_type'])
    # name this more informative
    trials_df = trials_df.rename(columns={'class_idx': 'stimulus_class'})

    # the n/a and baseline stimulus_superclass mean the same thing: blank
    # midgray screen
    trials_df.stimulus_superclass = trials_df.stimulus_superclass.fillna('baseline')
    # get the count of each outcome per stimulus superclass
    outcomes = trials_df.groupby('stimulus_superclass').outcome.value_counts()
    outcomes = outcomes.reset_index(name='n_trials')

    def group_outcome(x):
        if x in ['correct_rejection', 'false_alarm']:
            return 'absent'
        elif x in ['hit', 'miss']:
            return 'present'

    # we'll want this for the heatmap, because we want to know the what
    # percentage of hits they got out of the the trials where the target was
    # present, not out of all trials
    outcomes['outcome_supercategory'] = outcomes.outcome.apply(group_outcome)
    return outcomes
