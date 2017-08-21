#!/usr/bin/python
"""script to run the experiment

make sure you've already generated and saved the stimuli you want (using stimuli.main). They should
be in the order you want for that run.

also make sure that you've already set up the monitor you'll be using in PsychoPy's monitor center
"""

import numpy as np
from psychopy import visual, core, event
import h5py
import datetime
import glob


def _set_params(stim_filename, session_length=30, refresh_rate=60, on_msec_length=300,
                off_msec_length=200, fix_length_range=(1, 3), final_blank_sec_length=8,
                size=[1080, 1080], monitor='test', units='pix', fullscr=True, screen=1,
                **monitor_kwargs):
    """set the various experiment parameters
    """
    stimuli = np.load(stim_filename)
    expt_params = {'session_length_frames': session_length * refresh_rate}
    expt_params['on_frame_length'] = np.round(on_msec_length / (1000. / refresh_rate))
    expt_params['off_frame_length'] = np.round(off_msec_length / (1000. / refresh_rate))
    # we want to convert the range of time that the fixation dot will remain the same color from
    # seconds to frames
    expt_params['fix_length_range'] = np.array(fix_length_range) * float(refresh_rate)
    # the first dimension of stimuli (retrieved by len) is how many stimuli we have. the next two
    # are the size of the stimuli
    expt_params['stim_size'] = stimuli.shape[1:]
    repeat_num = (session_length - final_blank_sec_length) / (len(stimuli) * ((on_msec_length + off_msec_length) / 1000.))
    # with these two, we guarantee that stimuli will either end at the same time or before
    # session_length
    if repeat_num > 1:
        # this rounds down if repeat_num is not an integer (i.e., repeat(2.5) is the same as
        # repeat(2)), which is what we want because we'd rather have a bit of dead time
        stimuli = np.tile(stimuli, (int(repeat_num), 1, 1))
    elif repeat_num < 1:
        stimuli = stimuli[:int(len(stimuli) * repeat_num)]

    monitor_kwargs.update({'size': size, 'monitor': monitor, 'units': units, 'fullscr': fullscr,
                           'screen': screen})
    return stimuli, expt_params, monitor_kwargs


def run(stim_filename, session_length=30, refresh_rate=60, on_msec_length=300, off_msec_length=200,
        fix_length_range=(1, 3), final_blank_sec_length=8, fix_size=10, fix_color_change_pctg=.2,
        **monitor_kwargs):
    """run one run of the experiment

    Before running this, you need to make sure the stimuli specified are in the correct order. This
    function will simply go through those stimuli in order, showing each stimuli for 300 msecs and
    then a blank screen for 100 msecs (or as close as possible, given the refresh_rate). A fixation
    dot will be shown in the center of the screen (with a width of 10 pixels), whose color will
    change pseudo-randomly from red to green (change will happen at least twice a session, always
    separated by at least one second).

    you should create enough stimuli that, allowing for 400 msecs per stimuli, they last the number
    of seconds specified by session_length. However, session_length takes priority, so stimuli will
    be cut short or looped if the lengths don't match up.


    Arguments
    ============

    stim_filename: string, path to .npy file where stimuli are stored (as 3d array)

    session_length: int, length in seconds

    refresh_rate: int or float, the refresh rate of the monitor, in Hz. If you don't know this,
    psychopy.info.RunTimeInfo() will give you the windowRefreshTime, and 1000 over that number is
    your refresh_rate. most monitors have a 60 Hz refresh rate, but the projecter at NYU's primsa
    scanner goes up to 360 Hz depending on the resolution

    on_msec_length: int, length of the ON blocks in milliseconds; that is, the length of time to
    display each stimulus before moving on

    off_msec_length: int, length of the OFF blocks in milliseconds; that is, the length of time to
    between stimuli

    fix_length_range: 2-tuple of ints or floats. the length of time the fixation dot will remain
    the same color is picked from a uniform distribution with these endpoints

    fix_size: int, the size of the fixation dot, in monitor units (probably pixels, unless you set
    units in the monitor_kwargs)

    fix_color_change_pctg: float, the chance, on each frame, that the fixation dot will change
    color
    """
    stimuli, expt_params, monitor_kwargs = _set_params(stim_filename, session_length, refresh_rate,
                                                       on_msec_length, off_msec_length,
                                                       fix_length_range, final_blank_sec_length,
                                                       **monitor_kwargs)

    win = visual.Window(**monitor_kwargs)

    fixation = visual.GratingStim(win, size=fix_size, pos=[0, 0], color=(1, 0, 0), sf=0, mask='circle')
    gratings = [visual.ImageStim(win, image=stim, size=expt_params['stim_size']) for stim in stimuli]

    wait_text = visual.TextStim(win, ("Press 5 to start\nq will quit this run\nescape will quit "
                                      "this session"))
    wait_text.draw()
    win.flip()

    clock = core.Clock()
    # wait until receive 5, which is the scanner trigger
    all_keys = event.waitKeys(keyList=['5', 'q', 'escape'], timeStamped=clock)
    if 'q' in [k[0] for k in all_keys] or 'escape' in [k[0] for k in all_keys]:
        win.close()
        return None, all_keys
    win.recordFrameIntervals = True

    last_fix_change = 0
    next_fix_change = np.floor(np.random.uniform(*expt_params['fix_length_range']))
    last_stim_change = 0
    stim_num = 0
    keys_pressed = [(key[0], key[1], -1) for key in all_keys]
    fixation_dot = [(0, clock.getTime(), fixation.color)]
    for frame_num in xrange(expt_params['session_length_frames']):
        if stim_num < len(stimuli-1):
            if frame_num < (last_stim_change + expt_params['on_frame_length']):
                # gratings[stim_num].draw()
                gratings[stim_num].draw()
                # print frame_num, stim_num, clock.getTime(), last_stim_change
            # there's a hidden condition here, where we don't want to draw anything (and, in order
            # to make sure things line up exactly, we reset the last_stim_change and stim_num right
            # before)
            elif frame_num >= (last_stim_change + expt_params['on_frame_length'] + expt_params['off_frame_length'] - 1):
                last_stim_change = frame_num+1
                stim_num += 1
                # print frame_num, 'switching', clock.getTime(), last_stim_change
            # else:
                # print frame_num, '-', clock.getTime(), last_stim_change
        # else:
        #     print frame_num, 'waiting', clock.getTime()
        if frame_num >= (last_fix_change + next_fix_change):
            if (fixation.color == (0, 1, 0)).all():
                fixation.color = (1, 0, 0)
            elif (fixation.color == (1, 0, 0)).all():
                fixation.color = (0, 1, 0)
            last_fix_change = frame_num
            next_fix_change = np.floor(np.random.uniform(*expt_params['fix_length_range']))
            fixation_dot.append((frame_num, clock.getTime(), fixation.color))
        fixation.draw()
        win.flip()
        all_keys = event.getKeys(timeStamped=clock)
        if all_keys:
            keys_pressed.extend([(key[0], key[1], frame_num) for key in all_keys])
        if 'q' in [k[0] for k in all_keys] or 'escape' in [k[0] for k in all_keys]:
            break
    win.close()
    return win, keys_pressed, fixation_dot


def expt(stims_path, subj_name, **kwargs):
    """run a full experiment

    this just loops through the specified stims_path, passing each one to the run function in
    turn. any other kwargs are sent directly to run as well. it then saves the returned
    keys_pressed and frame intervals
    """
    file_path = "../data/raw_behavioral/%s_%s_sess{sess}.hdf5" % (datetime.datetime.now().strftime("%Y-%b-%d"), subj_name)
    sess_num = 0
    while glob.glob(file_path.format(sess=sess_num)):
        sess_num += 1
    for i, path in enumerate(stims_path):
        win, keys = run(path, **kwargs)
        with h5py.File(file_path.format(sess=sess_num), 'a') as f:
            f.create_dataset("run_%s_button_presses" % i, data=np.array(keys))
            if win is not None:
                f.create_dataset("run_%s_frame_intervals" % i, data=np.array(win.frameIntervals))
        if 'escape' in [k[0] for k in keys]:
            break
