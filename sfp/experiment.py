#!/usr/bin/python
"""script to run the experiment

make sure you've already generated and saved the stimuli you want (using stimuli.main). They should
be in the order you want for that run.

also make sure that you've already set up the monitor you'll be using in PsychoPy's monitor center
"""

import numpy as np
from psychopy import visual, core, event


def main(stim_filename, monitor_kwargs={}, session_length=30, refresh_rate=60):
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

    session_length: int, length in seconds

    refresh_rate: int or float, the refresh rate of the monitor, in Hz. If you don't know this,
    psychopy.info.RunTimeInfo() will give you the windowRefreshTime, and 1000 over that number is
    your refresh_rate. most monitors have a 60 Hz refresh rate, but the projecter at NYU's primsa
    scanner goes up to 360 Hz depending on the resolution
    """
    stimuli = np.load(stim_filename)
    session_length_frames = session_length * refresh_rate
    on_frame_length = np.round(300 / (1000. / refresh_rate))
    off_frame_length = np.round(100 / (1000. / refresh_rate))
    # this is just going to be the refresh_rate, but if we decide to change it from 1 sec, this
    # would do it
    fix_change_min_length = np.round(1000 / (1000. / refresh_rate))
    # the first dimension of stimuli (retrieved by len) is how many stimuli we have. the next two
    # are the size of the stimuli
    stim_size = stimuli.shape[1:]
    repeat_num = session_length / (len(stimuli) * .4)
    # with these two, we guarantee that stimuli will either end at the same time or before
    # session_length
    if repeat_num > 1:
        # this rounds down if repeat_num is not an integer (i.e., repeat(2.5) is the same as
        # repeat(2)), which is what we want because we'd rather have a bit of dead time
        stimuli = np.tile(stimuli, (int(repeat_num), 1, 1))
    elif repeat_num < 1:
        stimuli = stimuli[:int(len(stimuli) * repeat_num)]
    print len(stimuli), session_length_frames, on_frame_length, off_frame_length

    window_res = monitor_kwargs.get('size', [1080, 1080])
    monitor_name = monitor_kwargs.get('monitor', 'test')
    units = monitor_kwargs.get('units', 'pix')
    fullscreen = monitor_kwargs.get('fullscr', True)
    screen_num = monitor_kwargs.get('screen', 1)

    win = visual.Window(window_res, monitor=monitor_name, units=units,
                        fullscr=fullscreen, screen=screen_num, **monitor_kwargs)

    fixation = visual.GratingStim(win, size=10, pos=[0, 0], color=(1, 0, 0), sf=0, mask='circle')
    gratings = [visual.ImageStim(win, image=stim, size=stim_size) for stim in stimuli]

    wait_text = visual.TextStim(win, "Press 5 to start")
    wait_text.draw()
    win.flip()

    clock = core.Clock()
    # wait until receive 5, which is the scanner trigger
    all_keys = event.waitKeys(keyList=['5', 'q', 'escape'], timeStamped=clock)
    if 'q' in [k[0] for k in all_keys] or 'escape' in [k[0] for k in all_keys]:
        win.close()
        return all_keys
    win.recordFrameIntervals = True

    last_fix_change = 0
    last_stim_change = 0
    stim_num = 0
    keys_pressed = all_keys
    for frame_num in range(session_length_frames):
        if stim_num < len(stimuli-1):
            if frame_num < (last_stim_change + on_frame_length):
                gratings[stim_num].draw()
                # print frame_num, stim_num, clock.getTime(), last_stim_change
            # there's a hidden condition here, where we don't want to draw anything (and, in order
            # to make sure things line up exactly, we reset the last_stim_change and stim_num right
            # before)
            elif frame_num >= (last_stim_change + on_frame_length + off_frame_length - 1):
                last_stim_change = frame_num+1
                stim_num += 1
                # print frame_num, 'switching', clock.getTime(), last_stim_change
            # else:
                # print frame_num, '-', clock.getTime(), last_stim_change
        # else:
        #     print frame_num, 'waiting', clock.getTime()
        if np.random.uniform() > .8 and frame_num > (last_fix_change + fix_change_min_length):
            if (fixation.color == (0, 1, 0)).all():
                fixation.color = (1, 0, 0)
            elif (fixation.color == (1, 0, 0)).all():
                fixation.color = (0, 1, 0)
            last_fix_change = frame_num
        fixation.draw()
        win.flip()
        all_keys = event.getKeys(timeStamped=clock)
        keys_pressed.extend((all_keys[0], all_keys[1], frame_num))
        if 'q' in [k[0] for k in all_keys] or 'escape' in [k[0] for k in all_keys]:
            break
    win.close()
    return win, keys_pressed
