#!/usr/bin/python
"""small script to test displays
"""

import numpy as np
from psychopy import visual, event
from psychopy.tools import imagetools
import pandas as pd
import argparse
import warnings


def create_alternating_stimuli(size):
    """create a square stimuli that alternates every other pixel black-white

    the point of this stimuli is to make sure you can resolve the individual pixels on whatever
    screen you're using.

    size: int. the size of the image, in pixels. image will be square
    """
    assert int(size) == size, "Size must be an integer!"
    size = int(size)
    x = np.array(range(size)) / float(size)
    x, _ = np.meshgrid(x, x)
    # this is frequency * 2pi * x, but since our frequency is half the size, the 2s cancel.
    return np.cos(size * np.pi * x)


def test_display(screen_size, stimulus=None, stimulus_description_csv=None, freqs=None, text=None):
    """create a psychopy window and display a stimulus

    if stimulus is None, display create_alternating_stimuli. if a filename ending in npy, load that
    array in and show that array. if it's an array, display that array
    """
    if not hasattr(screen_size, "__iter__"):
        screen_size = [screen_size, screen_size]
    if isinstance(stimulus, str):
        stimulus = np.load(stimulus)
        if stimulus_description_csv is not None:
            stim_df = pd.read_csv(stimulus_description_csv)
            print(freqs)
            print(type(freqs))
            if 'w_r' in stim_df.columns:
                stim_idx = stim_df[(stim_df.w_a==freqs[0]) & (stim_df.w_r==freqs[1])].index[0]
            else:
                stim_idx = stim_df[(stim_df.w_x==freqs[0]) & (stim_df.w_y==freqs[1])].index[0]
            stimulus = stimulus[stim_idx, :, :]
    elif stimulus is None and text is None:
        stimulus = create_alternating_stimuli(min(screen_size))
    win = visual.Window(screen_size, fullscr=True, screen=1, colorSpace='rgb255', color=127,
                        units='pix')
    if text is None:
        if stimulus.ndim > 2:
            warnings.warn("stimulus is more than 2d, assuming it's three and [0,:,:] ...")
            stimulus = stimulus[0, :, :]
        stim_shape = stimulus.shape
        if stimulus.min() < -1 or stimulus.max() > 1:
            stimulus = imagetools.array2image(stimulus)
        thing_to_display = visual.ImageStim(win, stimulus, size=stim_shape)
    else:
        thing_to_display = visual.TextStim(win, text)
    thing_to_display.draw()
    win.flip()
    all_keys = event.waitKeys(keyList=['q', 'escape'])
    if 'q' in [k[0] for k in all_keys] or 'escape' in [k[0] for k in all_keys]:
        win.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test your display")
    parser.add_argument("screen_size", help="Screen size, in pixels. Can be one or two integers",
                        nargs='+', type=int)
    parser.add_argument("--stimulus", "-s",
                        help=("Optional, path to stimulus. If not used, will create alternating "
                              "black and white pixels the size of the screen"))
    parser.add_argument("--stimulus_description_csv", '-d',
                        help=("Optional, path to csv containing description of stimuli. Used with"
                              " --freqs arg to find stimuli with specified frequency"))
    parser.add_argument("--freqs", '-f', nargs=2, type=float,
                        help=("Optional, 2 floats specifying the frequency of the stimulus to "
                              "display. Should be either w_x, w_y or w_a, w_r"))
    parser.add_argument("--text", "-t", type=str,
                        help=("Optional, text to display. If set, will not show a grating but "
                              "instead whatever text you enter. Text can be easier to check for "
                              "blur"))
    args = vars(parser.parse_args())
    test_display(**args)
