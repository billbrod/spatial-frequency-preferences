#!/usr/bin/python
"""small script to test displays
"""

import numpy as np
from psychopy import visual, event
from psychopy.tools import imagetools
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


def test_display(screen_size, stimulus=None):
    """create a psychopy window and display a stimulu

    if stimulus is None, display create_alternating_stimuli. if a filename ending in npy, load that
    array in and show that array. if it's an array, display that array
    """
    if not hasattr(screen_size, "__iter__"):
        screen_size = [screen_size, screen_size]
    if isinstance(stimulus, str):
        stimulus = np.load(stimulus)
    elif stimulus is None:
        stimulus = create_alternating_stimuli(min(screen_size))
    if stimulus.ndim > 2:
        warnings.warn("stimulus is more than 2d, assuming it's three and [0,:,:] ...")
        stimulus = stimulus[0, :, :]
    stim_shape = stimulus.shape
    if stimulus.min() < -1 or stimulus.max() > 1:
        stimulus = imagetools.array2image(stimulus)
    win = visual.Window(screen_size, fullscr=True, screen=1, colorSpace='rgb255', color=127,
                        units='pix')
    grating = visual.ImageStim(win, stimulus, size=stim_shape)
    grating.draw()
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
    args = vars(parser.parse_args())
    test_display(args['screen_size'], args['stimulus'])
