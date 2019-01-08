#!/usr/bin/python
"""script to run the experiment

make sure you've already generated and saved the stimuli you want (using stimuli.main). They should
be in the order you want for that run.

also make sure that you've already set up the monitor you'll be using in PsychoPy's monitor center
"""

import numpy as np
import os
from psychopy import visual, core, event
from psychopy.tools import imagetools
import h5py
import datetime
import glob
import argparse


def _create_blanks(blank_sec_length, on_msec_length, off_msec_length, stimulus_shape, blank_loc):
    """create the blank stimuli

    stimulus_shape: 2-tuple of ints. specifies the shape of a single stimulus.
    """
    nblanks = blank_sec_length / ((on_msec_length + off_msec_length) / 1000.)
    if nblanks != int(nblanks):
        raise Exception("Because of your timing ({loc}_blank_sec_length, on_msec_length, and "
                        "off_msec_length), I can't show blanks for the {loc} {length:.02f} seconds"
                        ". {loc}_blank_sec_length must be a multiple of on_msec_length+"
                        "off_msec_length!".format(loc=blank_loc, length=blank_sec_length))
    nblanks = int(nblanks)
    # 128 corresponds to the mid-value between 0 and 255
    blanks = (128 * np.ones((nblanks, stimulus_shape[0], stimulus_shape[1]))).astype(np.uint8)
    return nblanks, blanks


def _set_params(stim_path, idx_path, session_length=30, on_msec_length=300, off_msec_length=200,
                fixation_type='digit', fix_button_prob=1/6., fix_dot_length_range=(1, 3),
                final_blank_sec_length=16, init_blank_sec_length=16, size=[1920, 1080],
                monitor='CBI-prisma-projector', units='pix', fullscr=True, screen=1, color=128,
                colorSpace='rgb255',  **monitor_kwargs):
    """set the various experiment parameters
    """
    stimuli = np.load(stim_path)
    idx = np.load(idx_path)
    stimuli = stimuli[idx]
    expt_params = {}
    # the first dimension of stimuli (retrieved by len) is how many stimuli we have. the next two
    # are the size of the stimuli
    expt_params['stim_size'] = stimuli.shape[1:]
    if session_length is not None:
        # when session_length is None, we show all of them!
        repeat_num = (session_length - final_blank_sec_length) / (len(stimuli) * ((on_msec_length + off_msec_length) / 1000.))
        # with these two, we guarantee that stimuli will either end at the same time or before
        # session_length
        if repeat_num > 1:
            # this rounds down if repeat_num is not an integer (i.e., repeat(2.5) is the same as
            # repeat(2)), which is what we want because we'd rather have a bit of dead time
            stimuli = np.tile(stimuli, (int(repeat_num), 1, 1))
            idx = np.tile(idx, (int(repeat_num), 1, 1))
        elif repeat_num < 1:
            stimuli = stimuli[:int(len(stimuli) * repeat_num)]
            idx = idx[:int(len(idx) * repeat_num)]
    expt_params['non_blank_stimuli_num'] = stimuli.shape[0]
    # In order to get the right amount of blank time at the end of the run, we insert an
    # appropriate amount of blank stimuli.
    final_nblanks, final_blanks = _create_blanks(final_blank_sec_length, on_msec_length,
                                                 off_msec_length, stimuli.shape[1:], 'final')
    init_nblanks, init_blanks = _create_blanks(init_blank_sec_length, on_msec_length,
                                               off_msec_length, stimuli.shape[1:], 'init')
    stimuli = np.concatenate([init_blanks, stimuli, final_blanks])
    expt_params['final_nblanks'] = final_nblanks
    expt_params['init_nblanks'] = init_nblanks

    if fixation_type == 'dot':
        # we need enough colors for each stimuli ON and OFF
        dot_num = stimuli.shape[0] * 2
        colors = ['red']
        while len(colors) < dot_num:
            current_color = [colors[-1]]
            next_flip = np.random.uniform(*fix_dot_length_range)
            i, j = 0, 0
            while i*(on_msec_length/1000.) + j*(off_msec_length/1000.) < next_flip:
                i += 1
                if i*(on_msec_length/1000.) + j*(off_msec_length/1000.) < next_flip:
                    j += 1
            colors.extend(current_color*(i+j-1))
            colors.append({'red': 'green', 'green': 'red'}.get(current_color[0]))
        expt_params['fixation_color'] = iter(colors)
    elif fixation_type == 'digit':
        # we show a digit with every other stimuli.
        digit_num = int(stimuli.shape[0] / 2)
        probs = np.ones(10)/9
        digits = [int(np.random.uniform(0, 10)), ""]
        for i in range(digit_num-1):
            # because -1 and -3 (and all negative odd indices) will be blanks, we check two and
            # four ago to make sure there are no repeats here
            if np.random.uniform() < fix_button_prob and (len(digits) == 2 or digits[-4] != digits[-2]):
                digits.append(digits[-2])
            else:
                probs_tmp = probs.copy()
                probs_tmp[digits[-2]] = 0
                digits.append(np.random.choice(range(10), p=probs_tmp))
            # after every digit, we show a blank
            digits.append("")
        expt_params['fixation_text'] = iter(digits)
        expt_params['fixation_color'] = iter(['white', 'white', 'black', 'black'] * int(np.ceil(digit_num/2.)))
    else:
        raise Exception("Don't know what to do with fixation_type %s!" % fixation_type)
    # these are all a variety of kwargs used by monitor
    monitor_kwargs.update({'size': size, 'monitor': monitor, 'units': units, 'fullscr': fullscr,
                           'screen': screen, 'color': color, 'colorSpace': colorSpace})
    return stimuli, idx, expt_params, monitor_kwargs


def run(stim_path, idx_path, session_length=30, on_msec_length=300, off_msec_length=200,
        final_blank_sec_length=16, init_blank_sec_length=16, fixation_type="digit",
        fix_pix_size=10, fix_deg_size=None, fix_button_prob=1/6., fix_dot_length_range=(1, 3),
        max_visual_angle=24, save_frames=None, **monitor_kwargs):
    """run one run of the experiment

    stim_path specifies the path of the unshuffled experiment stimuli, while idx_path specifies the
    path of the shuffled indices to use for this run. This function will load in the stimuli at
    stim_path and rearrange them using the indices found at idx_path, then simply go through those
    stimuli in order, showing each stimuli for `on_msec_length` msecs and then a blank screen for
    `off_msec_length` msecs (or as close as possible, given the monitor's refresh rate).

    For fixation, you can either choose a dot whose color changes from red to green
    (`fixation_type='dot'`) or a stream of digits whose colors alternate between black and white,
    with a `fix_button_prob` chance of repeating (`fixation_type='digit'`). For the digit, digits
    are presented with alternating stimuli ON and OFF blocks, so that a digit will be shown for
    on_msec_length+off_msec_length msecs and then there will be nothing at fixation for the next
    on_msec_length+off_msec_length msecs. For now, you can't change this. For the dot,
    `fix_dot_length_range` determines the range of time, in seconds, that the dot will change color
    in. It will be rounded to the nearest stimuli on or stimuli off.

    If `session_length` is None, all stimuli loaded in from stim_path will be shown. Else, the
    session will last exactly that long, so the stimuli will be cut short so that it ends after
    that amount of time (with the `final_blank_sec_length`) or looped to reach that amount of
    time. If you opt for this, make sure you think about your session_length carefully, because
    this presentation code does not know about your classes or anything else and so could very
    easily end in the middle of one.


    Arguments
    ============

    stim_path: string, path to .npy file where stimuli are stored (as 3d array)

    idx_path: string, path to .npy file where shuffled indices are stored (as 1d array)

    session_length: int or None, length in seconds. If None, then will be long enough to use all
    stimuli found at stim_path.

    on_msec_length: int, length of the ON blocks in milliseconds; that is, the length of time to
    display each stimulus before moving on

    off_msec_length: int, length of the OFF blocks in milliseconds; that is, the length of time to
    between stimuli

    fixation_type: {"digit", "dot"}. whether to use a fixation dot or digit for the distractor
    task. The fixation dot is a small circle that randomly changes color from red to green, while
    the digit is a stream of digits alternating between black and white. note that only a subset of
    the following fix_* arguments apply to each of these fixation types.

    fix_pix_size: int, the size of the fixation dot or digits, in pixels.

    fix_deg_size: int, float or None. the size of the fixation dot or digits, in degrees of visual
    angle. If this is None, then fix_pix_size will be used, otherwise this will be used (converted
    into pixels based on monitor_kwargs['size'], default 1080x1080, and max_visual_angle, default
    24)

    fix_button_prob: float. the probability that the fixation digit will repeat or the fixation dot
    will change color (will never repeat more than once in a row). For fixation digit, this
    probability is relative to each stimulus presentation / ON block starting; for fixation dot,
    it's each stimulus change (stimulus ON or OFF block starting).

    fix_dot_length_range: 2-tuple of ints. A random pull from a uniform distribution with these end
    points will determine when the next dot color change is (in seconds). It will be rounded to the
    nearest stimuli on or stimuli off.

    max_visual_angle: int or float. the max visual angle (in degrees) of the full screen. used to
    convert fix_deg_size to pixels.

    save_frames: None or str. if not None, this should be the filename you wish to save frames at
    (one image will be made for each frame). WARNING: typically a large number of files will be
    saved (depends on the length of your session), which means this may make the end of the run
    (with the screen completely blank) take a while
    """
    stimuli, idx, expt_params, monitor_kwargs = _set_params(
        stim_path, idx_path, session_length, on_msec_length, off_msec_length, fixation_type,
        fix_button_prob, fix_dot_length_range, final_blank_sec_length, init_blank_sec_length,
        **monitor_kwargs)

    win = visual.Window(**monitor_kwargs)
    win.mouseVisible = False
    # linear gamma ramp, since the projector at the scanner is linearized
    win.gammaRamp = np.tile(np.linspace(0, 1, 256), (3, 1))

    if fix_deg_size is not None:
        fix_pix_size = fix_deg_size * (monitor_kwargs['size'][0] / float(max_visual_angle))

    if fixation_type == 'dot':
        fixation = visual.GratingStim(win, size=fix_pix_size, pos=[0, 0], sf=0, color=None,
                                      mask='circle')
    else:
        fixation = visual.TextStim(win, next(expt_params['fixation_text']), height=fix_pix_size,
                                   color=None)
    fixation.color = next(expt_params['fixation_color'])
    # first one is special: we preload it, but we still want to include it in the iterator so the
    # numbers all match up (we don't draw or wait during the on part of the first iteration)
    grating = visual.ImageStim(win, image=imagetools.array2image(stimuli[0]),
                               size=expt_params['stim_size'])

    wait_text = visual.TextStim(win, ("Press 5 to start\nq will quit this run\nescape will quit "
                                      "this session"))
    wait_text.draw()
    win.flip()
    if save_frames is not None:
        win.getMovieFrame()
    # preload these to save time
    grating.draw()
    fixation.draw()

    clock = core.Clock()
    # wait until receive 5, which is the scanner trigger
    all_keys = event.waitKeys(keyList=['5', 'q', 'escape'], timeStamped=clock)
    if 'q' in [k[0] for k in all_keys] or 'escape' in [k[0] for k in all_keys]:
        win.close()
        return all_keys, [], [], expt_params, idx

    keys_pressed = [(key[0], key[1]) for key in all_keys]
    timings = [("start", "off", clock.getTime())]
    fixation_info = []
    for i, stim in enumerate(stimuli):
        if i > 0:
            # we don't wait the first time, and all these have been preloaded while we were waiting
            # for the scan trigger
            if "fixation_text" in expt_params:
                fixation.text = next(expt_params['fixation_text'])
            grating.image = imagetools.array2image(stim)
            fixation.color = next(expt_params['fixation_color'])
            grating.draw()
            fixation.draw()
            next_stim_time = (i*on_msec_length + i*off_msec_length - 2)/1000.
            core.wait(abs(clock.getTime() - timings[0][2] - next_stim_time))
        win.flip()
        if save_frames is not None:
            win.getMovieFrame()
        timings.append(("stimulus_%d" % i, "on", clock.getTime()))
        if fixation_type == "digit":
            fixation_info.append((fixation.text, clock.getTime()))
        elif fixation_type == 'dot':
            fixation_info.append((fixation.color, clock.getTime()))
            # the dot advances its color and stays drawn during the stimulus off segments
            fixation.color = next(expt_params['fixation_color'])
        fixation.draw()
        next_stim_time = ((i+1)*on_msec_length + i*off_msec_length - 1)/1000.
        core.wait(abs(clock.getTime() - timings[0][2] - next_stim_time))
        win.flip()
        if save_frames is not None:
            win.getMovieFrame()
        timings.append(("stimulus_%d" % i, "off", clock.getTime()))
        if fixation_type == 'dot':
            fixation_info.append((fixation.color, clock.getTime()))
        all_keys = event.getKeys(timeStamped=clock)
        if all_keys:
            keys_pressed.extend([(key[0], key[1]) for key in all_keys])
        if 'q' in [k[0] for k in all_keys] or 'escape' in [k[0] for k in all_keys]:
            break
    if save_frames is not None:
        win.saveMovieFrames(save_frames)
    win.close()
    return keys_pressed, fixation_info, timings, expt_params, idx


def _convert_str(list_of_strs):
    """convert strs to hdf5-savable format

    python 3 strings are more complicated than python 2, see
    http://docs.h5py.org/en/latest/strings.html and https://github.com/h5py/h5py/issues/892
    """
    list_of_strs = np.array(list_of_strs)
    saveable_list = []
    for x in list_of_strs:
        try:
            x = x.encode()
        except AttributeError:
            # then this is not a string but another list of strings
            x = [i.encode() for i in x]
        saveable_list.append(x)
    return saveable_list


def expt(stim_path, number_of_runs, first_run, subj_name, output_dir="../data/raw_behavioral",
         input_dir="../data/stimuli", **kwargs):
    """run a full experiment

    this just loops through the specified stims_path, passing each one to the run function in
    turn. any other kwargs are sent directly to run as well. it then saves the returned
    keys_pressed and frame intervals
    """
    if output_dir[-1] != '/':
        output_dir += '/'
    if input_dir[-1] != '/':
        input_dir += '/'
    file_path = "%s%s_%s_sess{sess:02d}.hdf5" % (output_dir, datetime.datetime.now().strftime("%Y-%b-%d"), subj_name)
    sess_num = 0
    while glob.glob(file_path.format(sess=sess_num)):
        sess_num += 1
    idx_paths = [input_dir + "%s_run%02d_idx.npy" % (subj_name, i) for i in range(first_run, first_run+number_of_runs)]
    for p in idx_paths:
        if not os.path.isfile(p):
            raise IOError("Unable to find array of stimulus indices %s!" % p)
    print("Running %d runs, with the following stimulus:" % number_of_runs)
    print("\t%s" % stim_path)
    print("Will use the following indices:")
    print("\t%s" % "\n\t".join(idx_paths))
    print("Will save at the following location:\n\t%s" % file_path.format(sess=sess_num))
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    for i, path in enumerate(idx_paths):
        keys, fixation, timings, expt_params, idx = run(stim_path, path, **kwargs)
        with h5py.File(file_path.format(sess=sess_num), 'a') as f:
            f.create_dataset("run_%02d_button_presses" % i, data=_convert_str(keys))
            f.create_dataset("run_%02d_fixation_data" % i, data=_convert_str(fixation))
            f.create_dataset("run_%02d_timing_data" % i, data=_convert_str(timings))
            f.create_dataset("run_%02d_stim_path" % i, data=stim_path.encode())
            f.create_dataset("run_%02d_idx_path" % i, data=path.encode())
            f.create_dataset("run_%02d_shuffled_indices" % i, data=idx)
            for k, v in expt_params.items():
                if k in ['fixation_color', 'fixation_text']:
                    continue
                f.create_dataset("run_%02d_%s" % (i, k), data=v)
            # also note differences from default options
            for k, v in kwargs.items():
                if v is None:
                    f.create_dataset("run_%02d_%s" % (i, k), data=str(v).encode())
                else:
                    f.create_dataset("run_%02d_%s" % (i, k), data=v)
        if 'escape' in [k[0] for k in keys]:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Run an experiment! This takes in the path to your unshuffled stimuli, the "
                     "name of your subject, and the number of runs, and passes that to expt. This "
                     "will then assume that your run indices (which shuffle the stimuli) are saved"
                     "in the INPUT_DIR at SUBJ_NAME_runNUM_idx.npy, where NUM runs from FIRST_RUN "
                     "to FIRST_RUN+NUMBER_OF_RUNS-1 (because this is python, 0-based indexing), "
                     "with all single-digit numbers represented as 0#."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("stimuli_path",
                        help="path to your unshuffled stimuli.")
    parser.add_argument("number_of_runs", help="number of runs you want to run", type=int)
    parser.add_argument("subj_name", help="name of the subject")
    parser.add_argument("--input_dir", '-i', help=("path to directory that contains your shuffled"
                                                   " run indices"),
                        default="data/stimuli")
    parser.add_argument("--output_dir", '-o', help="directory to place output in",
                        default="data/raw_behavioral")
    parser.add_argument("--first_run", '-f', type=int, default=0,
                        help=("Which run to run first. Useful if, for instance, you ran the first "
                              "two runs without problem and then had to quit out in the third. You"
                              " should then set this to 2 (because they're 0-indexed)."))
    args = vars(parser.parse_args())
    expt(args['stimuli_path'], args['number_of_runs'], args['first_run'], args['subj_name'],
         args['output_dir'], args['input_dir'], session_length=None, fix_deg_size=.25)
