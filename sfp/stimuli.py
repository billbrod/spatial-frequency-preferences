#!/usr/bin/python
"""script to generate stimuli
"""
import pyPyrTools as ppt
import numpy as np


def log_polar_grating(size, alpha, w_r=0, w_a=0, phi=0, ampl=1, origin=None):
    """Make a sinusoidal grating in logPolar space.

    this allows for the easy creation of stimuli whose spatial frequency decreases with
    eccentricity, as the peak spatial frequency of neurons in the early visual cortex does.

    Examples
    ============

    circular: `log_polar_grating(512, 4, 10)`

    radial: `log_polar_grating(512, 4, w_a=10)`

    spiral: `log_polar_grating(512, 4, 10, 10)`

    plaid: `log_polar_grating(512, 4, 10) + log_polar_grating(512, 4, w_a=10)`


    Parameters
    =============

    size: 2-tuple or scalar. if tuple: (Y,X) dimensions of image.  If scalar, image is square.

    alpha: int, radius (in pixel spacing) of the "fovea".  IE: log_rad = log(r^2 + alpha^2)

    w_r: int, logRadial frequency.  Units are matched to those of the angular frequency (`w_a`).

    w_a: int, angular frequency.  Units are cycles per revolution around the origin.

    phi: int, phase (in radians).

    ampl: int, amplitude

    origin: 2-tuple of floats, the origin of the image, from which all distances will be measured
    and angles will be relative to. By default, the center of the image
    """
    rad = ppt.mkR(size, origin=origin)
    # if the origin is set such that it lies directly on a pixel, then one of the pixels will have
    # distance 0 and, if alpha is also 0, that means we'll have a -inf out of np.log2 and thus a
    # nan from the cosine. this little hack avoids that issue.
    if alpha == 0 and 0 in rad:
        rad += 1e-12
    lrad = np.log2(rad**2 + alpha**2)
    theta = ppt.mkAngle(size, origin=origin)

    return ampl * np.cos((w_r/np.pi) * lrad + w_a * theta + phi)
