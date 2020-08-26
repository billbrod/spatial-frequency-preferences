#!/usr/bin/python
"""functions to put the final touches on figures for publication
"""
import re
import os.path as op
from svgutils import compose
LEGEND_PATH = op.join(op.dirname(op.realpath(__file__)), '..', 'reports', 'figures',
                      'stimulus-legend-relative.svg')


def calc_scale():
    """Convert to correct size in pixels.

    There's a bizarre issue that makes it so the size of the figure saved out
    from matplotlib as an svg is not the same as the SVG element created here.
    this wouldn't be an issue (because SVGs scale nicely and all that) except
    for the font size. So, here's the issue:

    - matplotlib saves out svgs, specifying the size in the header in pts,
      converting from inches to pts with a hard-coded dpi of 72 (see definition
      of class FigureCanvasSVG, defind in backend_svg.py)

    - svgutils loads in svgs, but assumes the values in the height/width
      definitions are in pixels, not points, and that pixels have a hard-coded
      dpi of 90

    Therefore, in order to make sure everything is the right size, we need to
    scale all SVG figures up by 90/72

    weirdly, ImageMagick (via identify on the command line) correctly does the
    conversion

    """
    return 90/72


def _convert_to_pix(val):
    """Convert value into pixels to make our lives easier.
    """
    if not isinstance(val, str):
        return val
    else:
        v, units = re.findall('([\d\.]+)([a-z]+)', val)[0]
        # svgutils can't convert out of inches or pts, so we do that ourselves
        # (it supposedly can do pts, but it says there is pt per inch? which is
        # just totally wrong)
        if units == 'in':
            return float(v) * 90
        elif units == 'pt':
            return float(v) * 72
        else:
            return compose.Unit(val).to('px').value


def crossvalidation(annotated_model_schematic, horizontal_cv_loss, save_path,
                    figure_width='6.5in',
                    text_params={'font': 'Helvetica', 'size': '12pt'}):
    """create the crossvalidation figure

    Note that we don't do any resizing, but the sizes used to create the
    input figures should work for this

    Parameters
    ----------
    annotated_model_schematic, horizontal_cv_loss : str
        path to the svg files containing the annotated model schematic
        and horizontal cv loss figures, respectively
    save_path : str
        path to save the composed figure at
    figure_width : str or float
        width of the figure to create. will attempt to scale things properly to
        different sizes, but will probably require some manual tuning
    text_params: dict
        parameters to pass to the svgutils.compose.Text objects for the panel
        labels.

    """
    figure_width = _convert_to_pix(figure_width)
    figure_height = figure_width * .6
    # this scaling doesn't really work right now, will properly require more
    # manual adjustment if you want different sizes
    scale = figure_width / _convert_to_pix('6.5in')
    compose.Figure(figure_width, figure_height,
                   compose.SVG(horizontal_cv_loss).scale(calc_scale()).move(240*scale, 64*scale),
                   compose.SVG(annotated_model_schematic).scale(calc_scale()),
                   compose.Text("B", (280+25)*scale, 40*scale, **text_params),
                   compose.Text("A", 25*scale, 40*scale, **text_params),
                   ).save(save_path)


def add_legend(figure, figsize, legend_location, save_path):
    """add legend to figure

    Note that we scale the legend by 3, but don't change the size of the figure
    at all. will probably eventually want to determine how best to do this.

    Parameters
    ----------
    figure : str
        path to the svg file containing the figure to add legend to
    figsize : tuple
        tuple specifying the width, height of the finished figure
    legend_location: tuple
        tuple specifying the x, y position of the legend
    save_path : str
        path to save the composed figure at

    """
    compose.Figure(*figsize,
                   compose.SVG(figure),
                   compose.SVG(LEGEND_PATH).scale(3).move(*legend_location),
                   ).save(save_path)
