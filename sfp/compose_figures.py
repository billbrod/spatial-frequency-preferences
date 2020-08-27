#!/usr/bin/python
"""functions to put the final touches on figures for publication
"""
import re
import os.path as op
from svgutils import compose
from . import style
REL_LEGEND_PATH = op.join(op.dirname(op.realpath(__file__)), '..', 'reports', 'figures',
                          'stimulus-legend-relative.svg')
ABS_LEGEND_PATH = op.join(op.dirname(op.realpath(__file__)), '..', 'reports', 'figures',
                          'stimulus-legend-absolute.svg')


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


class SVG(compose.SVG):
    """SVG from file.

    This is the same as svgutils.compose.SVG, except we automatically scale it
    appropriately (see docstring of calc_scale() for details)

    Parameters
    ----------
    fname : str
       full path to the file

    """

    def __init__(self, fname=None):
        super().__init__(fname)
        self.scale(calc_scale())


def _convert_to_pix(val):
    """Convert value into pixels to make our lives easier."""
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
            return float(v) * (90/72)
        else:
            return compose.Unit(val).to('px').value


def crossvalidation(annotated_model_schematic, horizontal_cv_loss, save_path,
                    context='paper'):
    """Create the crossvalidation figure.

    Note that we don't do any resizing, but the sizes used to create the
    input figures should work for this

    Parameters
    ----------
    annotated_model_schematic, horizontal_cv_loss : str
        path to the svg files containing the annotated model schematic
        and horizontal cv loss figures, respectively
    save_path : str
        path to save the composed figure at
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils', 'full')
    figure_width = _convert_to_pix(figure_width)
    figure_height = figure_width * .6
    compose.Figure(figure_width, figure_height,
                   SVG(horizontal_cv_loss).move(240, 64),
                   SVG(annotated_model_schematic),
                   compose.Text("B", (280+25), 40, **text_params),
                   compose.Text("A", 25, 40, **text_params),
                   ).save(save_path)


def feature_df_summary(rel_feature_df_plots, abs_feature_df_plots, save_path,
                       context='paper'):
    """Create the figure summarizing feature df plots.

    Note that we don't do any resizing, but the sizes used to create the
    input figures should work for this

    Parameters
    ----------
    rel_feature_df_plots, abs_feature_df_plots : list
        lists of strs giving the paths to the svgs containing the
        feature_df_plots to add to this image. They should be in the
        appropriate order: each should have the preferred period, then the
        preferred period contour, then max amplitude
    save_path : str
        path to save the composed figure at
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils', 'full')
    figure_width = _convert_to_pix(figure_width)
    figure_height = figure_width
    compose.Figure(
        figure_width, figure_height,
        SVG(rel_feature_df_plots[0]).move(0, -10),
        SVG(rel_feature_df_plots[1]).move(figure_width / 2, -10),
        SVG(rel_feature_df_plots[2]).move(
            figure_width / 2 + _convert_to_pix('14pt'),
            figure_height / 4 - 24),
        SVG(REL_LEGEND_PATH).scale(2.5).move(figure_width / 4,
                                             figure_height / 4 - 10),
        compose.Text("A", 10, _convert_to_pix('12pt'), **text_params),
        SVG(abs_feature_df_plots[0]).move(0, figure_height / 2 - 10),
        SVG(abs_feature_df_plots[1]).move(figure_width / 2,
                                          figure_height / 2 - 10),
        SVG(abs_feature_df_plots[2]).move(
            figure_width / 2 + _convert_to_pix('14pt'),
            figure_height / 2 + figure_height / 4 - 24),
        SVG(ABS_LEGEND_PATH).scale(2.5).move(
            figure_width / 4, figure_height / 2 + figure_height / 4 - 10),
        compose.Text("B", 10, figure_height / 2 + _convert_to_pix('12pt') - 10,
                     **text_params),
    ).save(save_path)


def add_legend(figure, figsize, legend_location, save_path):
    """Add legend to figure.

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
