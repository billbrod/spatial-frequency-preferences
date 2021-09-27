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
SCALING_CARTOON_PATH = op.join(op.dirname(op.realpath(__file__)), '..', 'reports', 'figures',
                               'scaling-cartoon.svg')
CONSTANT_CARTOON_PATH = op.join(op.dirname(op.realpath(__file__)), '..', 'reports', 'figures',
                                'constant-cartoon.svg')
ANNULUS_PATH = op.join(op.dirname(op.realpath(__file__)), '..', 'reports', 'figures', 'annulus.svg')
PINWHEEL_PATH = op.join(op.dirname(op.realpath(__file__)), '..', 'reports', 'figures', 'pinwheel.svg')


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
    if 'doubleup' in horizontal_cv_loss:
        height_frac = .55
        vert_shift = 57
        horiz_shift = 270
    else:
        height_frac = .63
        vert_shift = 67
        horiz_shift = 290
    figure_height = figure_width * height_frac
    compose.Figure(figure_width, figure_height,
                   SVG(horizontal_cv_loss).move(horiz_shift, vert_shift),
                   SVG(annotated_model_schematic),
                   compose.Text("B", (290+25), 40, **text_params),
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
        preferred period contour, then max relative amplitude
    save_path : str
        path to save the composed figure at
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils', 'full')
    figure_width = _convert_to_pix(figure_width)
    figure_height = figure_width * 1.05
    font_size = _convert_to_pix(text_params.pop('size'))
    compose.Figure(
        figure_width, figure_height,
        # want the abs features plots below the corresponding rel feature plots
        SVG(abs_feature_df_plots[0]).move(-8, figure_height / 2 - 1),
        SVG(abs_feature_df_plots[1]).move(figure_width / 2 - 16,
                                          figure_height / 2 - 9.5),
        SVG(abs_feature_df_plots[2]).move(
            figure_width / 2 - 16,
            figure_height / 2 + figure_height / 4 - 24),
        SVG(rel_feature_df_plots[0]).move(-8, -1),
        SVG(rel_feature_df_plots[1]).move(figure_width / 2 - 16, -9.5),
        SVG(rel_feature_df_plots[2]).move(
            figure_width / 2 - 16,
            figure_height / 4 - 24),
        SVG(REL_LEGEND_PATH).scale(2.5).move(figure_width / 4 - 16,
                                             figure_height / 4 + 20),
        compose.Text("A", 3, font_size - 5, size=font_size, **text_params),
        SVG(ABS_LEGEND_PATH).scale(2.5).move(
            figure_width / 4 - 16, figure_height / 2 + figure_height / 4 + 19),
        compose.Text("B", 3, figure_height / 2 + font_size - 6, size=font_size,
                     **text_params),
    ).save(save_path)


def add_legend(figure, figsize, legend_location, save_path,
               figure_location=(0, 0), aspect=1, legend='rel',
               context='paper'):
    """Add legend to figure.

    Note that we scale the legend by 3, but don't change the size of the figure
    at all. will probably eventually want to determine how best to do this.

    Parameters
    ----------
    figure : str
        path to the svg file containing the figure to add legend to
    figsize : {'half', 'full'}
        whether this figure should be full or half-width
    legend_location: tuple
        tuple specifying the x, y position of the legend
    save_path : str
        path to save the composed figure at
    figure_location: tuple, optional
        tuple specifying the x, y position of the figure
    aspect : float, optional
        aspect ratio of the figure to create
    legend : {'rel', 'abs'}
        whether to use the relative or absolute legend
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils', figsize)
    figure_width = _convert_to_pix(figure_width)
    figure_height = figure_width * aspect
    if isinstance(legend_location, str):
        legend_location = [eval(i.replace('height', figure_height).replace('width', figure_width))
                           for i in legend_location]
    if legend == 'rel':
        legend = REL_LEGEND_PATH
    elif legend == 'abs':
        legend = ABS_LEGEND_PATH
    compose.Figure(
        figure_width, figure_height,
        SVG(figure).move(*figure_location),
        SVG(legend).scale(2.5).move(*legend_location),
    ).save(save_path)


def summary_1d(pref_period_fig, bw_fig, save_path, context='paper'):
    """Create the 1d results summary figure

    This combines the figure showing the showing the overall preferred period
    results with 1d model fits (with the legend already added -- this is
    IMPORTANT, because otherwise the size will look wrong) with the figure
    showing the overall bandwidth results.

    Parameters
    ----------
    pref_period_fig, bw_fig : str
        paths to the svg files containing the overall 1d preferred period
        figure (with legend) and the bandwidth figure, respectively
    save_path : str
        path to save the composed figure at
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils', 'full')
    figure_width = _convert_to_pix(figure_width)
    figure_height = figure_width / 2.2
    font_size = _convert_to_pix(text_params.pop('size'))

    compose.Figure(
        figure_width, figure_height,
        # we use the regular SVG here, because this has the legend applied, and
        # so has been saved out correctly
        compose.SVG(pref_period_fig).move(10, -2),
        compose.Text("A", 10, 25, size=font_size, **text_params),
        SVG(bw_fig).move(figure_width/2+10, -2),
        compose.Text("B", figure_width/2, 25, size=font_size, **text_params),
    ).save(save_path)


def stimulus_figure(base_freq_fig, stim_fig, presented_freq_fig, save_path,
                    context='paper'):
    """Create the stimulus description figure.

    Parameters
    ----------
    base_freq_fig, stim_fig, presented_freq_fig
        paths to the svg files containing the base frequency, stimulus
        schematic, and presented frequencies figures, respectively
    save_path : str
        path to save the composed figure at
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils', 'full')
    figure_width = _convert_to_pix(figure_width)
    figure_height = figure_width * .72
    font_size = _convert_to_pix(text_params.pop('size'))

    compose.Figure(
        figure_width, figure_height,
        SVG(base_freq_fig).move(-3, figure_height/5),
        # SVG(CONSTANT_CARTOON_PATH).scale(3).move(40, 127),
        # SVG(SCALING_CARTOON_PATH).scale(3).move(40, 243),
        compose.Text("A", 5, figure_height/5+25, size=font_size, **text_params),
        SVG(presented_freq_fig).move(figure_width/2, figure_height/2+25),
        compose.Text("C", figure_width/2+10, figure_height/2+50, size=font_size,
                     **text_params),
        SVG(stim_fig).move(figure_width/2, -0),
        compose.Text("B", figure_width/2+10, 25, size=font_size,
                     **text_params),
    ).save(save_path)


def background_figure(theory_fig, save_path, context='paper'):
    """Create background theory figure.

    Parameters
    ----------
    theory_fig :
        path to the svg file containing the background theory figure.
    save_path : str
        path to save the composed figure at
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils',
                                                     'full')
    figure_width = _convert_to_pix(figure_width)
    figure_height = figure_width * .5
    font_size = _convert_to_pix(text_params.pop('size'))

    compose.Figure(
        figure_width, figure_height,
        SVG(theory_fig).move(6, -7),
        SVG(CONSTANT_CARTOON_PATH).scale(5).move(54, 22),
        SVG(SCALING_CARTOON_PATH).scale(5).move(54, 22+140),
        compose.Text("A", 14, -7+25, size=font_size, **text_params),
        compose.Text("B", figure_width/4+19, figure_height/3-15, size=font_size,
                     **text_params),
        compose.Text("C", 3*figure_width/4-16, -7+25, size=font_size,
                     **text_params),
    ).save(save_path)


def example_voxels(peakiness_fig, example_voxel_fig, save_path, context='paper'):
    """Create the example voxels figure.

    Which combines the panel with 3 example voxels and the peakiness check one,
    which plots all voxels on top of each other

    Parameters
    ----------
    peakiness_fig, example_voxel_fig
        paths to the svg files containing the peakiness check and 3 example
        voxel figures, respectively
    save_path : str
        path to save the composed figure at
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils', 'full')
    figure_width = _convert_to_pix(figure_width)
    figure_height = figure_width / 2.8
    font_size = _convert_to_pix(text_params.pop('size'))

    compose.Figure(
        figure_width, figure_height,
        SVG(peakiness_fig).move(figure_width*3/4-10, -2),
        SVG(example_voxel_fig).move(-8, 0),
        compose.Text("A", 0, 25, size=font_size, **text_params),
        compose.Text("B", figure_width*3/4-10, 25, size=font_size, **text_params),
    ).save(save_path)


def parameters(individual_fig, overall_fig, save_path, context='paper'):
    """Create the parameters figure.

    Which combines the figures showing the overall and individual parameters.

    Parameters
    ----------
    individual_fig, overall_fig
        paths to the svg files containing the individual and overall parameter
        values, respectively
    save_path : str
        path to save the composed figure at
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils', 'full')
    figure_width = _convert_to_pix(figure_width)
    figure_height = figure_width * 1.2
    font_size = _convert_to_pix(text_params.pop('size'))

    compose.Figure(
        figure_width, figure_height,
        SVG(overall_fig).move(0, 0),
        SVG(individual_fig).move(0, figure_height/2.25),
        compose.Text("A", 0, 25, size=font_size, **text_params),
        compose.Text("B", 0, figure_height/2.25+25, size=font_size, **text_params),
    ).save(save_path)


def visual_field_differences(comparison_fig, diff_fig, save_path,
                             context='paper'):
    """Create the visual field differences figure.

    Which combines the figures showing the preferred period of the two, and the
    differences between them

    Parameters
    ----------
    comparison_fig, diff_fig
        paths to the svg files containing the comparison and diff figures,
        respectively
    save_path : str
        path to save the composed figure at
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils', 'full')
    figure_width = _convert_to_pix(figure_width)
    figure_height = figure_width * .5
    font_size = _convert_to_pix(text_params.pop('size'))

    compose.Figure(
        figure_width, figure_height,
        SVG(diff_fig).move(figure_width/2-10, 0),
        SVG(comparison_fig).move(-5, 0),
        compose.Text("A", 0, 17, size=font_size, **text_params),
        compose.Text("B", figure_width/2-10, 17, size=font_size, **text_params),
    ).save(save_path)


def example_ecc_bins(ecc_bin_fig, save_path, context='paper'):
    """Add example stimuli to example eccentricity bin figure.

    Adds the example annulus and pinwheel stimuli.

    Parameters
    ----------
    comparison_fig, diff_fig
        paths to the svg files containing the comparison and diff figures,
        respectively
    save_path : str
        path to save the composed figure at
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked
    """
    text_params, figure_width = style.plotting_style(context, 'svgutils', 'full')
    figure_width = _convert_to_pix(figure_width)
    figure_height = figure_width / 2.2

    compose.Figure(
        figure_width, figure_height,
        SVG(ecc_bin_fig).move(45, 0),
        SVG(PINWHEEL_PATH).scale(5).move(130+45, 170),
        SVG(ANNULUS_PATH).scale(5).move(270+45, 170),
    ).save(save_path)


def schematic_model_2d(inputs_fig, model_schematic_fig, save_path,
                       context='paper'):
    """Combine the schematics showing 2d model input and example preferred period contours.

    Parameters
    ----------
    inputs_fig, model_schematic_fig
        paths to the svg files containing the inputs and 2d model schematics
        (showing preferred period contours), respectively
    save_path : str
        path to save the composed figure at
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils',
                                                     'full')
    figure_width = _convert_to_pix(figure_width)
    figure_height = figure_width / 2.2
    font_size = _convert_to_pix(text_params.pop('size'))

    compose.Figure(
        figure_width, figure_height,
        SVG(inputs_fig).move(0, 0),
        # we use the regular SVG here, because this has the legend applied, and
        # so has been saved out correctly
        compose.SVG(model_schematic_fig).move(figure_width/2-20, 10),
        compose.Text("A", 0, 25, size=font_size, **text_params),
        compose.Text("B", figure_width/2-10, 25, size=font_size, **text_params),
    ).save(save_path)
