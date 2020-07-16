#!/usr/bin/python
"""functions to put the final touches on figures for publication
"""
from svgutils import compose


def crossvalidation(annotated_model_schematic, horizontal_cv_loss, save_path):
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

    """
    compose.Figure(740, 400,
                   compose.SVG(annotated_model_schematic),
                   compose.Text("A", 25, 40, size=20, weight='bold'),
                   compose.SVG(horizontal_cv_loss).move(380, 60),
                   compose.Text("B", 380+25, 40, size=20, weight='bold'),
                   ).save(save_path)
