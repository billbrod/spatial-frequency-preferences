#!/usr/bin/env python3
"""code related to styling plots
"""


def plotting_style(context='paper', for_package='matplotlib', figsize='full'):
    """Get dictionary of formatting options for figure creation

    inspired by seaborn's plotting_context and axes_style (and using several of
    the values from there as a starting point), this is to make sure our
    figures are consistently-styled.

    if `for_package='matplotlib'`, this will be a dictionary to pass to
    `plt.style.use()`.

    This has mainly been tested for the paper context, may not work quite as
    well for poster

    Parameters
    ----------
    context : {'paper', 'poster'}, optional
        whether the figure will be used in a paper or presentation
    for_package : {'matplotlib', 'svgutils'}
        whether this is for matplotlib/seaborn or svgutils package
    figsize : {'full', 'half'}
        our figures come in two sizes: full or half, which correspond to
        whether they take up a full page / slide, or half of one.

    Returns
    -------
    params : dict
        dictionary of formatting options. see above for how to use
    figure_width : float
        the width of the figure, in inches. height will be determined by the
        different plotting/compose functions, because different figures have
        different aspect ratios

    """
    if for_package == 'matplotlib':
        params = {'figure.facecolor': 'white', 'axes.labelcolor': '.15',
                  'xtick.direction': 'out', 'ytick.direction': 'out',
                  'xtick.color': '.15', 'ytick.color': '.15',
                  'axes.axisbelow': True, 'grid.linestyle': '-',
                  'text.color': '.15', 'font.family': ['sans-serif'],
                  'font.sans-serif': ['Helvetica'],
                  'lines.solid_capstyle': 'round', 'patch.edgecolor': 'w',
                  'patch.force_edgecolor': True, 'image.cmap': 'rocket',
                  'xtick.top': False, 'ytick.right': False,
                  'axes.grid': False, 'axes.facecolor': 'white',
                  'axes.edgecolor': '.15', 'grid.color': '.8',
                  'axes.spines.left': True, 'axes.spines.bottom': True,
                  'axes.spines.right': False, 'axes.spines.top': False,
                  'xtick.bottom': False, 'ytick.left': False,
                  'figure.dpi': 90,
                  'text.usetex': False, }
                  # this is necessary for dealing with underscores in column
                  # names, see
                  # https://github.com/matplotlib/matplotlib/issues/17774
                  # 'text.latex.preamble': r"\usepackage{underscore}"}
        if context == 'paper':
            params.update({'font.size': 10,
                           'axes.labelsize': 10,
                           'axes.titlesize': 10,
                           'xtick.labelsize': 8,
                           'ytick.labelsize': 8,
                           'legend.fontsize': 10,
                           'axes.linewidth': 1.0,
                           'grid.linewidth': 0.8,
                           'lines.linewidth': 1.2,
                           'lines.markersize': 4.8,
                           'patch.linewidth': 1.2,
                           'xtick.major.width': 1.0,
                           'ytick.major.width': 1.0,
                           'xtick.minor.width': 0.8,
                           'ytick.minor.width': 0.8,
                           'xtick.major.size': 4.8,
                           'ytick.major.size': 4.8,
                           'xtick.minor.size': 3.2,
                           'ytick.minor.size': 3.2})
            if figsize == 'full':
                figure_width = 6.5
            elif figsize == 'half':
                figure_width = 3.25
        elif context == 'poster':
            params.update({'font.size': 24,
                           'axes.labelsize': 24,
                           'axes.titlesize': 24,
                           'xtick.labelsize': 20,
                           'ytick.labelsize': 20,
                           'legend.fontsize': 24,
                           'axes.linewidth': 2.5,
                           'grid.linewidth': 2,
                           'lines.linewidth': 3.0,
                           'lines.markersize': 12,
                           'patch.linewidth': 2.5,
                           'xtick.major.width': 2.5,
                           'ytick.major.width': 2.5,
                           'xtick.minor.width': 2,
                           'ytick.minor.width': 2,
                           'xtick.major.size': 12,
                           'ytick.major.size': 12,
                           'xtick.minor.size': 8,
                           'ytick.minor.size': 8})
            # matplotlib figures have to be specified in inches, which requires
            # making sure we know the dpi and converting it back. this will
            # work for pngs (and other raster graphics), but for svgs (which is
            # what we actually use), it will only be approximately correct
            if figsize == 'full':
                figure_width = 1920 / params['figure.dpi']
            elif figsize == 'half':
                figure_width = (1920/2) / params['figure.dpi']
        params['figure.titlesize'] = params['axes.titlesize']
    elif for_package == 'svgutils':
        if context == 'paper':
            params = {'font': 'Helvetica', 'size': '10pt'}
            if figsize == 'full':
                figure_width = '6.5in'
            elif figsize == 'half':
                figure_width = '3.25in'
        if context == 'poster':
            params = {'font': 'Helvetica', 'size': '28.8pt'}
            if figsize == 'full':
                figure_width = '1080px'
            elif figsize == 'half':
                figure_width = '540px'
    return params, figure_width
