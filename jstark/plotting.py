"""Common figure style for the jstark diagnostic plots.

Every plotting function in jstark draws its figure with :data:`STYLE`
(through :func:`styled`), so that the plots look the same whatever the
user's matplotlib settings. :func:`use_style` applies the same style to
your own figures.
"""
import functools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

__all__ = ['STYLE', 'WIDTH', 'PALETTE', 'INK', 'INK2', 'MUTED', 'MASK_CMAP', 'DIV_CMAP',
           'use_style', 'styled', 'image_height', 'colorbar', 'blank_colorbar']

#: Default figure width (inches): about the width of a documentation page or
#: a notebook cell at 100 dpi, so that the fonts are shown near their size.
WIDTH = 9.

#: Categorical colours, in this order (``'C0'`` ... ``'C7'`` after :func:`use_style`):
#: blue, orange, aqua, yellow, magenta, green, violet, red. The order keeps
#: neighbouring series distinguishable for colour-blind readers. jstark's own
#: plot functions use these values directly: a ``'CN'`` name is resolved when
#: the figure is drawn, after :func:`styled` has restored the user's settings.
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']

#: Text and axes ink: primary, secondary (data points, ticks) and muted (references).
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'

#: Diverging colour map (``'jstark_div'``): blue - neutral grey - red, for residuals.
DIV_CMAP = LinearSegmentedColormap.from_list(
    'jstark_div', ['#0d366b', '#2a78d6', '#9ec5f4', '#f0efec', '#f3a9a8', '#e34948', '#8c1d1d'])
DIV_CMAP.set_bad('#ffffff')

#: Colour map for boolean masks (False white, True blue).
MASK_CMAP = ListedColormap(['#ffffff', '#3987e5'])

if DIV_CMAP.name not in matplotlib.colormaps:
    matplotlib.colormaps.register(DIV_CMAP)


def _font_families(preferred=('Arial', 'Helvetica Neue', 'Helvetica', 'Liberation Sans')):
    """The preferred sans-serif families that are installed, then DejaVu Sans.

    Actual family names (not the generic ``'sans-serif'``) keep every text in
    this font after a styled plot function has restored the rcParams; only
    installed ones, since matplotlib warns about every missing family.
    """
    have = {f.name for f in font_manager.fontManager.ttflist}
    return [f for f in preferred if f in have] + ['DejaVu Sans']


#: matplotlib ``rcParams`` used by all jstark plots.
STYLE = {
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.family': _font_families(),
    'mathtext.fontset': 'stixsans',
    'font.size': 10,
    'text.color': INK,
    'axes.titlesize': 11,
    'axes.titlelocation': 'left',
    'axes.titlepad': 6,
    'axes.labelsize': 10,
    'axes.labelcolor': INK,
    'axes.edgecolor': INK2,
    'axes.linewidth': 0.7,
    'axes.formatter.useoffset': False,
    'axes.prop_cycle': cycler(color=PALETTE),
    'legend.fontsize': 9,
    'legend.frameon': False,
    'legend.handlelength': 1.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.color': INK2,
    'ytick.color': INK2,
    'xtick.labelcolor': INK,
    'ytick.labelcolor': INK,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'lines.linewidth': 1.4,
    'lines.markersize': 3,
    'errorbar.capsize': 0,
    'image.cmap': 'plasma',           # images and counts; wavelength maps use 'turbo'
    'image.interpolation': 'none',
    'image.origin': 'lower',
}


def use_style():
    """Apply :data:`STYLE` to all later figures (e.g. at the top of a notebook)."""
    plt.rcParams.update(STYLE)


def styled(func):
    """Decorator: draw the figure of ``func`` with :data:`STYLE`.

    Only the keys of :data:`STYLE` are restored afterwards. ``plt.rc_context``
    would restore every rcParam, including ``interactive``: if the first figure
    of a Jupyter session is drawn here, the inline backend is activated inside
    the context and switches ``interactive`` on, and restoring it to False
    stops all later figures from being shown.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        orig = {k: plt.rcParams[k] for k in STYLE}
        plt.rcParams.update(STYLE)
        try:
            return func(*args, **kwargs)
        finally:
            plt.rcParams.update(orig)
    return wrapper


def image_height(img, width=WIDTH, lo=1.3, hi=3.2):
    """Panel height (inches) that shows a detector image at a sensible aspect."""
    ny, nx = np.shape(img)[-2:]
    return float(np.clip(0.9 * width * ny / max(nx, 1), lo, hi))


def colorbar(fig, im, ax, label=None):
    """A slim colour bar next to ``ax``."""
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04, aspect=15)
    cb.ax.minorticks_off()
    cb.outline.set_linewidth(0.5)
    if label:
        cb.set_label(label)
    return cb


def blank_colorbar(fig, ax):
    """Reserve the space of a colour bar, so that stacked panels stay aligned."""
    cb = colorbar(fig, plt.cm.ScalarMappable(cmap=MASK_CMAP), ax)
    cb.ax.set_visible(False)
    return cb
