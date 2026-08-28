"""
plot_style.py
=============
Shared dark theme for all standalone study scripts (convergence_study.py,
advanced_study.py, etc.) so every output PNG looks like it belongs to the
same report, matching main.py's PPT-image aesthetic.

Usage: `from plot_style import apply_style, PALETTE` then `apply_style()`
once at import time.
"""
import matplotlib.pyplot as plt

BG, PANEL, GRID_C = '#080C14', '#0E1420', '#1E2A3A'
TEXT, DIM = '#D0DCF0', '#6A7A9A'

PALETTE = dict(
    BG=BG, PANEL=PANEL, GRID=GRID_C, TEXT=TEXT, DIM=DIM,
    CONV='#FF5B35', STLTH='#00E5FF',
    HH='#00E5FF', VV='#FF5B35', HV='#FFD040', VH='#C084FC',
    ACCENT='#39FF88', WARN='#FF4000', BASELINE='#8892A6',
)


def apply_style():
    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': PANEL, 'axes.edgecolor': GRID_C,
        'axes.labelcolor': TEXT, 'text.color': TEXT,
        'xtick.color': DIM, 'ytick.color': DIM,
        'grid.color': GRID_C, 'grid.alpha': 0.35, 'grid.linewidth': 0.6,
        'font.size': 10.5, 'font.family': 'sans-serif',
        'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlecolor': TEXT,
        'axes.labelsize': 10, 'legend.facecolor': PANEL, 'legend.edgecolor': GRID_C,
        'legend.framealpha': 0.9, 'savefig.facecolor': BG,
        'axes.spines.top': False, 'axes.spines.right': False,
    })


def style_axis(ax, title=None, xlabel=None, ylabel=None):
    """Common per-axis polish: title/labels + a light fill under each line."""
    if title:  ax.set_title(title, pad=10)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35, lw=0.6)
    ax.tick_params(length=0)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(PALETTE['GRID'])
