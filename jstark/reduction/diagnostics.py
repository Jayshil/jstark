"""Diagnostic plots for the reduction steps.

All functions return ``(fig, axes)`` so that plots can be adjusted or saved.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from .robust import nanmedian
from ..plotting import (styled, WIDTH, MASK_CMAP, INK2, PALETTE, image_height, colorbar,
                        blank_colorbar)

__all__ = ['load_stage1_diagnostics', 'plot_masks', 'plot_group_bkg', 'plot_jumps',
           'plot_rateints', 'plot_cleaning', 'plot_wavelength', 'imshow_robust']


def imshow_robust(ax, img, pct=(1, 99), cmap='plasma', **kwargs):
    """``imshow`` with percentile-based colour limits."""
    finite = np.isfinite(img)
    vmin, vmax = np.nanpercentile(img[finite], pct) if finite.any() else (0, 1)
    return ax.imshow(img, origin='lower', aspect='auto', interpolation='none', cmap=cmap,
                     vmin=vmin, vmax=vmax, **kwargs)


def load_stage1_diagnostics(fname):
    """Load a ``*_stage1diag.npz`` file into a nested dictionary ``{step: {key: value}}``."""
    out = {}
    with np.load(fname) as f:
        for k in f.files:
            if '__' in k:
                step, key = k.split('__', 1)
                out.setdefault(step, {})[key] = f[k]
            else:
                out[k] = f[k]
    return out


def _stacked(fig_heights, sharex=True):
    """Figure with vertically stacked panels of the given heights (inches)."""
    return plt.subplots(len(fig_heights), 1, figsize=(WIDTH, sum(fig_heights) + 0.3),
                        sharex=sharex, layout='constrained', squeeze=False,
                        gridspec_kw={'height_ratios': fig_heights})


@styled
def plot_masks(signal_image, trace_mask, bkg_mask=None, title=None):
    """Show the quick source image and the trace / background masks."""
    h = image_height(signal_image)
    n = 3 if bkg_mask is not None else 2
    fig, axs = _stacked([h] * n)
    axs = axs[:, 0]
    im = imshow_robust(axs[0], signal_image, pct=(1, 99.5))
    axs[0].set_title('Quick source image (median CDS)' if title is None else title)
    colorbar(fig, im, axs[0], 'DN')
    masks = [(trace_mask, f'Trace mask ({trace_mask.mean() * 100:.1f}% of pixels)')]
    if bkg_mask is not None:
        masks.append((bkg_mask, f'Background pixels ({bkg_mask.mean() * 100:.1f}%)'))
    for ax, (m, t) in zip(axs[1:], masks):
        ax.imshow(m, origin='lower', aspect='auto', interpolation='none', cmap=MASK_CMAP)
        ax.set_title(t)
        blank_colorbar(fig, ax)
    for ax in axs:
        ax.set_ylabel('row')
    axs[-1].set_xlabel('column')
    return fig, axs


@styled
def plot_group_bkg(profile, group=-1, axis='column', dt=None):
    """Visualise the group-level background / 1/f stripes that were removed.

    Parameters
    ----------
    profile : ndarray
        ``(nint, ngroup, n)`` (columns) or ``(nint, ngroup, ny, namp)``
        (rows) stripe values from the ``group_bkg`` step.
    group : int
        Group to display.
    dt : float or None
        Time between consecutive stripes (s), to label the power spectrum
        axis in Hz; otherwise in cycles per stripe.
    """
    prof = np.asarray(profile)
    if prof.ndim == 4:
        prof = prof.mean(axis=-1)
    p = prof[:, group]
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 3.0), layout='constrained',
                            gridspec_kw={'width_ratios': [1.3, 1, 1]})
    im = imshow_robust(axs[0], p, cmap='jstark_div')
    axs[0].set_xlabel('column' if axis == 'column' else 'row')
    axs[0].set_ylabel('integration')
    axs[0].set_title(f'Removed {axis} offsets (group {group})')
    colorbar(fig, im, axs[0], 'DN')
    axs[1].plot(np.nanmedian(prof, axis=(0, 2)), 'o-', color=PALETTE[0], ms=4)
    axs[1].set_xlabel('group')
    axs[1].set_ylabel('median offset [DN]')
    axs[1].set_title('Static part')
    # power spectrum of the stripe pattern (time-variable part)
    var = p - np.nanmedian(p, axis=0)
    var = np.where(np.isfinite(var), var, 0.)
    ps = np.mean(np.abs(np.fft.rfft(var, axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(var.shape[1], d=dt if dt else 1.)
    axs[2].loglog(f[1:], ps[1:], color=PALETTE[0])
    axs[2].set_xlabel('frequency [Hz]' if dt else 'frequency [1/stripe]')
    axs[2].set_ylabel('power')
    axs[2].set_title('Power spectrum')
    return fig, axs


@styled
def plot_jumps(frac_per_int, count_map=None, times=None):
    """Fraction of pixels with a jump per integration and the 2D jump-count map."""
    heights = [2.2] + ([image_height(count_map)] if count_map is not None else [])
    fig, axs = _stacked(heights, sharex=False)
    axs = axs[:, 0]
    t = np.arange(len(frac_per_int)) if times is None else times
    axs[0].plot(t, np.asarray(frac_per_int) * 100, '.', color=INK2, ms=2)
    axs[0].set_ylabel('pixels with a jump [%]')
    axs[0].set_xlabel('integration' if times is None else 'time')
    if count_map is not None:
        im = axs[1].imshow(count_map, origin='lower', aspect='auto', interpolation='none',
                           cmap='plasma', vmax=max(1, np.nanpercentile(count_map, 99.9)))
        axs[1].set_title('Detected jumps per pixel')
        axs[1].set_xlabel('column')
        axs[1].set_ylabel('row')
        colorbar(fig, im, axs[1], 'jumps')
        blank_colorbar(fig, axs[0])
    return fig, axs


@styled
def plot_rateints(fname_or_data, trace_mask=None):
    """Median rate image and a quick white-light curve of a rateints file/cube."""
    if isinstance(fname_or_data, str):
        with fits.open(fname_or_data) as h:
            d = np.asarray(h['SCI'].data, dtype=np.float32)
            dq = np.asarray(h['DQ'].data)
        d = np.where((dq & 1) > 0, np.nan, d)
    else:
        d = np.asarray(fname_or_data)
    med = nanmedian(d, axis=0)
    if trace_mask is None:
        trace_mask = med > np.nanpercentile(med, 90)
    wl = np.nansum(np.where(np.isfinite(d[:, trace_mask]), d[:, trace_mask], med[trace_mask]), axis=1)
    ny, nx = med.shape
    if nx < 2.5 * ny:      # tall or square image: image and light curve side by side
        fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.4), layout='constrained',
                                gridspec_kw={'width_ratios': [1, 2]})
    else:
        fig, axs = _stacked([image_height(med), 2.2], sharex=False)
        axs = axs[:, 0]
    im = imshow_robust(axs[0], med, pct=(1, 99.5))
    axs[0].set_title('Median rate image')
    axs[0].set_xlabel('column')
    axs[0].set_ylabel('row')
    colorbar(fig, im, axs[0], 'DN/s')
    if nx >= 2.5 * ny:
        blank_colorbar(fig, axs[1])
    f = wl / np.nanmedian(wl)
    axs[1].plot(f, '.', color=INK2, ms=2)
    lo, hi = np.nanpercentile(f, [0.5, 99.5])
    pad = 0.3 * (hi - lo)
    axs[1].set_ylim(lo - pad, hi + pad)
    axs[1].set_xlabel('integration')
    axs[1].set_ylabel('normalised flux')
    axs[1].set_title('Quick white-light curve')
    return fig, axs


@styled
def plot_cleaning(cc):
    """Summary of a :class:`~jstark.reduction.CubeCleaner`: masks, outliers, white light."""
    med = cc.median_frame()
    h = image_height(med)
    fig, axs = _stacked([h, h, h, 2.2], sharex=False)
    axs = axs[:, 0]
    for ax in axs[1:3]:
        ax.sharex(axs[0])
        ax.sharey(axs[0])
    im = imshow_robust(axs[0], med, pct=(1, 99.5))
    axs[0].set_title('Median frame')
    colorbar(fig, im, axs[0])
    axs[1].imshow(cc.badpix == 0, origin='lower', aspect='auto', interpolation='none', cmap=MASK_CMAP)
    axs[1].set_title(f'Persistent bad pixels: {(cc.badpix == 0).sum()}')
    blank_colorbar(fig, axs[1])
    cnt = cc.crmask.sum(axis=0)
    im = axs[2].imshow(cnt, origin='lower', aspect='auto', interpolation='none', cmap='plasma',
                       vmax=max(1, np.percentile(cnt, 99.9)))
    axs[2].set_title(f'Transient flags per pixel (total {cc.crmask.sum()})')
    colorbar(fig, im, axs[2], 'flags')
    for ax in axs[:3]:
        ax.set_ylabel('row')
    for ax in axs[:2]:
        ax.tick_params(labelbottom=False)
    axs[2].set_xlabel('column')
    t = cc.times if cc.times is not None else np.arange(cc.shape[0])
    axs[3].plot(t, cc.white_light(), '.', color=INK2, ms=2)
    ax2 = axs[3].twinx()
    ax2.plot(t, cc.crmask.reshape(cc.shape[0], -1).mean(axis=1) * 100, '-', color=PALETTE[1], lw=0.6,
             alpha=0.7)
    ax2.set_ylabel('flagged [%]', color=PALETTE[1])
    ax2.tick_params(axis='y', which='both', colors=PALETTE[1])
    axs[3].tick_params(axis='y', which='both', right=False)
    axs[3].set_ylabel('white light (norm.)')
    axs[3].set_xlabel('BJD_TDB' if cc.times is not None else 'integration')
    blank_colorbar(fig, axs[3])
    return fig, axs


@styled
def plot_wavelength(wavelength, title='Wavelength map'):
    """Display a wavelength map (one panel per order for SOSS)."""
    w = np.asarray(wavelength)
    w = w[None] if w.ndim == 2 else w
    fig, axs = _stacked([image_height(w[0])] * len(w))
    for k, wk in enumerate(w):
        im = axs[k, 0].imshow(wk, origin='lower', aspect='auto', interpolation='none', cmap='turbo')
        axs[k, 0].set_title(title if len(w) == 1 else f'{title} - order {k + 1}')
        axs[k, 0].set_ylabel('row')
        colorbar(fig, im, axs[k, 0], 'wavelength [µm]')
    axs[-1, 0].set_xlabel('column')
    return fig, axs[:, 0]
