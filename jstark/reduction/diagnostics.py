"""Diagnostic plots for the reduction steps.

All functions return ``(fig, axes)`` so that plots can be adjusted or saved.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from .robust import nanmedian

__all__ = ['load_stage1_diagnostics', 'plot_masks', 'plot_group_bkg', 'plot_jumps',
           'plot_rateints', 'plot_cleaning', 'plot_wavelength', 'imshow_robust']


def imshow_robust(ax, img, pct=(1, 99), cmap='viridis', **kwargs):
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


def plot_masks(signal_image, trace_mask, bkg_mask=None, title=None):
    """Show the quick source image and the trace / background masks."""
    n = 3 if bkg_mask is not None else 2
    fig, axs = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)
    im = imshow_robust(axs[0], signal_image, pct=(1, 99.5))
    axs[0].set_title('Quick source image (median CDS)' if title is None else title)
    fig.colorbar(im, ax=axs[0], pad=0.01)
    axs[1].imshow(trace_mask, origin='lower', aspect='auto', interpolation='none', cmap='gray_r')
    axs[1].set_title(f'Trace mask ({trace_mask.mean() * 100:.1f}% of pixels)')
    if bkg_mask is not None:
        axs[2].imshow(bkg_mask, origin='lower', aspect='auto', interpolation='none', cmap='gray_r')
        axs[2].set_title(f'Background pixels ({bkg_mask.mean() * 100:.1f}%)')
    axs[-1].set_xlabel('x [pix]')
    fig.tight_layout()
    return fig, axs


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
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    im = imshow_robust(axs[0], p, cmap='RdBu_r')
    axs[0].set_xlabel('column' if axis == 'column' else 'row')
    axs[0].set_ylabel('integration')
    axs[0].set_title(f'Removed {axis} offsets, group {group}')
    fig.colorbar(im, ax=axs[0])
    axs[1].plot(np.nanmedian(prof, axis=(0, 2)), 'o-')
    axs[1].set_xlabel('group')
    axs[1].set_ylabel('median offset [DN]')
    axs[1].set_title('Static part')
    # power spectrum of the stripe pattern (time-variable part)
    var = p - np.nanmedian(p, axis=0)
    var = np.where(np.isfinite(var), var, 0.)
    ps = np.mean(np.abs(np.fft.rfft(var, axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(var.shape[1], d=dt if dt else 1.)
    axs[2].loglog(f[1:], ps[1:])
    axs[2].set_xlabel('frequency [Hz]' if dt else 'frequency [1/stripe]')
    axs[2].set_title('Power spectrum of the stripes')
    fig.tight_layout()
    return fig, axs


def plot_jumps(frac_per_int, count_map=None, times=None):
    """Fraction of pixels with a jump per integration and the 2D jump-count map."""
    n = 2 if count_map is not None else 1
    fig, axs = plt.subplots(n, 1, figsize=(12, 3.2 * n))
    axs = np.atleast_1d(axs)
    t = np.arange(len(frac_per_int)) if times is None else times
    axs[0].plot(t, np.asarray(frac_per_int) * 100, 'k.', ms=2)
    axs[0].set_ylabel('% pixels with a jump')
    axs[0].set_xlabel('integration' if times is None else 'time')
    if count_map is not None:
        im = axs[1].imshow(count_map, origin='lower', aspect='auto', interpolation='none',
                           cmap='magma', vmax=np.nanpercentile(count_map, 99.9))
        axs[1].set_title('Number of detected jumps per pixel')
        fig.colorbar(im, ax=axs[1])
    fig.tight_layout()
    return fig, axs


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
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    im = imshow_robust(axs[0], med, pct=(1, 99.5))
    axs[0].set_title('Median rate image [DN/s]')
    fig.colorbar(im, ax=axs[0], pad=0.01)
    axs[1].plot(wl / np.nanmedian(wl), 'k.', ms=2)
    axs[1].set_xlabel('integration')
    axs[1].set_ylabel('normalised flux')
    axs[1].set_title('Quick white-light curve')
    fig.tight_layout()
    return fig, axs


def plot_cleaning(cc):
    """Summary of a :class:`~jstark.reduction.CubeCleaner`: masks, outliers, white light."""
    fig, axs = plt.subplots(4, 1, figsize=(12, 11))
    im = imshow_robust(axs[0], cc.median_frame(), pct=(1, 99.5))
    axs[0].set_title('Median frame')
    fig.colorbar(im, ax=axs[0], pad=0.01)
    axs[1].imshow(cc.badpix == 0, origin='lower', aspect='auto', interpolation='none', cmap='gray_r')
    axs[1].set_title(f'Persistent bad pixels: {(cc.badpix == 0).sum()}')
    cnt = cc.crmask.sum(axis=0)
    im = axs[2].imshow(cnt, origin='lower', aspect='auto', interpolation='none', cmap='magma',
                       vmax=max(1, np.percentile(cnt, 99.9)))
    axs[2].set_title(f'Transient flags per pixel (total {cc.crmask.sum()})')
    fig.colorbar(im, ax=axs[2], pad=0.01)
    t = cc.times if cc.times is not None else np.arange(cc.shape[0])
    axs[3].plot(t, cc.white_light(), 'k.', ms=2)
    ax2 = axs[3].twinx()
    ax2.plot(t, cc.crmask.reshape(cc.shape[0], -1).mean(axis=1) * 100, 'r-', lw=0.5, alpha=0.6)
    ax2.set_ylabel('% flagged', color='r')
    axs[3].set_ylabel('white light (norm.)')
    axs[3].set_xlabel('BJD_TDB' if cc.times is not None else 'integration')
    fig.tight_layout()
    return fig, axs


def plot_wavelength(wavelength, title='Wavelength map [micron]'):
    """Display a wavelength map (one panel per order for SOSS)."""
    w = np.asarray(wavelength)
    w = w[None] if w.ndim == 2 else w
    fig, axs = plt.subplots(len(w), 1, figsize=(12, 2.5 * len(w)), squeeze=False)
    for k, wk in enumerate(w):
        im = axs[k, 0].imshow(wk, origin='lower', aspect='auto', interpolation='none', cmap='turbo')
        axs[k, 0].set_title(title if len(w) == 1 else f'{title} - order {k + 1}')
        fig.colorbar(im, ax=axs[k, 0], pad=0.01)
    fig.tight_layout()
    return fig, axs[:, 0]
