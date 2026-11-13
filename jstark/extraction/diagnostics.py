"""Diagnostic plots for spectral extraction and photometry."""
import numpy as np
import matplotlib.pyplot as plt

from .metrics import p2p_sigma, binned_noise
from .trace import fill_bad
from .contamination import load_detection_image

__all__ = ['plot_trace', 'plot_psf', 'plot_psf_width', 'plot_residuals', 'plot_optimisation',
           'plot_white_light', 'plot_regressors', 'plot_spectra', 'plot_photometry',
           'plot_contaminants']


def _imshow(ax, img, pct=(1, 99), **kw):
    v = img[np.isfinite(img)]
    lo, hi = np.percentile(v, pct) if v.size else (0, 1)
    return ax.imshow(img, origin='lower', aspect='auto', interpolation='none', vmin=lo, vmax=hi, **kw)


def plot_trace(ex, ax=None):
    """Median frame (canonical orientation) with the traces and the background region."""
    if not ex.traces:
        ex.find_trace()
    ex._ensure_background_region()
    img = fill_bad(ex.src.median, ex.src.badpix)
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    _imshow(ax, np.arcsinh(img / max(np.nanpercentile(img, 90), 1e-9)), cmap='viridis')
    ax.contour(ex.bkg_mask, levels=[0.5], colors='w', linewidths=0.5)
    for o, tr in ex.traces.items():
        a = ex.p('extract', o)['aperture']
        ax.plot(tr.cols, tr.y0, 'r-', lw=0.8, label=f'order {o}')
        ax.plot(tr.cols, tr.y0 - a, 'r:', lw=0.6)
        ax.plot(tr.cols, tr.y0 + a, 'r:', lw=0.6)
    ax.set_xlabel('spectral pixel')
    ax.set_ylabel('spatial pixel')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'{ex.mode}: traces, apertures (dotted) and background region (white contour)')
    return ax


def plot_psf(ex, order=None, cols=None, nint=50):
    """Normalised pixel values of a few columns against the fitted profile."""
    o = order or ex.orders[0]
    tr, psf = ex.traces[o], ex.psfs[o]
    cols = cols if cols is not None else np.linspace(0, tr.ncol - 1, 4).astype(int)
    d, e, g, gi = ex.src.read_ints(ex.src.sample_ints(nint))
    d = np.array(d, np.float32, copy=True)
    prep = ex._prepare(d, e, g)
    W = ex._window(o, d, e, g, prep[o]['dy'])
    fig, axs = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.5), sharey=True)
    for ax, c in zip(np.atleast_1d(axs), cols):
        f = np.sum(np.where(W['G'], W['D'], 0)[:, :, c] * W['P'][:, :, c], 1) / \
            np.sum(np.where(W['G'], W['P'], 0)[:, :, c] ** 2, 1)
        u = W['u'][:, :, c]
        y = W['D'][:, :, c] / f[:, None]
        ok = W['G'][:, :, c]
        ax.plot(u[ok], y[ok], 'k.', ms=1, alpha=0.3)
        uu, pp = psf.profile(c)
        ax.plot(uu, pp, 'r-', lw=1)
        lam = tr.info.get('wavelength')
        ax.set_title(f'column {tr.cols[c]}' + (f' ({lam[c]:.2f} um)' if lam is not None else ''))
        ax.set_xlabel('offset from trace [px]')
    np.atleast_1d(axs)[0].set_ylabel('normalised flux (x pixel response)')
    fig.tight_layout()
    return fig


def plot_psf_width(ex, order=None):
    """FWHM and peak of the profile along the trace."""
    o = order or ex.orders[0]
    tr, psf = ex.traces[o], ex.psfs[o]
    x = tr.info.get('wavelength', tr.cols)
    cx, norm = psf.column_coeffs()
    u = np.linspace(-psf.halfwidth, psf.halfwidth, 1001)
    peak = np.array([psf.profile(c, u)[1].max() for c in range(tr.ncol)])
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axs[0].plot(x, psf.width(), 'k-')
    axs[0].set_ylabel('FWHM [px]')
    axs[1].plot(x, peak, 'k-')
    axs[1].set_ylabel('peak (unit integral)')
    axs[1].set_xlabel('wavelength [um]' if 'wavelength' in tr.info else 'spectral pixel')
    fig.tight_layout()
    return fig


def plot_residuals(ex, segment=0, ints=slice(0, 200), order=None):
    """Median normalised residual image ``(data - model) / sigma`` of some integrations."""
    o = order or ex.orders[0]
    d, e, g, gi = ex.src.read(segment, ints)
    d = np.array(d, np.float32, copy=True)
    prep = ex._prepare(d, e, g)
    r = ex._extract_order(o, d, e, g, prep[o]['dy'], prep[o]['var_bkg'], method='optimal')
    W = ex._window(o, d, e, g, r['dy'])
    z = np.where(W['G'] & (W['e'] > 0), (W['D'] - r['flux'][:, None, :] * W['P']) / np.sqrt(W['V']),
                 np.nan)
    med = np.nanmedian(z, axis=0)
    fig, ax = plt.subplots(figsize=(14, 3))
    im = ax.imshow(med, origin='lower', aspect='auto', interpolation='none', cmap='RdBu_r',
                   vmin=-3, vmax=3)
    plt.colorbar(im, ax=ax, label='median residual [sigma]')
    ax.set_xlabel('trace column')
    ax.set_ylabel('window row')
    ax.set_title(f'order {o}: median residual of {len(d)} integrations; '
                 f'chi2/pixel = {np.nanmean(z ** 2):.2f}')
    return fig


def plot_optimisation(res, ax=None):
    """White-light scatter versus aperture for every background inner radius."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for i, rin in enumerate(res['bkg_inner']):
        ax.plot(res['apertures'], 1e6 * res['sigma'][i], 'o-', label=f'bkg inner {rin:g}')
    b = res['best']
    ax.plot(b['aperture'], 1e6 * b['sigma'], 'k*', ms=12)
    ax.set_xlabel('aperture half-width [px]')
    ax.set_ylabel('white-light p2p scatter [ppm]')
    ax.legend(fontsize=8)
    return ax


def plot_white_light(spec, order=None, bins=(1, 2, 4, 8, 16, 32, 64)):
    """White-light curves (optimal and box) and their binned scatter."""
    o = order or spec.orders[0]
    t = spec.times - spec.times[0] if spec.times is not None else np.arange(len(spec.gidx))
    fig, axs = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [3, 1]})
    for method, c in (('optimal', 'k'), ('box', 'C1')):
        if method == 'box' and o not in spec.box:
            continue
        f, e = spec.white_light(o, method=method)
        axs[0].plot(t * 24, f, '.', color=c, ms=2,
                    label=f'{method}: {1e6 * p2p_sigma(f):.0f} ppm p2p, '
                          f'ratio {p2p_sigma(f) / np.median(e):.2f}')
        nb, s = binned_noise(f, bins)
        axs[1].loglog(nb, 1e6 * s, 'o-', color=c)
        axs[1].loglog(nb, 1e6 * s[0] / np.sqrt(nb), ':', color=c)
    axs[0].set_xlabel('time [h]')
    axs[0].set_ylabel('relative flux')
    axs[0].legend(fontsize=8)
    axs[1].set_xlabel('bin size [integrations]')
    axs[1].set_ylabel('scatter [ppm]')
    fig.tight_layout()
    return fig


def plot_regressors(spec, order=None):
    """Trace offsets, width changes, spectral shifts and background against time."""
    o = order or spec.orders[0]
    r = spec.regressors[o]
    t = spec.times - spec.times[0] if spec.times is not None else np.arange(len(spec.gidx))
    keys = [k for k in ('dy', 'ds', 'dx', 'bkg', 'maskfrac', 'white') if k in r]
    fig, axs = plt.subplots(len(keys), 1, figsize=(12, 1.8 * len(keys)), sharex=True)
    for ax, k in zip(axs, keys):
        ax.plot(t * 24, r[k], 'k-', lw=0.6)
        ax.set_ylabel(k)
    axs[-1].set_xlabel('time [h]')
    fig.tight_layout()
    return fig


def plot_spectra(spec, order=None, nshow=50):
    """Extracted spectra (a subset) and their per-column precision."""
    o = order or spec.orders[0]
    f, e = spec.flux[o], spec.err[o]
    x = spec.wavelength[o] if np.isfinite(spec.wavelength[o]).any() else spec.cols[o]
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for i in np.linspace(0, len(f) - 1, min(nshow, len(f))).astype(int):
        axs[0].plot(x, f[i], 'k-', lw=0.3, alpha=0.3)
    axs[0].set_ylabel('flux')
    rel = p2p_sigma(f / np.nanmedian(f, 0), axis=0)
    axs[1].plot(x, 1e6 * rel, 'k-', lw=0.6, label='p2p scatter')
    axs[1].plot(x, 1e6 * np.nanmedian(e / np.nanmedian(f, 0), 0), 'r-', lw=0.6, label='errors')
    axs[1].set_yscale('log')
    axs[1].set_ylabel('per-column precision [ppm]')
    axs[1].legend(fontsize=8)
    axs[1].set_xlabel('wavelength [um]')
    fig.tight_layout()
    return fig


def plot_photometry(res):
    """Aperture and PSF-weighted light curves, and the regressors."""
    t = res.times - res.times[0] if res.times is not None else np.arange(len(res.flux['psf']))
    fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for k, c in (('aperture', 'C1'), ('psf', 'k')):
        f = res.flux[k] / np.median(res.flux[k])
        e = res.err[k] / np.median(res.flux[k])
        axs[0].plot(t * 24, f, '.', color=c, ms=2,
                    label=f'{k}: {1e6 * p2p_sigma(f):.0f} ppm, ratio {p2p_sigma(f) / np.median(e):.2f}')
    axs[0].legend(fontsize=8)
    axs[0].set_ylabel('relative flux')
    axs[1].plot(t * 24, res.table['xc'] - np.median(res.table['xc']), label='x')
    axs[1].plot(t * 24, res.table['yc'] - np.median(res.table['yc']), label='y')
    axs[1].legend(fontsize=8)
    axs[1].set_ylabel('centroid [px]')
    axs[2].plot(t * 24, res.table['width'], 'k-', lw=0.6)
    axs[2].set_ylabel('width [px]')
    axs[3].plot(t * 24, res.table['bkg'], 'k-', lw=0.6)
    axs[3].set_ylabel('background')
    axs[3].set_xlabel('time [h]')
    fig.tight_layout()
    return fig


def plot_contaminants(ex, image=None, ax=None):
    """Detection image with the contaminant mask, field-star traces and the target's traces.

    ``image`` is the detection image given to
    :meth:`~jstark.extraction.SpectralExtractor.find_contaminants` (e.g. the
    F277W exposure); ``None`` shows the exposure median frame.
    """
    if not ex.traces:
        ex.find_trace()
    if image is None:
        img = fill_bad(ex.src.median, ex._badpix0)
        title = 'exposure median'
    else:
        img = load_detection_image(image, ex.src)
        title = 'detection image'
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    _imshow(ax, np.arcsinh(img / max(np.nanpercentile(np.abs(img), 90), 1e-9)), cmap='magma')
    if ex.contam_mask is not None and ex.contam_mask.any():
        ax.contour(ex.contam_mask, levels=[0.5], colors='c', linewidths=0.8)
    for s in ex.contaminants:
        ax.annotate(f"S/N {s['peak']:.0f}", (s['x'], s['bbox'][1]), color='c', fontsize=7,
                    ha='center', va='bottom')
    for o, tr in ex.traces.items():
        ax.plot(tr.cols, tr.y0, 'w-', lw=0.6)
    for ft in ex.field_traces:
        g = ex._field_geometry(ft)
        if g is not None:
            ax.plot(g['x'], g['y'], 'y--', lw=0.8)
    ax.set_xlabel('spectral pixel')
    ax.set_ylabel('spatial pixel')
    ax.set_title(f'{title}: contaminant mask (cyan), traces (white), field-star traces (yellow)')
    return ax
