"""Diagnostic plots for spectral extraction and photometry."""
import numpy as np
import matplotlib.pyplot as plt

from .metrics import p2p_sigma, binned_noise
from .trace import fill_bad
from .contamination import load_detection_image
from ..plotting import styled, WIDTH, INK2, PALETTE, image_height, colorbar

__all__ = ['plot_trace', 'plot_psf', 'plot_psf_width', 'plot_residuals', 'plot_optimisation',
           'plot_white_light', 'plot_regressors', 'plot_spectra', 'plot_photometry',
           'plot_contaminants']


def _imshow(ax, img, pct=(1, 99), **kw):
    v = img[np.isfinite(img)]
    lo, hi = np.percentile(v, pct) if v.size else (0, 1)
    return ax.imshow(img, origin='lower', aspect='auto', interpolation='none', vmin=lo, vmax=hi, **kw)


@styled
def plot_trace(ex, ax=None):
    """Median frame (canonical orientation) with the traces and the background region."""
    if not ex.traces:
        ex.find_trace()
    ex._ensure_background_region()
    img = fill_bad(ex.src.median, ex.src.badpix)
    if ax is None:
        fig, ax = plt.subplots(figsize=(WIDTH, image_height(img, hi=3.6) + 0.7), layout='constrained')
    _imshow(ax, np.arcsinh(img / max(np.nanpercentile(img, 90), 1e-9)), cmap='plasma')
    ax.contour(ex.bkg_mask, levels=[0.5], colors='w', linewidths=0.8, linestyles='--')
    for o, tr in ex.traces.items():
        a = ex.p('extract', o)['aperture']
        ax.plot(tr.cols, tr.y0, '-', color=PALETTE[2], lw=1, label=f'order {o}')
        ax.plot(tr.cols, tr.y0 - a, ':', color=PALETTE[2], lw=1.2)
        ax.plot(tr.cols, tr.y0 + a, ':', color=PALETTE[2], lw=1.2)
    ax.set_xlabel('spectral pixel')
    ax.set_ylabel('spatial pixel')
    ax.legend(loc='upper right', labelcolor='w')
    ax.set_title(f'{ex.mode}: trace, aperture (dotted) and background region (white, dashed)')
    return ax


@styled
def plot_psf(ex, order=None, cols=None, nint=50):
    """Normalised pixel values of a few columns against the fitted profile."""
    o = order or ex.orders[0]
    tr, psf = ex.traces[o], ex.psfs[o]
    cols = cols if cols is not None else np.linspace(0, tr.ncol - 1, 4).astype(int)
    d, e, g, gi = ex.src.read_ints(ex.src.sample_ints(nint))
    d = np.array(d, np.float32, copy=True)
    prep = ex._prepare(d, e, g)
    W = ex._window(o, d, e, g, prep[o]['dy'])
    fig, axs = plt.subplots(1, len(cols), figsize=(WIDTH, 2.9), sharey=True, layout='constrained')
    for ax, c in zip(np.atleast_1d(axs), cols):
        f = np.sum(np.where(W['G'], W['D'], 0)[:, :, c] * W['P'][:, :, c], 1) / \
            np.sum(np.where(W['G'], W['P'], 0)[:, :, c] ** 2, 1)
        u = W['u'][:, :, c]
        y = W['D'][:, :, c] / f[:, None]
        ok = W['G'][:, :, c]
        ax.plot(u[ok], y[ok], '.', color=INK2, ms=1.5, alpha=0.3)
        uu, pp = psf.profile(c)
        ax.plot(uu, pp, '-', color=PALETTE[1], lw=1.4)
        lam = tr.info.get('wavelength')
        ax.set_title(f'column {tr.cols[c]}' + (f' ({lam[c]:.2f} µm)' if lam is not None else ''))
        ax.set_xlabel('offset from trace [px]')
    np.atleast_1d(axs)[0].set_ylabel('normalised flux')
    return fig


@styled
def plot_psf_width(ex, order=None):
    """FWHM and peak of the profile along the trace."""
    o = order or ex.orders[0]
    tr, psf = ex.traces[o], ex.psfs[o]
    x = tr.info.get('wavelength', tr.cols)
    cx, norm = psf.column_coeffs()
    u = np.linspace(-psf.halfwidth, psf.halfwidth, 1001)
    peak = np.array([psf.profile(c, u)[1].max() for c in range(tr.ncol)])
    fig, axs = plt.subplots(2, 1, figsize=(WIDTH, 4.2), sharex=True, layout='constrained')
    axs[0].plot(x, psf.width(), '-', color=PALETTE[0])
    axs[0].set_ylabel('FWHM [px]')
    axs[1].plot(x, peak, '-', color=PALETTE[0])
    axs[1].set_ylabel('peak (unit integral)')
    axs[1].set_xlabel('wavelength [µm]' if 'wavelength' in tr.info else 'spectral pixel')
    return fig


@styled
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
    fig, ax = plt.subplots(figsize=(WIDTH, image_height(med) + 0.7), layout='constrained')
    im = ax.imshow(med, origin='lower', aspect='auto', interpolation='none', cmap='jstark_div',
                   vmin=-3, vmax=3)
    colorbar(fig, im, ax, 'median residual [σ]')
    ax.set_xlabel('trace column')
    ax.set_ylabel('window row')
    ax.set_title(f'order {o}: median residual of {len(d)} integrations; '
                 f'chi2/pixel = {np.nanmean(z ** 2):.2f}')
    return fig


@styled
def plot_optimisation(res, ax=None):
    """White-light scatter versus aperture for every background inner radius."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 3.6), layout='constrained')
    for i, rin in enumerate(res['bkg_inner']):
        ax.plot(res['apertures'], 1e6 * res['sigma'][i], 'o-', label=f'bkg inner {rin:g}')
    b = res['best']
    ax.plot(b['aperture'], 1e6 * b['sigma'], '*', color=INK2, ms=12, label='best')
    ax.set_xlabel('aperture half-width [px]')
    ax.set_ylabel('white-light p2p scatter [ppm]')
    ax.legend()
    return ax


@styled
def plot_white_light(spec, order=None, bins=(1, 2, 4, 8, 16, 32, 64)):
    """White-light curves (optimal and box) and their binned scatter."""
    o = order or spec.orders[0]
    t = spec.times - spec.times[0] if spec.times is not None else np.arange(len(spec.gidx))
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.3), layout='constrained',
                            gridspec_kw={'width_ratios': [3, 1]})
    for method, c in (('box', PALETTE[1]), ('optimal', PALETTE[0])):
        if method == 'box' and o not in spec.box:
            continue
        f, e = spec.white_light(o, method=method)
        axs[0].plot(t * 24, f, '.', color=c, ms=1.5, alpha=0.8,
                    label=f'{method}: {1e6 * p2p_sigma(f):.0f} ppm p2p, '
                          f'ratio {p2p_sigma(f) / np.median(e):.2f}')
        nb, s = binned_noise(f, bins)
        axs[1].loglog(nb, 1e6 * s, 'o-', color=c, ms=4)
        axs[1].loglog(nb, 1e6 * s[0] / np.sqrt(nb), ':', color=c)
    axs[0].set_xlabel('time [h]')
    axs[0].set_ylabel('relative flux')
    axs[0].legend(markerscale=5)
    axs[1].set_xlabel('bin size [integrations]')
    axs[1].set_ylabel('scatter [ppm]')
    axs[1].set_title('dotted: white noise', fontsize=9)
    return fig


@styled
def plot_regressors(spec, order=None):
    """Trace offsets, width changes, spectral shifts and background against time."""
    o = order or spec.orders[0]
    r = spec.regressors[o]
    t = spec.times - spec.times[0] if spec.times is not None else np.arange(len(spec.gidx))
    keys = [k for k in ('dy', 'ds', 'dx', 'bkg', 'maskfrac', 'white') if k in r]
    labels = {'dy': 'dy [px]', 'ds': 'width', 'dx': 'dx [px]', 'bkg': 'background',
              'maskfrac': 'masked', 'white': 'white light'}
    fig, axs = plt.subplots(len(keys), 1, figsize=(WIDTH, 1.25 * len(keys) + 0.5), sharex=True,
                            layout='constrained')
    for ax, k in zip(np.atleast_1d(axs), keys):
        ax.plot(t * 24, r[k], '-', color=INK2, lw=0.5)
        ax.set_ylabel(labels[k])
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    fig.align_ylabels(axs)
    np.atleast_1d(axs)[-1].set_xlabel('time [h]')
    return fig


@styled
def plot_spectra(spec, order=None, nshow=50):
    """Extracted spectra (a subset) and their per-column precision."""
    o = order or spec.orders[0]
    f, e = spec.flux[o], spec.err[o]
    x = spec.wavelength[o] if np.isfinite(spec.wavelength[o]).any() else spec.cols[o]
    fig, axs = plt.subplots(2, 1, figsize=(WIDTH, 5), sharex=True, layout='constrained')
    for i in np.linspace(0, len(f) - 1, min(nshow, len(f))).astype(int):
        axs[0].plot(x, f[i], '-', color=INK2, lw=0.3, alpha=0.3)
    axs[0].set_ylabel('flux')
    rel = p2p_sigma(f / np.nanmedian(f, 0), axis=0)
    axs[1].plot(x, 1e6 * rel, '-', color=INK2, lw=0.8, label='p2p scatter')
    axs[1].plot(x, 1e6 * np.nanmedian(e / np.nanmedian(f, 0), 0), '-', color=PALETTE[1], lw=0.8,
                label='errors')
    axs[1].set_yscale('log')
    axs[1].set_ylabel('per-column precision [ppm]')
    axs[1].legend()
    axs[1].set_xlabel('wavelength [µm]')
    return fig


@styled
def plot_photometry(res):
    """Aperture and PSF-weighted light curves, and the regressors."""
    t = res.times - res.times[0] if res.times is not None else np.arange(len(res.flux['psf']))
    fig, axs = plt.subplots(4, 1, figsize=(WIDTH, 7), sharex=True, layout='constrained',
                            gridspec_kw={'height_ratios': [2, 1, 1, 1]})
    for k, c in (('aperture', PALETTE[1]), ('psf', PALETTE[0])):
        f = res.flux[k] / np.median(res.flux[k])
        e = res.err[k] / np.median(res.flux[k])
        axs[0].plot(t * 24, f, '.', color=c, ms=2, alpha=0.8,
                    label=f'{k}: {1e6 * p2p_sigma(f):.0f} ppm, ratio {p2p_sigma(f) / np.median(e):.2f}')
    axs[0].legend(markerscale=4)
    axs[0].set_ylabel('relative flux')
    axs[1].plot(t * 24, res.table['xc'] - np.median(res.table['xc']), lw=0.6, label='x')
    axs[1].plot(t * 24, res.table['yc'] - np.median(res.table['yc']), lw=0.6, label='y')
    axs[1].legend(ncol=2, loc='upper right')
    axs[1].set_ylabel('centroid [px]')
    axs[2].plot(t * 24, res.table['width'], '-', color=INK2, lw=0.6)
    axs[2].set_ylabel('width [px]')
    axs[3].plot(t * 24, res.table['bkg'], '-', color=INK2, lw=0.6)
    axs[3].set_ylabel('background')
    axs[3].set_xlabel('time [h]')
    fig.align_ylabels(axs)
    return fig


@styled
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
        fig, ax = plt.subplots(figsize=(WIDTH, image_height(img, hi=3.6) + 0.7), layout='constrained')
    _imshow(ax, np.arcsinh(img / max(np.nanpercentile(np.abs(img), 90), 1e-9)), cmap='plasma')
    if ex.contam_mask is not None and ex.contam_mask.any():
        ax.contour(ex.contam_mask, levels=[0.5], colors=PALETTE[2], linewidths=0.9)
    for s in ex.contaminants:
        ax.annotate(f"S/N {s['peak']:.0f}", (s['x'], s['bbox'][1]), color=PALETTE[2], fontsize=7,
                    ha='center', va='bottom')
    for o, tr in ex.traces.items():
        ax.plot(tr.cols, tr.y0, '-', color='w', lw=0.8)
    for ft in ex.field_traces:
        g = ex._field_geometry(ft)
        if g is not None:
            ax.plot(g['x'], g['y'], '--', color=PALETTE[3], lw=0.9)
    ax.set_xlabel('spectral pixel')
    ax.set_ylabel('spatial pixel')
    ax.set_title(f'{title}: contaminant mask (aqua), traces (white), field-star traces (yellow)')
    return ax
