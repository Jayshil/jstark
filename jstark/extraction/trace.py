"""Spectral trace: static shape, usable column range and per-integration offsets.

All functions work on canonical frames ``(spatial, spectral)`` (see
:mod:`jstark.extraction.source`).

The trace of an integration ``t`` is ``y(t, x) = y0(x) + dy(t)``: a smooth
static shape measured on the exposure median frame, plus one offset per
integration (pointing jitter / drift). ``dy`` is first measured by centre of
flux and later refined by fitting the PSF model (see
:mod:`jstark.extraction.extract`).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import LSQUnivariateSpline

from ..reduction.robust import interp_nans_1d, mad_sigma, running_median

__all__ = ['Trace', 'fill_bad', 'find_trace', 'trace_from_reference', 'edge_weights', 'window', 'column_range',
           'cof_offsets', 'reference_cof']


@dataclass
class Trace:
    """Trace of one spectral order.

    Attributes
    ----------
    order : int
    cols : ndarray of int
        Spectral pixels (canonical frame, contiguous) covered by the extraction.
    y0 : ndarray
        Static spatial position of the trace in each column.
    dy : ndarray or None
        Offset of every integration of the exposure (``nint``), once measured.
    snr : ndarray or None
        Per-integration S/N of each column in the median frame.
    """
    order: int
    cols: np.ndarray
    y0: np.ndarray
    dy: np.ndarray = None
    snr: np.ndarray = None
    info: dict = field(default_factory=dict)

    @property
    def ncol(self):
        return len(self.cols)

    def y(self, gidx=None, dy=None):
        """Trace positions ``(n, ncol)`` for integrations ``gidx`` (or offsets ``dy``)."""
        if dy is None:
            dy = np.zeros(1) if self.dy is None or gidx is None else self.dy[np.asarray(gidx)]
        return self.y0[None, :] + np.asarray(dy, float)[:, None]


def fill_bad(img, good):
    """Copy of ``img`` with bad / non-finite pixels interpolated along the spatial axis."""
    a = np.where(good & np.isfinite(img), img, np.nan).astype(float)
    a = interp_nans_1d(a, axis=0)
    return np.where(np.isfinite(a), a, 0.)


def edge_weights(u, h):
    """Fraction of each pixel (centre offset ``u``) inside the window ``[-h, h]``."""
    return np.clip(h + 0.5 - np.abs(u), 0., 1.)


def window(y, nspat, h):
    """Rows of a window of half-width ``h`` around trace positions ``y``.

    Parameters
    ----------
    y : ndarray ``(n, ncol)``
    nspat : int
        Number of spatial pixels.
    h : float

    Returns
    -------
    rows : ndarray of int ``(n, nwin, ncol)``
        Row indices (clipped to the frame).
    u : ndarray ``(n, nwin, ncol)``
        Offset of each pixel centre from the trace.
    inside : ndarray of bool
        Pixel exists and ``|u| <= h + 0.5`` (it overlaps the window).
    """
    hh = int(np.ceil(h)) + 1
    nwin = 2 * hh + 1
    r0 = np.floor(y).astype(int) - hh
    rows = r0[:, None, :] + np.arange(nwin)[None, :, None]
    u = rows - y[:, None, :]
    inside = (rows >= 0) & (rows < nspat) & (np.abs(u) < h + 0.5)
    return np.clip(rows, 0, nspat - 1), u, inside


def _gather(a, rows, cols):
    """``a[n, rows, cols]`` for a chunk ``a`` ``(n, nspat, nspec)``."""
    sub = a[:, :, cols]
    if sub.shape[0] != rows.shape[0]:
        sub = np.broadcast_to(sub, (rows.shape[0],) + sub.shape[1:])
    return np.take_along_axis(sub, rows, axis=1)


def _column_background(img, y, h, gap=3):
    """Robust per-column level of the pixels more than ``h + gap`` from ``y``."""
    rows = np.arange(img.shape[0])[:, None]
    far = np.abs(rows - y[None, :]) > h + gap
    a = np.where(far, img, np.nan)
    with np.errstate(all='ignore'):
        return np.nan_to_num(np.nanmedian(a, axis=0))


def _cof(img, y, h, niter=3):
    """Iterative centre of flux of each column in a window of half-width ``h``."""
    nspat = img.shape[0]
    rows = np.arange(nspat)[:, None].astype(float)
    y = np.asarray(y, float).copy()
    for _ in range(niter):
        e = edge_weights(rows - y[None, :], h)
        wsum = np.sum(e * img, axis=0)
        ok = wsum > 0
        ynew = np.sum(e * img * rows, axis=0) / np.where(ok, wsum, 1.)
        y = np.where(ok, np.clip(ynew, 0, nspat - 1), y)
    return y


def column_range(snr, valid, snr_min=5., gap=5, min_len=20):
    """Largest contiguous run of columns with ``snr > snr_min`` and ``valid``.

    Gaps of up to ``gap`` columns (e.g. single bad columns) are bridged.
    """
    ok = (np.asarray(snr) > snr_min) & np.asarray(valid, bool)
    if gap:
        bad = ~ok
        edges = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
        for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            if e - s <= gap and s > 0 and e < len(ok):
                ok[s:e] = True
    edges = np.diff(np.concatenate([[0], ok.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    if len(starts) == 0:
        raise ValueError('No columns with enough signal for the trace.')
    k = np.argmax(ends - starts)
    if ends[k] - starts[k] < min_len:
        raise ValueError(f'Longest usable column range is only {ends[k] - starts[k]} columns.')
    return int(starts[k]), int(ends[k])


def find_trace(median, good, halfwidth=4., std=None, wavelength=None, columns=None,
               snr_min=5., min_good_frac=0.5, nknots=8, clip=3., guess=None, order=1,
               smooth_peak=15, gap=5):
    """Find the static trace on the (exposure) median frame.

    Parameters
    ----------
    median : ndarray ``(nspat, nspec)``
        Median frame (canonical orientation).
    good : ndarray of bool
        Persistent good-pixel map.
    halfwidth : float
        Half-width of the centre-of-flux window (about the PSF FWHM).
    std : ndarray or None
        Per-pixel temporal noise (for the per-integration S/N per column).
    wavelength : ndarray or None
        Wavelength map; columns without a finite wavelength at the trace are
        excluded.
    columns : (int, int) or None
        Column range ``[x0, x1)``; found automatically when ``None``.
    snr_min : float
        Minimum per-integration S/N of a column for the automatic range.
    min_good_frac : float
        Minimum fraction of good pixels in the window.
    nknots : int
        Interior knots of the cubic spline describing the trace shape.
    clip : float
        Sigma clipping of the per-column centroids.
    guess : ndarray or None
        Approximate trace position per column (e.g. from a reference trace);
        by default the brightest row of the smoothed column.

    Returns
    -------
    Trace
    """
    nspat, nspec = median.shape
    img = fill_bad(median, good)
    if guess is None:
        sm = running_median(img, smooth_peak, axis=1) if smooth_peak else img
        guess = np.argmax(sm, axis=0).astype(float)
    guess = np.asarray(guess, float)
    bkg = _column_background(img, guess, 3 * halfwidth)
    img_b = img - bkg[None, :]
    y = _cof(np.clip(img_b, 0, None), guess, halfwidth)
    rows = np.arange(nspat)[:, None]
    e = edge_weights(rows - y[None, :], halfwidth)
    flux = np.sum(e * img_b, axis=0)
    if std is not None:
        noise = np.sqrt(np.sum(e ** 2 * np.where(np.isfinite(std), std, 0.) ** 2, axis=0))
    else:
        noise = mad_sigma(img_b[np.abs(rows - y[None, :]) > 3 * halfwidth]) * \
            np.sqrt(np.sum(e ** 2, axis=0))
    snr = flux / np.where(noise > 0, noise, np.inf)
    gfrac = np.sum(e * good, axis=0) / np.maximum(np.sum(e, axis=0), 1e-9)
    valid = gfrac >= min_good_frac
    valid &= (y - halfwidth >= -0.5) & (y + halfwidth <= nspat - 0.5)
    if wavelength is not None:
        lam = _sample_spatial(wavelength, y)
        valid &= np.isfinite(lam) & (lam > 0)
    if columns is None:
        x0, x1 = column_range(snr, valid, snr_min=snr_min, gap=gap)
    else:
        x0, x1 = int(columns[0]), int(columns[1])
    cols = np.arange(x0, x1)
    # smooth shape: weighted cubic spline with sigma clipping
    yc, w = y[cols], np.clip(snr[cols], 0, None) ** 2 * valid[cols]
    xs = cols.astype(float)
    use = w > 0
    nk = max(0, min(int(nknots), use.sum() // 10))
    knots = np.linspace(xs[0], xs[-1], nk + 2)[1:-1]
    fit = yc.copy()
    for _ in range(5):
        spl = LSQUnivariateSpline(xs[use], yc[use], knots, w=np.sqrt(w[use]), k=3)
        fit = spl(xs)
        res = yc - fit
        s = mad_sigma(res[use])
        new = (w > 0) & (np.abs(res) < clip * max(s, 1e-4))
        if np.array_equal(new, use):
            break
        use = new
    return Trace(order=order, cols=cols, y0=fit, snr=snr[cols],
                 info={'centroids': yc, 'halfwidth': halfwidth, 'nknots': nk,
                       'rms': float(mad_sigma(yc[use] - fit[use])), 'columns': (x0, x1),
                       'snr_all': snr, 'valid_all': valid})


def _sample_spatial(a, y):
    """Linear interpolation of ``a[:, x]`` at spatial position ``y[x]`` for every column."""
    nspat = a.shape[0]
    yy = np.clip(y, 0, nspat - 1)
    i0 = np.clip(np.floor(yy).astype(int), 0, nspat - 2)
    f = yy - i0
    x = np.arange(a.shape[1])
    return a[i0, x] * (1 - f) + a[i0 + 1, x] * f


def reference_cof(img, trace, h):
    """Centre of flux of each trace column in a static window (reference for offsets)."""
    y = trace.y0
    rows = np.arange(img.shape[0])[:, None]
    sub = img[:, trace.cols]
    e = edge_weights(rows - y[None, :], h)
    w = np.sum(e * sub, axis=0)
    return np.sum(e * sub * rows, axis=0) / np.where(w != 0, w, np.nan), w


def cof_offsets(data, trace, h, ref_cof, weights):
    """Per-integration trace offsets from the centre of flux in static windows.

    ``dy(t)`` is the weighted, clipped mean over columns of
    ``cof(t, x) - ref_cof(x)``. Windows are fixed (not re-centred), so the
    result is a slightly damped but very low-noise measure of the motion;
    it is refined later with the PSF model.
    """
    rows = np.arange(data.shape[1])[None, :, None]
    sub = data[:, :, trace.cols]
    e = edge_weights(rows - trace.y0[None, None, :], h)
    s = np.sum(e * sub, axis=1)
    c = np.sum(e * sub * rows, axis=1) / np.where(s != 0, s, np.nan)
    d = c - ref_cof[None, :]
    w = np.where(np.isfinite(d), weights[None, :], 0.)
    d = np.where(np.isfinite(d), d, 0.)
    out = np.zeros(len(data))
    for i in range(len(data)):
        wi, di = w[i], d[i]
        keep = wi > 0
        for _ in range(3):
            m = np.sum(wi[keep] * di[keep]) / max(np.sum(wi[keep]), 1e-300)
            s_ = mad_sigma(di[keep] - m)
            new = (wi > 0) & (np.abs(di - m) < 5 * max(s_, 1e-6))
            if np.array_equal(new, keep):
                break
            keep = new
        out[i] = m
    return out


def trace_from_reference(median, good, yref, others=(), halfwidth=10., std=None, wavelength=None,
                         columns=None, snr_min=5., min_good_frac=0.5, ref_order=1, clip=3.,
                         order=1, gap=5, min_sep=None):
    """Trace from a reference shape (e.g. NIRISS/SOSS PASTASOSS traces) plus a fitted correction.

    The reference ``yref(x)`` is shifted to the data: centres of flux are
    measured on the median frame in *clean* columns (where every other order
    is at least ``min_sep`` rows away) and a polynomial of degree
    ``ref_order`` in ``x`` is fitted to ``cof - yref``. Where orders overlap
    the shape comes from the reference, so their light does not bias the
    trace.

    Parameters
    ----------
    yref : ndarray ``(nspec,)``
        Reference position in every column (NaN where undefined).
    others : list of ndarray
        Reference positions of the other orders on the detector.

    Returns
    -------
    Trace
        ``info['clean']`` flags the clean columns.
    """
    nspat, nspec = median.shape
    img = fill_bad(median, good)
    x = np.arange(nspec, dtype=float)
    yref = np.asarray(yref, float)
    min_sep = 2 * halfwidth + 4 if min_sep is None else min_sep
    sep = np.full(nspec, np.inf)
    for yo in others:
        yo = np.asarray(yo, float)
        sep = np.minimum(sep, np.where(np.isfinite(yo), np.abs(yref - yo), np.inf))
    inframe = np.isfinite(yref) & (yref - halfwidth >= -0.5) & (yref + halfwidth <= nspat - 0.5)
    clean = inframe & (sep > min_sep)
    yr = np.where(np.isfinite(yref), yref, 0.)
    bkg = _column_background(img, yr, 3 * halfwidth)
    imgb = np.clip(img - bkg[None, :], 0, None)
    rows = np.arange(nspat)[:, None]
    corr = np.zeros(nspec)
    for _ in range(4):
        cof = _cof(imgb, yr + corr, halfwidth)
        e = edge_weights(rows - (yr + corr)[None, :], halfwidth)
        flux = np.sum(e * imgb, axis=0)
        w = np.where(clean & np.isfinite(cof) & (flux > 0), flux, 0.)
        d = cof - yr
        use = w > 0
        if use.sum() < ref_order + 3:
            raise ValueError(f'Order {order}: too few clean columns to place the reference trace.')
        for _ in range(5):
            c = np.polyfit(x[use], d[use], ref_order, w=np.sqrt(w[use]))
            res = d - np.polyval(c, x)
            s = mad_sigma(res[use])
            new = (w > 0) & (np.abs(res) < clip * max(s, 1e-3))
            if np.array_equal(new, use):
                break
            use = new
        corr = np.polyval(c, x)
    y = yr + corr
    e = edge_weights(rows - y[None, :], halfwidth)
    flux = np.sum(e * (img - bkg[None, :]), axis=0)
    if std is not None:
        noise = np.sqrt(np.sum(e ** 2 * np.where(np.isfinite(std), std, 0.) ** 2, axis=0))
    else:
        noise = mad_sigma(img[np.abs(rows - y[None, :]) > 3 * halfwidth]) * np.sqrt(np.sum(e ** 2, 0))
    snr = flux / np.where(noise > 0, noise, np.inf)
    gfrac = np.sum(e * good, axis=0) / np.maximum(np.sum(e, axis=0), 1e-9)
    valid = inframe & (gfrac >= min_good_frac)
    if wavelength is not None:
        lam = _sample_spatial(wavelength, np.clip(y, 0, nspat - 1))
        valid &= np.isfinite(lam) & (lam > 0)
    if columns is None:
        x0, x1 = column_range(snr, valid, snr_min=snr_min, gap=gap)
    else:
        x0, x1 = int(columns[0]), int(columns[1])
    cols = np.arange(x0, x1)
    return Trace(order=order, cols=cols, y0=y[cols], snr=snr[cols],
                 info={'halfwidth': halfwidth, 'clean': clean[cols], 'correction': c.tolist(),
                       'rms': float(mad_sigma(res[use])), 'columns': (x0, x1),
                       'centroids': cof[cols], 'snr_all': snr, 'valid_all': valid,
                       'reference': yref[cols]})
