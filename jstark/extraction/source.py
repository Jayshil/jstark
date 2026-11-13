"""Segment-by-segment access to the cleaned products of an exposure.

Everything in :mod:`jstark.extraction` works in a common *canonical*
orientation: a frame is ``(spatial, spectral)``, i.e. the dispersion runs
along the last axis. NIRSpec, NIRISS/SOSS and NIRCam grism data are already
in this orientation; MIRI/LRS (dispersion along detector y) is transposed on
read. Imaging data keep their orientation.

Data are never stitched: :meth:`ExposureSource.read` returns any range of
integrations of one segment (via ``astropy.io.fits`` sections, so only those
integrations are read from disk), and :meth:`ExposureSource.chunks` walks
through the exposure in chunks whose size is set by a memory budget.
"""
import os

import numpy as np
from astropy.io import fits

from ..reduction.cleaning import ExposureCleaner
from ..reduction.instruments import get_mode_config

__all__ = ['ExposureSource']


class ExposureSource:
    """Read-only access to an exposure cleaned by :class:`~jstark.reduction.ExposureCleaner`.

    Parameters
    ----------
    info : str or ExposureCleaner
        The ``<exposure>_cleaninfo.fits`` file (or the loaded object).
    max_mem_gb : float
        Memory budget used to size the chunks of integrations.

    Attributes
    ----------
    mode : str
    transpose : bool
        ``True`` when frames are transposed into the canonical orientation.
    seg_nint, seg_start : list / ndarray
        Integrations per segment and index of the first integration of each
        segment in the exposure.
    badpix, median, std, cr_sigma, wavelength : ndarray
        Exposure-level maps in canonical orientation (``wavelength`` may be
        ``(norder, nspat, nspec)`` for SOSS).
    times : ndarray
        Mid-integration times (BJD_TDB - 2400000.5) of all integrations.
    """

    def __init__(self, info, max_mem_gb=2.):
        ec = info if isinstance(info, ExposureCleaner) else ExposureCleaner.load(info, verbose=False)
        self.ec = ec
        self.info_file = getattr(ec, 'info_file', None)
        self.mode = ec.mode
        self.config = get_mode_config(self.mode)
        self.exposure = ec.exposure
        self.header = dict(ec.header)
        self.dispersion_axis = ec.dispersion_axis
        self.transpose = self.dispersion_axis == 0
        self.outdir = ec.outdir
        self.files = [ec.outputs[i] for i in sorted(ec.outputs)]
        self.segments = sorted(ec.outputs)
        self.seg_nint = [int(ec.seg_nint[i]) for i in self.segments]
        self.seg_start = np.concatenate([[0], np.cumsum(self.seg_nint)[:-1]]).astype(int)
        self.input_names = [os.path.basename(ec.files[i]) for i in self.segments]
        self.max_mem_gb = max_mem_gb
        c = self.canon
        self.badpix = c(ec.badpix).astype(bool) if ec.badpix is not None else None
        self.median = c(ec.median)
        self.std = c(ec.std)
        self.cr_sigma = c(ec.cr_sigma)
        self.trace_mask = c(ec.trace_mask)
        self.bkg_mask = c(ec.bkg_mask)
        self.wavelength = c(ec.wavelength)
        self.int_times = dict(getattr(ec, 'int_times', {}) or {})
        if 'int_mid_BJD_TDB' in self.int_times and len(self.int_times['int_mid_BJD_TDB']) == self.nint:
            self.times = np.asarray(self.int_times['int_mid_BJD_TDB'], float)
        else:
            self.times = None
        with fits.open(self.files[0]) as h:
            self.frame_shape = tuple(h['SCI'].shape[1:])
        self.shape = self.canon(np.empty(self.frame_shape)).shape   # canonical (nspat, nspec)
        if self.badpix is None:
            self.badpix = np.ones(self.shape, bool)

    # ------------------------------------------------------------------
    @property
    def nint(self):
        return int(sum(self.seg_nint))

    @property
    def nseg(self):
        return len(self.files)

    def canon(self, a):
        """Detector -> canonical orientation (last two axes)."""
        if a is None:
            return None
        a = np.asarray(a)
        return np.swapaxes(a, -1, -2) if self.transpose else a

    def decanon(self, a):
        """Canonical -> detector orientation (last two axes)."""
        return self.canon(a)

    def segment_of(self, gidx):
        """Segment number (position in :attr:`files`) of global integration ``gidx``."""
        return int(np.searchsorted(self.seg_start, gidx, side='right') - 1)

    # ------------------------------------------------------------------
    def read(self, k, ints=None):
        """Read integrations ``ints`` (slice) of segment ``k`` (0-based position).

        Returns
        -------
        data, err : ndarray (float32), ``(n, nspat, nspec)``
        good : ndarray of bool
            ``True`` for usable pixels (not persistently bad, not flagged as
            transient / cosmic ray).
        gidx : ndarray of int
            Global (exposure) integration numbers, 0-based.
        """
        ints = slice(0, self.seg_nint[k]) if ints is None else ints
        start, stop, _ = ints.indices(self.seg_nint[k])
        with fits.open(self.files[k]) as h:
            data = np.array(h['SCI'].section[start:stop], dtype=np.float32)
            err = np.array(h['ERR'].section[start:stop], dtype=np.float32)
            cr = np.array(h['CRMASK'].section[start:stop]).astype(bool)
        good = ~cr & self.decanon(self.badpix)[None] & np.isfinite(data) & np.isfinite(err) & (err > 0)
        gidx = self.seg_start[k] + np.arange(start, stop)
        return self.canon(data), self.canon(err), self.canon(good), gidx

    def read_ints(self, gidx):
        """Read an arbitrary sorted list of global integrations (grouped by segment)."""
        gidx = np.sort(np.asarray(gidx, int))
        parts = []
        for k in range(self.nseg):
            sel = gidx[(gidx >= self.seg_start[k]) & (gidx < self.seg_start[k] + self.seg_nint[k])]
            if len(sel) == 0:
                continue
            loc = sel - self.seg_start[k]
            if len(loc) > 0.5 * (loc.max() - loc.min() + 1):
                # dense: read the covering range once and subsample
                d, e, g, gi = self.read(k, slice(loc.min(), loc.max() + 1))
                pick = loc - loc.min()
                parts.append((d[pick], e[pick], g[pick], gi[pick]))
            else:
                # sparse: one integration at a time (bounded memory)
                for j in loc:
                    parts.append(self.read(k, slice(j, j + 1)))
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))

    def chunk_size(self, factor=12):
        """Integrations per chunk for the memory budget (``factor`` arrays per pixel)."""
        per_int = np.prod(self.shape) * 4 * factor
        return max(1, int(self.max_mem_gb * 1024 ** 3 / per_int))

    def chunks(self, segments=None, chunk=None):
        """Yield ``(k, data, err, good, gidx)`` chunks, one segment at a time."""
        chunk = chunk or self.chunk_size()
        for k in (range(self.nseg) if segments is None else segments):
            n = self.seg_nint[k]
            for a in range(0, n, chunk):
                d, e, g, gi = self.read(k, slice(a, min(n, a + chunk)))
                yield k, d, e, g, gi

    def sample_ints(self, n):
        """``n`` integrations spread evenly over the exposure (global indices)."""
        n = min(int(n), self.nint)
        return np.unique(np.linspace(0, self.nint - 1, n).round().astype(int))
