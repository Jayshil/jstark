"""Light curves prepared for fitting.

One :class:`LightCurves` object holds one *dataset*: one visit observed with
one instrument mode, order or detector (e.g. NIRSpec PRISM visit 1, NRS2 of a
G395H visit, or one MIRI F1500W eclipse). It contains

* the times (BJD_TDB), and a flux and an error per integration and channel
  (spectroscopic channels, or a single channel for photometry);
* a pool of named per-integration time series (*regressors*: trace offsets,
  centroids, PLD components, ...). The data only stores them; a fit decides
  which ones enter the linear model and which one is the GP input
  (see :class:`jstark.fitting.BroadbandFit`).

Several datasets are combined in a fit through a dict ``{name: LightCurves}``;
the names become juliet instrument names.
"""
import re
import warnings

import numpy as np

from ..reduction.robust import MAD_TO_SIGMA, running_median

__all__ = ['LightCurves', 'as_lightcurves', 'check_name', 'MIRI_PIVOT']

# Pivot wavelengths (micron) of the MIRI imaging filters
MIRI_PIVOT = {'F560W': 5.635, 'F770W': 7.639, 'F1000W': 9.953, 'F1130W': 11.309,
              'F1280W': 12.810, 'F1500W': 15.064, 'F1800W': 17.984, 'F2100W': 20.795,
              'F2550W': 25.365}

# Words that occur as parts of juliet parameter names. juliet splits names on
# '_' and matches instruments against the parts, so a dataset may not be
# called like any of them.
_RESERVED = {'p', 'P', 't0', 'a', 'b', 'rho', 'inc', 'ecc', 'e', 'omega', 'w', 'sigma',
             'q1', 'q2', 'q3', 'q4', 'u1', 'u2', 'c1', 'c2', 'c3', 'c4', 'fp', 'C1', 'C2',
             'D1', 'D2', 'L', 'mflux', 'mdilution', 'GP', 'lc', 'rv', 'phaseoffset', 'dt', 'T',
             'secosomega', 'sesinomega', 'esinomega', 'ecosomega', 'r1', 'r2', 'K', 'mu'}

# Regressors computed on request (not stored)
BUILTIN_REGRESSORS = ('time', 'time2', 'time3', 'phase4')


def check_name(name):
    """Raise ``ValueError`` unless ``name`` is usable as a juliet instrument name.

    juliet splits parameter names on ``_`` (and per-instrument limb-darkening
    laws on ``-``), so names must be letters and digits only, start with a
    letter, and must not look like a parameter or planet name (``p1``,
    ``q1``, ``theta0``, ...).
    """
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9]*', name):
        raise ValueError(f'Dataset name {name!r}: use letters and digits only, starting with a '
                         'letter (juliet splits names on "_" and "-").')
    if name in _RESERVED or re.fullmatch(r'(p|theta)\d+', name):
        raise ValueError(f'Dataset name {name!r} clashes with a juliet parameter name; '
                         'choose another (e.g. "V1", "NRS1", "PRISM").')
    return name


class LightCurves:
    """Time series of one dataset, with optional regressors.

    Parameters
    ----------
    times : array ``(nint,)``
        Mid-integration times, BJD_TDB (full Julian date).
    flux, err : array ``(nint,)`` or ``(nint, nchan)``
        Flux and 1-sigma error per integration (and channel). Any units;
        light curves are normalised by their median when they are fitted.
    wavelength : float, array ``(nchan,)`` or None
        Wavelength of each channel (micron).
    wave_edges : array ``(nchan, 2)`` or None
        Lower and upper wavelength edge of each channel.
    regressors : dict or None
        ``{name: array}``; arrays of shape ``(nint,)`` or ``(nint, k)``
        (e.g. ``k`` PLD components). Use :meth:`add_regressor` with
        ``per_channel=True`` for regressors that differ between channels.
    index : array ``(nint,)`` or None
        Integration number within the exposure (``gidx`` of the spectra);
        used by :meth:`exclude` and the ``phase4`` regressor. Default
        ``0 .. nint-1``.
    name : str or None
        Default dataset name.
    meta : dict or None
        Free-form information (mode, file, ...).
    groups : array ``(nchan,)`` of int or None
        A group label (>= 0) per channel, e.g. the readout amplifier of each
        column (read from NIRCam spectra). Used by :meth:`split` and for
        per-group common modes; -1 means no group.
    """

    def __init__(self, times, flux, err, wavelength=None, wave_edges=None, regressors=None,
                 index=None, name=None, meta=None, groups=None):
        t = np.asarray(times, float).ravel()
        f = np.asarray(flux, float)
        e = np.asarray(err, float)
        if f.ndim == 1:
            f, e = f[:, None], e[:, None]
        if f.ndim != 2 or f.shape != e.shape or f.shape[0] != t.size:
            raise ValueError(f'times {t.shape}, flux {np.shape(flux)} and err {np.shape(err)} '
                             'do not match (expected (nint,) and (nint,) or (nint, nchan)).')
        self.times, self.flux, self.err = t, f, e
        nch = f.shape[1]
        if wavelength is None:
            wl = np.full(nch, np.nan)
        else:
            wl = np.atleast_1d(np.asarray(wavelength, float))
            if wl.size == 1 and nch > 1:
                raise ValueError('One wavelength given for several channels.')
            if wl.size != nch:
                raise ValueError(f'{wl.size} wavelengths for {nch} channels.')
        self.wavelength = wl
        if wave_edges is None:
            wave_edges = _edges_from_centres(wl)
        self.wave_edges = np.asarray(wave_edges, float).reshape(nch, 2)
        self.index = np.arange(t.size) if index is None else np.asarray(index).ravel()
        if self.index.size != t.size:
            raise ValueError('index must have one entry per integration.')
        self.name = name
        self.meta = dict(meta or {})
        if groups is not None:
            groups = np.asarray(groups, int).ravel()
            if groups.size != nch:
                raise ValueError(f'{groups.size} group labels for {nch} channels.')
        self.groups = groups
        self.regressors, self.channel_regressors = {}, {}
        for k, v in (regressors or {}).items():
            self.add_regressor(k, v)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_spectra(cls, fname, order=None, method='optimal', name=None):
        """Read a jstark ``<exposure>_spectra.fits`` file (one order).

        ``method='optimal'`` uses the main extraction (``FLUX_O<n>``),
        ``'box'`` the box extraction. All per-integration regressors of the
        extraction (``dy``, ``ds``, ``dx``, ``chi2``, ...) are kept. For
        NIRCam the readout amplifier of each column becomes :attr:`groups`.
        """
        from ..extraction.extractor import Spectra
        s = Spectra(fname)
        o = order or s.orders[0]
        if o not in s.orders:
            raise ValueError(f'Order {o} not in {fname} (orders {s.orders}).')
        if method == 'optimal':
            f, e = s.flux[o], s.err[o]
        elif method == 'box':
            if o not in s.box:
                raise ValueError('No box extraction in this file.')
            f, e = s.box[o], s.box_err[o]
        else:
            raise ValueError("method must be 'optimal' or 'box'.")
        if s.bjd is None:
            raise ValueError(f'{fname} has no integration times.')
        regs = {k: np.asarray(v, float) for k, v in s.regressors[o].items()}
        meta = dict(mode=s.mode, order=o, method=method, file=fname, cols=s.cols[o],
                    satgrp=s.satgrp.get(o), ngroups=s.ngroups, contam=s.contam.get(o))
        return cls(s.bjd, f, e, wavelength=s.wavelength[o], regressors=regs, index=s.gidx,
                   name=name, meta=meta, groups=s.amp.get(o))

    @classmethod
    def from_photometry(cls, fname, method='psf', pld=None, wavelength=None, name=None):
        """Read a jstark ``<exposure>_photometry.fits`` file.

        Parameters
        ----------
        method : {'psf', 'aperture'}
        pld : int, True or None
            ``int``: add the first ``pld`` principal components of the
            normalised pixel fractions as the multi-column regressor
            ``'pld'``; ``True``: add the normalised pixel fractions
            themselves; ``None``: no PLD regressor.
        wavelength : float or None
            Filter wavelength (micron); default the MIRI pivot wavelength
            when the filter is known.
        """
        from astropy.io import fits
        from ..extraction.photometry import PhotometryResult
        r = PhotometryResult(fname)
        if method not in r.flux:
            raise ValueError(f'method must be one of {list(r.flux)}.')
        if r.times is None:
            raise ValueError(f'{fname} has no integration times.')
        hdr = fits.getheader(fname, 0)
        mode = hdr.get('JSMODE', '')
        if wavelength is None:
            filt = mode.split('/')[-1] if '/' in mode else None
            wavelength = MIRI_PIVOT.get(filt, np.nan)
        regs = {k: np.asarray(v, float) for k, v in r.regressors().items()}
        if 'maskfrac' in r.table:
            regs['maskfrac'] = np.asarray(r.table['maskfrac'], float)
        if pld:
            if pld is True:
                regs['pld'] = r.pld_basis()
            else:
                _, pcs = r.pld_basis(npc=int(pld))
                regs['pld'] = np.asarray(pcs).T
        index = r.table.get('gidx')
        meta = dict(mode=mode, method=method, file=fname, exposure=hdr.get('EXPOSURE'))
        return cls(r.times + 2400000.5, r.flux[method], r.err[method], wavelength=wavelength,
                   regressors=regs, index=index, name=name, meta=meta)

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def nint(self):
        return self.times.size

    @property
    def nchan(self):
        return self.flux.shape[1]

    @property
    def photometric(self):
        """True for a single-channel dataset (photometry or one bin)."""
        return self.nchan == 1

    def __repr__(self):
        wl = self.wavelength
        w = '' if not np.isfinite(wl).any() else f', {np.nanmin(wl):.3g}-{np.nanmax(wl):.3g} um'
        return (f'LightCurves({self.name or "unnamed"}: {self.nint} integrations, '
                f'{self.nchan} channel(s){w}; regressors {self.regressor_names()})')

    def regressor_names(self):
        return list(self.regressors) + list(self.channel_regressors)

    # ------------------------------------------------------------------
    # Regressors
    # ------------------------------------------------------------------
    def add_regressor(self, name, values, per_channel=False):
        """Store a regressor.

        ``values`` has shape ``(nint,)`` or ``(nint, k)``; with
        ``per_channel=True`` it has shape ``(nint, nchan)`` and each channel
        uses its own column.
        """
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', str(name)):
            raise ValueError(f'Regressor name {name!r}: letters, digits and "_" only.')
        if name in BUILTIN_REGRESSORS:
            raise ValueError(f'{name!r} is a built-in regressor name.')
        v = np.asarray(values, float)
        if v.shape[0] != self.nint:
            raise ValueError(f'Regressor {name!r} has {v.shape[0]} rows for {self.nint} integrations.')
        if per_channel:
            if v.shape != (self.nint, self.nchan):
                raise ValueError(f'A per-channel regressor must have shape {(self.nint, self.nchan)}.')
            self.channel_regressors[name] = v
            self.regressors.pop(name, None)
        else:
            if v.ndim > 2:
                raise ValueError('Regressors must be 1D or 2D.')
            self.regressors[name] = v
            self.channel_regressors.pop(name, None)

    def regressor(self, name, channel=None):
        """Values of one regressor as ``(nint, k)`` plus column labels.

        Built-in regressors: ``time`` (days from the mid-time of the
        dataset), ``time2``, ``time3`` (its powers) and ``phase4`` (three
        indicator columns for integrations 1, 2, 3 mod 4; the NIRCam fast
        frame-reset pattern).
        """
        if name in ('time', 'time2', 'time3'):
            x = self.times - 0.5 * (self.times.min() + self.times.max())
            p = {'time': 1, 'time2': 2, 'time3': 3}[name]
            return (x ** p)[:, None], [name]
        if name == 'phase4':
            ph = np.asarray(self.index) % 4
            return np.stack([(ph == k).astype(float) for k in (1, 2, 3)], axis=1), \
                ['phase4a', 'phase4b', 'phase4c']
        if name in self.channel_regressors:
            if channel is None and self.nchan == 1:
                channel = 0
            if channel is None:
                raise ValueError(f'{name!r} is a per-channel regressor; give a channel.')
            return self.channel_regressors[name][:, channel][:, None], [name]
        if name in self.regressors:
            v = self.regressors[name]
            if v.ndim == 1:
                return v[:, None], [name]
            return v, [f'{name}{k}' for k in range(v.shape[1])]
        raise KeyError(f'Unknown regressor {name!r}; available: '
                       f'{list(BUILTIN_REGRESSORS) + self.regressor_names()}')

    def design_matrix(self, names, channel=None, standardise=True):
        """Linear-model matrix of the named regressors.

        Returns ``X`` ``(nint, ncol)``, the column labels and the
        ``(mean, std)`` of every column (``X`` is ``(x - mean) / std`` when
        ``standardise``; then the fitted coefficients are the flux change for
        a one-sigma change of the regressor).
        """
        cols, labels = [], []
        for n in names:
            v, lab = self.regressor(n, channel=channel)
            cols.append(v)
            labels += lab
        if not cols:
            return np.zeros((self.nint, 0)), [], np.zeros((0, 2))
        X = np.concatenate(cols, axis=1).astype(float)
        mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
        bad = ~(sd > 0)
        if np.any(bad):
            raise ValueError(f'Regressor column(s) {[l for l, b in zip(labels, bad) if b]} are '
                             'constant (degenerate with the flux offset).')
        if standardise:
            X = (X - mu) / sd
        else:
            mu, sd = np.zeros_like(mu), np.ones_like(sd)
        return X, labels, np.stack([mu, sd], axis=1)

    # ------------------------------------------------------------------
    # Light curves
    # ------------------------------------------------------------------
    def good_channels(self, max_nan_frac=0.5):
        """Channels that are finite in at least ``1 - max_nan_frac`` of the integrations."""
        ok = np.isfinite(self.flux) & np.isfinite(self.err)
        return ok.mean(axis=0) >= 1. - max_nan_frac

    def white_light(self, normalise=True, channels=None, max_nan_frac=0.5):
        """Sum of the channels (the white-light curve) and its error.

        Channels that are NaN in more than ``max_nan_frac`` of the
        integrations are left out (e.g. saturated or contaminated columns);
        an integration in which any remaining channel is NaN is NaN. For a
        single-channel dataset this is the light curve itself.

        Returns
        -------
        flux, err : ndarray ``(nint,)``
        """
        good = self.good_channels(max_nan_frac)
        if channels is not None:
            sel = np.zeros(self.nchan, bool)
            sel[np.atleast_1d(channels)] = True
            good &= sel
        if not good.any():
            raise ValueError('No usable channel for the white light.')
        f = np.sum(self.flux[:, good], axis=1)
        e = np.sqrt(np.sum(self.err[:, good] ** 2, axis=1))
        if normalise:
            m = np.nanmedian(f)
            f, e = f / m, e / m
        return f, e

    def select_channels(self, wave_range=None, channels=None):
        """New object with a subset of the channels.

        ``wave_range=(lo, hi)`` keeps channels with ``lo <= wavelength < hi``
        (micron); ``channels`` gives indices or a boolean mask. Both may be
        combined. Per-channel regressors are subset too.
        """
        keep = np.ones(self.nchan, bool)
        if wave_range is not None:
            lo, hi = wave_range
            keep &= (self.wavelength >= lo) & (self.wavelength < hi)
        if channels is not None:
            sel = np.zeros(self.nchan, bool)
            sel[np.atleast_1d(channels)] = True
            keep &= sel
        if not keep.any():
            raise ValueError('No channel selected.')
        lc = LightCurves(self.times, self.flux[:, keep], self.err[:, keep],
                         wavelength=self.wavelength[keep], wave_edges=self.wave_edges[keep],
                         regressors=dict(self.regressors), index=self.index, name=self.name,
                         meta=dict(self.meta, channels=np.flatnonzero(keep).tolist()),
                         groups=None if self.groups is None else self.groups[keep])
        for k, v in self.channel_regressors.items():
            lc.add_regressor(k, v[:, keep], per_channel=True)
        return lc

    def bin(self, nbins=None, edges=None, max_nan_frac=0.5):
        """Spectroscopic light curves summed in wavelength bins.

        Parameters
        ----------
        nbins : int or None
            Number of bins with (nearly) the same number of channels.
        edges : sequence or None
            Bin edges in wavelength (micron); overrides ``nbins``.

        Returns
        -------
        LightCurves
            One channel per bin (empty bins are dropped). Bin wavelengths are
            the mean wavelength of the summed channels; ``wave_edges`` are
            the given edges (or the outer channel edges with ``nbins``).
            Per-channel regressors become the mean over the bin's channels,
            weighted by their median flux (so a per-channel common mode
            becomes that of the summed light curve). A bin's group is the
            group of its channels, or -1 if they belong to several.
        """
        good = self.good_channels(max_nan_frac)
        if edges is not None:
            edges = np.asarray(edges, float)
            idx = np.digitize(self.wavelength, edges) - 1
            nb = len(edges) - 1
            idx[(self.wavelength < edges[0]) | (self.wavelength >= edges[-1])] = -1
        else:
            nb = int(nbins or 20)
            gidx = np.flatnonzero(good)
            idx = np.full(self.nchan, -1)
            idx[gidx] = (np.arange(gidx.size) * nb) // gidx.size
        f, e, wl, we, keep = [], [], [], [], []
        creg = {k: [] for k in self.channel_regressors}
        grp = []
        for b in range(nb):
            sel = (idx == b) & good
            if not sel.any():
                continue
            f.append(np.sum(self.flux[:, sel], axis=1))
            e.append(np.sqrt(np.sum(self.err[:, sel] ** 2, axis=1)))
            wl.append(np.nanmean(self.wavelength[sel]))
            if edges is not None:
                we.append((edges[b], edges[b + 1]))
            else:
                we.append((np.nanmin(self.wave_edges[sel, 0]), np.nanmax(self.wave_edges[sel, 1])))
            keep.append(np.flatnonzero(sel))
            if creg:
                wt = np.nan_to_num(np.nanmedian(self.flux[:, sel], axis=0))
                wt = wt / wt.sum() if wt.sum() > 0 else np.full(wt.size, 1. / wt.size)
                for k, v in self.channel_regressors.items():
                    creg[k].append(v[:, sel] @ wt)
            if self.groups is not None:
                g = np.unique(self.groups[sel])
                grp.append(int(g[0]) if g.size == 1 else -1)
        if not f:
            raise ValueError('All bins are empty.')
        meta = dict(self.meta, binned_from=[k.tolist() for k in keep])
        lc = LightCurves(self.times, np.stack(f, 1), np.stack(e, 1), wavelength=wl,
                         wave_edges=we, regressors=dict(self.regressors), index=self.index,
                         name=self.name, meta=meta, groups=grp if self.groups is not None else None)
        for k, v in creg.items():
            lc.add_regressor(k, np.stack(v, 1), per_channel=True)
        return lc

    # ------------------------------------------------------------------
    # Selecting integrations
    # ------------------------------------------------------------------
    def subset(self, keep):
        """New object with the integrations where ``keep`` is True (or with indices ``keep``)."""
        keep = np.asarray(keep)
        if keep.dtype != bool:
            m = np.zeros(self.nint, bool)
            m[keep] = True
            keep = m
        lc = LightCurves(self.times[keep], self.flux[keep], self.err[keep],
                         wavelength=self.wavelength, wave_edges=self.wave_edges,
                         regressors={k: v[keep] for k, v in self.regressors.items()},
                         index=self.index[keep], name=self.name, meta=dict(self.meta),
                         groups=self.groups)
        for k, v in self.channel_regressors.items():
            lc.add_regressor(k, v[keep], per_channel=True)
        return lc

    def match_series(self, times, values, tol=None):
        """Values of a time series at this dataset's integrations (NaN where it has none).

        ``times`` are matched to :attr:`times` within ``tol`` (default a
        tenth of the median time step), so a series from a fit that dropped
        integrations (trimmed, clipped, excluded) lines up correctly.
        """
        ts = np.asarray(times, float)
        v = np.asarray(values, float)
        if ts.shape[0] != v.shape[0]:
            raise ValueError('times and values do not match.')
        if tol is None:
            dt = np.diff(np.sort(self.times))
            dt = dt[dt > 0]
            tol = 0.1 * (np.median(dt) if dt.size else 1e-6)
        o = np.argsort(ts)
        ts, v = ts[o], v[o]
        out = np.full((self.nint,) + v.shape[1:], np.nan)
        if not ts.size:
            return out
        j = np.clip(np.searchsorted(ts, self.times), 1, ts.size - 1) if ts.size > 1 \
            else np.zeros(self.nint, int)
        if ts.size > 1:
            j = np.where(np.abs(ts[j - 1] - self.times) < np.abs(ts[j] - self.times), j - 1, j)
        ok = np.abs(ts[j] - self.times) <= tol
        out[ok] = v[j[ok]]
        return out

    def apply_common_mode(self, times, values, mode='divide', name='cm'):
        """New object corrected by (or regressing on) a common-mode time series.

        Parameters
        ----------
        times, values : array
            The common mode (e.g. from :meth:`BroadbandFit.common_mode`),
            normalised to a median of 1; matched by time
            (:meth:`match_series`). ``values`` is ``(n,)`` (one for all
            channels) or ``(n, nchan)`` (one per channel, e.g. from
            :meth:`BroadbandFit.channel_common_mode`).
        mode : {'divide', 'regressor'}
            ``'divide'``: every channel (flux and error) is divided by it;
            ``'regressor'``: it is stored as the regressor ``name`` (list it
            in ``linear`` to use it; per channel if ``values`` is 2D).
            Integrations without a value are NaN (divide) or have a NaN
            regressor; fits drop them either way.
        """
        cm = self.match_series(times, values)
        if cm.ndim == 2 and cm.shape[1] != self.nchan:
            raise ValueError(f'The common mode has {cm.shape[1]} channels, the data {self.nchan}.')
        cm2 = cm[:, None] if cm.ndim == 1 else cm
        if mode == 'divide':
            lc = LightCurves(self.times, self.flux / cm2, self.err / cm2,
                             wavelength=self.wavelength, wave_edges=self.wave_edges,
                             regressors=dict(self.regressors), index=self.index, name=self.name,
                             meta=dict(self.meta), groups=self.groups)
            for k, v in self.channel_regressors.items():
                lc.add_regressor(k, v, per_channel=True)
        elif mode == 'regressor':
            lc = self.subset(np.ones(self.nint, bool))
            lc.add_regressor(name, cm, per_channel=cm.ndim == 2)
        else:
            raise ValueError("mode must be 'divide' or 'regressor'.")
        lc.meta['common_mode'] = dict(mode=mode,
                                      n_missing=int(np.sum(~np.all(np.isfinite(cm2), axis=1))))
        return lc

    def split(self, groups=None, prefix='A'):
        """One dataset per channel group (e.g. per readout amplifier).

        ``groups``: a label per channel (default :attr:`groups`). Returns
        ``{f'{prefix}{g + 1}': LightCurves}`` for every label ``g >= 0``
        (``'A1'`` ... ``'A4'`` for the four NIRCam amplifiers), e.g. to fit
        the white light of each amplifier for
        :meth:`BroadbandFit.channel_common_mode`.
        """
        g = self.groups if groups is None else np.asarray(groups, int).ravel()
        if g is None:
            raise ValueError('This dataset has no channel groups; give groups=.')
        if g.size != self.nchan:
            raise ValueError(f'{g.size} group labels for {self.nchan} channels.')
        out = {}
        for k in np.unique(g[g >= 0]):
            name = check_name(f'{prefix}{k + 1}')
            lc = self.select_channels(channels=np.flatnonzero(g == k))
            lc.name = name
            out[name] = lc
        return out

    def exclude(self, ints=None, time_range=None):
        """Remove integrations.

        Parameters
        ----------
        ints : list of (start, stop) or None
            Ranges of integration numbers (``index``; ``stop`` exclusive),
            e.g. ``[(0, 3000), (20750, 20766)]`` to trim the start and drop
            a high-gain-antenna move.
        time_range : list of (t1, t2) or None
            Time ranges (same units as ``times``) to remove.
        """
        drop = np.zeros(self.nint, bool)
        for a, b in (ints or []):
            drop |= (self.index >= a) & (self.index < b)
        for a, b in (time_range or []):
            drop |= (self.times >= a) & (self.times <= b)
        lc = self.subset(~drop)
        lc.meta.setdefault('excluded', []).append(dict(ints=ints, time_range=time_range,
                                                       n=int(drop.sum())))
        return lc

    @classmethod
    def concatenate(cls, lcs, name=None, common_channels=False):
        """Join consecutive exposures of one visit into one dataset.

        For a time series split into several exposures (e.g. the WASP-43 b
        MIRI phase curve, three exposures), which must share one baseline:
        fitting them as separate datasets would give each its own flux
        offset, which trades off against a phase curve.

        Parameters
        ----------
        lcs : list of LightCurves
            In time order, with the same channels (wavelengths).
        name : str or None
            Dataset name (default: that of the first).
        common_channels : bool
            Keep only the channels whose wavelengths (to 1e-6 um) are in
            every exposure, e.g. when each exposure's extraction kept a
            slightly different column range. Otherwise the channels must
            match exactly.

        Returns
        -------
        LightCurves
            ``index`` continues across exposures (exposure ``k`` starts after
            the last integration number of ``k - 1``), so :meth:`exclude`
            uses visit-wide integration numbers. Regressors present in every
            exposure are kept; the 2D regressor ``expstep`` (one column per
            exposure after the first, 1 inside that exposure) is added, so
            ``linear=['expstep']`` fits a flux step per exposure.
            ``meta['exposures']`` lists the integration-number range of each.
        """
        lcs = list(lcs)
        if len(lcs) < 2:
            raise ValueError('Give at least two exposures.')
        if common_channels:
            keys = [set(np.round(lc.wavelength, 6)) for lc in lcs]
            common = set.intersection(*keys)
            if not common:
                raise ValueError('The exposures have no channel in common.')
            lcs = [lc.select_channels(channels=np.flatnonzero(np.isin(np.round(lc.wavelength, 6),
                                                                      list(common))))
                   for lc in lcs]
        ref = lcs[0]
        for lc in lcs[1:]:
            if lc.nchan != ref.nchan or not np.allclose(lc.wavelength, ref.wavelength,
                                                        equal_nan=True):
                raise ValueError('The exposures do not have the same channels.')
        idx, off, ranges = [], 0, []
        for lc in lcs:
            i = np.asarray(lc.index) - np.min(lc.index) + off
            idx.append(i)
            ranges.append((int(i.min()), int(i.max()) + 1))
            off = int(i.max()) + 1
        keys = [k for k in ref.regressors if all(k in lc.regressors for lc in lcs)]
        ckeys = [k for k in ref.channel_regressors if all(k in lc.channel_regressors for lc in lcs)]
        step = np.zeros((sum(lc.nint for lc in lcs), len(lcs) - 1))
        n0 = 0
        for k, lc in enumerate(lcs):
            if k:
                step[n0:n0 + lc.nint, k - 1] = 1.
            n0 += lc.nint
        regs = {k: np.concatenate([lc.regressors[k] for lc in lcs]) for k in keys}
        regs['expstep'] = step
        out = cls(np.concatenate([lc.times for lc in lcs]),
                  np.concatenate([lc.flux for lc in lcs]), np.concatenate([lc.err for lc in lcs]),
                  wavelength=ref.wavelength, wave_edges=ref.wave_edges, regressors=regs,
                  index=np.concatenate(idx), name=name or ref.name,
                  meta=dict(ref.meta, exposures=ranges,
                            files=[lc.meta.get('file') for lc in lcs]),
                  groups=ref.groups)
        for k in ckeys:
            out.add_regressor(k, np.concatenate([lc.channel_regressors[k] for lc in lcs]),
                              per_channel=True)
        if np.any(np.diff(out.times) <= 0):
            warnings.warn('The exposures overlap or are not in time order.')
        return out

    def bin_time(self, n=None, dt=None, max_gap=3., max_nan_frac=0.5):
        """Average consecutive integrations (time binning).

        Parameters
        ----------
        n : int or None
            Integrations per bin.
        dt : float or None
            Bin width in days (used if ``n`` is None), e.g. ``5 / 1440``.
        max_gap : float
            A bin never spans a gap longer than ``max_gap`` times the median
            cadence (a new bin starts after it).
        max_nan_frac : float
            A channel is NaN in a bin when more than this fraction of its
            integrations are NaN.

        Returns
        -------
        LightCurves
            Times, flux and regressors are means over the bin; errors are
            ``sqrt(sum err^2) / N``; ``index`` is the first integration of
            the bin (so :meth:`exclude` by integration number still works,
            but ``phase4`` does not). ``meta['bin_time']`` holds ``n``,
            ``dt``, the number of bins and the median bin span in days
            (``exptime``, for supersampling the model).

        Notes
        -----
        Binning smears the transit and eclipse ingress and egress; with bins
        that are not short compared with them, supersample the model
        (``BroadbandFit(supersample=...)``).
        """
        if n is None and dt is None:
            raise ValueError('Give n (integrations per bin) or dt (days).')
        o = np.argsort(self.times, kind='stable')
        t = self.times[o]
        cad = np.median(np.diff(t)) if t.size > 1 else 1.
        new = np.r_[True, np.diff(t) > max_gap * cad]
        seg = np.cumsum(new) - 1
        lab = np.empty(t.size, int)
        nxt = 0
        for s in np.unique(seg):
            j = np.flatnonzero(seg == s)
            k = (np.arange(j.size) // int(n)) if n is not None \
                else np.floor((t[j] - t[j[0]]) / dt).astype(int)
            _, k = np.unique(k, return_inverse=True)
            lab[j] = nxt + k
            nxt += k.max() + 1
        nb = nxt

        def mean(x):
            x = np.asarray(x, float)[o]
            out = np.full((nb,) + x.shape[1:], np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                for b in range(nb):
                    out[b] = np.nanmean(x[lab == b], axis=0)
            return out

        f = self.flux[o]
        e = self.err[o]
        flux = np.full((nb, self.nchan), np.nan)
        err = np.full((nb, self.nchan), np.nan)
        times = np.empty(nb)
        index = np.empty(nb, self.index.dtype)
        span = np.empty(nb)
        for b in range(nb):
            s = lab == b
            ok = np.isfinite(f[s]) & np.isfinite(e[s])
            cnt = ok.sum(0)
            with np.errstate(invalid='ignore', divide='ignore'):
                flux[b] = np.where(ok, f[s], 0.).sum(0) / cnt
                err[b] = np.sqrt(np.where(ok, e[s] ** 2, 0.).sum(0)) / cnt
            bad = cnt < (1. - max_nan_frac) * s.sum()
            flux[b, bad] = err[b, bad] = np.nan
            times[b] = t[s].mean()
            index[b] = self.index[o][s][0]
            span[b] = t[s][-1] - t[s][0] + cad
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            lc = LightCurves(times, flux, err, wavelength=self.wavelength, wave_edges=self.wave_edges,
                             regressors={k: mean(v) for k, v in self.regressors.items()},
                             index=index, name=self.name, meta=dict(self.meta), groups=self.groups)
            for k, v in self.channel_regressors.items():
                lc.add_regressor(k, mean(v), per_channel=True)
        lc.meta['bin_time'] = dict(n=n, dt=dt, nbins=int(nb), exptime=float(np.median(span)))
        return lc

    def outliers(self, nsigma=5., window=31, channel=None):
        """Mask of outlying integrations in the white light (or one channel).

        An integration is an outlier when it deviates from the running
        median of ``window`` integrations by more than ``nsigma`` robust
        standard deviations of the point-to-point-free residuals. The window
        must be short compared with the transit or eclipse ingress so that
        the running median follows the event.
        """
        if channel is None:
            y, _ = self.white_light()
        else:
            y = self.flux[:, channel] / np.nanmedian(self.flux[:, channel])
        r = y - running_median(y, window)
        s = MAD_TO_SIGMA * np.nanmedian(np.abs(r - np.nanmedian(r)))
        return ~np.isfinite(r) | (np.abs(r) > nsigma * s)

    def clip(self, nsigma=5., window=31, channel=None, verbose=True):
        """New object without the outliers found by :meth:`outliers`."""
        bad = self.outliers(nsigma, window, channel)
        if verbose:
            print(f'{self.name or "LightCurves"}: removed {int(bad.sum())} of {self.nint} '
                  f'integrations (> {nsigma} sigma from a {window}-integration running median)')
        lc = self.subset(~bad)
        lc.meta.setdefault('clipped', []).append(dict(nsigma=nsigma, window=window,
                                                      n=int(bad.sum())))
        return lc


def as_lightcurves(obj, name=None):
    """Convert ``obj`` to :class:`LightCurves`.

    Accepts a :class:`LightCurves`, a dict with keys ``times``, ``flux``,
    ``err`` (and optionally ``wavelength``, ``regressors``, ...; the keyword
    arguments of :class:`LightCurves`), or a tuple ``(times, flux, err)``.
    """
    if isinstance(obj, LightCurves):
        return obj
    if isinstance(obj, dict):
        kw = dict(obj)
        if 'lc' in kw and 'flux' not in kw:
            kw['flux'] = kw.pop('lc')
        if 'errors' in kw and 'err' not in kw:
            kw['err'] = kw.pop('errors')
        kw.setdefault('name', name)
        return LightCurves(**kw)
    if isinstance(obj, (tuple, list)) and len(obj) == 3:
        return LightCurves(*obj, name=name)
    raise TypeError(f'Cannot make LightCurves from {type(obj).__name__}.')


def _edges_from_centres(wl):
    wl = np.asarray(wl, float)
    n = wl.size
    if n == 0 or not np.isfinite(wl).all():
        return np.full((n, 2), np.nan)
    if n == 1:
        return np.array([[wl[0], wl[0]]])
    mid = 0.5 * (wl[1:] + wl[:-1])
    lo = np.concatenate([[wl[0] - (mid[0] - wl[0])], mid])
    hi = np.concatenate([mid, [wl[-1] + (wl[-1] - mid[-1])]])
    return np.stack([np.minimum(lo, hi), np.maximum(lo, hi)], axis=1)
