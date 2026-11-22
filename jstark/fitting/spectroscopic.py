"""Spectroscopic light-curve fits: one juliet fit per wavelength bin.

:class:`SpectroscopicFit` puts every dataset on one wavelength grid. For each
bin it sums the channels of each dataset that covers the bin and fits those
light curves jointly (a :class:`~jstark.fitting.BroadbandFit` per bin): the
planet parameters are shared as your priors say (normally one depth per bin,
the orbit fixed from the broadband fit), and every dataset keeps its own
flux offset, jitter and systematics. Bins run in parallel; finished bins are
skipped when the run is started again.

:class:`Spectrum` is the resulting table (one row per bin) with a plot
method.
"""
import json
import os
import time
import traceback
import warnings

import numpy as np
from astropy.table import Table

from ..plotting import styled, WIDTH, colorbar
from .fit import BroadbandFit, _as_list, _per_dataset
from .lightcurve import as_lightcurves, check_name
from .priors import normalise_priors

__all__ = ['SpectroscopicFit', 'Spectrum', 'restrict_priors', 'make_bins']

_NUISANCE_PREFIXES = ('mflux', 'mdilution', 'sigma', 'GP', 'theta', 'ramp', 'unnamed', 'loglike')
_THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
_BIN_JSON = 'jstark_spectroscopic.json'


def _is_nuisance(name):
    head = name.split('_')[0]
    return any(head.startswith(p) for p in _NUISANCE_PREFIXES)


def restrict_priors(priors, keep, names):
    """Priors for a fit of the datasets ``keep`` only.

    ``names`` are all dataset names of the full fit. In each parameter name
    the parts that are dataset names not in ``keep`` are removed
    (``q1_V1_V2`` becomes ``q1_V1`` when only ``V1`` is fitted); a parameter
    whose datasets are all missing is dropped.
    """
    keep, names = set(keep), set(names)
    out = {}
    for k, v in priors.items():
        parts = k.split('_')
        ins = [p for p in parts if p in names]
        if not ins:
            out[k] = v
            continue
        if not any(p in keep for p in ins):
            continue
        out['_'.join(p for p in parts if p not in names or p in keep)] = v
    return out


def _channels(lc, max_nan_frac):
    """Wavelength, lower and upper edge of the good channels of one dataset."""
    good = lc.good_channels(max_nan_frac) & np.isfinite(lc.wavelength)
    return lc.wavelength[good], lc.wave_edges[good, 0], lc.wave_edges[good, 1]


def make_bins(datasets, edges=None, nbins=None, max_nan_frac=0.5):
    """Wavelength bins ``(nbin, 2)`` (lower, upper edge; micron).

    Parameters
    ----------
    datasets : dict
        ``{name: LightCurves}``.
    edges : array or None
        1D: contiguous bin edges; 2D ``(nbin, 2)``: lower and upper edge of
        each bin (gaps allowed, e.g. a published table).
    nbins : int or None
        Without ``edges``: ``nbins`` bins with (nearly) equal numbers of
        good channels over all datasets. With neither: the native channels
        (all datasets must share one channel grid).
    """
    if edges is not None:
        e = np.asarray(edges, float)
        if e.ndim == 1:
            if e.size < 2:
                raise ValueError('edges needs at least two values.')
            pairs = np.stack([e[:-1], e[1:]], axis=1)
        elif e.ndim == 2 and e.shape[1] == 2:
            pairs = e
        else:
            raise ValueError('edges: a 1D array of edges or an (nbin, 2) array of [lower, upper].')
        if np.any(pairs[:, 1] <= pairs[:, 0]):
            raise ValueError('Every bin needs upper edge > lower edge.')
        return pairs
    wl, lo, hi = [], [], []
    for lc in datasets.values():
        w, l, h = _channels(lc, max_nan_frac)
        wl.append(w)
        lo.append(l)
        hi.append(h)
    wl, lo, hi = (np.concatenate(x) for x in (wl, lo, hi))
    if wl.size == 0:
        raise ValueError('No channel with a wavelength.')
    o = np.argsort(wl, kind='stable')
    wl, lo, hi = wl[o], lo[o], hi[o]
    if nbins is None:
        pairs = np.unique(np.round(np.stack([lo, hi], axis=1), 9), axis=0)
        if np.any(pairs[1:, 0] < pairs[:-1, 1] - 1e-9):
            raise ValueError('The datasets have different channel grids: give edges or nbins.')
        return pairs
    nb = int(nbins)
    if nb < 1 or nb > wl.size:
        raise ValueError(f'nbins must be between 1 and the number of channels ({wl.size}).')
    grp = (np.arange(wl.size) * nb) // wl.size
    e = [lo[0]]
    for g in range(nb - 1):
        a, b = np.flatnonzero(grp == g)[-1], np.flatnonzero(grp == g + 1)[0]
        e.append(0.5 * (hi[a] + lo[b]))
    e.append(hi[-1])
    return np.stack([e[:-1], e[1:]], axis=1)


class SpectroscopicFit:
    """One joint juliet fit per wavelength bin, over all datasets covering it.

    Parameters
    ----------
    datasets : dict
        ``{name: LightCurves}`` (or anything :func:`as_lightcurves`
        accepts); spectroscopic datasets (single-channel datasets are left
        out). Names are juliet instrument names.
    priors : dict, str or callable
        Priors of each bin in juliet naming: the same dict for every bin, or
        a function ``priors(info) -> dict`` with ``info`` a dict with keys
        ``index``, ``wave``, ``lo``, ``hi`` and ``datasets`` (e.g. for
        limb-darkening coefficients per bin). Typically the orbit fixed by
        :meth:`BroadbandFit.channel_priors` plus ``p_p1`` (or ``fp_p1``)
        and limb darkening. Parameters naming datasets that do not cover a
        bin are adapted (see :func:`restrict_priors`). Nuisance priors are
        added per bin as in :class:`BroadbandFit`.
    out_folder : str
        Folder of the fit; bin ``i`` is fitted in ``<out_folder>/bin<i>``.
    edges, nbins
        Wavelength bins (see :func:`make_bins`).
    min_cover : float
        A dataset enters a bin when its good channels cover at least this
        fraction of the bin width (default 0.5), e.g. at the edge of a
        detector or next to saturated columns.
    linear, gp, ramp, gp_kernel, ld_laws, nuisance, standardise, clip, max_nan_frac
        As in :class:`BroadbandFit`, used for every bin (dicts per dataset).
    bin_time, supersample
        As in :class:`BroadbandFit` (off by default): time binning of each
        bin's light curve and supersampling of the model.
    common_mode : BroadbandFit, dict or None
        Common-mode correction (off by default). A finished
        :class:`BroadbandFit` (its datasets of the same names are used;
        :meth:`BroadbandFit.common_mode`), or ``{name: BroadbandFit or
        (times, cm)}`` per dataset. Datasets without one are fitted as
        they are. Per channel group (e.g. NIRCam amplifiers): a fit of the
        white light of each group (datasets ``A1``, ``A2``, ... or
        ``<name>A1``, ...: :meth:`LightCurves.split`) gives every channel
        the common mode of its group
        (:meth:`BroadbandFit.channel_common_mode`); ``cm`` may also be
        ``(nint, nchan)``.
    cm_mode : {'divide', 'regressor'}
        ``'divide'``: every channel is divided by the common mode before
        binning; ``'regressor'``: the common mode becomes the linear
        regressor ``cm`` of every bin (free coefficient per bin, so the
        correction may scale with wavelength). The integrations missing in
        the white-light fit are dropped.

    Examples
    --------
    >>> bb = BroadbandFit(lcs, priors=pri, out_folder='white', linear=['time', 'dy'])
    >>> bb.fit(nthreads=8)
    >>> pri_ch = bb.channel_priors()          # orbit fixed to the white-light medians
    >>> pri_ch.update({'p_p1': ('uniform', [0., 0.3]),
    ...                'q1_V1_V2': ('uniform', [0., 1.]), 'q2_V1_V2': ('uniform', [0., 1.])})
    >>> sf = SpectroscopicFit(lcs, priors=pri_ch, out_folder='spec', nbins=40,
    ...                       linear=['time', 'dy'])
    >>> spec = sf.run(nproc=10)                # resumable
    >>> spec.plot(white=bb)

    Run it from a script with an ``if __name__ == '__main__':`` guard: the
    worker processes are spawned.
    """

    def __init__(self, datasets, priors=None, out_folder='spectroscopic', edges=None, nbins=None,
                 min_cover=0.5, linear=None, gp=None, ramp=None, gp_kernel='matern',
                 ld_laws='quadratic', nuisance=None, standardise=True, clip=None,
                 max_nan_frac=0.5, common_mode=None, cm_mode='divide', bin_time=None,
                 supersample=None, verbose=True):
        if not datasets:
            raise ValueError('No datasets given.')
        self.datasets = {}
        for name, obj in datasets.items():
            check_name(name)
            lc = as_lightcurves(obj, name=name)
            if lc.photometric:
                warnings.warn(f'Dataset {name} has one channel; it is left out of the '
                              'spectroscopic fit.')
                continue
            self.datasets[name] = lc
        if not self.datasets:
            raise ValueError('No spectroscopic (multi-channel) dataset.')
        self.names = list(self.datasets)
        self.priors = priors if callable(priors) else normalise_priors(priors)
        self.out_folder = os.path.abspath(out_folder)
        self.min_cover = float(min_cover)
        self.max_nan_frac = max_nan_frac
        self.verbose = verbose
        if cm_mode not in ('divide', 'regressor'):
            raise ValueError("cm_mode must be 'divide' or 'regressor'.")
        self.cm_mode = cm_mode
        self.common_mode = self._apply_common_mode(common_mode)
        if self.common_mode and cm_mode == 'regressor':
            lin = _per_dataset(linear, self.names, 'linear')
            linear = {n: _as_list(v) + (['cm'] if n in self.common_mode else [])
                      for n, v in lin.items()}
        self.options = dict(linear=linear, gp=gp, ramp=ramp, gp_kernel=gp_kernel, ld_laws=ld_laws,
                            nuisance=nuisance, standardise=standardise, clip=clip,
                            max_nan_frac=max_nan_frac, bin_time=bin_time, supersample=supersample)
        self.edges = make_bins(self.datasets, edges=edges, nbins=nbins, max_nan_frac=max_nan_frac)
        self.bins = self._define_bins()
        if verbose:
            n = sum(1 for b in self.bins if b['datasets'])
            print(f'{len(self.bins)} bins, {n} covered by at least one dataset '
                  f'({self.edges[0, 0]:.4g}-{self.edges[-1, 1]:.4g} um); datasets {self.names}')

    def _apply_common_mode(self, source):
        """Correct the datasets by their common mode; returns ``{name: (times, cm)}``."""
        if source is None:
            return {}
        out = {}
        for n in self.names:
            s = source.get(n) if isinstance(source, dict) else source
            if s is None:
                warnings.warn(f'No common mode given for dataset {n}; fitted without one.')
                continue
            if isinstance(s, BroadbandFit):
                lc0 = self.datasets[n]
                pre = None
                if n not in s.names and lc0.groups is not None:
                    labels = np.unique(lc0.groups[lc0.groups >= 0])
                    pre = next((p for p in (f'{n}A', 'A')
                                if all(f'{p}{g + 1}' in s.names for g in labels)), None)
                if pre is not None:
                    t, cm = s.channel_common_mode(lc0, prefix=pre)
                elif n in s.names:
                    t, cm = s.common_mode(n)
                else:
                    warnings.warn(f'The common-mode fit has no dataset {n}; fitted without one.')
                    continue
            else:
                t, cm = s
            t, cm = np.asarray(t, float), np.asarray(cm, float)
            lc = self.datasets[n].apply_common_mode(t, cm, mode=self.cm_mode)
            self.datasets[n] = lc
            out[n] = (t, cm)
            if self.verbose:
                miss = lc.meta['common_mode']['n_missing']
                kind = 'per channel group' if np.ndim(cm) == 2 else 'white light'
                print(f'{n}: common mode ({self.cm_mode}, {kind}) from {t.size} integrations'
                      + (f'; {miss} integrations without one are dropped' if miss else ''))
        return out

    # ------------------------------------------------------------------
    # Bins
    # ------------------------------------------------------------------
    def _define_bins(self):
        chans = {n: _channels(lc, self.max_nan_frac) for n, lc in self.datasets.items()}
        bins = []
        for i, (lo, hi) in enumerate(self.edges):
            cover, waves = {}, []
            for n, (w, l, h) in chans.items():
                sel = (w >= lo) & (w < hi)
                if not sel.any():
                    continue
                ov = np.clip(np.minimum(h[sel], hi) - np.maximum(l[sel], lo), 0., None)
                c = float(np.sum(ov) / (hi - lo)) if np.all(np.isfinite(ov)) else 1.
                if c >= self.min_cover:
                    cover[n] = min(c, 1.)
                    waves.append(np.mean(w[sel]))
            bins.append(dict(index=i, lo=float(lo), hi=float(hi),
                             wave=float(np.mean(waves)) if waves else 0.5 * float(lo + hi),
                             datasets=list(cover), cover=cover,
                             folder=os.path.join(self.out_folder, f'bin{i:03d}')))
        return bins

    def bin_priors(self, i):
        """Priors of bin ``i`` (before the nuisance priors are added)."""
        b = self.bins[i]
        if callable(self.priors):
            info = {k: b[k] for k in ('index', 'wave', 'lo', 'hi', 'datasets')}
            pri = normalise_priors(self.priors(info))
        else:
            pri = self.priors
        return restrict_priors(pri, b['datasets'], self.names)

    def bin_lightcurves(self, i):
        """``{name: LightCurves}`` (one channel each) of bin ``i``."""
        b = self.bins[i]
        out = {}
        for n in b['datasets']:
            out[n] = self.datasets[n].bin(edges=[b['lo'], b['hi']], max_nan_frac=self.max_nan_frac)
        return out

    def bin_fit(self, i):
        """The :class:`BroadbandFit` of bin ``i`` (not run)."""
        b = self.bins[i]
        if not b['datasets']:
            raise ValueError(f'Bin {i} ({b["lo"]:.4g}-{b["hi"]:.4g} um) has no data.')
        opts = {}
        for k, v in self.options.items():
            if isinstance(v, dict) and k != 'nuisance':
                v = {n: x for n, x in v.items() if n in b['datasets']}
            opts[k] = v
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            bb = BroadbandFit(self.bin_lightcurves(i), priors=self.bin_priors(i),
                              out_folder=b['folder'], verbose=False, **opts)
        bb._construction_warnings = sorted({str(x.message) for x in w})
        return bb

    @staticmethod
    def _state(bb):
        """'done', 'new' or 'different' for the folder of a bin fit."""
        files = bb._juliet_files()
        if not files:
            return 'new'
        js = os.path.join(bb.out_folder, 'jstark_fit.json')
        if not os.path.isfile(js):
            return 'different'
        with open(js) as f:
            same = json.load(f).get('fingerprint') == bb.fingerprint()
        if not same:
            return 'different'
        need = ('jstark_summary.ecsv', 'jstark_diagnostics.ecsv', 'jstark_residuals.npz')
        if all(os.path.isfile(os.path.join(bb.out_folder, f)) for f in need):
            return 'done'
        return 'new'

    def _write_json(self, sampler, fit_kwargs):
        os.makedirs(self.out_folder, exist_ok=True)
        if self.common_mode:
            np.savez(os.path.join(self.out_folder, 'common_mode.npz'),
                     **{f'{n}__{k}': v for n, (t, cm) in self.common_mode.items()
                        for k, v in (('t', t), ('cm', cm))})
        rec = dict(datasets=self.names, sampler=sampler, min_cover=self.min_cover,
                   fit_kwargs={k: repr(v) for k, v in fit_kwargs.items()},
                   common_mode=dict(mode=self.cm_mode, datasets=list(self.common_mode))
                   if self.common_mode else None,
                   options={k: v for k, v in self.options.items() if k != 'nuisance'},
                   bins=[{k: b[k] for k in ('index', 'lo', 'hi', 'wave', 'datasets', 'cover')}
                         | {'folder': os.path.basename(b['folder'])} for b in self.bins])
        with open(os.path.join(self.out_folder, _BIN_JSON), 'w') as f:
            json.dump(rec, f, indent=1, default=str)

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    def run(self, nproc=1, sampler='dynamic_dynesty', overwrite=False, bins=None, nsamples=500,
            **fit_kwargs):
        """Fit the bins (skipping finished ones) and return the :class:`Spectrum`.

        Parameters
        ----------
        nproc : int
            Bins fitted in parallel (one process each; juliet single-threaded
            inside). With ``nproc=1`` the bins run one after the other and
            ``nthreads`` may be passed to juliet.
        sampler : str
            juliet sampler (default ``'dynamic_dynesty'``).
        overwrite : bool
            ``False`` (default): finished bins are skipped; bins whose folder
            holds a different fit (data, regressors or priors changed) stop
            the run before any fitting. ``True``: fit all (requested) bins
            again. Their earlier files are deleted at the start, so an
            interrupted run continues with ``overwrite=False``.
        bins : sequence of int or None
            Only these bins (default all bins with data); bins without
            data are skipped.
        nsamples : int
            Posterior samples for the saved median models.
        **fit_kwargs
            Passed to ``juliet.load.fit`` (e.g. ``n_live_points``, ``dlogz``,
            ``bound``). In parallel runs dynesty's progress output is off
            unless ``print_progress`` is given.

        Returns
        -------
        Spectrum
        """
        idx = [b['index'] for b in self.bins if b['datasets']]
        if bins is not None:
            bins = [int(i) for i in bins]
            bad = [i for i in bins if not 0 <= i < len(self.bins)]
            if bad:
                raise ValueError(f'Bin indices out of range (0-{len(self.bins) - 1}): {bad}')
            empty = [i for i in bins if i not in idx]
            if empty and self.verbose:
                print(f'Skipping bins without data: {empty}')
            idx = [i for i in bins if i in idx]
        if nproc > 1:
            if fit_kwargs.pop('nthreads', None):
                warnings.warn('nthreads is ignored when bins run in parallel (nproc > 1).')
            fit_kwargs.setdefault('print_progress', False)
        todo, done, different, msgs = [], [], [], set()
        for i in idx:
            bb = self.bin_fit(i)
            msgs.update(bb._construction_warnings)
            st = self._state(bb)
            if st == 'done' and not overwrite:
                done.append(i)
            else:
                if st == 'different':
                    different.append(i)
                todo.append((i, bb))
        for m in sorted(msgs):
            warnings.warn(m)
        if different and not overwrite:
            raise RuntimeError(f'{len(different)} bin folder(s) hold a different fit (e.g. '
                               f'{self.bins[different[0]]["folder"]}); use overwrite=True or '
                               'another out_folder.')
        if overwrite:
            for _, bb in todo:
                bb._check_eclipse_timing(fit_kwargs)      # errors before anything is deleted
            for _, bb in todo:
                bb._check_folder(True)
        self._write_json(sampler, fit_kwargs)
        if self.verbose:
            print(f'{len(done)} bins already done, {len(todo)} to fit with {sampler} '
                  f'({nproc} process{"es" if nproc > 1 else ""})')
        failed = []
        t0 = time.time()

        def report(k, i, r):
            b = self.bins[i]
            if r['ok']:
                msg = f'{r["seconds"]:.0f} s'
            else:
                failed.append(i)
                msg = 'FAILED: ' + r['error'].strip().splitlines()[-1]
            if self.verbose:
                print(f'[{k}/{len(todo)}] bin {i} ({b["lo"]:.4g}-{b["hi"]:.4g} um, '
                      f'{"+".join(b["datasets"])}): {msg}', flush=True)

        if nproc <= 1:
            for k, (i, bb) in enumerate(todo, 1):
                report(k, i, _run_bin(bb, sampler, False, fit_kwargs, nsamples))
        elif todo:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            old = {v: os.environ.get(v) for v in _THREAD_VARS}
            for v in _THREAD_VARS:
                os.environ[v] = '1'
            try:
                ctx = multiprocessing.get_context('spawn')
                with ProcessPoolExecutor(max_workers=int(nproc), mp_context=ctx) as ex:
                    fut = {ex.submit(_run_bin, bb, sampler, False, fit_kwargs, nsamples): i
                           for i, bb in todo}
                    for k, f in enumerate(as_completed(fut), 1):
                        i = fut[f]
                        try:
                            r = f.result()
                        except Exception:
                            r = dict(ok=False, error=traceback.format_exc(), seconds=np.nan)
                        report(k, i, r)
            finally:
                for v, x in old.items():
                    if x is None:
                        os.environ.pop(v, None)
                    else:
                        os.environ[v] = x
        if self.verbose and todo:
            print(f'Fitted {len(todo) - len(failed)} bins in {time.time() - t0:.0f} s'
                  + (f'; {len(failed)} failed: {failed}' if failed else ''))
        self.failed = failed
        return self.spectrum()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, out_folder, verbose=False):
        """Results of a fit from its folder (no data needed): :meth:`spectrum`, plots."""
        out_folder = os.path.abspath(out_folder)
        with open(os.path.join(out_folder, _BIN_JSON)) as f:
            rec = json.load(f)
        self = cls.__new__(cls)
        self.out_folder, self.verbose = out_folder, verbose
        self.datasets = None
        self.names = rec['datasets']
        self.bins = [dict(b, folder=os.path.join(out_folder, b['folder'])) for b in rec['bins']]
        self.edges = np.array([[b['lo'], b['hi']] for b in self.bins])
        self.record = rec
        return self

    def bin_result(self, i):
        """The finished :class:`BroadbandFit` of bin ``i`` (loaded from its folder)."""
        return BroadbandFit.load(self.bins[i]['folder'])

    def _finished(self):
        for b in self.bins:
            f = os.path.join(b['folder'], 'jstark_summary.ecsv')
            if b['datasets'] and os.path.isfile(f):
                yield b

    def spectrum(self, save=True):
        """Collect the bins into a :class:`Spectrum` (and ``spectrum_diagnostics.ecsv``).

        One row per finished bin: ``bin``, ``wave`` (mean wavelength of the
        summed channels), ``wave_lo``, ``wave_hi``, ``datasets``, ``lnZ``
        and, for every fitted or derived planet and limb-darkening parameter
        ``x``, the columns ``x``, ``x_errp``, ``x_errm`` (median and 68%
        interval). Saved as ``<out_folder>/spectrum.ecsv``.
        """
        rows, params, diag = [], [], []
        for b in self._finished():
            s = Table.read(os.path.join(b['folder'], 'jstark_summary.ecsv'))
            vals = {}
            for r in s:
                if r['kind'] in ('fitted', 'derived') and not _is_nuisance(r['name']):
                    vals[r['name']] = (r['median'], r['err_plus'], r['err_minus'])
                    if r['name'] not in params:
                        params.append(r['name'])
            rows.append((b, vals, s.meta.get('lnZ', np.nan)))
            fd = os.path.join(b['folder'], 'jstark_diagnostics.ecsv')
            if os.path.isfile(fd):
                d = Table.read(fd)
                d['bin'] = b['index']
                d['wave'] = b['wave']
                d['cover'] = [b['cover'].get(n, np.nan) for n in d['dataset']]
                diag.append(d)
        cols = dict(bin=[], wave=[], wave_lo=[], wave_hi=[], datasets=[], lnZ=[])
        for p in params:
            cols[p], cols[p + '_errp'], cols[p + '_errm'] = [], [], []
        for b, vals, lnz in rows:
            cols['bin'].append(b['index'])
            cols['wave'].append(b['wave'])
            cols['wave_lo'].append(b['lo'])
            cols['wave_hi'].append(b['hi'])
            cols['datasets'].append(','.join(b['datasets']))
            cols['lnZ'].append(lnz)
            for p in params:
                v = vals.get(p, (np.nan, np.nan, np.nan))
                cols[p].append(v[0])
                cols[p + '_errp'].append(v[1])
                cols[p + '_errm'].append(v[2])
        spec = Spectrum(cols, names=list(cols))
        spec.meta.update(out_folder=self.out_folder, datasets=self.names,
                         nbins=len(self.bins), finished=len(rows))
        self.diagnostics_table = None
        if diag:
            from astropy.table import vstack
            d = vstack(diag, metadata_conflicts='silent')
            first = ['bin', 'wave', 'dataset', 'cover']
            self.diagnostics_table = d[first + [c for c in d.colnames if c not in first]]
        if save and rows:
            spec.write(os.path.join(self.out_folder, 'spectrum.ecsv'), overwrite=True)
            if self.diagnostics_table is not None:
                self.diagnostics_table.write(os.path.join(self.out_folder,
                                                          'spectrum_diagnostics.ecsv'),
                                             overwrite=True)
        return spec

    def diagnostics(self):
        """Noise of the residuals per bin and dataset (see :meth:`BroadbandFit.diagnostics`)."""
        self.spectrum(save=False)
        return self.diagnostics_table

    def residuals(self, dataset, quantity='resid'):
        """Image ``(nbin, nint)`` of one quantity of one dataset over all finished bins.

        ``quantity``: ``'y'`` (normalised data), ``'model'``, ``'planet'``,
        ``'detrended'`` (data with the systematics removed) or ``'resid'``.
        Returns ``(image, times, bins)``: integrations not fitted in a bin
        are NaN; ``times`` are the times of the columns; ``bins`` the bin
        indices of the rows.
        """
        per_bin, idx_all, t_all = [], [], []
        for b in self._finished():
            if dataset not in b['datasets']:
                continue
            f = os.path.join(b['folder'], 'jstark_residuals.npz')
            if not os.path.isfile(f):
                continue
            z = np.load(f)
            if f'{dataset}__{quantity}' not in z:
                raise KeyError(f'No {quantity!r} in {f}.')
            t = z[f'{dataset}__t']
            idx = z[f'{dataset}__index'] if f'{dataset}__index' in z else None
            key = idx if idx is not None else np.round((t - t.min()) * 86400., 3)
            per_bin.append((b['index'], key, t, z[f'{dataset}__{quantity}']))
            idx_all.append(key)
            t_all.append(t)
        if not per_bin:
            raise ValueError(f'No finished bin with dataset {dataset}.')
        keys, first = np.unique(np.concatenate(idx_all), return_index=True)
        times = np.concatenate(t_all)[first]
        img = np.full((len(per_bin), keys.size), np.nan)
        for r, (_, key, _, v) in enumerate(per_bin):
            img[r, np.searchsorted(keys, key)] = v
        return img, times, np.array([p[0] for p in per_bin])

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_spectrum(self, param=None, ax=None, white=None, **kwargs):
        """Plot the spectrum (see :meth:`Spectrum.plot`)."""
        return self.spectrum(save=False).plot(param=param, ax=ax, white=white, **kwargs)

    @styled
    def plot_2d(self, dataset=None, quantities=('detrended', 'planet', 'resid'), tbin=1,
                cmap='plasma', ppm=True, figsize=None, vlim=None):
        """Data, model and residuals of all bins as images (wavelength vs time).

        Parameters
        ----------
        dataset : str or None
            Dataset to show (default the first).
        quantities : sequence
            Panels, from ``'y'``, ``'model'``, ``'planet'``, ``'detrended'``,
            ``'resid'``.
        tbin : int
            Average this many consecutive integrations per column.
        vlim : (float, float) or None
            Colour range (ppm, relative to 1, or relative units) of the data
            panels; default the 1-99 percentiles. Residual panels use a
            symmetric range from their own percentiles.
        """
        import matplotlib.pyplot as plt
        dataset = dataset or self.names[0]
        imgs = {q: self.residuals(dataset, q) for q in quantities}
        fig, axs = plt.subplots(len(quantities), 1, sharex=True, layout='constrained',
                                figsize=figsize or (WIDTH, 2.2 * len(quantities) + 0.4), squeeze=False)
        axs = axs[:, 0]
        for ax, q in zip(axs, quantities):
            img, t, rows = imgs[q]
            if q != 'resid':
                img = img - 1.
            if ppm:
                img = img * 1e6
            if tbin > 1:
                n = img.shape[1] // tbin
                img = np.nanmean(img[:, :n * tbin].reshape(img.shape[0], n, tbin), axis=2)
                t = t[:n * tbin].reshape(n, tbin).mean(axis=1)
            th = (t - t[0]) * 24.
            if q == 'resid':
                v = np.nanpercentile(np.abs(img), 99)
                lim = (-v, v)
                cm = 'jstark_div'
            else:
                lim = vlim or tuple(np.nanpercentile(img, [1, 99]))
                cm = cmap
            im = ax.imshow(img, aspect='auto', origin='lower', cmap=cm, vmin=lim[0], vmax=lim[1],
                           extent=[th[0], th[-1], -0.5, img.shape[0] - 0.5],
                           interpolation='nearest')
            waves = [self.bins[r]['wave'] for r in rows]
            tk = np.linspace(0, len(rows) - 1, min(6, len(rows))).round().astype(int)
            ax.set_yticks(tk)
            ax.set_yticklabels([f'{waves[k]:.2f}' for k in tk])
            ax.set_ylabel('wavelength [µm]')
            ax.minorticks_off()
            colorbar(fig, im, ax, f'{q} [{"ppm" if ppm else "rel."}]')
        axs[-1].set_xlabel('time since first integration [h]')
        axs[0].set_title(f'{dataset}: {len(imgs[quantities[0]][2])} bins')
        return fig, axs

    @styled
    def plot_noise(self, figsize=(WIDTH, 3.4)):
        """Residual noise: rms vs bin size for every bin, and p2p / error vs wavelength."""
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize, layout='constrained')
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for k, n in enumerate(self.names):
            try:
                img, _, _ = self.residuals(n, 'resid')
            except ValueError:
                continue
            c = colors[k % len(colors)]
            for row in img:
                r = row[np.isfinite(row)]
                if r.size < 64:
                    continue
                sizes = np.unique(np.geomspace(1, r.size // 10, 25).astype(int))
                rms = []
                for s in sizes:
                    m = r.size // s
                    rms.append(np.std(r[:m * s].reshape(m, s).mean(axis=1)))
                rms = np.asarray(rms)
                a1.loglog(sizes, rms / rms[0], color=c, alpha=0.15, lw=0.8)
            a1.plot([], [], color=c, label=n)
        x = np.geomspace(1, 1e4, 10)
        a1.loglog(x, x ** -0.5, 'k--', lw=1, label='white noise')
        a1.set_xlabel('integrations per bin')
        a1.set_ylabel('rms / unbinned rms')
        a1.legend()
        d = self.diagnostics()
        if d is not None:
            for k, n in enumerate(self.names):
                s = d['dataset'] == n
                if s.any():
                    o = np.argsort(d['wave'][s])
                    a2.plot(np.asarray(d['wave'][s])[o], np.asarray(d['p2p_over_err'][s])[o],
                            '.-', color=colors[k % len(colors)], label=f'{n} p2p/err')
                    a2.plot(np.asarray(d['wave'][s])[o], np.asarray(d['p2p_over_total'][s])[o],
                            ':', color=colors[k % len(colors)], label=f'{n} incl. jitter')
            a2.axhline(1, color='k', lw=0.8)
            a2.set_xlabel('wavelength [µm]')
            a2.set_ylabel('residual p2p scatter / error')
            a2.legend()
        return fig, (a1, a2)


def _run_bin(bb, sampler, overwrite, fit_kwargs, nsamples):
    """Worker: fit one bin, save summary, diagnostics and residuals (module level for pickling)."""
    t0 = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bb.fit(sampler=sampler, overwrite=overwrite, **fit_kwargs)
            bb.diagnostics(save=True)
            bb.save_residuals(nsamples=nsamples)
        return dict(ok=True, seconds=time.time() - t0)
    except Exception:
        return dict(ok=False, error=traceback.format_exc(), seconds=time.time() - t0)


class Spectrum(Table):
    """Table of fitted parameters per wavelength bin (see :meth:`SpectroscopicFit.spectrum`).

    ``Spectrum.read('<out_folder>/spectrum.ecsv')`` reads a saved one.
    """

    def params(self):
        """Names of the parameters with values (columns with ``_errp`` / ``_errm``)."""
        return [c for c in self.colnames if c + '_errp' in self.colnames]

    def default_param(self):
        for p in self.params():
            if p.endswith('_ppm') and p.startswith(('depth_', 'fp_')):
                return p
        ps = self.params()
        if not ps:
            raise ValueError('The spectrum has no parameter columns.')
        return ps[0]

    def values(self, param=None):
        """``(wave, half_width, value, err_plus, err_minus)`` of one parameter."""
        p = param or self.default_param()
        w = np.asarray(self['wave'], float)
        hw = 0.5 * (np.asarray(self['wave_hi'], float) - np.asarray(self['wave_lo'], float))
        return (w, hw, np.asarray(self[p], float), np.asarray(self[p + '_errp'], float),
                np.asarray(self[p + '_errm'], float))

    @styled
    def plot(self, param=None, ax=None, white=None, label=None, xerr=True, **kwargs):
        """Plot one parameter against wavelength.

        Parameters
        ----------
        param : str or None
            Column, default the transit depth (``depth_p1_ppm``) or eclipse
            depth (``fp_p1_ppm``).
        white : BroadbandFit, (value, error) or None
            Broadband value of the same parameter, drawn as a line and band
            (from a :class:`BroadbandFit` its summary row ``param``).
        """
        import matplotlib.pyplot as plt
        p = param or self.default_param()
        if ax is None:
            _, ax = plt.subplots(figsize=(WIDTH, 3.6), layout='constrained')
        w, hw, v, ep, em = self.values(p)
        kw = dict(fmt='o', ms=3, elinewidth=1, capsize=0, label=label or p)
        kw.update(kwargs)
        ax.errorbar(w, v, yerr=[em, ep], xerr=hw if xerr else None, **kw)
        if white is not None:
            if isinstance(white, BroadbandFit):
                s = white.summary(save=False)
                r = s[s['name'] == p]
                if len(r) == 0:
                    raise ValueError(f'{p} not in the broadband summary.')
                val, err = float(r['median'][0]), 0.5 * float(r['err_plus'][0] + r['err_minus'][0])
            else:
                val, err = white
            ax.axhline(val, color='k', lw=1, label='broadband')
            ax.axhspan(val - err, val + err, color='k', alpha=0.1, lw=0)
        ax.set_xlabel('wavelength [µm]')
        ax.set_ylabel(p)
        return ax

    def write_text(self, fname, param=None):
        """Plain text table: wavelength, half-width, value and errors of one parameter."""
        w, hw, v, ep, em = self.values(param)
        np.savetxt(fname, np.stack([w, hw, v, ep, em], axis=1), fmt='%.8g',
                   header=f'wave half_width {param or self.default_param()} err_plus err_minus')
