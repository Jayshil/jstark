"""juliet fits of broadband light curves of one or several datasets.

:class:`BroadbandFit` fits one light curve per dataset, all datasets jointly
in one juliet run: the white light (sum of the channels) of a spectroscopic
dataset, or the light curve itself of a photometric one. Planet and orbit
parameters are shared or per dataset as your priors say (juliet naming);
every dataset gets its own flux offset, jitter and systematics model.
"""
import glob
import hashlib
import json
import os
import re
import warnings

import numpy as np

from .lightcurve import as_lightcurves, check_name
from .priors import build_priors, model_report, normalise_priors, priors_to_text
from .utils import b_to_inc, rho_to_ar
from ..extraction.metrics import p2p_sigma

__all__ = ['BroadbandFit', 'ExpRamp', 'red_noise_beta', 'JULIET_FILES']

# Files juliet writes into its output folder (removed by ``overwrite=True``)
JULIET_FILES = ('lc.dat', 'priors.dat', 'LM_lc_regressors.dat', 'GP_lc_regressors.dat',
                'posteriors.dat', '*posteriors.pkl', '*_checkpoint.pkl', '_dynesty_*',
                'jomnest_*', 'model_resids.dat')
JSTARK_FILES = ('jstark_fit.json', 'jstark_summary.ecsv', 'jstark_diagnostics.ecsv',
                'jstark_residuals.npz')

_PLANET_GEOMETRY = ('P', 't0', 'a', 'rho', 'b', 'inc', 'ecc', 'omega', 'secosomega',
                    'sesinomega', 'esinomega', 'ecosomega', 'dt', 'T')


def _per_dataset(value, names, what):
    """Expand a setting to ``{name: value}`` (dict: per dataset; anything else: for all)."""
    if isinstance(value, dict):
        unknown = set(value) - set(names)
        if unknown:
            raise ValueError(f'{what}: unknown dataset(s) {sorted(unknown)}; datasets are {names}.')
        return {n: value.get(n) for n in names}
    return {n: value for n in names}


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return list(v)


class ExpRamp:
    """Multiplicative exponential ramp ``1 + sum_k A_k exp(-x / tau_k)`` of one dataset.

    ``x`` is the time since the first fitted integration (days). Parameters
    ``rampamp_<ds>``, ``ramptau_<ds>`` (and ``ramp2amp_<ds>``,
    ``ramp2tau_<ds>`` for a second term) are juliet parameters. Used through
    juliet's ``non_linear_functions`` (a module-level class, so it can be
    pickled for multiprocessing).
    """

    def __init__(self, name, nterms=1):
        self.name, self.nterms = name, int(nterms)

    def names(self):
        out = []
        for k in range(1, self.nterms + 1):
            sfx = '' if k == 1 else str(k)
            out += [f'ramp{sfx}amp_{self.name}', f'ramp{sfx}tau_{self.name}']
        return out

    def neutral(self):
        """Parameter values that make the ramp 1 (used to plot planet-only models)."""
        return {n: 0. for n in self.names()[0::2]}

    def __call__(self, x, pv):
        r = np.ones_like(x, dtype=float)
        n = self.names()
        for k in range(self.nterms):
            r = r + pv[n[2 * k]] * np.exp(-x / pv[n[2 * k + 1]])
        return r


def red_noise_beta(resid, bins=(8, 16, 32, 64, 128)):
    """Red-noise factor beta: rms of non-overlapping bin means over the white-noise expectation."""
    r = np.asarray(resid, float)
    r = r[np.isfinite(r)]
    s1 = np.std(r)
    out = []
    for n in bins:
        m = r.size // n
        if m < 8:
            out.append(np.nan)
            continue
        rb = r[:m * n].reshape(m, n).mean(axis=1)
        out.append(np.std(rb) / (s1 / np.sqrt(n) * np.sqrt(m / (m - 1.))))
    return np.asarray(bins), np.asarray(out)


class BroadbandFit:
    """Joint juliet fit of the broadband light curve of every dataset.

    Parameters
    ----------
    datasets : dict
        ``{name: LightCurves}`` (or anything :func:`as_lightcurves`
        accepts). Names are juliet instrument names: letters and digits
        only (e.g. ``'V1'``, ``'V4a'``, ``'NRS2'``).
    priors : dict, str or None
        Astrophysical priors in juliet naming, e.g. ``{'P_p1': ('fixed',
        4.055), 't0_p1': ('normal', [t0, 0.001]), 'fp_p1': ('uniform', [0,
        1e-3]), ...}``. Planet parameters without an instrument suffix are
        shared by all datasets; ``p_p1_V1`` is specific to ``V1``, and
        ``q1_V1_V2`` is shared by ``V1`` and ``V2``. Limb darkening is never
        added for you: a dataset gets a transit model only when a
        ``q1``/``u1``/``c1`` prior names it. Missing nuisance priors are
        added (see :func:`jstark.fitting.priors.build_priors`).
    out_folder : str
        juliet output folder.
    linear : list, dict or None
        Regressors of the linear model, by name (see
        :meth:`LightCurves.regressor`; e.g. ``['time', 'dy', 'ds']``). A
        list applies to every dataset; a dict gives them per dataset.
    gp : str, list, dict or None
        GP input per dataset (one name, e.g. ``'time'``; a list gives a
        multi-dimensional input for george kernels). A str or list applies
        to every dataset; a dict gives it per dataset.
    ramp : int, dict or None
        Number of multiplicative exponential ramp terms (:class:`ExpRamp`;
        0, 1 or 2), for every dataset or ``{name: n}`` (e.g. MIRI start-of-
        visit settling).
    gp_kernel : str or dict
        Kernel used to add default GP priors when you give none
        (``'matern'``, ``'exp'``, ``'matern-exp'``, ``'sho'``). The kernel
        juliet uses is set by the names of the GP priors.
    ld_laws : str or dict
        Limb-darkening law for all datasets, or ``{name: law}``.
    nuisance : dict or None
        Overrides of the default nuisance priors (``mflux``, ``sigma_w``,
        ``theta``, ``mdilution`` or full parameter names).
    standardise : bool
        Standardise the linear regressors (mean 0, standard deviation 1).
    clip : dict or None
        If given, :meth:`LightCurves.clip` is applied with these keywords
        before fitting (e.g. ``{'nsigma': 5, 'window': 31}``).
    max_nan_frac : float
        Channels NaN in more than this fraction of the integrations are left
        out of the white light.

    Examples
    --------
    >>> lcs = {'V1': LightCurves.from_photometry(f1), 'V2': LightCurves.from_photometry(f2)}
    >>> fit = BroadbandFit(lcs, priors=pri, out_folder='eclipses', linear=['time', 'xc', 'yc'])
    >>> fit.fit(sampler='dynamic_dynesty', nthreads=8)
    >>> fit.summary(); fit.diagnostics(); fit.plots().phase_folded_lc()
    """

    def __init__(self, datasets, priors=None, out_folder='broadband', linear=None, gp=None,
                 ramp=None, gp_kernel='matern', ld_laws='quadratic', nuisance=None,
                 standardise=True, clip=None, max_nan_frac=0.5, verbose=True):
        if not datasets:
            raise ValueError('No datasets given.')
        self.datasets = {}
        for name, obj in datasets.items():
            check_name(name)
            lc = as_lightcurves(obj, name=name)
            if clip:
                lc = lc.clip(verbose=verbose, **clip)
            self.datasets[name] = lc
        names = list(self.datasets)
        self.names = names
        self.linear = {n: _as_list(v) for n, v in _per_dataset(linear, names, 'linear').items()}
        self.gp = {n: _as_list(v) for n, v in _per_dataset(gp, names, 'gp').items()}
        self.gp_kernel = _per_dataset(gp_kernel, names, 'gp_kernel')
        self.ramp = {n: int(v or 0) for n, v in _per_dataset(ramp, names, 'ramp').items()}
        if any(v not in (0, 1, 2) for v in self.ramp.values()):
            raise ValueError('ramp: 0, 1 or 2 exponential terms per dataset.')
        self.ld_laws = ld_laws
        self.out_folder = os.path.abspath(out_folder)
        self.standardise = standardise
        self.max_nan_frac = max_nan_frac
        self.verbose = verbose
        self.user_priors = normalise_priors(priors)
        self.nuisance = nuisance
        self.dataset = self.results = None
        self.sampler = None
        self.prepare()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def prepare(self):
        """Build the light curves, regressor matrices and complete priors."""
        self.data = {}
        info = {}
        for n, lc in self.datasets.items():
            y, e = lc.white_light(normalise=False, max_nan_frac=self.max_nan_frac)
            X, labels, stats = lc.design_matrix(self.linear[n], standardise=self.standardise)
            G = None
            if self.gp[n]:
                G = np.concatenate([lc.regressor(g)[0] for g in self.gp[n]], axis=1)
            ok = np.isfinite(y) & np.isfinite(e) & (e > 0) & np.all(np.isfinite(X), axis=1)
            if G is not None:
                ok &= np.all(np.isfinite(G), axis=1)
            if ok.sum() < 10:
                raise ValueError(f'Dataset {n}: only {int(ok.sum())} usable integrations.')
            m = np.median(y[ok])
            d = dict(t=lc.times[ok], y=y[ok] / m, e=e[ok] / m, X=X[ok], labels=labels,
                     stats=stats, index=lc.index[ok], nint_input=lc.nint, norm=float(m))
            d['ramp_x'] = d['t'] - d['t'].min()
            if G is not None:
                G = G[ok]
                d['G'] = G[:, 0] if G.shape[1] == 1 else G
                if d['G'].ndim == 1 and np.any(np.diff(d['G']) < 0):
                    # celerite needs a sorted regressor: order the dataset by it
                    o = np.argsort(d['G'], kind='stable')
                    for k in ('t', 'y', 'e', 'X', 'index', 'G', 'ramp_x'):
                        d[k] = d[k][o]
                    d['sorted_by_gp'] = True
            self.data[n] = d
            info[n] = dict(nlin=X.shape[1], gp_kernel=self.gp_kernel[n] if G is not None else None,
                           gp_x=d.get('G'), ramp=self.ramp[n], ramp_x=d['ramp_x'])
            if self.verbose:
                r = f'; ramp {self.ramp[n]} exp' if self.ramp[n] else ''
                print(f'{n}: {ok.sum()} of {lc.nint} integrations; linear {labels or "none"}; '
                      f'GP {self.gp[n] or "none"}{r}')
        self.priors, self.added = build_priors(self.user_priors, info, nuisance=self.nuisance,
                                               verbose=self.verbose)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.models = model_report(self.priors, self.names)
        for x in w:
            warnings.warn(str(x.message))
        if self.verbose:
            for n, r in self.models.items():
                kinds = [k for k in ('transit', 'eclipse', 'phasecurve') if r[k]] or ['systematics only']
                print(f'{n}: model = {" + ".join(kinds)}')

    def _ld_string(self):
        if isinstance(self.ld_laws, dict):
            laws = {n: self.ld_laws.get(n, 'quadratic') for n in self.names}
            return ','.join(f'{n}-{laws[n]}' for n in self.names)
        return self.ld_laws

    def fingerprint(self):
        """Hash of the data, regressors, priors and model settings."""
        h = hashlib.sha1()
        for n in self.names:
            d = self.data[n]
            h.update(n.encode())
            for k in ('t', 'y', 'e', 'X'):
                h.update(np.ascontiguousarray(d[k], float).tobytes())
            if 'G' in d:
                h.update(np.ascontiguousarray(d['G'], float).tobytes())
        h.update(priors_to_text(self.priors).encode())
        h.update(str(self._ld_string()).encode())
        h.update(json.dumps(self.ramp, sort_keys=True).encode())
        return h.hexdigest()

    def _juliet_files(self):
        out = set()
        for p in JULIET_FILES + JSTARK_FILES:
            out.update(glob.glob(os.path.join(self.out_folder, p)))
        return sorted(f for f in out if not os.path.basename(f).startswith('._'))

    def _check_folder(self, overwrite):
        """Refuse to reuse a folder that holds a different fit (juliet would silently reuse it)."""
        js = os.path.join(self.out_folder, 'jstark_fit.json')
        files = self._juliet_files()
        if not files:
            return
        same = False
        if os.path.isfile(js):
            with open(js) as f:
                same = json.load(f).get('fingerprint') == self.fingerprint()
        if same:
            return
        if not overwrite:
            raise RuntimeError(f'{self.out_folder} holds a different fit (data, regressors or '
                               'priors changed). juliet would reuse its files; use another '
                               'out_folder or overwrite=True.')
        for f in files:
            os.remove(f)
        if self.verbose:
            print(f'Removed {len(files)} files of the previous fit in {self.out_folder}')

    def _write_json(self, sampler, fit_kwargs, extra=None):
        os.makedirs(self.out_folder, exist_ok=True)
        rec = dict(fingerprint=self.fingerprint(), sampler=sampler,
                   fit_kwargs={k: repr(v) for k, v in fit_kwargs.items()},
                   ld_laws=self._ld_string(), added_priors=self.added, ramp=self.ramp,
                   datasets={n: dict(n_fit=int(self.data[n]['t'].size),
                                     n_input=int(self.data[n]['nint_input']),
                                     linear=self.data[n]['labels'],
                                     linear_mean=self.data[n]['stats'][:, 0].tolist(),
                                     linear_std=self.data[n]['stats'][:, 1].tolist(),
                                     gp=self.gp[n], norm=self.data[n]['norm'],
                                     meta={k: v for k, v in self.datasets[n].meta.items()
                                           if isinstance(v, (str, int, float, list, dict))
                                           and k not in ('binned_from',)},
                                     model=self.models[n])
                             for n in self.names})
        if extra:
            rec.update(extra)
        with open(os.path.join(self.out_folder, 'jstark_fit.json'), 'w') as f:
            json.dump(rec, f, indent=1, default=str)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def load_juliet(self):
        """Create the ``juliet.load`` object (writes juliet's input files)."""
        import juliet
        os.makedirs(self.out_folder, exist_ok=True)
        t = {n: d['t'] for n, d in self.data.items()}
        y = {n: d['y'] for n, d in self.data.items()}
        e = {n: d['e'] for n, d in self.data.items()}
        lin = {n: d['X'] for n, d in self.data.items() if d['X'].shape[1] > 0} or None
        gp = {n: d['G'] for n, d in self.data.items() if 'G' in d} or None
        nlf = {n: dict(function=ExpRamp(n, self.ramp[n]), regressor=self.data[n]['ramp_x'],
                       multiplicative=True) for n in self.names if self.ramp[n]}
        self.dataset = juliet.load(priors=self.priors, t_lc=t, y_lc=y, yerr_lc=e,
                                   linear_regressors_lc=lin, GP_regressors_lc=gp,
                                   out_folder=self.out_folder, ld_laws=self._ld_string(),
                                   non_linear_functions=nlf, verbose=False)
        return self.dataset

    def fit(self, sampler='dynamic_dynesty', overwrite=False, **fit_kwargs):
        """Run juliet (or load the finished fit from ``out_folder``).

        Parameters
        ----------
        sampler : str
            juliet sampler (``'dynamic_dynesty'``, ``'dynesty'``,
            ``'ultranest'``, ``'emcee'``, ...).
        overwrite : bool
            Delete the files of a *different* earlier fit in ``out_folder``.
            An identical earlier fit is always loaded instead of re-run.
        **fit_kwargs
            Passed to ``juliet.load.fit``: juliet options (``n_live_points``,
            ``nthreads``, ``light_travel_delay``, ``stellar_radius``, ...)
            and any keyword of the sampler (for dynesty e.g. ``bound``,
            ``sample``, ``dlogz``, ``maxiter``; juliet forwards those it
            recognises).

        Returns
        -------
        juliet.fit
        """
        self._check_folder(overwrite)
        self.load_juliet()
        self._write_json(sampler, fit_kwargs)
        self.sampler = sampler
        self.results = self.dataset.fit(sampler=sampler, **fit_kwargs)
        try:
            self.summary(save=True)
        except Exception as exc:  # the fit itself is saved by juliet; do not lose it
            warnings.warn(f'Could not write the summary: {exc}')
        return self.results

    @classmethod
    def load(cls, out_folder, verbose=False):
        """Reload a finished fit from its folder (no data objects needed).

        Returns an object with ``dataset``, ``results`` and the
        result methods (:meth:`summary`, :meth:`diagnostics`, :meth:`plots`,
        :meth:`channel_priors`).
        """
        import juliet
        out_folder = os.path.abspath(out_folder)
        with open(os.path.join(out_folder, 'jstark_fit.json')) as f:
            rec = json.load(f)
        self = cls.__new__(cls)
        self.out_folder, self.verbose = out_folder, verbose
        self.datasets, self.data = None, None
        self.sampler = rec['sampler']
        self.ramp = rec.get('ramp', {})
        nlf = {}
        if any(self.ramp.values()):
            # juliet does not store non-linear functions: rebuild the ramps from lc.dat
            tl, _, _, inst = np.genfromtxt(os.path.join(out_folder, 'lc.dat'), dtype=str,
                                           usecols=(0, 1, 2, 3), unpack=True)
            tl = tl.astype(float)
            for n, k in self.ramp.items():
                if k:
                    tn = tl[inst == n]
                    nlf[n] = dict(function=ExpRamp(n, k), regressor=tn - tn.min(),
                                  multiplicative=True)
        self.dataset = juliet.load(input_folder=out_folder, ld_laws=rec.get('ld_laws', 'quadratic'),
                                   non_linear_functions=nlf, verbose=False)
        # juliet (2.2.10) keeps the functions but does not switch them on when reading a folder
        for n in nlf:
            self.dataset.nlm_lc_boolean[n] = True
        self.names = list(self.dataset.inames_lc)
        self.priors = self.dataset.priors
        self.results = self.dataset.fit(sampler=rec['sampler'])
        self.record = rec
        return self

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def _require(self):
        if self.results is None:
            raise RuntimeError('Run fit() first.')

    @property
    def posteriors(self):
        """Posterior samples (``{name: array}``)."""
        self._require()
        return self.results.posteriors['posterior_samples']

    def values(self, name):
        """Posterior samples of ``name``, or its fixed value, or None."""
        post = self.posteriors
        if name in post:
            return np.asarray(post[name])
        p = self.priors.get(name)
        if p is not None and p['distribution'] == 'fixed':
            return np.atleast_1d(float(np.atleast_1d(p['hyperparameters'])[0]))
        return None

    def summary(self, save=True):
        """Table of parameters: fitted (median and 68% interval), fixed and derived.

        Derived quantities: transit depth ``(Rp/R*)^2``, eclipse depth in
        ppm, ``a/R*`` from ``rho`` and the inclination from ``b``.
        Saved to ``<out_folder>/jstark_summary.ecsv``.
        """
        from astropy.table import Table
        self._require()
        post = self.posteriors
        rows = []

        def q(v):
            lo, md, hi = np.percentile(v, [15.865, 50., 84.135])
            return md, hi - md, md - lo

        for k, v in post.items():
            if k.startswith('unnamed') or k.startswith('loglike'):
                continue
            rows.append((k, *q(np.asarray(v)), 'fitted'))
        for k, p in self.priors.items():
            if p['distribution'] == 'fixed' and k not in post:
                rows.append((k, float(np.atleast_1d(p['hyperparameters'])[0]), 0., 0., 'fixed'))
        planets = sorted({k.split('_')[1] for k in list(post) + list(self.priors)
                          if len(k.split('_')) > 1 and k.split('_')[1][:1] == 'p'
                          and k.split('_')[1][1:].isdigit()})
        for k in list(post):
            parts = k.split('_')
            if parts[0] == 'p':
                rows.append(('depth_' + '_'.join(parts[1:]) + '_ppm',
                             *q(np.asarray(post[k]) ** 2 * 1e6), 'derived'))
            if parts[0] == 'fp':
                rows.append((k + '_ppm', *q(np.asarray(post[k]) * 1e6), 'derived'))
        for pl in planets:
            per = self.values(f'P_{pl}')
            rho = self.values('rho')
            if self.values(f'a_{pl}') is None and rho is not None and per is not None:
                a = rho_to_ar(rho, per)
                if np.size(a) > 1:
                    rows.append((f'a_{pl}', *q(a), 'derived'))
            a = self.values(f'a_{pl}')
            if a is None and rho is not None and per is not None:
                a = rho_to_ar(rho, per)
            b = self.values(f'b_{pl}')
            if a is not None and b is not None and (np.size(a) > 1 or np.size(b) > 1):
                ecc = self.values(f'ecc_{pl}')
                om = self.values(f'omega_{pl}')
                inc = b_to_inc(b, a, ecc=0. if ecc is None else ecc,
                               omega=90. if om is None else om)
                rows.append((f'inc_{pl}', *q(inc), 'derived'))
        tab = Table(rows=rows, names=('name', 'median', 'err_plus', 'err_minus', 'kind'),
                    dtype=(str, float, float, float, str))
        lnz = self.results.posteriors.get('lnZ')
        if lnz is not None:
            tab.meta['lnZ'] = float(lnz)
            tab.meta['lnZerr'] = float(self.results.posteriors.get('lnZerr', np.nan))
        tab.meta['sampler'] = self.sampler
        if save:
            tab.write(os.path.join(self.out_folder, 'jstark_summary.ecsv'), overwrite=True)
        return tab

    def residuals(self, name, nsamples=1000, components=False):
        """Data, median model and residuals of one dataset (sorted in time).

        With ``components=True`` also ``planet`` (the planet model alone,
        1 out of transit/eclipse), ``detrended`` (``y - model + planet``)
        and ``index`` (integration numbers; only for a fit made in this
        session, None after :meth:`load`).
        """
        self._require()
        t = np.asarray(self.dataset.times_lc[name])
        y = np.asarray(self.dataset.data_lc[name])
        e = np.asarray(self.dataset.errors_lc[name])
        if components:
            m, comp = self.results.lc.evaluate(name, nsamples=nsamples, return_components=True)
            planet = comp.get('transit')
            if isinstance(planet, dict):
                planet = planet[name]
            planet = np.ones_like(t) if planet is None else np.asarray(planet, float)
        else:
            m = self.results.lc.evaluate(name, nsamples=nsamples)
        m = np.asarray(m, float)
        i = np.argsort(t)
        out = dict(t=t[i], y=y[i], err=e[i], model=m[i], resid=(y - m)[i])
        if components:
            out['planet'] = planet[i]
            out['detrended'] = (y - m + planet)[i]
            idx = None
            if self.data is not None:
                # juliet keeps the order of the input arrays
                idx = np.asarray(self.data[name]['index'])[i]
            out['index'] = idx
        return out

    def common_mode(self, name, nsamples=500):
        """Common-mode time series of one dataset: the data divided by the planet model.

        This is the white light without its transit (or eclipse): the
        systematics shared by all wavelengths plus the white-light noise,
        normalised to a median of 1. Pass it to :class:`SpectroscopicFit`
        (``common_mode=``) or :meth:`LightCurves.apply_common_mode`.

        Returns
        -------
        times, cm : ndarray
            Sorted in time (the integrations used in this fit).
        """
        r = self.residuals(name, nsamples=nsamples, components=True)
        cm = r['y'] / r['planet']
        return r['t'], cm / np.nanmedian(cm)

    def channel_common_mode(self, lc, prefix='A', nsamples=500):
        """Per-channel common mode of ``lc`` from per-group white-light fits.

        This fit must hold one dataset per channel group of ``lc``, named as
        :meth:`LightCurves.split` names them (``f'{prefix}{g + 1}'``), e.g.
        the white light of each NIRCam amplifier, each with its own depth
        and limb darkening. Every channel gets the common mode of its group.

        Returns
        -------
        times, cm : ndarray
            ``lc.times`` and ``(nint, nchan)``; NaN in integrations a group
            fit did not use. Pass ``{name: (times, cm)}`` as
            ``SpectroscopicFit(common_mode=...)``.
        """
        if lc.groups is None:
            raise ValueError('lc has no channel groups (LightCurves.groups).')
        labels = np.unique(lc.groups[lc.groups >= 0])
        missing = [f'{prefix}{g + 1}' for g in labels if f'{prefix}{g + 1}' not in self.names]
        if missing:
            raise ValueError(f'No dataset {missing} in this fit (datasets {self.names}).')
        cm = np.full((lc.nint, lc.nchan), np.nan)
        for g in labels:
            t, c = self.common_mode(f'{prefix}{g + 1}', nsamples=nsamples)
            cm[:, lc.groups == g] = lc.match_series(t, c)[:, None]
        if np.any(lc.groups < 0):
            warnings.warn(f'{int(np.sum(lc.groups < 0))} channels without a group get no '
                          'common mode (NaN: left out of the bins).')
        return lc.times.copy(), cm

    def save_residuals(self, fname=None, nsamples=500):
        """Save data, model, planet model and residuals of every dataset (``.npz``).

        Keys ``<dataset>__<quantity>`` (``t``, ``y``, ``err``, ``model``,
        ``planet``, ``detrended``, ``resid``, ``index``). Default file
        ``<out_folder>/jstark_residuals.npz``.
        """
        fname = fname or os.path.join(self.out_folder, 'jstark_residuals.npz')
        arrays = {}
        for n in self.names:
            r = self.residuals(n, nsamples=nsamples, components=True)
            for k, v in r.items():
                if v is not None:
                    arrays[f'{n}__{k}'] = np.asarray(v)
        np.savez(fname, **arrays)
        return fname

    def diagnostics(self, bins=(8, 16, 32, 64, 128), save=True):
        """Noise of the residuals per dataset.

        Columns: number of points, p2p scatter of the residuals (ppm), median
        error (ppm), fitted jitter ``sigma_w`` (ppm), the ratios
        ``p2p / err`` and ``p2p / sqrt(err^2 + sigma_w^2)``, and the
        red-noise factor beta for each bin size (1 = white noise).
        """
        from astropy.table import Table
        self._require()
        rows = []
        for n in self.names:
            r = self.residuals(n)
            p2p = p2p_sigma(r['resid'])
            err = np.median(r['err'])
            sw = self.values(f'sigma_w_{n}')
            if sw is None:
                sw = next((self.values(k) for k in self.posteriors
                           if k.startswith('sigma_w_') and n in k.split('_')[2:]), None)
            sw = 0. if sw is None else float(np.median(sw)) * 1e-6
            _, beta = red_noise_beta(r['resid'], bins)
            rows.append([n, r['t'].size, p2p * 1e6, err * 1e6, sw * 1e6, p2p / err,
                         p2p / np.sqrt(err ** 2 + sw ** 2)] + list(beta))
        names = ['dataset', 'n', 'p2p_ppm', 'err_ppm', 'sigma_w_ppm', 'p2p_over_err',
                 'p2p_over_total'] + [f'beta_{b}' for b in bins]
        tab = Table(rows=rows, names=names)
        if save:
            tab.write(os.path.join(self.out_folder, 'jstark_diagnostics.ecsv'), overwrite=True)
        return tab

    def plots(self, N=1000):
        """:class:`jstark.fitting.plots.julietPlots` for this fit."""
        from .plots import julietPlots
        self._require()
        return julietPlots(input_folder=self.out_folder, dataset=self.dataset, res=self.results, N=N)

    def channel_priors(self, fix=_PLANET_GEOMETRY, mode='fixed', planet=None):
        """Priors for spectroscopic channels from this fit.

        Parameters
        ----------
        fix : sequence of str
            Parameters to carry over, as name prefixes ending at an
            underscore (default: the orbit and geometry, ``P``, ``t0``,
            ``a``/``rho``, ``b``/``inc``, eccentricity terms and per-transit
            times). ``'GP'`` takes every GP hyperparameter, ``'GP_rho'``
            only the Matern timescale (the amplitude stays as you give it
            per bin), ``'ramptau'`` the ramp timescale, ``'GP_rho_V1'``
            one parameter of one dataset.
        mode : {'fixed', 'normal'}
            ``'fixed'``: fix each to its posterior median; ``'normal'``:
            normal priors with the posterior median and standard deviation.
        planet : str or None
            Only this planet's parameters (e.g. ``'p1'``); default all.
            Dataset parameters (GP, ramp, ...) are not affected.

        Returns
        -------
        dict
            Short-form priors ``{name: (distribution, hyperparameters)}``.
        """
        self._require()
        if mode not in ('fixed', 'normal'):
            raise ValueError("mode must be 'fixed' or 'normal'.")
        fix = (fix,) if isinstance(fix, str) else tuple(fix)
        out = {}
        for k, p in self.priors.items():
            if not any(k == f or k.startswith(f + '_') for f in fix):
                continue
            tags = [x for x in k.split('_')[1:] if re.fullmatch(r'p\d+', x)]
            if planet and tags and planet not in tags:
                continue
            v = self.values(k)
            if k in self.posteriors:
                out[k] = (('fixed', float(np.median(v))) if mode == 'fixed'
                          else ('normal', [float(np.median(v)), float(np.std(v))]))
            else:
                out[k] = (p['distribution'], p['hyperparameters'])
        return out
