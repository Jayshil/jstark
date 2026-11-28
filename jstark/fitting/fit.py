"""juliet fits of broadband light curves of one or several datasets.

:class:`BroadbandFit` fits one light curve per dataset, all datasets jointly
in one juliet run: the white light (sum of the channels) of a spectroscopic
dataset, or the light curve itself of a photometric one. Planet and orbit
parameters are shared or per dataset as your priors say (juliet naming);
every dataset gets its own flux offset, jitter and systematics model.
"""
import ast
import contextlib
import functools
import glob
import hashlib
import inspect
import json
import os
import pickle
import re
import warnings

import numpy as np

from .lightcurve import as_lightcurves, check_bjd, check_name
from .priors import (build_priors, check_groups, expand_group_options, expand_group_priors,
                     model_report, normalise_priors, priors_to_text, stray_priors)
from .utils import b_to_inc, rho_to_ar, phase_curve_quantities
from ..extraction.metrics import p2p_sigma

__all__ = ['BroadbandFit', 'ExpRamp', 'NonLinear', 'red_noise_beta', 'JULIET_FILES']

# Files juliet writes into its output folder (removed by ``overwrite=True``)
JULIET_FILES = ('lc.dat', 'rvs.dat', 'priors.dat', 'LM_lc_regressors.dat', 'GP_lc_regressors.dat',
                'posteriors.dat', '*posteriors.pkl', '*_checkpoint.pkl', '_dynesty_*',
                'jomnest_*', 'model_resids.dat')
JSTARK_FILES = ('jstark_fit.json', 'jstark_summary.ecsv', 'jstark_diagnostics.ecsv',
                'jstark_residuals.npz', 'jstark_nonlinear.npz', 'jstark_rv_diagnostics.ecsv',
                'jstark_jax_posteriors.npz', 'jstark_jax_diagnostics.ecsv', 'jstark_jax_inputs.npz',
                'jstark_jax_priors.json')
# Samplers of the JAX models (jstark.fitting.jaxfit); every other sampler name goes to juliet
JAX_SAMPLERS = ('blackjax-nuts', 'numpyro-nuts', 'blackjax-nss')
# juliet.fit options that change the JAX model too
_JAX_MODEL_OPTIONS = ('light_travel_delay', 'stellar_radius', 'ecclim', 'pl', 'pu', 'ta')

# juliet.fit options that change the model (needed again when a fit is reloaded)
_MODEL_OPTIONS = ('light_travel_delay', 'stellar_radius', 'ta', 'ecclim', 'pl', 'pu')
# Cadence (days) from which a dataset without supersampling gets a warning (about 10 min)
_LONG_CADENCE = 9.5 / 1440.
# Column of array regressors of non-linear functions while the dataset is clipped and binned
_NL_ARRAY = 'nonlinearinput'


def jax_sampler_name(name):
    """Standard name of a JAX sampler (``'blackjax_nuts'`` -> ``'blackjax-nuts'``), or None
    for a juliet sampler."""
    if not isinstance(name, str):
        return None
    s = name.strip().lower().replace('_', '-')
    return s if s in JAX_SAMPLERS else None


_PLANET_GEOMETRY = ('P', 't0', 'a', 'rho', 'b', 'inc', 'ecc', 'omega', 'secosomega',
                    'sesinomega', 'esinomega', 'ecosomega', 'dt', 'T', 't_secondary')


def _per_dataset(value, names, what, groups=None):
    """Expand a setting to ``{name: value}`` (dict: per dataset or group; anything else: for all)."""
    if isinstance(value, dict):
        unknown = set(value) - set(names) - set(groups or {})
        if unknown:
            raise ValueError(f'{what}: unknown dataset(s) {sorted(unknown)}; datasets are {names}'
                             + (f', groups {list(groups)}.' if groups else '.'))
        value = expand_group_options(value, groups or {})
        return {n: value.get(n) for n in names}
    return {n: value for n in names}


def _supersample_kwargs(ss):
    """juliet.load keywords for ``{name: (n, exptime)}``."""
    if not ss:
        return {}
    names = list(ss)
    return dict(lc_instrument_supersamp=names, lc_n_supersamp=[ss[n][0] for n in names],
                lc_exptime_supersamp=[ss[n][1] for n in names])


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

    def __init__(self, name, nterms=1, priors=None):
        self.name, self.nterms = name, int(nterms)
        self.params = {}
        for k in priors or {}:                 # shared by several datasets: ramptau_V1_V2
            parts = k.split('_')
            if name in parts[1:]:
                self.params[parts[0]] = k

    def names(self):
        out = []
        for k in range(1, self.nterms + 1):
            sfx = '' if k == 1 else str(k)
            out += [self.params.get(f'ramp{sfx}{q}', f'ramp{sfx}{q}_{self.name}')
                    for q in ('amp', 'tau')]
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


class NonLinear:
    """Your non-linear function of one dataset, in the form juliet calls it.

    ``function(x, parameters)`` gets the regressor(s) and a dict with the
    value of every parameter of the fit, and returns the model of each
    integration. juliet allows one function per dataset, so a multiplicative
    function also carries the exponential ramp of the dataset, if any (then
    ``x`` is ``(ramp time, regressor)``). ``aliases`` lets the function use
    group names (``amp_TESS`` for ``amp_TESS51_TESS91``).
    """

    def __init__(self, function, multiplicative=False, ramp=None, aliases=None):
        if ramp is not None and not multiplicative:
            raise ValueError('A ramp cannot be combined with an additive non-linear function '
                             '(juliet allows one function per dataset).')
        self.function, self.multiplicative, self.ramp = function, bool(multiplicative), ramp
        self.aliases = dict(aliases or {})
        self.enabled = True

    def neutral(self):
        return self.ramp.neutral() if self.ramp is not None else {}

    def __call__(self, x, pv):
        if not self.enabled:
            return 1. if self.multiplicative else 0.
        if self.aliases:
            pv = dict(pv, **{a: pv[k] for a, k in self.aliases.items() if k in pv})
        if self.ramp is None:
            return self.function(x, pv)
        return self.ramp(x[0], pv) * self.function(x[1], pv)


@contextlib.contextmanager
def _without_non_linear(dataset):
    """Switch off the user's non-linear functions of a juliet dataset (planet-only models)."""
    funcs = [f['function'] for f in (getattr(dataset, 'non_linear_functions', None) or {}).values()
             if isinstance(f['function'], NonLinear)]
    try:
        for f in funcs:
            f.enabled = False
        yield
    finally:
        for f in funcs:
            f.enabled = True


def _describe_function(f):
    """``(name, text)`` identifying a function: its source code and simple stored values."""
    def values(x):
        if isinstance(x, np.ndarray):
            return hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()
        if isinstance(x, (list, tuple)):
            return [values(i) for i in x]
        if isinstance(x, dict):
            return {str(k): values(v) for k, v in sorted(x.items(), key=lambda i: str(i[0]))}
        if callable(x):
            return list(_describe_function(x))
        return repr(x) if isinstance(x, (str, bytes, int, float, bool, type(None), np.generic)) \
            else type(x).__name__
    if isinstance(f, functools.partial):
        name, text = _describe_function(f.func)
        return name, text + repr(values([f.args, f.keywords]))
    target = f if inspect.isroutine(f) else type(f)
    name = getattr(target, '__qualname__', type(f).__name__)
    try:
        text = inspect.getsource(target)
    except (OSError, TypeError):               # e.g. defined in an interactive session
        code = getattr(f, '__code__', None)
        text = repr((code.co_code, code.co_consts, code.co_names)) if code is not None else ''
    extra = []
    if inspect.isfunction(f):
        extra = [f.__defaults__, f.__kwdefaults__] + [c.cell_contents for c in f.__closure__ or ()]
        extra = [e for e in extra if e is not None]
    elif not inspect.isroutine(f):
        extra = [vars(f)] if hasattr(f, '__dict__') else []
    return name, text + (repr(values(extra)) if extra else '')


def _prior_centre(p):
    """A typical value of a parameter under its prior."""
    h = np.atleast_1d(np.asarray(p['hyperparameters'], float))
    d = p['distribution']
    if d == 'uniform':
        return float(0.5 * (h[0] + h[1]))
    if d in ('loguniform', 'jeffreys'):
        return float(np.sqrt(h[0] * h[1]))
    if d == 'beta':
        return float(h[0] / (h[0] + h[1]))
    return float(h[0])                          # fixed, normal, truncatednormal, ...


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
        only (e.g. ``'V1'``, ``'V4a'``, ``'NRS2'``). Data of other
        observatories are read with :meth:`LightCurves.from_table`. The
        times of every dataset must be full Julian dates (BJD_TDB).
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
    non_linear : dict or None
        Your own non-linear models, ``{dataset: {'function': f, 'regressor':
        ..., 'multiplicative': bool}}`` (or ``{dataset: f}``).
        ``f(x, parameters)`` returns the model of each integration from the
        regressor ``x`` and the dict ``parameters`` of all parameter values;
        give priors for the parameters it uses. ``regressor`` is a regressor
        name (default ``'time'``), a list of names (``x`` is then ``(nint,
        k)``), or an array with one row per integration of the dataset as
        passed in. The full model is ``(planet + linear) * f``
        (``multiplicative=True``) or ``planet + linear + f`` (default, as in
        juliet). juliet allows one function per dataset: a ``ramp`` can be
        combined with a multiplicative function, not with an additive one.
        For ``nthreads > 1`` define ``f`` at the top level of a module or
        script. :meth:`load` needs the functions again.
    groups : dict or None
        Optional short names for several datasets, e.g. ``{'TESS':
        ['TESS51', 'TESS91']}``. In a prior name, a group is one parameter
        *shared* by its datasets (``q1_TESS`` is ``q1_TESS51_TESS91``); in
        ``linear``, ``gp``, ``ramp`` and the other per-dataset options
        (``nuisance`` included) the setting goes to each of its datasets.
        Full juliet names always work too.
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
    rvs : dict or None
        Radial velocities fitted jointly, ``{instrument: (times, rv, err)}``
        (BJD_TDB; m/s, or any unit, then used for ``K`` and the jitters too),
        or ``{instrument: {'times': ..., 'rv': ..., 'err': ...}}``. Needs a
        ``K_p<n>`` prior (Keplerian orbits with the ``P``, ``t0`` and
        eccentricity priors of the planet; planets without a radius-ratio
        prior are fitted to the RVs only, with increasing periods ``p1``,
        ``p2``, ...). Missing offsets ``mu_<instrument>`` and jitters
        ``sigma_w_<instrument>`` are added. ``rv_slope`` (and ``rv_quad``)
        priors, per day, add a linear (quadratic) trend about the middle of
        the RV times. Names must differ from the dataset names. Needs the
        radvel package. See :meth:`rv_diagnostics` and
        :meth:`julietPlots.plot_rv`.
    clip : dict or None
        If given, :meth:`LightCurves.clip` is applied with these keywords
        before fitting (e.g. ``{'nsigma': 5, 'window': 31}``).
    max_nan_frac : float
        Channels NaN in more than this fraction of the integrations are left
        out of the white light.
    bin_time : int, dict or None
        Off by default. Average this many consecutive integrations before
        fitting (:meth:`LightCurves.bin_time`; after ``clip``), for every
        dataset or ``{name: n}``. For a bin width in time, bin the
        :class:`LightCurves` yourself (``bin_time(dt=...)``).
    supersample : int, tuple, dict or None
        Off by default. Evaluate the planet model on this many
        sub-exposures per point (juliet's supersampling), for every dataset
        or ``{name: n}``; the exposure time is the bin span of time-binned
        data, else the median cadence, or give it: ``(n, exposure time in
        days)``. Use it with long bins or exposures (there is a warning
        from about 10 min).

    Examples
    --------
    >>> lcs = {'V1': LightCurves.from_photometry(f1), 'V2': LightCurves.from_photometry(f2)}
    >>> fit = BroadbandFit(lcs, priors=pri, out_folder='eclipses', linear=['time', 'xc', 'yc'])
    >>> fit.fit(sampler='dynamic_dynesty', nthreads=8)
    >>> fit.summary(); fit.diagnostics(); fit.plots().phase_folded_lc()
    """

    def __init__(self, datasets, priors=None, out_folder='broadband', linear=None, gp=None,
                 ramp=None, gp_kernel='matern', ld_laws='quadratic', nuisance=None,
                 standardise=True, clip=None, max_nan_frac=0.5, bin_time=None, supersample=None,
                 verbose=True, non_linear=None, groups=None, rvs=None):
        if not datasets:
            raise ValueError('No datasets given.')
        self.datasets = {}
        for name in datasets:
            check_name(name)
        self.groups = check_groups(groups, list(datasets))
        self.rvs = self._check_rvs(rvs, list(datasets) + list(self.groups))
        self.rv_names = list(self.rvs)
        self.non_linear = self._check_non_linear(non_linear, list(datasets))
        bt = _per_dataset(bin_time, list(datasets), 'bin_time', self.groups)
        for name, obj in datasets.items():
            lc = as_lightcurves(obj, name=name)
            check_bjd(lc.times, name)
            nl = self.non_linear.get(name)
            if nl and not isinstance(nl['regressor'], list):
                # an array: keep it aligned with the data while clipping and binning
                lc = lc.subset(np.ones(lc.nint, bool))
                lc.add_regressor(_NL_ARRAY, nl['regressor'])
            if clip:
                lc = lc.clip(verbose=verbose, **clip)
            if bt[name]:
                lc = lc.bin_time(n=int(bt[name]))
            self.datasets[name] = lc
        names = list(self.datasets)
        self.names = names
        self.supersample = {}
        for n, k in _per_dataset(supersample, names, 'supersample', self.groups).items():
            exp = None
            if isinstance(k, (tuple, list)):
                k, exp = k
                if not 0 < exp < 1:
                    raise ValueError(f'supersample of {n}: exposure time {exp} (it must be in days).')
            if k and int(k) > 1:
                self.supersample[n] = (int(k), float(exp) if exp is not None
                                       else self._exposure_time(n))

        def option(value, what):
            return _per_dataset(value, names, what, self.groups)

        self.linear = {n: _as_list(v) for n, v in option(linear, 'linear').items()}
        self.gp = {n: _as_list(v) for n, v in option(gp, 'gp').items()}
        self.gp_kernel = option(gp_kernel, 'gp_kernel')
        self.ramp = {n: int(v or 0) for n, v in option(ramp, 'ramp').items()}
        if any(v not in (0, 1, 2) for v in self.ramp.values()):
            raise ValueError('ramp: 0, 1 or 2 exponential terms per dataset.')
        for n, nl in self.non_linear.items():
            if self.ramp[n] and not nl['multiplicative']:
                raise ValueError(f'Dataset {n}: a ramp cannot be combined with an additive non-linear '
                                 'function (juliet allows one function per dataset). Put the ramp '
                                 'in your function or make the function multiplicative.')
        self.ld_laws = expand_group_options(ld_laws, self.groups)
        self.out_folder = os.path.abspath(out_folder)
        self.standardise = standardise
        self.max_nan_frac = max_nan_frac
        self.verbose = verbose
        self.user_priors, self.aliases = expand_group_priors(normalise_priors(priors), self.groups, names)
        self.nuisance = self._expand_nuisance(nuisance)
        self.dataset = self.results = None
        self.sampler = None
        self.jax_model = self._evaluator = None
        self.prepare()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def _check_non_linear(self, non_linear, names):
        """``{dataset: {'function', 'regressor' (list of names or array), 'multiplicative'}}``."""
        if non_linear is None:
            return {}
        if not isinstance(non_linear, dict) or 'function' in non_linear:
            raise ValueError("non_linear: give {dataset: {'function': ..., 'regressor': ..., "
                             "'multiplicative': ...}}.")
        out = {}
        for n, spec in _per_dataset(non_linear, names, 'non_linear', self.groups).items():
            if spec is None:
                continue
            spec = dict(spec) if isinstance(spec, dict) else {'function': spec}
            unknown = set(spec) - {'function', 'regressor', 'multiplicative'}
            if unknown:
                raise ValueError(f'non_linear of {n}: unknown key(s) {sorted(unknown)}.')
            if not callable(spec.get('function')):
                raise ValueError(f"non_linear of {n}: 'function' must be a function f(x, parameters).")
            reg = spec.get('regressor', 'time')
            if isinstance(reg, str) or (isinstance(reg, (list, tuple)) and len(reg)
                                        and all(isinstance(r, str) for r in reg)):
                reg = _as_list(reg)
            else:
                reg = np.asarray(reg, float)
            out[n] = dict(function=spec['function'], regressor=reg,
                          multiplicative=bool(spec.get('multiplicative', False)))
        return out

    @staticmethod
    def _check_rvs(rvs, taken):
        """``{instrument: {'t', 'rv', 'err'}}``, finite and sorted in time."""
        out = {}
        for n, v in (rvs or {}).items():
            check_name(n)
            if n in taken:
                raise ValueError(f'RV instrument {n!r} has the name of a dataset or group (juliet uses '
                                 'one sigma_w per name).')
            if isinstance(v, dict):
                unknown = set(v) - {'times', 'rv', 'err'}
                if unknown or len(v) < 3:
                    raise ValueError(f'RVs of {n}: give times, rv and err (got {sorted(v)}).')
                v = (v['times'], v['rv'], v['err'])
            if len(v) != 3:
                raise ValueError(f'RVs of {n}: give (times, rv, err).')
            t, y, e = (np.asarray(x, float).ravel() for x in v)
            if not t.size == y.size == e.size:
                raise ValueError(f'RVs of {n}: {t.size} times, {y.size} velocities, {e.size} errors.')
            ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(e) & (e > 0)
            if not ok.any():
                raise ValueError(f'RVs of {n}: no usable point.')
            check_bjd(t[ok], n)
            o = np.flatnonzero(ok)[np.argsort(t[ok], kind='stable')]
            out[n] = dict(t=t[o], rv=y[o], err=e[o])
        return out

    def _expand_nuisance(self, nuisance):
        """Full parameter names with a group (``sigma_w_TESS``): one entry per dataset."""
        if not nuisance or not self.groups:
            return nuisance
        out = {}
        for k, v in nuisance.items():
            parts = k.split('_')
            g = [x for x in parts[1:] if x in self.groups]
            for m in self.groups[g[0]] if g else [None]:
                key = k if m is None else '_'.join(m if x == g[0] else x for x in parts)
                if key == k or key not in nuisance:
                    out[key] = v
        return out

    def _exposure_time(self, n):
        """Time span of one point (days): the bin span of time-binned data, else the cadence."""
        lc = self.datasets[n]
        exp = lc.meta.get('bin_time', {}).get('exptime')
        if exp is None:
            dt = np.diff(np.sort(lc.times))
            dt = dt[dt > 0]
            exp = np.median(dt) if dt.size else 0.
        return float(exp)

    def _check_non_linear_functions(self):
        """Call every non-linear function once with typical parameter values."""
        pv = {k: _prior_centre(p) for k, p in self.priors.items()}
        for n, f in self._non_linear_functions().items():
            if n not in self.non_linear:
                continue
            try:
                r = np.asarray(f['function'](f['regressor'], pv))
            except KeyError as exc:
                raise ValueError(f'The non-linear function of {n} uses the parameter {exc}, which has '
                                 'no prior.') from exc
            nint = self.data[n]['t'].size
            if r.shape not in ((), (nint,)) or not np.issubdtype(r.dtype, np.floating):
                raise ValueError(f'The non-linear function of {n} returned shape {r.shape} '
                                 f'({r.dtype}) for {nint} integrations.')

    def _non_linear_functions(self):
        """juliet's ``non_linear_functions``: your functions and the exponential ramps."""
        out = {}
        for n in self.names:
            ramp = ExpRamp(n, self.ramp[n], self.priors) if self.ramp[n] else None
            nl = self.non_linear.get(n)
            if nl:
                f = NonLinear(nl['function'], nl['multiplicative'], ramp=ramp, aliases=self.aliases)
                x = self.data[n]['N']
                out[n] = dict(function=f, multiplicative=nl['multiplicative'],
                              regressor=x if ramp is None else (self.data[n]['ramp_x'], x))
            elif ramp:
                out[n] = dict(function=ramp, regressor=self.data[n]['ramp_x'], multiplicative=True)
        return out

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
            N = None
            if n in self.non_linear:
                reg = self.non_linear[n]['regressor']
                N = np.concatenate([lc.regressor(r)[0] for r in
                                    (reg if isinstance(reg, list) else [_NL_ARRAY])], axis=1)
            ok = np.isfinite(y) & np.isfinite(e) & (e > 0) & np.all(np.isfinite(X), axis=1)
            for R in (G, N):
                if R is not None:
                    ok &= np.all(np.isfinite(R), axis=1)
            if ok.sum() < 10:
                raise ValueError(f'Dataset {n}: only {int(ok.sum())} usable integrations.')
            m = np.median(y[ok])
            d = dict(t=lc.times[ok], y=y[ok] / m, e=e[ok] / m, X=X[ok], labels=labels,
                     stats=stats, index=lc.index[ok], nint_input=lc.nint, norm=float(m))
            d['ramp_x'] = d['t'] - d['t'].min()
            if N is not None:
                d['N'] = N[ok][:, 0] if N.shape[1] == 1 else N[ok]
            if G is not None:
                G = G[ok]
                d['G'] = G[:, 0] if G.shape[1] == 1 else G
                if d['G'].ndim == 1 and np.any(np.diff(d['G']) < 0):
                    # celerite needs a sorted regressor: order the dataset by it
                    o = np.argsort(d['G'], kind='stable')
                    for k in ('t', 'y', 'e', 'X', 'index', 'G', 'ramp_x', 'N'):
                        if k in d:
                            d[k] = d[k][o]
                    d['sorted_by_gp'] = True
            self.data[n] = d
            info[n] = dict(nlin=X.shape[1], gp_kernel=self.gp_kernel[n] if G is not None else None,
                           gp_x=d.get('G'), ramp=self.ramp[n], ramp_x=d['ramp_x'])
            if self.verbose:
                r = f'; ramp {self.ramp[n]} exp' if self.ramp[n] else ''
                if n in self.non_linear:
                    nl = self.non_linear[n]
                    r += (f"; non-linear {_describe_function(nl['function'])[0]}"
                          f"({', '.join(nl['regressor']) if isinstance(nl['regressor'], list) else 'array'})")
                print(f'{n}: {ok.sum()} of {lc.nint} integrations; linear {labels or "none"}; '
                      f'GP {self.gp[n] or "none"}{r}')
        if self.verbose:
            for n, v in self.rvs.items():
                print(f'{n}: {v["t"].size} radial velocities')
        self.priors, self.added = build_priors(self.user_priors, info, nuisance=self.nuisance,
                                               verbose=self.verbose, rvs=self.rvs)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.models = model_report(self.priors, self.names)
        for x in w:
            warnings.warn(str(x.message))
        stray = stray_priors(self.priors, self.names, self.rv_names)
        if stray:
            warnings.warn(f'Priors of datasets that are not in this fit: {stray}. juliet ignores them '
                          '(free ones only add unused dimensions); check the names, or remove them.')
        for n, r in self.models.items():
            exp = self._exposure_time(n)
            if exp > _LONG_CADENCE and n not in self.supersample and (r['transit'] or r['eclipse']):
                warnings.warn(f'Dataset {n}: one point covers {exp * 1440:.3g} min and the planet model '
                              f'is not supersampled; consider supersample={{{n!r}: 10}}.')
        self._check_non_linear_functions()
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
        if getattr(self, 'supersample', None):
            h.update(json.dumps(self.supersample, sort_keys=True).encode())
        for n, v in self.rvs.items():
            h.update(f'RV {n}'.encode())
            for k in ('t', 'rv', 'err'):
                h.update(np.ascontiguousarray(v[k], float).tobytes())
        for n, nl in self.non_linear.items():
            h.update(f"{n} {nl['multiplicative']} {_describe_function(nl['function'])}".encode())
            h.update(np.ascontiguousarray(self.data[n]['N'], float).tobytes())
        return h.hexdigest()

    def _juliet_files(self):
        out = set()
        for p in JULIET_FILES + JSTARK_FILES:
            out.update(glob.glob(os.path.join(self.out_folder, p)))
        return sorted(f for f in out if not os.path.basename(f).startswith('._'))

    def _check_folder(self, overwrite):
        """Refuse to reuse a folder that holds a different fit (juliet would silently reuse it);
        ``overwrite=True`` empties the folder in any case."""
        js = os.path.join(self.out_folder, 'jstark_fit.json')
        files = self._juliet_files()
        if not files:
            return
        if not overwrite:
            same = False
            if os.path.isfile(js):
                with open(js) as f:
                    same = json.load(f).get('fingerprint') == self.fingerprint()
            if same:
                if self.verbose:
                    print(f'{self.out_folder} holds the same fit: loaded if finished '
                          '(overwrite=True to fit again)')
                return
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
                   model_options={k: fit_kwargs[k] for k in _MODEL_OPTIONS
                                  if isinstance(fit_kwargs.get(k), (bool, int, float))},
                   rvs={n: int(v['t'].size) for n, v in self.rvs.items()},
                   supersample=self.supersample, groups=self.groups, aliases=self.aliases,
                   non_linear={n: dict(zip(('function', 'source'), _describe_function(nl['function'])),
                                       multiplicative=nl['multiplicative'],
                                       regressor=nl['regressor'] if isinstance(nl['regressor'], list)
                                       else 'array')
                               for n, nl in self.non_linear.items()},
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
        if self.non_linear:                    # juliet does not save the regressors
            np.savez(os.path.join(self.out_folder, 'jstark_nonlinear.npz'),
                     **{n: self.data[n]['N'] for n in self.non_linear})
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
        nlf = self._non_linear_functions()
        rv = {}
        if self.rvs:
            rv = dict(t_rv={n: v['t'] for n, v in self.rvs.items()},
                      y_rv={n: v['rv'] for n, v in self.rvs.items()},
                      yerr_rv={n: v['err'] for n, v in self.rvs.items()})
        self.dataset = juliet.load(priors=self.priors, t_lc=t, y_lc=y, yerr_lc=e, **rv,
                                   linear_regressors_lc=lin, GP_regressors_lc=gp,
                                   out_folder=self.out_folder, ld_laws=self._ld_string(),
                                   non_linear_functions=nlf, verbose=False,
                                   **_supersample_kwargs(self.supersample))
        return self.dataset

    def fit(self, sampler='dynamic_dynesty', overwrite=False, **fit_kwargs):
        """Run juliet or a JAX sampler (or load the finished fit from ``out_folder``).

        Parameters
        ----------
        sampler : str
            juliet sampler (``'dynamic_dynesty'``, ``'dynesty'``,
            ``'ultranest'``, ``'emcee'``, ...), or a JAX sampler:
            ``'blackjax-nuts'`` or ``'numpyro-nuts'`` (NUTS), or
            ``'blackjax-nss'`` (nested slice sampling, with the evidence);
            underscores work too. JAX samplers fit the same model with
            :mod:`jstark.fitting.jaxfit` (jax, jaxoplanet, blackjax or
            numpyro); every result method then uses the JAX model.
            Non-linear functions must be written with ``jax.numpy``.
        overwrite : bool
            ``False`` (default): an identical earlier fit (same data,
            regressors, priors and model settings) in ``out_folder`` is
            loaded instead of re-run, a different one raises an error.
            ``True``: delete the files of any earlier fit and fit again.
            The sampler settings are not compared.
        **fit_kwargs
            juliet: passed to ``juliet.load.fit``, juliet options
            (``n_live_points``, ``nthreads``, ``light_travel_delay``,
            ``stellar_radius``, ...) and any keyword of the sampler (for
            dynesty e.g. ``bound``, ``sample``, ``dlogz``, ``maxiter``;
            juliet forwards those it recognises).
            JAX NUTS samplers: ``nchains`` (4), ``nwarmup`` (1000), ``nsamples``
            (1000 per chain), ``seed`` (0), ``target_accept`` (0.9),
            ``max_tree_depth`` (10), ``dense_mass`` (True),
            ``chain_method`` (``'vectorized'`` or ``'sequential'``),
            ``init`` (``'map'``: chains start around the posterior maximum;
            ``'prior'``; or ``{name: value}``), ``nstart`` (8 optimiser
            starts), ``progress``. ``blackjax-nss`` (nested slice
            sampling, gives ``lnZ``): ``nlive`` (500; or juliet's
            ``n_live_points``), ``num_delete`` (nlive / 10), ``num_inner_steps``
            (5 x dimension), ``dlogz`` (0.01), ``max_steps``, ``nsamples``
            (default: the effective sample size), ``seed``, ``progress``.
            All JAX samplers take juliet's model options ``pl``, ``pu``,
            ``ecclim``, ``light_travel_delay``, ``stellar_radius``, ``ta``.
            Other juliet options are ignored with a warning.

        Returns
        -------
        juliet.fit or :class:`jstark.fitting.jaxfit.JaxResults`
        """
        self._check_eclipse_timing(fit_kwargs)
        if self.rvs:
            # reference time of RV trends
            tr = np.concatenate([v['t'] for v in self.rvs.values()])
            fit_kwargs.setdefault('ta', float(0.5 * (tr.min() + tr.max())))
        name = jax_sampler_name(sampler)
        if name is not None:
            return self._fit_jax(name, overwrite, fit_kwargs)
        if (fit_kwargs.get('nthreads') or 1) > 1:
            for n, nl in self.non_linear.items():
                try:
                    pickle.dumps(nl['function'])
                except Exception as exc:
                    raise ValueError(f'The non-linear function of {n} cannot be sent to other processes '
                                     f'(nthreads > 1): define it at the top level of a module or script '
                                     f'({exc}).') from exc
        self._check_folder(overwrite)
        self.load_juliet()
        self._write_json(sampler, fit_kwargs)
        self.sampler = sampler
        self.jax_model = self._evaluator = None
        self.results = self.dataset.fit(sampler=sampler, **fit_kwargs)
        self._write_summary()
        return self.results

    def _write_summary(self):
        try:
            self.summary(save=True)
        except Exception as exc:  # the fit itself is saved; do not lose it
            warnings.warn(f'Could not write the summary: {exc}')

    def _fit_jax(self, sampler, overwrite, fit_kwargs):
        """Fit with a JAX sampler: same model, priors and files as juliet's fit."""
        from .jaxfit import run_sampler
        opts, res = self._jax_prepare(sampler, overwrite, fit_kwargs)
        new = res is None
        if new:
            res = run_sampler(self.jax_model, sampler, **opts)
        return self._jax_finish(sampler, res, new)

    def _jax_prepare(self, sampler, overwrite, fit_kwargs):
        """Model, folder and input files of a JAX fit; returns ``(sampler options, results)``,
        results being the finished fit found in the folder or None."""
        from .jaxfit import JaxModel, JaxResults
        from .jaxfit.samplers import check_run, options_of
        OPTIONS = options_of(sampler)
        fit_kwargs = dict(fit_kwargs)
        if sampler == 'blackjax-nss' and 'n_live_points' in fit_kwargs:     # juliet's name
            fit_kwargs.setdefault('nlive', fit_kwargs.pop('n_live_points'))
        model_opts = {k: fit_kwargs[k] for k in _JAX_MODEL_OPTIONS if k in fit_kwargs}
        ignored = sorted(set(fit_kwargs) - set(OPTIONS) - set(_JAX_MODEL_OPTIONS))
        if ignored:
            warnings.warn(f'{ignored}: juliet options, ignored by the JAX sampler {sampler} (its '
                          f'options are {sorted(OPTIONS)}).')
        sampler_opts = {k: v for k, v in fit_kwargs.items() if k in OPTIONS}
        # errors (unknown options, features not available in JAX) before the folder is touched
        check_run(sampler, sampler_opts)
        self.jax_model = JaxModel.from_fit(self, **model_opts)
        self._check_folder(overwrite)
        self._write_json(sampler, fit_kwargs)
        self._save_jax_inputs()
        res = None
        if JaxResults.exists(self.out_folder):
            res = JaxResults.load(self.out_folder)
            if res.sampler != sampler or res.names != self.jax_model.prior_set.free:
                res = None                     # another sampler's run of the same fit: run again
            elif self.verbose:
                print(f'Loaded the finished {sampler} fit from {self.out_folder}')
        return sampler_opts, res

    def _jax_finish(self, sampler, res, new=True):
        """Save (``new``) and report a JAX run, and make it the results of this fit."""
        if new:
            tab = res.save(self.out_folder)
            report = res.report(tab).split('\n')
            if self.verbose:
                print(report[0])
            for line in report[1:]:
                warnings.warn(line.replace('WARNING: ', ''))
        self.dataset, self.results, self.sampler = None, res, sampler
        self._evaluator = None
        self._write_summary()
        return self.results

    def _save_jax_inputs(self):
        """Arrays and priors of a JAX fit (``load`` needs them; juliet keeps its own files)."""
        arrays = {}
        for n, d in self.data.items():
            for k in ('t', 'y', 'e', 'X', 'ramp_x', 'N', 'G', 'index'):
                if d.get(k) is not None:
                    arrays[f'{n}__{k}'] = np.asarray(d[k])
        for n, v in self.rvs.items():
            for k in ('t', 'rv', 'err'):
                arrays[f'rv:{n}__{k}'] = np.asarray(v[k])
        np.savez(os.path.join(self.out_folder, 'jstark_jax_inputs.npz'), **arrays)
        pri = {k: dict(distribution=p['distribution'],
                       hyperparameters=np.asarray(p['hyperparameters'], float).tolist())
               for k, p in self.priors.items()}
        with open(os.path.join(self.out_folder, 'jstark_jax_priors.json'), 'w') as f:
            json.dump(pri, f, indent=1)

    def _check_eclipse_timing(self, fit_kwargs):
        """juliet reads ``t_secondary_p<n>`` for eclipses only when ``light_travel_delay``
        is off; with ``light_travel_delay=True, stellar_radius=...`` it computes the
        eclipse time from the orbit, so no ``t_secondary`` prior is needed."""
        if fit_kwargs.get('light_travel_delay'):
            return
        planets = sorted({k.split('_')[1] for k in self.priors if k.split('_')[0] == 'fp'})
        missing = [f't_secondary_{pl}' for pl in planets if f't_secondary_{pl}' not in self.priors]
        if missing:
            raise ValueError(f'Eclipse fit without {missing}: either fit with '
                             'light_travel_delay=True, stellar_radius=<R*/Rsun> (juliet then '
                             'computes the eclipse time from P, t0, ecc, omega), or give a '
                             f'{missing[0]} prior (e.g. normal around t0 + P/2).')
        # A *fixed* t_secondary is ignored while the orbit varies: batman converts
        # t_secondary into a conjunction time only when t_secondary changes between
        # calls, and juliet writes the given t0 back into the batman parameters on every
        # call, so the eclipse is placed at the time implied by t0, ecc, omega (t0 + P/2
        # for a circular orbit), not at t_secondary. A free t_secondary changes on every
        # call and is handled correctly.
        orbit = ('P', 't0', 'a', 'b', 'inc', 'ecc', 'omega', 'r1', 'r2', 'esinomega',
                 'ecosomega', 'sesinomega', 'secosomega')
        for pl in planets:
            p = self.priors.get(f't_secondary_{pl}')
            if p is None or p['distribution'] != 'fixed':
                continue
            free = [k for k, v in self.priors.items() if v['distribution'] != 'fixed'
                    and ((k.split('_')[0] in orbit and k.split('_')[1:2] == [pl]) or k == 'rho')]
            # harmless when it is where t0 puts it anyway (circular orbit, t0 and P fixed)
            fixed = {n: self.priors.get(f'{n}_{pl}') for n in ('t0', 'P', 'ecc')}
            if all(v is not None and v['distribution'] == 'fixed' for v in fixed.values()):
                t0, per, ecc = (float(fixed[n]['hyperparameters']) for n in ('t0', 'P', 'ecc'))
                dt = (float(p['hyperparameters']) - t0 - per / 2) % per
                if ecc == 0 and min(dt, per - dt) < 1. / 86400:
                    free = []
            if free:
                raise ValueError(
                    f'Fixed t_secondary_{pl} with a varying orbit ({", ".join(free)}): juliet '
                    'would place the eclipse at the time implied by t0, ecc and omega (t0 + P/2 '
                    'for a circular orbit), not at t_secondary_' + pl + '. Give it a narrow free '
                    "prior instead (e.g. ('normal', [t_sec, sigma])), or fix ecc and omega to "
                    'values that put the eclipse at that time and fit with '
                    'light_travel_delay=True, stellar_radius=<R*/Rsun>.')

    @classmethod
    def load(cls, out_folder, verbose=False, non_linear=None):
        """Reload a finished fit from its folder (no data objects needed).

        A fit with non-linear functions needs them again (only their source
        text is stored, for reference): ``non_linear={dataset: function}``.

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
        self.groups, self.aliases = rec.get('groups') or {}, rec.get('aliases') or {}
        functions = cls._reload_non_linear(rec, non_linear, self.groups)
        self.jax_model = self._evaluator = None
        if jax_sampler_name(self.sampler):
            return self._load_jax(rec, functions)
        self.non_linear = {}
        with open(os.path.join(out_folder, 'priors.dat')) as f:
            priors = [line.split()[0] for line in f if line.strip() and line.strip()[0] != '#']
        nlf = {}
        if any(self.ramp.values()):
            # juliet does not store non-linear functions: rebuild the ramps from lc.dat
            tl, _, _, inst = np.genfromtxt(os.path.join(out_folder, 'lc.dat'), dtype=str,
                                           usecols=(0, 1, 2, 3), unpack=True)
            tl = tl.astype(float)
            for n, k in self.ramp.items():
                if k:
                    tn = tl[inst == n]
                    nlf[n] = dict(function=ExpRamp(n, k, priors), regressor=tn - tn.min(),
                                  multiplicative=True)
        if functions:
            with np.load(os.path.join(out_folder, 'jstark_nonlinear.npz')) as z:
                for n, f in functions.items():
                    mult, x = rec['non_linear'][n]['multiplicative'], z[n]
                    ramp = nlf.get(n)
                    self.non_linear[n] = dict(function=f, multiplicative=mult,
                                              regressor=rec['non_linear'][n]['regressor'])
                    nlf[n] = dict(function=NonLinear(f, mult, ramp=ramp and ramp['function'],
                                                     aliases=self.aliases),
                                  regressor=x if ramp is None else (ramp['regressor'], x),
                                  multiplicative=mult)
        self.supersample = {k: tuple(v) for k, v in rec.get('supersample', {}).items()}
        self.dataset = juliet.load(input_folder=out_folder, ld_laws=rec.get('ld_laws', 'quadratic'),
                                   non_linear_functions=nlf, verbose=False,
                                   **_supersample_kwargs(self.supersample))
        # juliet (2.2.10) keeps the functions but does not switch them on when reading a folder
        for n in nlf:
            self.dataset.nlm_lc_boolean[n] = True
        self.names = list(self.dataset.inames_lc)
        self.priors = self.dataset.priors
        self.rv_names = list(self.dataset.inames_rv if rec.get('rvs') else [])
        # juliet's fit() reads the results when they exist and otherwise starts sampling (in
        # this folder, perhaps while the real fit is still running): load only finished fits
        if not glob.glob(os.path.join(glob.escape(out_folder), '*posteriors.pkl')):
            raise FileNotFoundError(f'{out_folder} holds no finished juliet fit (no *posteriors.pkl): '
                                    'load() only reads results; use fit() to run it.')
        self.results = self.dataset.fit(sampler=rec['sampler'], **cls._model_options(rec))
        self.record = rec
        return self

    def _load_jax(self, rec, functions):
        """Rebuild the JAX model of a stored fit and read its chains."""
        from .jaxfit import JaxModel, JaxResults
        folder = self.out_folder
        with open(os.path.join(folder, 'jstark_jax_priors.json')) as f:
            self.priors = json.load(f)
        self.names = list(rec['datasets'])
        self.ld_laws = rec.get('ld_laws', 'quadratic')
        self.supersample = {k: tuple(v) for k, v in rec.get('supersample', {}).items()}
        self.non_linear = {n: dict(function=f, multiplicative=rec['non_linear'][n]['multiplicative'],
                                   regressor=rec['non_linear'][n]['regressor'])
                           for n, f in functions.items()}
        data = {n: {} for n in self.names}
        self.rvs = {n: {} for n in rec.get('rvs') or {}}
        with np.load(os.path.join(folder, 'jstark_jax_inputs.npz')) as z:
            for k in z.files:
                n, q = k.split('__', 1)
                if n.startswith('rv:'):
                    self.rvs[n[3:]][q] = z[k]
                else:
                    data[n][q] = z[k]
        self.rv_names = list(self.rvs)
        self._jax_index = {n: d.get('index') for n, d in data.items()}
        opts = {k: v for k, v in self._model_options(rec).items() if k in _JAX_MODEL_OPTIONS}
        self.jax_model = JaxModel.from_fit(self, data=data, **opts)
        self.results = JaxResults.load(folder)
        self.dataset = None
        self.record = rec
        return self

    @staticmethod
    def _model_options(rec):
        """juliet.fit options of a stored fit that change the model (e.g. light_travel_delay)."""
        out = rec.get('model_options')
        if out is None:                        # fits before 2026-11-23: only the repr of the options
            out = {}
            for k in _MODEL_OPTIONS:
                try:
                    v = ast.literal_eval(rec.get('fit_kwargs', {}).get(k, ''))
                except (ValueError, SyntaxError):
                    continue
                if isinstance(v, (bool, int, float)):
                    out[k] = v
        return out

    @staticmethod
    def _reload_non_linear(rec, non_linear, groups):
        """Functions of the stored fit ``{dataset: function}``, checked against the record."""
        stored = rec.get('non_linear') or {}
        given = expand_group_options(dict(non_linear or {}), groups)
        given = {n: (f['function'] if isinstance(f, dict) else f) for n, f in given.items()}
        if set(given) != set(stored):
            raise ValueError('This fit has non-linear functions for the datasets '
                             f'{sorted(stored)} (got {sorted(given)}): pass them as '
                             'load(..., non_linear={dataset: function}).')
        for n, f in given.items():
            if list(_describe_function(f)) != [stored[n]['function'], stored[n]['source']]:
                warnings.warn(f'The non-linear function of {n} differs from the one that was fitted '
                              '(see jstark_fit.json).')
        return given

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def _require(self):
        if self.results is None:
            raise RuntimeError('Run fit() first.')

    @property
    def evaluator(self):
        """Data and posterior models of the fit (:mod:`jstark.fitting.evaluators`): the juliet
        model for juliet samplers, the JAX model for JAX samplers."""
        self._require()
        ev = getattr(self, '_evaluator', None)
        if ev is None or ev.results is not self.results:
            if jax_sampler_name(self.sampler):
                from .jaxfit import JaxEvaluator
                index = getattr(self, '_jax_index', None)
                if self.data is not None:
                    index = {n: d['index'] for n, d in self.data.items()}
                ev = JaxEvaluator(self.jax_model, self.results, index=index)
            else:
                from .evaluators import JulietEvaluator
                ev = JulietEvaluator(self.dataset, self.results)
            self._evaluator = ev
        return ev

    @property
    def posteriors(self):
        """Posterior samples (``{name: array}``)."""
        self._require()
        return self.results.posteriors['posterior_samples']

    def values(self, name):
        """Posterior samples of ``name`` (group names allowed), or its fixed value, or None."""
        post = self.posteriors
        name = getattr(self, 'aliases', {}).get(name, name)
        if name in post:
            return np.asarray(post[name])
        p = self.priors.get(name)
        if p is not None and p['distribution'] == 'fixed':
            return np.atleast_1d(float(np.atleast_1d(p['hyperparameters'])[0]))
        return None

    def summary(self, save=True):
        """Table of parameters: fitted (median and 68% interval), fixed and derived.

        Derived quantities: transit depth ``(Rp/R*)^2``, eclipse depth in
        ppm, ``a/R*`` from ``rho`` and the inclination from ``b``; for a
        Cowan & Agol phase curve (:meth:`phase_curve`) the nightside flux
        ``fnight`` (at the transit; the dayside flux at the eclipse is
        ``fp``), the maximum and minimum planet flux, the peak-to-peak
        amplitude ``pcamp`` (all in ppm) and the phase offsets of the
        maximum before the eclipse and of the minimum before the transit
        (``pcoffset``, ``pcoffmin``, degrees; positive = eastward).
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
                ecc, om = self._eccentricity(pl)
                inc = b_to_inc(b, a, ecc=0. if ecc is None else ecc,
                               omega=90. if om is None else om)
                rows.append((f'inc_{pl}', *q(inc), 'derived'))
            if self.values(f'ecc_{pl}') is None:
                for k, v in zip(('ecc', 'omega'), self._eccentricity(pl)):
                    if np.size(v) > 1:
                        rows.append((f'{k}_{pl}', *q(v), 'derived'))
        rows += self._phase_curve_rows(q)
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

    def _eccentricity(self, planet):
        """Eccentricity and omega (degrees, stellar orbit) of a planet, whatever the
        parametrisation; (None, None) if there is none."""
        ecc, om = self.values(f'ecc_{planet}'), self.values(f'omega_{planet}')
        if ecc is not None:
            return ecc, om
        for c, s, power in (('secosomega', 'sesinomega', 2), ('ecosomega', 'esinomega', 1)):
            c, s = self.values(f'{c}_{planet}'), self.values(f'{s}_{planet}')
            if c is not None and s is not None:
                return np.sqrt(c ** 2 + s ** 2) ** power, np.degrees(np.arctan2(s, c))
        return None, None

    # ------------------------------------------------------------------
    # Radial velocities
    # ------------------------------------------------------------------
    def rv_residuals(self, name, nsamples=1000):
        """Radial velocities, median model and residuals of one RV instrument.

        Returns a dict with ``t``, ``rv``, ``err``, ``model`` (offset, trend
        and Keplerian orbits) and ``resid``.
        """
        self._require()
        if name not in self.rv_names:
            raise ValueError(f'No RV instrument {name!r} in this fit ({self.rv_names}).')
        t, y, e = self.evaluator.rv_data(name)
        m = self.evaluator.rv(name, nsamples=nsamples)
        return dict(t=t, rv=y, err=e, model=m, resid=y - m)

    def rv_diagnostics(self, save=True):
        """Residuals of the radial velocities per instrument (in their unit, e.g. m/s).

        Columns: number of points, offset ``mu``, rms of the residuals,
        median error, fitted jitter ``sigma_w``, and the reduced chi2 of the
        median model with the errors alone and with the jitter added.
        Saved to ``<out_folder>/jstark_rv_diagnostics.ecsv``.
        """
        from astropy.table import Table
        self._require()
        if not self.rv_names:
            raise ValueError('This fit has no radial velocities.')
        rows = []
        for n in self.rv_names:
            r = self.rv_residuals(n)
            sw = float(np.median(self.values(f'sigma_w_{n}')))
            rows.append([n, r['t'].size, float(np.median(self.values(f'mu_{n}'))),
                         np.std(r['resid']), np.median(r['err']), sw,
                         np.mean((r['resid'] / r['err']) ** 2),
                         np.mean(r['resid'] ** 2 / (r['err'] ** 2 + sw ** 2))])
        tab = Table(rows=rows, names=['instrument', 'n', 'mu', 'rms', 'err', 'sigma_w',
                                      'chi2_err', 'chi2_total'])
        if save:
            tab.write(os.path.join(self.out_folder, 'jstark_rv_diagnostics.ecsv'), overwrite=True)
        return tab

    def phase_curve(self, suffix='_p1'):
        """Posterior samples of the Cowan & Agol phase-curve quantities.

        ``suffix`` is the part after ``C1`` in the prior names (``'_p1'``,
        or ``'_p1_MIRI'`` for a per-dataset phase curve). Returns the dict
        of :func:`jstark.fitting.utils.phase_curve_quantities` (relative
        flux; offsets in degrees, positive = maximum before the eclipse),
        or None if this fit has no such phase curve.
        """
        vals = [self.values(c + suffix) for c in ('fp', 'C1', 'D1', 'C2', 'D2')]
        if vals[0] is None or vals[1] is None or vals[2] is None:
            return None
        vals = [np.zeros(1) if v is None else v for v in vals]
        n = max(np.size(v) for v in vals)
        vals = [np.broadcast_to(v, (n,)) if np.size(v) in (1, n) else v[:n] for v in vals]
        return phase_curve_quantities(*vals)

    def _phase_curve_rows(self, q):
        """Derived rows for every Cowan & Agol phase curve of the fit."""
        rows = []
        names = set(self.posteriors) | set(self.priors)
        for sfx in sorted({k[len('C1'):] for k in names if k.split('_')[0] == 'C1'}):
            pc = self.phase_curve(sfx)
            if pc is None or np.size(pc['fnight']) < 2:
                continue
            for k, label in (('fnight', 'fnight'), ('fmax', 'fmax'), ('fmin', 'fmin'),
                             ('amp', 'pcamp')):
                rows.append((f'{label}{sfx}_ppm', *q(pc[k] * 1e6), 'derived'))
            rows.append((f'pcoffset{sfx}_deg', *q(pc['offset']), 'derived'))
            rows.append((f'pcoffmin{sfx}_deg', *q(pc['offset_min']), 'derived'))
        return rows

    def residuals(self, name, nsamples=1000, components=False):
        """Data, median model and residuals of one dataset (sorted in time).

        With ``components=True`` also ``planet`` (the planet model alone,
        1 out of transit/eclipse), ``detrended`` (``y - model + planet``)
        and ``index`` (integration numbers; None after :meth:`load` of a
        juliet fit). JAX fits evaluate the JAX model.
        """
        ev = self.evaluator
        t, y, e = ev.data(name)
        r = ev.lc(name, nsamples=nsamples, components=components)
        m = r['model']
        i = np.argsort(t)
        out = dict(t=t[i], y=y[i], err=e[i], model=m[i], resid=(y - m)[i])
        if components:
            planet = r['planet']
            out['planet'] = planet[i]
            out['detrended'] = (y - m + planet)[i]
            # the fit keeps the order of the input arrays
            idx = self.data[name]['index'] if self.data is not None else ev.index(name)
            out['index'] = None if idx is None else np.asarray(idx)[i]
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

    def plots(self, N=1000, **kwargs):
        """:class:`jstark.fitting.plots.julietPlots` for this fit."""
        from .plots import julietPlots
        self._require()
        if jax_sampler_name(self.sampler):
            if kwargs:
                raise TypeError(f'plots() of a JAX fit takes no juliet.fit options ({sorted(kwargs)}).')
            ev = self.evaluator
        else:
            from .evaluators import JulietEvaluator
            ev = JulietEvaluator(self.dataset, self.results, fit_kwargs=kwargs)
        pl = julietPlots(input_folder=self.out_folder, N=N, evaluator=ev, **kwargs)
        pl.groups, pl.fit = getattr(self, 'groups', {}), self
        return pl

    def channel_priors(self, fix=_PLANET_GEOMETRY, mode='fixed', planet=None, datasets=None):
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
        datasets : sequence or None
            Only parameters that apply to these datasets (or groups), e.g.
            the JWST datasets of a joint fit with other observatories that
            go into a :class:`SpectroscopicFit`. Names shared with other
            datasets are shortened (``GP_rho_V1_TESS51`` becomes
            ``GP_rho_V1``). Default all.

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
        if datasets is not None:
            from .spectroscopic import restrict_priors
            groups = getattr(self, 'groups', {})
            keep = [m for d in _as_list(datasets) for m in groups.get(d, [d])]
            unknown = [d for d in keep if d not in self.names]
            if unknown:
                raise ValueError(f'Unknown dataset(s) {unknown}; datasets are {list(self.names)}.')
            rv = list(getattr(self, 'rv_names', []))
            out = {k: v for k, v in restrict_priors(out, keep, list(self.names) + rv).items()
                   if k.split('_')[0] not in ('K', 'rv')}
        return out
