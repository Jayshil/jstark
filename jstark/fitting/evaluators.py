"""The model interface of the fit results: data and posterior models of each dataset.

The result methods of :class:`~jstark.fitting.BroadbandFit` (residuals,
common mode, diagnostics, ...) and the plots
(:class:`~jstark.fitting.plots.julietPlots`) get the data and the models
through one of these objects, so they are the same code for juliet and JAX
fits:

* :class:`JulietEvaluator`: juliet's ``results.lc.evaluate`` (batman, the
  GPs and systematics of juliet);
* :class:`jstark.fitting.jaxfit.JaxEvaluator`: the JAX model of the fit.

Interface
---------
``names``, ``rv_names``
    Light-curve datasets and RV instruments.
``posteriors``, ``priors``
    Posterior samples (``{name: array}``) and the priors (juliet's dict).
``data(name)``
    ``(t, y, err)`` of a dataset, in the order of the fit.
``regressors(name)``
    ``{'linear': X or None, 'gp': GP regressor or None}`` (fit order).
``fit_type(name)``
    ``'transit'``, ``'eclipse'`` or ``'phasecurve'`` (transit and eclipse).
``lc(name, nsamples=1000, components=False)``
    ``{'model': ...}``: the median over ``nsamples`` posterior draws of
    the full model at the data; with ``components=True`` also ``planet``
    (``1 + sum_i (planet_i - 1)``, before dilution: juliet's ``transit``
    component).
``lc_models(name, nsamples=1000)``
    Draws and 68% intervals of the full model at the data: ``samples``
    ``(n, npoints)``, ``median``, ``upper``, ``lower``, ``linear`` (median
    linear model or None), and with a GP ``gp``, ``gp_upper``,
    ``gp_lower`` (the GP prediction at the data).
``planet_models(name, nsamples=1000, t=None)``
    The same (``samples``, ``median``, ``upper``, ``lower``) for the planet
    model alone (no dilution, offsets, systematics or GP), at the data or
    at times ``t``.
``non_linear(name)``
    ``(values, multiplicative)`` of the non-linear model (ramps, your
    functions) at the posterior medians, or ``(None, False)``.
``orbit(planet='p1', name=None)``
    Posterior medians of ``P``, ``t0``, ``p``, ``a``, ``b``, ``ecc``,
    ``w`` (deg) and ``eclipse_phase`` (phase of the eclipse centre, None
    if unknown) of a planet, in the model of dataset ``name``.
``gp_predict(name, x, resid, err, x_pred, params=None)``
    ``(mean, std)`` of the GP of dataset ``name`` at the posterior medians
    (``params``: ``{prior name: value}`` replacing some) conditioned on
    residuals ``resid`` (errors ``err``, which include ``sigma_w``) at
    regressor values ``x``, at ``x_pred``.
``rv_data(name)``, ``rv(name, nsamples=1000)``
    Data and median model of radial velocities.
``keplerian(t, P, t0, ecc, w, K)``, ``rv_ta``
    Keplerian RVs (``(n, nt)``, for ``n`` parameter sets; ``w`` in
    radians; ``t`` ``(nt,)`` or ``(n, nt)``) and the reference time of the
    RV trend.
``index(name)``
    Integration numbers of the points of a dataset, or None if unknown.
"""
import glob
import os
import shutil
import tempfile
from contextlib import contextmanager

import numpy as np

__all__ = ['JulietEvaluator', 'median_or_fixed']


def median_or_fixed(posteriors, priors, name):
    """Posterior median of ``name``, else its fixed prior value (KeyError if neither)."""
    if name in posteriors:
        return float(np.nanmedian(posteriors[name]))
    p = priors[name]
    if p['distribution'] == 'fixed':
        return float(np.atleast_1d(p['hyperparameters'])[0])
    raise KeyError(name)


@contextmanager
def _juliet_state(lc, name):
    """Restore the juliet model state of dataset ``name`` after ``lc.evaluate(t=...)``.

    juliet's ``evaluate`` at other times leaves the dataset's model arrays
    (``ones``, ``M``, ...) at the new length, so any later evaluation at the
    data fails (shape mismatch); the arrays, times, number of points and
    linear regressors are put back here."""
    model = lc.model[name]
    saved = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in model.items()}
    keep = {}
    for attr in ('times', 'ndatapoints_per_instrument', 'lm_arguments'):
        d = getattr(lc, attr, None)
        if isinstance(d, dict) and name in d:
            keep[attr] = d[name]
    try:
        yield
    finally:
        model.clear()
        model.update(saved)
        for attr, v in keep.items():
            getattr(lc, attr)[name] = v


class JulietEvaluator:
    """Data and posterior models of a juliet fit (``juliet.load``, ``juliet.fit``).

    ``fit_kwargs`` are the juliet.fit options of the fit (used to rebuild it
    for high-resolution models of light-travel-delayed eclipses).
    """

    kind = 'juliet'

    def __init__(self, dataset, results, fit_kwargs=None):
        self.dataset, self.results = dataset, results
        self.names = list(dataset.inames_lc or [])
        self.rv_names = list(getattr(dataset, 'inames_rv', None) or [])
        self.fit_kwargs = dict(fit_kwargs or {})

    @property
    def posteriors(self):
        return self.results.posteriors['posterior_samples']

    @property
    def priors(self):
        return self.dataset.priors

    def data(self, name):
        d = self.dataset
        return tuple(np.asarray(getattr(d, k)[name], float) for k in ('times_lc', 'data_lc', 'errors_lc'))

    def regressors(self, name):
        lm = (self.dataset.lm_lc_arguments or {}).get(name)
        gp = (self.dataset.GP_lc_arguments or {}).get(name)
        return dict(linear=None if lm is None else np.asarray(lm), gp=None if gp is None else np.asarray(gp))

    def fit_type(self, name):
        o = self.dataset.lc_options[name]
        return 'phasecurve' if o['TranEclFit'] else ('eclipse' if o['EclipseFit'] else 'transit')

    def lc(self, name, nsamples=1000, components=False):
        if not components:
            return dict(model=np.asarray(self.results.lc.evaluate(name, nsamples=nsamples), float))
        m, comp = self.results.lc.evaluate(name, nsamples=nsamples, return_components=True)
        planet = comp.get('transit')
        if isinstance(planet, dict):
            planet = planet[name]
        m = np.asarray(m, float)
        planet = np.ones_like(m) if planet is None else np.asarray(planet, float)
        return dict(model=m, planet=planet)

    def lc_models(self, name, nsamples=1000):
        s, m, u, lo, comp = self.results.lc.evaluate(name, nsamples=nsamples, return_err=True,
                                                     return_components=True, return_samples=True)
        reg = self.regressors(name)
        out = dict(samples=s, median=m, upper=u, lower=lo,
                   linear=np.asarray(comp['lm']) if reg['linear'] is not None else None)
        mod = self.results.lc.model[name]
        if reg['gp'] is not None:
            out.update(gp=np.copy(mod['GP']), gp_upper=np.copy(mod['GP_uerror']),
                       gp_lower=np.copy(mod['GP_lerror']))
        return out

    # ------------------------------------------------------------------
    def _planet_only_samples(self):
        """Posterior samples with the flux offsets, linear coefficients and GP set to 0, the
        dilution to 1 (no dilution: juliet's model is then the planet model) and non-linear
        (e.g. ramp) models made neutral, so that only uncertainties of the planet model
        propagate to the planet-only model."""
        post = self.results.posteriors['posterior_samples'].copy()
        for i in post.keys():
            if (i[0:5] == 'mflux') or (i[0:5] == 'theta') or (i[0:2] == 'GP'):
                post[i] = np.zeros(len(post[i]))
            elif i[0:9] == 'mdilution':
                post[i] = np.ones(len(post[i]))
        nlf = getattr(self.dataset, 'non_linear_functions', None) or {}
        for f in nlf.values():
            neutral = getattr(f['function'], 'neutral', None)
            if callable(neutral):
                for k, v in neutral().items():
                    if k in post:
                        post[k] = np.full(len(post[k]), v)
        return post

    def planet_models(self, name, nsamples=1000, t=None):
        from .fit import _without_non_linear
        with _without_non_linear(self.dataset):
            post = self._planet_only_samples()
            kw = dict(nsamples=nsamples, return_err=True, parameter_values=post, return_components=True,
                      return_samples=True, evaluate_transit=True)
            if t is None:
                out = self.results.lc.evaluate(name, **kw)
                order = None
            else:
                t, gp_reg, lin_reg, order = self._highres_inputs(name, np.asarray(t, float))
                try:
                    # does not work for occultations/phase curves with light_travel_delay=True
                    with _juliet_state(self.results.lc, name):
                        out = self.results.lc.evaluate(name, t=t, GPregressors=gp_reg, LMregressors=lin_reg, **kw)
                except Exception:
                    out = self._highres_workaround(name, t, gp_reg, lin_reg, post, nsamples)
        s, m, u, lo = (np.asarray(x) for x in out[:4])
        if order is not None:                 # back to the order of the times given
            inv = np.argsort(order)
            s, m, u, lo = s[:, inv], m[inv], u[inv], lo[inv]
        return dict(samples=s, median=m, upper=u, lower=lo)

    def _highres_inputs(self, name, t):
        """Times, interpolated GP and linear regressors for juliet at times ``t`` (sorted by the GP
        regressor, as juliet needs; ``order`` is that sorting or None)."""
        from scipy.interpolate import interp1d
        tim = self.dataset.times_lc[name]
        srt = np.argsort(tim)
        reg = self.regressors(name)
        gp_reg, order = None, None
        if reg['gp'] is not None:
            try:
                g = reg['gp'] if reg['gp'].ndim == 1 else reg['gp'][:, 0]
                gp_reg = interp1d(x=tim[srt], y=g[srt], kind='cubic')(t)
                order = np.argsort(gp_reg)
                t, gp_reg = t[order], gp_reg[order]
            except Exception:
                gp_reg, order = None, None
        lin_reg = None
        if reg['linear'] is not None:
            try:
                lin_reg = interp1d(x=tim, y=reg['linear'], kind='cubic', axis=0)(t)
            except Exception:
                lin_reg = None
        return t, gp_reg, lin_reg, order

    def _highres_workaround(self, name, t, gp_reg, lin_reg, post, nsamples):
        """High-resolution planet models where juliet's ``evaluate(t=...)`` fails
        (light_travel_delay=True for occultations and phase curves): a juliet fit of a dummy
        dataset at times ``t`` that loads the posteriors (in a scratch folder)."""
        import juliet
        ones = np.ones(len(t))
        gp_pars = None if gp_reg is None else {name: gp_reg}
        lin_pars = None if lin_reg is None else {name: lin_reg}
        tmp = tempfile.mkdtemp(prefix='jstark_highres_')
        try:
            for f in glob.glob(os.path.join(self.dataset.out_folder, '*posteriors.pkl')):
                shutil.copy(f, tmp)
            fit_kwargs = dict(self.fit_kwargs)
            fit_kwargs.setdefault('sampler', self.results.sampler)
            for key in ('light_travel_delay', 'stellar_radius'):
                if hasattr(self.results, key):
                    fit_kwargs.setdefault(key, getattr(self.results, key))
            data = juliet.load(priors=self.dataset.priors, t_lc={name: t}, y_lc={name: ones},
                               yerr_lc={name: ones}, GP_regressors_lc=gp_pars,
                               linear_regressors_lc=lin_pars, out_folder=tmp)
            res = data.fit(**fit_kwargs)
            return res.lc.evaluate(name, nsamples=nsamples,
                                   return_err=True, parameter_values=post, return_components=True,
                                   return_samples=True, evaluate_transit=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    def median_params(self, values=None):
        """``{name: value}`` at the posterior medians and the fixed values (``values``
        replacing some)."""
        pv = {k: float(np.nanmedian(v)) for k, v in self.posteriors.items() if np.ndim(v) == 1}
        for k, v in self.priors.items():
            if v['distribution'] == 'fixed' and k not in pv:
                pv[k] = float(np.atleast_1d(v['hyperparameters'])[0])
        for k, v in (values or {}).items():
            if k not in self.priors:
                raise KeyError(f'No parameter {k!r} in the fit.')
            pv[k] = float(v)
        return pv

    def non_linear(self, name):
        nlf = getattr(self.dataset, 'non_linear_functions', None) or {}
        if name not in nlf:
            return None, False
        pv = self.median_params()
        f = nlf[name]
        return np.asarray(f['function'](f['regressor'], pv)), bool(f.get('multiplicative', False))

    def orbit(self, planet='p1', name=None):
        """Posterior medians (or fixed values); derived ones from juliet's batman parameters.

        Fixed priors are preferred over batman's parameter objects: with a
        ``t_secondary`` prior, batman shifts ``params.t0`` to the conjunction
        implied by the eclipse time, which would hide an offset eclipse."""
        from .utils import inc_to_b, rho_to_ar
        name = self.names[0] if name is None else name
        post, pri = self.posteriors, self.priors

        def prm():
            return self.results.lc.model[name]['params']

        def get(k, fallback):
            try:
                return median_or_fixed(post, pri, k)
            except KeyError:
                return fallback()

        P = get(f'P_{planet}', lambda: prm().per)
        out = dict(P=P, t0=get(f't0_{planet}', lambda: prm().t0), p=get(f'p_{planet}', lambda: prm().rp))
        if 'rho' in pri:
            out['a'] = get('rho', lambda: None)
            out['a'] = prm().a if out['a'] is None else rho_to_ar(out['a'], P)
        else:
            out['a'] = get(f'a_{planet}', lambda: prm().a)
        out['b'] = get(f'b_{planet}', lambda: inc_to_b(inc=prm().inc, ar=out['a'], ecc=prm().ecc,
                                                         omega=prm().w))
        try:
            pr = prm()
        except KeyError:                       # no planet model in this dataset
            out.update(ecc=None, w=None, eclipse_phase=None)
            return out
        out['ecc'], out['w'] = pr.ecc, pr.w
        tsec = getattr(pr, 't_secondary', None)
        out['eclipse_phase'] = (float((tsec - pr.t0) / pr.per % 1.)
                                if tsec is not None and np.isfinite(tsec) else None)
        return out

    def gp_predict(self, name, x, resid, err, x_pred, params=None):
        """juliet's GP object of the dataset at the posterior medians (``params``: ``{prior
        name: value}`` replacing some; or, as before, the GP object's own celerite/george
        parameter vector, used as given).

        ``err`` includes ``sigma_w``, so the jitter term of juliet's GP object is switched
        off (``sigma_w`` = 1e-30); with both, ``sigma_w`` would count twice."""
        nm = self.dataset.lc_options[name]['noise_model']
        gp = nm.GP
        if params is None or isinstance(params, dict):
            pv = self.median_params(params)
            pv[f'sigma_w_{name}'] = 1e-30
            nm.set_parameter_vector(pv)
        else:
            gp.set_parameter_vector(params)
        gp.compute(t=x, yerr=err)
        mean, var = gp.predict(y=resid, t=x_pred, return_var=True)
        return mean, np.sqrt(var)

    # ------------------------------------------------------------------
    def rv_data(self, name):
        d = self.dataset
        return tuple(np.asarray(getattr(d, k)[name], float) for k in ('times_rv', 'data_rv', 'errors_rv'))

    def rv(self, name, nsamples=1000):
        return np.asarray(self.results.rv.evaluate(name, nsamples=nsamples), float)

    @property
    def rv_ta(self):
        ta = getattr(self.results, 'ta', None)
        return self.results.rv.ta if ta is None else ta

    @staticmethod
    def keplerian(t, P, t0, ecc, w, K):
        """radvel's Keplerian (juliet's RV model)."""
        import radvel
        P, t0, ecc, w, K = (np.atleast_1d(np.asarray(v, float)) for v in (P, t0, ecc, w, K))
        t = np.asarray(t, float)
        out = np.empty((P.size, t.shape[-1]))
        for i in range(P.size):
            tp = radvel.orbit.timetrans_to_timeperi(t0[i], P[i], ecc[i], w[i])
            ti = t[i] if t.ndim == 2 else t
            out[i] = radvel.kepler.rv_drive(ti, np.array([P[i], tp, ecc[i], w[i], K[i]]))
        return out

    def index(self, name):
        return None
