"""Posterior models of a JAX fit (the interface of :mod:`jstark.fitting.evaluators`)."""
import numpy as np

import jax
import jax.numpy as jnp

from . import gp as _gp, planet as _planet, orbit as _orbit

__all__ = ['JaxEvaluator']

_ELEMENTS = 2e6        # model elements per vmapped chunk (bounds the memory)


def _quantiles(s):
    """Median and 68% interval (upper, lower) over the draws (axis 0)."""
    m, u, lo = np.nanpercentile(s, [50., 84., 16.], axis=0)
    return m, u, lo


class JaxEvaluator:
    """Data and posterior models of a :class:`~jstark.fitting.jaxfit.JaxModel` fit.

    Posterior draws are picked at random without replacement (as juliet),
    with a fixed ``seed``, so repeated calls give the same models.
    """

    kind = 'jax'

    def __init__(self, model, results, index=None, seed=0):
        self.model, self.results = model, results
        self.names, self.rv_names = list(model.names), list(model.rv_names)
        self._index = dict(index or {})
        self.seed = seed
        self._fn = {}

    @property
    def posteriors(self):
        return self.results.posteriors['posterior_samples']

    @property
    def priors(self):
        return self.model.priors

    def data(self, name):
        d = self.model.data[name]
        return tuple(np.asarray(d[k], float) for k in ('t', 'y', 'e'))

    def regressors(self, name):
        d = self.model.data[name]
        X = np.asarray(d['X'])
        return dict(linear=X if X.shape[1] else None, gp=np.asarray(d['G']) if 'G' in d else None)

    def fit_type(self, name):
        c = self.model.config[name]
        return 'phasecurve' if c['traneclipse'] else ('eclipse' if c['eclipse'] else 'transit')

    def median_params(self, values=None):
        """Parameter dict at the posterior medians (``values``: {name: value} replacing some)."""
        x = np.nanmedian(self.results.samples.reshape(-1, self.results.samples.shape[-1]), axis=0)
        pv = {k: jnp.asarray(v) for k, v in self.model.params(jnp.asarray(x)).items()}
        for k, v in (values or {}).items():
            if k not in pv:
                raise KeyError(f'No parameter {k!r} in the fit.')
            pv[k] = jnp.asarray(float(v))
        return pv

    def index(self, name):
        i = self._index.get(name)
        return None if i is None else np.asarray(i)

    def draws(self, nsamples=1000):
        """``(n, ndim)`` posterior draws (all of them if ``nsamples`` is None or larger)."""
        x = self.results.samples.reshape(-1, self.results.samples.shape[-1])
        if nsamples is None or nsamples >= len(x):
            return x
        idx = np.random.default_rng(self.seed).choice(len(x), int(nsamples), replace=False)
        return x[np.sort(idx)]

    def _chunk(self, name):
        d, cfg = self.model.data[name], self.model.config[name]
        n = int(d['t'].size) * (int(cfg['supersample'][0]) if cfg['supersample'] else 1)
        if cfg['law'] not in (None, 'quadratic', 'linear', 'uniform'):
            n *= 2 * _planet._NQ
        if cfg['gp'] is not None and cfg['gp'] not in _gp.QUASISEP:
            n *= int(d['t'].size)              # dense GP: N x N per draw
        return max(1, int(_ELEMENTS // max(n, 1)))

    @staticmethod
    def _map(f, x, b):
        """``f`` (vmapped, jitted) over the draws ``x`` in chunks of ``b``."""
        x = np.atleast_2d(np.asarray(x, float))
        outs = []
        for i in range(0, len(x), b):
            xc = x[i:i + b]
            k = len(xc)
            if k < b and len(x) > b:          # same shape for every chunk: compile once
                xc = np.concatenate([xc, np.repeat(xc[-1:], b - k, axis=0)])
            outs.append([np.asarray(o)[:k] for o in f(jnp.asarray(xc))])
        return [np.concatenate([o[j] for o in outs]) for j in range(len(outs[0]))]

    def components(self, name, x):
        """``(n, npoints)`` models of one dataset for draws ``x``: ``model`` (deterministic
        model plus the GP prediction, juliet's full model), ``planet``, ``gp``."""
        if name not in self._fn:
            m = self.model

            def one(q):
                pv = m.params(q)
                c = m.components(pv, name)
                g = m.gp_prediction(pv, name, comp=c)
                return c['model'] + g, c['planet'], g, c['linear']

            self._fn[name] = jax.jit(jax.vmap(one))
        mm, pp, gg, ll = self._map(self._fn[name], x, self._chunk(name))
        return dict(model=mm, planet=pp, gp=gg, linear=ll)

    def lc(self, name, nsamples=1000, components=False):
        c = self.components(name, self.draws(nsamples))
        out = dict(model=np.nanmedian(c['model'], axis=0))
        if components:
            out['planet'] = np.nanmedian(c['planet'], axis=0)
            out['gp'] = np.nanmedian(c['gp'], axis=0)
        return out

    def lc_models(self, name, nsamples=1000):
        c = self.components(name, self.draws(nsamples))
        med, up, lo = _quantiles(c['model'])
        reg = self.regressors(name)
        out = dict(samples=c['model'], median=med, upper=up, lower=lo,
                   linear=np.nanmedian(c['linear'], axis=0) if reg['linear'] is not None else None)
        if reg['gp'] is not None:
            out['gp'], out['gp_upper'], out['gp_lower'] = _quantiles(c['gp'])
        return out

    def planet_models(self, name, nsamples=1000, t=None):
        x = self.draws(nsamples)
        if t is None:
            s = self.components(name, x)['planet']
        else:
            t = np.asarray(t, float)
            key = ('planet', name, t.size)
            if key not in self._fn:
                m = self.model
                self._fn[key] = jax.jit(jax.vmap(lambda q, tt: (m.planet_model(m.params(q), name, tt),),
                                                 in_axes=(0, None)))
            f = self._fn[key]
            n = t.size * (int(self.model.config[name]['supersample'][0])
                          if self.model.config[name]['supersample'] else 1)
            if self.model.config[name]['law'] not in (None, 'quadratic', 'linear', 'uniform'):
                n *= 2 * _planet._NQ
            tj = jnp.asarray(t)
            (s,) = self._map(lambda xc: f(xc, tj), x, max(1, int(_ELEMENTS // max(n, 1))))
        med, up, lo = _quantiles(s)
        return dict(samples=s, median=med, upper=up, lower=lo)

    def non_linear(self, name):
        pv = self.median_params()
        nl = self.model._non_linear(pv, name, self.model.data[name])
        if nl is None:
            return None, False
        return np.asarray(nl, float), bool(self.model.config[name]['multiplicative'])

    def orbit(self, planet='p1', name=None):
        """Posterior medians of the orbit of a planet (derived quantities per draw, then the
        median), in the model of dataset ``name`` (default: the first with that planet)."""
        n = int(planet[1:])
        names = [name] if name is not None else self.names
        name = next((k for k in names if any(q['n'] == n for q in self.model.config[k]['planets'])), None)
        if name is None:
            raise KeyError(f'No light-curve model of planet {planet}.')
        key = ('orbit', name, n)
        if key not in self._fn:
            m = self.model
            self._fn[key] = jax.jit(jax.vmap(lambda q: m.geometry(m.params(q), name, n)))
        g = {k: np.asarray(v) for k, v in self._fn[key](jnp.asarray(self.draws(None))).items()}
        out = {k: float(np.nanmedian(v)) for k, v in g.items() if k != 't_secondary'}
        ph = ((g['t_secondary'] - g['t0']) / g['P']) % 1.
        out['eclipse_phase'] = float(np.nanmedian(ph))
        return out

    def gp_predict(self, name, x, resid, err, x_pred, params=None):
        """GP of dataset ``name`` at the posterior medians; ``params``: ``{prior name: value}``
        replacing some of them (e.g. ``{'GP_rho_V1': 0.1}``)."""
        cfg = self.model.config[name]
        if cfg['gp'] is None:
            raise ValueError(f'Dataset {name} has no GP.')
        if params is not None and not isinstance(params, dict):
            raise TypeError('For JAX fits, give GP parameters as {prior name: value}.')
        pv = self.median_params(params)
        x, resid, err = (jnp.asarray(np.asarray(v, float)) for v in (x, resid, err))
        x_pred = np.asarray(x_pred, float)
        if x.ndim != x_pred.ndim:
            raise ValueError('x_pred must have the dimensions of the GP regressor x.')
        key = ('gp_predict', name)
        if key not in self._fn:
            self._fn[key] = jax.jit(lambda p, xx, r, e, xp: self.model.gp_predict(p, name, xx, r, e ** 2, xp))
        b = max(1, int(_ELEMENTS // max(int(x.shape[0]), 1)))
        mean, std = [], []
        for i in range(0, len(x_pred), b):
            xp = x_pred[i:i + b]
            k = len(xp)
            if k < b and len(x_pred) > b:
                xp = np.concatenate([xp, np.repeat(xp[-1:], b - k, axis=0)])
            mu, sd = self._fn[key](pv, x, resid, err, jnp.asarray(xp))
            mean.append(np.asarray(mu)[:k])
            std.append(np.asarray(sd)[:k])
        return np.concatenate(mean), np.concatenate(std)

    @property
    def rv_ta(self):
        return 0. if self.model.ta is None else self.model.ta

    @staticmethod
    def keplerian(t, P, t0, ecc, w, K):
        """Keplerian RVs of the JAX model (converged Kepler solver)."""
        P, t0, ecc, w, K = (jnp.atleast_1d(jnp.asarray(v, float)) for v in (P, t0, ecc, w, K))
        t = jnp.asarray(t, float)
        f = jax.vmap(_orbit.rv_keplerian, in_axes=(0 if t.ndim == 2 else None, 0, 0, 0, 0, 0))
        return np.asarray(jax.jit(f)(t, t0, P, K, ecc, w))

    def rv_data(self, name):
        d = self.model.rv_data[name]
        return tuple(np.asarray(d[k], float) for k in ('t', 'rv', 'err'))

    def rv(self, name, nsamples=1000):
        """Median RV model (offset, trend, Keplerians) of one instrument."""
        key = ('rv', name)
        if key not in self._fn:
            m = self.model
            self._fn[key] = jax.jit(jax.vmap(lambda q: (m.rv_components(m.params(q), name)['model'],)))
        (mm,) = self._map(self._fn[key], self.draws(nsamples), 2000)
        return np.nanmedian(mm, axis=0)
