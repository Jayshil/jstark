"""juliet priors in JAX: transforms to an unconstrained space and log-densities.

The distributions are the ones juliet's nested samplers draw from (its prior
transforms), so a JAX fit samples the same prior as a dynesty fit:

==================  ====================  ===================================
juliet name         hyperparameters       distribution
==================  ====================  ===================================
``uniform``         ``a, b``              uniform on (a, b)
``loguniform``,     ``a, b``              log-uniform on (a, b)
``jeffreys``
``normal``          ``mu, sigma``         normal
``truncatednormal`` ``mu, sigma, a, b``   normal truncated to (a, b)
``beta``            ``a, b``              beta(a, b) on (0, 1)
``exponential``     ``a``                 gamma with shape ``a`` and scale 1
                                          (juliet uses ``scipy.stats.gamma``)
``modjeffreys``     ``turn, hi``          density 1 / (x + turn) on (0, hi)
``fixed``           ``value``             not sampled
==================  ====================  ===================================

Bounded parameters are mapped from the real line with a logistic function,
positive ones with an exponential, normal ones linearly (``x = mu + sigma
z``), so the sampler works on ``z`` without hard walls.
"""
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, ndtr

__all__ = ['PriorSet', 'KINDS']

# distribution -> number of hyperparameters
KINDS = {'uniform': 2, 'loguniform': 2, 'normal': 2, 'truncatednormal': 4, 'beta': 2,
         'exponential': 1, 'modjeffreys': 2}
_ALIASES = {'jeffreys': 'loguniform'}
_LOG_SQRT_2PI = 0.5 * np.log(2. * np.pi)


def _log_sigmoid_jac(z):
    """log(s (1 - s)) of the logistic s = sigmoid(z), stable for large |z|."""
    return -jax.nn.softplus(z) - jax.nn.softplus(-z)


def _logit(u):
    u = np.clip(u, 1e-12, 1. - 1e-12)
    return float(np.log(u) - np.log1p(-u))


class PriorSet:
    """The free and fixed parameters of a fit, with their priors.

    Parameters
    ----------
    priors : dict
        juliet priors ``{name: {'distribution': ..., 'hyperparameters': ...}}``
        (the dict of :func:`jstark.fitting.priors.build_priors`). The order
        of the free parameters is the order of this dict, as in juliet.
    """

    def __init__(self, priors):
        self.free, self.kinds, self.fixed = [], [], {}
        hyper = []
        for name, p in priors.items():
            kind = str(p['distribution']).lower()
            kind = _ALIASES.get(kind, kind)
            h = np.atleast_1d(np.asarray(p['hyperparameters'], float))
            if kind == 'fixed':
                self.fixed[name] = float(h[0])
                continue
            if kind not in KINDS:
                raise ValueError(f'Prior {name}: distribution {kind!r} is not available in the JAX '
                                 f'fits (use one of {sorted(KINDS) + ["fixed", "jeffreys"]}).')
            if h.size != KINDS[kind]:
                raise ValueError(f'Prior {name}: {kind} needs {KINDS[kind]} hyperparameters, got {h.size}.')
            self._check(name, kind, h)
            self.free.append(name)
            self.kinds.append(kind)
            hyper.append(np.pad(h, (0, 4 - h.size)))
        self.ndim = len(self.free)
        # (ndim, 4): an array, so priors can differ between batched fits of the same structure
        self.hyper = np.asarray(hyper, float).reshape(self.ndim, 4)
        self.index = {n: i for i, n in enumerate(self.free)}

    @staticmethod
    def _check(name, kind, h):
        if kind in ('uniform', 'truncatednormal', 'modjeffreys', 'loguniform'):
            lo, hi = (h[2], h[3]) if kind == 'truncatednormal' else (h[0], h[1])
            if kind == 'modjeffreys':
                lo = 0.
            if not hi > lo:
                raise ValueError(f'Prior {name}: empty range {h.tolist()}.')
        if kind == 'loguniform' and not h[0] > 0:
            raise ValueError(f'Prior {name}: log-uniform needs a positive lower limit.')
        if kind in ('normal', 'truncatednormal') and not h[1] > 0:
            raise ValueError(f'Prior {name}: sigma must be positive.')
        if kind in ('beta', 'modjeffreys') and not np.all(h[:2] > 0):
            raise ValueError(f'Prior {name}: hyperparameters must be positive.')
        if kind == 'exponential' and not h[0] > 0:
            raise ValueError(f'Prior {name}: the shape must be positive.')

    # ------------------------------------------------------------------
    def constrain(self, z, hyper=None):
        """Parameter values ``x`` and ``log|dx/dz|`` from unconstrained ``z`` (shape ``(ndim,)``)."""
        h = self.hyper if hyper is None else hyper
        xs, lj = [], 0.
        for i, kind in enumerate(self.kinds):
            zi, (a, b, c, d) = z[i], (h[i, 0], h[i, 1], h[i, 2], h[i, 3])
            if kind == 'uniform':
                x = a + (b - a) * jax.nn.sigmoid(zi)
                lj = lj + jnp.log(b - a) + _log_sigmoid_jac(zi)
            elif kind == 'loguniform':
                la, lb = jnp.log(a), jnp.log(b)
                lx = la + (lb - la) * jax.nn.sigmoid(zi)
                x = jnp.exp(lx)
                lj = lj + jnp.log(lb - la) + _log_sigmoid_jac(zi) + lx
            elif kind == 'normal':
                x = a + b * zi
                lj = lj + jnp.log(b)
            elif kind == 'truncatednormal':
                x = c + (d - c) * jax.nn.sigmoid(zi)
                lj = lj + jnp.log(d - c) + _log_sigmoid_jac(zi)
            elif kind == 'beta':
                x = jax.nn.sigmoid(zi)
                lj = lj + _log_sigmoid_jac(zi)
            elif kind == 'exponential':
                x = jnp.exp(zi)
                lj = lj + zi
            elif kind == 'modjeffreys':
                x = b * jax.nn.sigmoid(zi)
                lj = lj + jnp.log(b) + _log_sigmoid_jac(zi)
            xs.append(x)
        return (jnp.stack(xs) if xs else jnp.zeros(0)), lj

    def log_prior(self, x, hyper=None):
        """Log prior density at ``x`` (normalised; -inf outside the support)."""
        h = self.hyper if hyper is None else hyper
        out = 0.
        for i, kind in enumerate(self.kinds):
            xi, (a, b, c, d) = x[i], (h[i, 0], h[i, 1], h[i, 2], h[i, 3])
            if kind == 'uniform':
                lp = jnp.where((xi > a) & (xi < b), -jnp.log(b - a), -jnp.inf)
            elif kind == 'loguniform':
                ok = (xi > a) & (xi < b)
                lp = jnp.where(ok, -jnp.log(jnp.where(ok, xi, 1.)) - jnp.log(jnp.log(b) - jnp.log(a)),
                               -jnp.inf)
            elif kind == 'normal':
                lp = -0.5 * ((xi - a) / b) ** 2 - jnp.log(b) - _LOG_SQRT_2PI
            elif kind == 'truncatednormal':
                norm = ndtr((d - a) / b) - ndtr((c - a) / b)
                lp = jnp.where((xi > c) & (xi < d),
                               -0.5 * ((xi - a) / b) ** 2 - jnp.log(b) - _LOG_SQRT_2PI - jnp.log(norm),
                               -jnp.inf)
            elif kind == 'beta':
                ok = (xi > 0) & (xi < 1)
                xs = jnp.where(ok, xi, 0.5)
                lp = jnp.where(ok, (a - 1) * jnp.log(xs) + (b - 1) * jnp.log1p(-xs)
                               - (gammaln(a) + gammaln(b) - gammaln(a + b)), -jnp.inf)
            elif kind == 'exponential':
                ok = xi > 0
                xs = jnp.where(ok, xi, 1.)
                lp = jnp.where(ok, (a - 1) * jnp.log(xs) - xs - gammaln(a), -jnp.inf)
            elif kind == 'modjeffreys':
                ok = (xi > 0) & (xi < b)
                lp = jnp.where(ok, -jnp.log(jnp.where(ok, xi, 0.) + a) - jnp.log(jnp.log((a + b) / a)),
                               -jnp.inf)
            out = out + lp
        return out

    def log_density(self, z, hyper=None):
        """Log prior density of the unconstrained ``z`` (prior at ``x(z)`` plus the Jacobian)."""
        x, lj = self.constrain(z, hyper)
        return self.log_prior(x, hyper) + lj, x

    # ------------------------------------------------------------------
    def unconstrain(self, x):
        """``z`` from parameter values (NumPy; for starting points)."""
        z = np.zeros(self.ndim)
        for i, kind in enumerate(self.kinds):
            a, b, c, d = self.hyper[i]
            xi = float(x[i])
            if kind == 'uniform':
                z[i] = _logit((xi - a) / (b - a))
            elif kind == 'loguniform':
                z[i] = _logit((np.log(xi) - np.log(a)) / (np.log(b) - np.log(a)))
            elif kind == 'normal':
                z[i] = (xi - a) / b
            elif kind == 'truncatednormal':
                z[i] = _logit((xi - c) / (d - c))
            elif kind == 'beta':
                z[i] = _logit(xi)
            elif kind == 'exponential':
                z[i] = np.log(xi)
            elif kind == 'modjeffreys':
                z[i] = _logit(xi / b)
        return z

    def centre(self):
        """A typical value of every free parameter (median or the middle of the range)."""
        out = np.zeros(self.ndim)
        for i, kind in enumerate(self.kinds):
            a, b, c, d = self.hyper[i]
            if kind == 'uniform':
                out[i] = 0.5 * (a + b)
            elif kind == 'loguniform':
                out[i] = np.sqrt(a * b)
            elif kind == 'normal':
                out[i] = a
            elif kind == 'truncatednormal':
                out[i] = min(max(a, c + 1e-3 * (d - c)), d - 1e-3 * (d - c))
            elif kind == 'beta':
                out[i] = a / (a + b)
            elif kind == 'exponential':
                out[i] = max(a - 1., 0.5 * a)
            elif kind == 'modjeffreys':
                out[i] = 0.5 * b
        return out

    def sample(self, rng, n):
        """``n`` draws from the prior (NumPy; ``(n, ndim)``)."""
        from scipy import stats
        out = np.zeros((n, self.ndim))
        for i, kind in enumerate(self.kinds):
            a, b, c, d = self.hyper[i]
            u = rng.uniform(size=n)
            if kind == 'uniform':
                out[:, i] = a + (b - a) * u
            elif kind == 'loguniform':
                out[:, i] = np.exp(np.log(a) + u * (np.log(b) - np.log(a)))
            elif kind == 'normal':
                out[:, i] = stats.norm.ppf(u, a, b)
            elif kind == 'truncatednormal':
                out[:, i] = stats.truncnorm.ppf(u, (c - a) / b, (d - a) / b, loc=a, scale=b)
            elif kind == 'beta':
                out[:, i] = stats.beta.ppf(u, a, b)
            elif kind == 'exponential':
                out[:, i] = stats.gamma.ppf(u, a)
            elif kind == 'modjeffreys':
                out[:, i] = a * (np.exp(u * np.log(b / a + 1)) - 1)
        return out

    def as_dict(self, x, fixed=None):
        """``{name: value}`` of the free parameters ``x`` and all fixed ones (``fixed``: values
        replacing the stored ones, e.g. traced arrays of a batched fit)."""
        out = dict(self.fixed)
        if fixed is not None:
            out.update(fixed)
        for i, n in enumerate(self.free):
            out[n] = x[i]
        return out
