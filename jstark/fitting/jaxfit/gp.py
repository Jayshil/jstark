"""juliet's GP kernels of light curves, in tinygp.

The kernel of a dataset is chosen by the names of its ``GP_`` priors, as in
juliet (``juliet.fit.gaussian_process``), and parametrised as juliet
parametrises celerite and george:

=========================  ===============  ======================================
juliet kernel              variables        k(tau)
=========================  ===============  ======================================
CeleriteExpKernel          sigma, timescale ``sigma exp(-timescale tau)`` (juliet
                                            puts ``log sigma`` and ``log
                                            timescale`` into celerite's
                                            ``RealTerm(log_a, log_c)``)
CeleriteMaternKernel       sigma, rho       Matern-3/2 (celerite's
                                            ``Matern32Term``, which approximates
                                            it with ``eps = 0.01``)
CeleriteMaternExpKernel    sigma, timescale ``RealTerm(sigma, timescale) x
                           rho              Matern32Term(1, rho)``
CeleriteSHOKernel          S0, Q, omega0    SHO (``sigma^2 = S0 omega0 Q``)
CeleriteQPKernel           B, L, Prot, C    juliet's ``RotationTerm``
CeleriteDoubleSHOKernel    sigma, Q0,       two SHOs (juliet's formulas, its
                           period, f, dQ    ``omega2`` included)
CeleriteMaternSHOKernel    sigma, rho, S0,  Matern + SHO
                           Q, omega0
SEKernel                   sigma, alpha<i>  ``n (1e-6 sigma)^2 exp(-sum alpha_i
                                            dx_i^2 / 2)`` (george, dense; n =
                                            number of regressors: george's
                                            constant kernel in n dimensions)
M32Kernel                  sigma, malpha<i> george Matern-3/2 (dense)
ExpSineSquaredSEKernel     sigma, alpha,    ``(1e-6 sigma)^2 exp(-alpha dx^2 / 2)
                           Gamma, Prot      exp(-log(Gamma) sin^2(pi dx / Prot))``
                                            (juliet writes ``log Gamma`` into
                                            george's linear ``gamma``)
=========================  ===============  ======================================

The celerite kernels need a sorted 1D regressor (``BroadbandFit`` sorts
the dataset by it) and cost O(N); the george ones are dense, O(N^3), and
exact (juliet uses george's approximate HODLR solver by default).
"""
import numpy as np

import jax.numpy as jnp

__all__ = ['KERNELS', 'QUASISEP', 'find_kernel', 'build_kernel', 'MATERN_EPS']

MATERN_EPS = 0.01        # juliet's default matern_eps

# juliet's kernel variables (juliet.fit.gaussian_process.all_kernel_variables)
KERNELS = {
    'CeleriteQPKernel': ('B', 'L', 'Prot', 'C'),
    'CeleriteExpKernel': ('sigma', 'timescale'),
    'CeleriteMaternKernel': ('sigma', 'rho'),
    'CeleriteMaternExpKernel': ('sigma', 'timescale', 'rho'),
    'CeleriteSHOKernel': ('S0', 'Q', 'omega0'),
    'CeleriteDoubleSHOKernel': ('sigma', 'Q0', 'period', 'f', 'dQ'),
    'CeleriteMaternSHOKernel': ('sigma', 'rho', 'S0', 'Q', 'omega0'),
    'ExpSineSquaredSEKernel': ('sigma', 'alpha', 'Gamma', 'Prot'),
}
QUASISEP = tuple(k for k in KERNELS if k.startswith('Celerite'))


def _variables(nx):
    out = {'SEKernel': ('sigma',) + tuple(f'alpha{i}' for i in range(nx)),
           'M32Kernel': ('sigma',) + tuple(f'malpha{i}' for i in range(nx))}
    out.update(KERNELS)
    return out


def find_kernel(priors, inst, nx=1):
    """juliet's kernel of dataset ``inst`` and the prior name of each of its variables.

    Returns ``(kernel, {variable: prior name})``, or ``(None, {})`` without GP priors.
    """
    found = {}
    for k in priors:
        vec = k.split('_')
        if vec[0] == 'GP' and len(vec) > 2 and inst in vec[2:]:
            found.setdefault(vec[1], k)
    if not found:
        return None, {}
    for name, var in _variables(nx).items():
        if set(var) == set(found) and len(var) == len(found):
            return name, {v: found[v] for v in var}
    raise ValueError(f'Dataset {inst}: GP priors {sorted(found.values())} match none of juliet\'s '
                     f'kernels ({ {k: list(v) for k, v in _variables(nx).items()} }).')


def _matern(sigma, rho):
    """celerite (v1) ``Matern32Term(log sigma, log rho, eps=0.01)``.

    celerite approximates the Matern-3/2 kernel by ``s^2 exp(-w t) (cos(eps
    t) + w / eps sin(eps t))`` (w = sqrt(3) / rho), (eps t)^2 / 6 from the
    exact kernel. That celerite form loses precision in tinygp's
    factorisation for small rho (rho ~ the cadence: -inf), so the exact
    kernel is used: its log-likelihood equals celerite's to ~1e-11
    (relative) on 300-point visits.
    """
    from tinygp.kernels import quasisep as q
    return q.Matern32(scale=rho, sigma=sigma)


def _sho(S0, Q, w0):
    from tinygp.kernels import quasisep as q
    return q.SHO(omega=w0, quality=Q, sigma=jnp.sqrt(S0 * w0 * Q))


def build_kernel(name, p):
    """tinygp kernel of juliet kernel ``name`` for variable values ``p`` ({variable: value}).

    For the george kernels the regressor scaling is part of the input
    (:func:`scale_input`); the kernel returned works on scaled inputs.
    """
    from tinygp import kernels
    from tinygp.kernels import quasisep as q
    if name == 'CeleriteExpKernel':
        return q.Exp(scale=1. / p['timescale'], sigma=jnp.sqrt(p['sigma']))
    if name == 'CeleriteMaternKernel':
        return _matern(p['sigma'], p['rho'])
    if name == 'CeleriteMaternExpKernel':
        return q.Exp(scale=1. / p['timescale'], sigma=jnp.sqrt(p['sigma'])) * _matern(1., p['rho'])
    if name == 'CeleriteSHOKernel':
        return _sho(p['S0'], p['Q'], p['omega0'])
    if name == 'CeleriteQPKernel':
        B, L, P, C = p['B'], p['L'], p['Prot'], p['C']
        real = q.Exp(scale=L, sigma=jnp.sqrt(B * (1. + C) / (2. + C)))
        return real + q.Celerite(a=B / (2. + C), b=0., c=1. / L, d=2. * jnp.pi / P)
    if name == 'CeleriteDoubleSHOKernel':
        sigma, Q0, P, f, dQ = p['sigma'], p['Q0'], p['period'], p['f'], p['dQ']
        Q1 = 0.5 + Q0 + dQ
        w1 = 4. * jnp.pi * Q1 / (P * jnp.sqrt(4. * Q1 ** 2 - 1.))
        S1 = sigma ** 2 / ((1. + f) * w1 * Q1)
        Q2 = 0.5 + Q0
        w2 = 8. * jnp.pi * Q1 / (P * jnp.sqrt(4. * Q1 ** 2 - 1.))     # juliet uses Q1 here
        S2 = f * sigma ** 2 / ((1. + f) * w2 * Q2)
        return _sho(S1, Q1, w1) + _sho(S2, Q2, w2)
    if name == 'CeleriteMaternSHOKernel':
        return _matern(p['sigma'], p['rho']) + _sho(p['S0'], p['Q'], p['omega0'])
    amp = (1e-6 * p['sigma']) ** 2
    if name in ('SEKernel', 'M32Kernel'):
        # george's ConstantKernel in n dimensions is n exp(log_constant): juliet's amplitude is
        # n (1e-6 sigma)^2 for n regressors
        n = sum(1 for k in p if k.startswith('alpha') or k.startswith('malpha'))
        base = kernels.ExpSquared(scale=1.) if name == 'SEKernel' else kernels.Matern32(scale=1.)
        return n * amp * base
    if name == 'ExpSineSquaredSEKernel':
        return amp * kernels.ExpSquared(scale=1. / jnp.sqrt(p['alpha'])) * \
            kernels.ExpSineSquared(scale=p['Prot'], gamma=jnp.log(p['Gamma']))
    raise ValueError(f'Unknown GP kernel {name!r}.')


def scale_input(name, p, X):
    """The GP input of the george kernels with the metric applied (``sqrt(alpha_i) x_i``)."""
    if name in ('SEKernel', 'M32Kernel'):
        pre = 'alpha' if name == 'SEKernel' else 'malpha'
        X2 = X if X.ndim == 2 else X[:, None]
        s = jnp.sqrt(jnp.stack([p[f'{pre}{i}'] for i in range(X2.shape[1])]))
        return X2 * s
    return X


def check_sorted(name, X, inst):
    """celerite kernels need a monotonic 1D input (juliet's condition)."""
    if name in QUASISEP:
        X = np.asarray(X)
        if X.ndim != 1:
            raise ValueError(f'Dataset {inst}: kernel {name} takes a 1D GP regressor.')
        d = np.diff(X)
        if not (np.all(d >= 0) or np.all(d <= 0)):
            raise ValueError(f'Dataset {inst}: the GP regressor of {name} must be sorted.')
