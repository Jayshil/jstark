"""Transit, eclipse and phase-curve models in JAX, matching batman and juliet.

* Quadratic and linear limb darkening: Agol et al. (2020), via
  ``jaxoplanet.core.limb_dark`` (to 1e-4 ppm), as batman's analytic Mandel &
  Agol model.
* Other laws (``squareroot``, ``logarithmic``, ``exponential``, ``power2``,
  ``nonlinear``): the occulted flux is integrated over the stellar radius
  with Gauss-Legendre quadrature (batman integrates numerically too, to
  1 ppm by default).
* Eclipses: the uniform planet disk occulted by the star (batman
  ``secondary``): ``1 + fp * visible fraction``.
"""
import numpy as np

import jax.numpy as jnp
from jaxoplanet.core.limb_dark import light_curve as _poly_ld

__all__ = ['LD_LAWS', 'ld_coefficients', 'transit_flux', 'eclipse_visible', 'overlap_area',
           'cowan_agol', 'sine_phase_curve']

# law -> number of coefficients (batman names; 'power-2' is accepted as 'power2')
LD_LAWS = {'linear': 1, 'quadratic': 2, 'squareroot': 2, 'logarithmic': 2, 'exponential': 2,
           'power2': 2, 'nonlinear': 4, 'uniform': 0}
_KIPPING = ('quadratic', 'squareroot', 'logarithmic', 'linear')
_NQ = 32            # quadrature nodes per stellar-radius integral (error < 1e-3 ppm)
_NNORM = 256        # nodes for the disk-integrated intensity
# Gauss-Legendre order of jaxoplanet's linear-term integral: <= 1e-4 ppm from the exact value
# (batman's own quadratic model is good to ~0.01 ppm); order 40 (jaxoplanet's default) is 30% slower
_POLY_ORDER = 20


def ld_coefficients(law, values, parametrization):
    """batman coefficients from juliet's parameters.

    ``values`` are (q1, q2) for ``parametrization='kipping2013'`` (Kipping
    2013, as juliet's ``reverse_ld_coeffs``), (u1, u2) or (c1, ..., c4) for
    ``'normal'``; one value for the linear law.
    """
    if law == 'linear':
        return [values[0]]
    if parametrization == 'kipping2013':
        q1, q2 = values
        s = jnp.sqrt(q1)
        if law == 'quadratic':
            return [2. * s * q2, s * (1. - 2. * q2)]
        if law == 'squareroot':
            return [s * (1. - 2. * q2), 2. * s * q2]
        if law == 'logarithmic':
            return [1. - s * q2, 1. - s]
        raise ValueError(f'juliet has no q1/q2 parametrisation of the {law} law; give u1, u2 priors.')
    return list(values)


def intensity(law, u, mu):
    """Normalised intensity profile I(mu) / I(1) of batman's laws."""
    if law in ('quadratic', 'linear'):
        u1 = u[0]
        u2 = u[1] if law == 'quadratic' else 0.
        return 1. - u1 * (1. - mu) - u2 * (1. - mu) ** 2
    if law == 'squareroot':
        return 1. - u[0] * (1. - mu) - u[1] * (1. - jnp.sqrt(mu))
    if law == 'logarithmic':
        mus = jnp.where(mu > 0, mu, 1.)
        return 1. - u[0] * (1. - mu) - u[1] * jnp.where(mu > 0, mus * jnp.log(mus), 0.)
    if law == 'exponential':
        return 1. - u[0] * (1. - mu) - u[1] / (1. - jnp.exp(mu))
    if law == 'power2':
        return 1. - u[0] * (1. - mu ** u[1])
    if law == 'nonlinear':
        return (1. - u[0] * (1. - mu ** 0.5) - u[1] * (1. - mu) - u[2] * (1. - mu ** 1.5)
                - u[3] * (1. - mu ** 2))
    if law == 'uniform':
        return jnp.ones_like(mu)
    raise ValueError(f'Unknown limb-darkening law {law!r}.')


def _safe(x, ok, fill):
    return jnp.where(ok, x, fill)


def overlap_area(z, p, r=1.):
    """Overlap area of a disk of radius ``r`` (centre 0) and one of radius ``p`` at distance ``z``."""
    z, p, r = jnp.asarray(z), jnp.abs(p), jnp.asarray(r)
    disjoint = z >= r + p
    inner = z <= jnp.abs(r - p)                      # one disk inside the other
    part = ~(disjoint | inner)
    zs = _safe(z, part, 1.)
    rs = _safe(r, part, 1.)
    ps = _safe(p, part, 0.5)
    k0 = jnp.arccos(jnp.clip((ps ** 2 + zs ** 2 - rs ** 2) / (2. * ps * zs), -1., 1.))
    k1 = jnp.arccos(jnp.clip((rs ** 2 - ps ** 2 + zs ** 2) / (2. * rs * zs), -1., 1.))
    q = 4. * zs ** 2 * rs ** 2 - (rs ** 2 + zs ** 2 - ps ** 2) ** 2
    lens = ps ** 2 * k0 + rs ** 2 * k1 - 0.5 * jnp.sqrt(jnp.where(part & (q > 0), q, 0.))
    small = jnp.pi * jnp.minimum(r, p) ** 2
    return jnp.where(disjoint, 0., jnp.where(inner, small, lens))


def eclipse_visible(z, p):
    """Visible fraction of the planet disk (radius ``p``) at separation ``z`` from the star."""
    p = jnp.abs(p)
    return 1. - overlap_area(z, p) / (jnp.pi * p ** 2)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(_NQ)
_GN_X, _GN_W = np.polynomial.legendre.leggauss(_NNORM)


def _cos_map(lo, hi):
    """Nodes and weights on (lo, hi) after r = lo + (hi - lo)(1 - cos theta) / 2 (square-root ends)."""
    th = 0.5 * np.pi * (_GL_X + 1.)
    w = 0.5 * np.pi * _GL_W
    r = lo[..., None] + (hi - lo)[..., None] * 0.5 * (1. - jnp.cos(th))
    dr = (hi - lo)[..., None] * 0.5 * jnp.sin(th) * w
    return r, dr


def _numeric_deficit(law, u, z, p):
    """Fraction of the stellar flux occulted by a planet of radius ``p`` at separation ``z``."""
    p = jnp.abs(p)
    # disk-integrated intensity: 2 pi int_0^1 I(mu) mu dmu
    mu_n = 0.5 * (_GN_X + 1.)
    total = 2. * jnp.pi * jnp.sum(0.5 * _GN_W * intensity(law, u, mu_n) * mu_n)

    def I_of_r(r):
        return intensity(law, u, jnp.sqrt(jnp.clip(1. - r ** 2, 0., 1.)))

    # annuli fully covered: r < p - z
    full_hi = jnp.clip(p - z, 0., 1.)
    r, dr = _cos_map(jnp.zeros_like(z), full_hi)
    full = jnp.sum(I_of_r(r) * 2. * jnp.pi * r * dr, axis=-1)
    # annuli partly covered: |z - p| < r < min(1, z + p); covered arc 2 r arccos(...)
    lo = jnp.clip(jnp.abs(z - p), 0., 1.)
    hi = jnp.clip(z + p, 0., 1.)
    hi = jnp.maximum(hi, lo)
    r, dr = _cos_map(lo, hi)
    zz = z[..., None]
    ok = (r > 0) & (zz > 0)
    arg = (r ** 2 + zz ** 2 - p ** 2) / (2. * _safe(r, ok, 1.) * _safe(zz, ok, 1.))
    kappa = jnp.where(ok, jnp.arccos(jnp.clip(arg, -1., 1.)), 0.)
    partial = jnp.sum(I_of_r(r) * 2. * r * kappa * dr, axis=-1)
    deficit = (full + partial) / total
    return jnp.where(z >= 1. + p, 0., deficit)


def transit_flux(law, u, z, p):
    """Relative stellar flux during a transit (batman ``primary``; ``p < 0``: inverse transit)."""
    pa = jnp.abs(p)
    if law in ('quadratic', 'linear'):
        uu = jnp.stack([u[0], u[1] if law == 'quadratic' else jnp.zeros_like(u[0])])
        lc = 1. + _poly_ld(uu, z, pa, order=_POLY_ORDER)
    elif law == 'uniform':
        lc = 1. - overlap_area(z, pa) / jnp.pi
    else:
        lc = 1. - _numeric_deficit(law, u, z, pa)
    return jnp.where(p < 0, 2. - lc, lc)


def cowan_agol(t, t_secondary, per, fp, C1, D1, C2, D2):
    """juliet's Cowan & Agol (2008) planet flux (equal to ``fp`` at the eclipse)."""
    wt = 2. * jnp.pi * (t - t_secondary) / per
    return fp + C1 * (jnp.cos(wt) - 1.) + D1 * jnp.sin(wt) + C2 * (jnp.cos(2 * wt) - 1.) + D2 * jnp.sin(2 * wt)


def sine_phase_curve(t, t0, per, fp, offset_deg):
    """juliet's sinusoidal phase curve (``phaseoffset``): 0 at the transit, ``fp`` at the eclipse."""
    phase = ((t - t0) / per) % 1.
    return fp * 0.5 * (jnp.sin(2. * jnp.pi * phase - jnp.pi / 2. + jnp.radians(offset_deg)) + 1.)
