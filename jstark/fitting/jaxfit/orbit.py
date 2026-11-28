"""Orbits with batman's conventions, in JAX.

batman (and so juliet) places the planet with ``t0`` = time of inferior
conjunction (``t_secondary`` for eclipse models), ``omega`` = argument of
periastron in degrees, and returns the sky-projected separation in stellar
radii, set to 100 on the half of the orbit where the event cannot happen
(planet behind the star for transits, in front of it for eclipses).
"""
import jax.numpy as jnp
from jaxoplanet.core.kepler import kepler

__all__ = ['G', 'conjunction_phase', 't_periastron', 't_secondary_from_t0',
           't0_from_t_secondary', 'sky_separation', 'a_from_rho', 'reverse_bp',
           'rv_keplerian', 'light_travel_times']

G = 6.67408e-11          # juliet's value (m^3 kg^-1 s^-2)
_FAR = 100.              # batman's separation outside the half-orbit of the event
_CIRCULAR = 1e-5         # batman: true anomaly = mean anomaly below this eccentricity


def _mean_anomaly_at(ta, ecc):
    """Mean anomaly of the true anomaly ``ta`` (batman's ``_get_phase`` before dividing by 2 pi)."""
    E = 2. * jnp.arctan(jnp.sqrt((1. - ecc) / (1. + ecc)) * jnp.tan(ta / 2.))
    return E - ecc * jnp.sin(E)


def conjunction_phase(ecc, w_rad, position='primary'):
    """Orbital phase (0-1, from periastron) of the transit or the eclipse centre."""
    ta = (jnp.pi / 2. - w_rad) if position == 'primary' else (3. * jnp.pi / 2. - w_rad)
    return _mean_anomaly_at(ta, ecc) / (2. * jnp.pi)


def t_periastron(t0, per, ecc, w_rad):
    return t0 - per * conjunction_phase(ecc, w_rad, 'primary')


def t_secondary_from_t0(t0, per, ecc, w_rad):
    """batman ``get_t_secondary``."""
    return t0 + per * (conjunction_phase(ecc, w_rad, 'secondary') - conjunction_phase(ecc, w_rad, 'primary'))


def t0_from_t_secondary(t_sec, per, ecc, w_rad):
    """batman ``get_t_conjunction``."""
    return t_sec + per * (conjunction_phase(ecc, w_rad, 'primary') - conjunction_phase(ecc, w_rad, 'secondary'))


def _safe_sqrt(x):
    ok = x > 0
    return jnp.where(ok, jnp.sqrt(jnp.where(ok, x, 1.)), 0.)


def sky_separation(t, t0, per, a, cosi, ecc, w_rad, secondary=False):
    """Separation of the centres in stellar radii at times ``t`` (batman ``_rsky``).

    ``t0`` is the conjunction time of the event (the transit for
    ``secondary=False``, the eclipse otherwise: batman computes the eclipse
    orbit from ``t_secondary``). Returns ``(d, true anomaly)``.
    """
    tp = t_periastron(t0, per, ecc, w_rad)
    M = 2. * jnp.pi / per * (t - tp)
    sinf, cosf = kepler(M, ecc)
    frac = (t - tp) / per
    fc = 2. * jnp.pi * (frac - jnp.trunc(frac))      # batman's circular branch (C int cast)
    circ = ecc <= _CIRCULAR
    sinf = jnp.where(circ, jnp.sin(fc), sinf)
    cosf = jnp.where(circ, jnp.cos(fc), cosf)
    r = a * (1. - ecc ** 2) / (1. + ecc * cosf)
    sw, cw = jnp.sin(w_rad), jnp.cos(w_rad)
    s_wf = sinf * cw + cosf * sw                       # sin(omega + f)
    c_wf = cosf * cw - sinf * sw                       # cos(omega + f)
    d = r * _safe_sqrt(c_wf ** 2 + (s_wf * cosi) ** 2)
    sini = _safe_sqrt(1. - cosi ** 2)
    front = s_wf * sini > 0.
    d = jnp.where(front != secondary, d, _FAR)
    return d, jnp.arctan2(sinf, cosf)


def a_from_rho(rho, per):
    """a/R* from the stellar density (kg/m^3) and the period (days), as juliet computes it."""
    return (rho * G * (per * 24. * 3600.) ** 2 / (3. * jnp.pi)) ** (1. / 3.)


def reverse_bp(r1, r2, pl, pu):
    """(b, p) from the Espinoza (2018) parameters r1, r2 (juliet's ``efficient_bp``)."""
    Ar = (pu - pl) / (2. + pl + pu)
    hi = r1 > Ar
    r1l = jnp.where(hi, Ar, r1)                         # keeps sqrt(r1/Ar) finite in the unused branch
    b = jnp.where(hi, (1 + pl) * (1. + (r1 - 1.) / (1. - Ar)),
                  (1. + pl) + jnp.sqrt(r1l / Ar) * r2 * (pu - pl))
    p = jnp.where(hi, (1 - r2) * pl + r2 * pu,
                  pu + (pl - pu) * jnp.sqrt(r1l / Ar) * (1. - r2))
    return b, p


C_LIGHT = 299792458.     # m/s (astropy)
R_SUN = 6.957e8          # m (astropy, IAU 2015 nominal)


def rv_keplerian(t, t0, per, K, ecc, w_rad):
    """Stellar radial velocity of one planet (radvel ``rv_drive`` with juliet's ``per tc e w k``).

    radvel clips the eccentricity to 0-0.99 and takes ``t0`` as the time of
    transit (periastron time from ``timetrans_to_timeperi``).
    """
    e = jnp.clip(ecc, 0., 0.99)
    tp = t_periastron(t0, per, e, w_rad)
    sinf, cosf = kepler(2. * jnp.pi / per * (t - tp), e)
    cos_fw = cosf * jnp.cos(w_rad) - sinf * jnp.sin(w_rad)
    return K * (cos_fw + e * jnp.cos(w_rad))


def light_travel_times(t, t0, per, a, sini, ecc, w_rad, stellar_radius):
    """juliet's ``correct_light_travel_time``: ``t`` minus the light-travel delay across the orbit.

    The delay is the line-of-sight position of the planet relative to its
    position at the transit, over c (a in stellar radii, the stellar radius
    in solar radii). juliet evaluates eclipse models (and phase curves) at
    these times.
    """
    a_m = a * stellar_radius * R_SUN
    tp = t_periastron(t0, per, ecc, w_rad)

    def los(tt):                          # Bell's KeplerOrbit x (Omega = 270 deg): r sin i sin(f + w)
        sinf, cosf = kepler(2. * jnp.pi / per * (tt - tp), ecc)
        r = a_m * (1. - ecc ** 2) / (1. + ecc * cosf)
        return r * sini * (sinf * jnp.cos(w_rad) + cosf * jnp.sin(w_rad))

    x0_c = a_m * sini
    x_c = x0_c * jnp.cos(2. * jnp.pi * (t - t0) / per)
    ecc_on = ecc > 0.
    x0 = jnp.where(ecc_on, los(t0), x0_c)
    x = jnp.where(ecc_on, los(t), x_c)
    return t - (x0 - x) / C_LIGHT / 86400.
