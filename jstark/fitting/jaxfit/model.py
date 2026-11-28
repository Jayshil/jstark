"""The light-curve model and likelihood of a jstark fit, in JAX.

:class:`JaxModel` reads what a :class:`~jstark.fitting.BroadbandFit` hands to
juliet (data, regressors, complete priors, limb-darkening laws, ramps,
non-linear functions, supersampling) and evaluates the same model as juliet:

* per planet (``b_p<n>`` or ``r1_p<n>`` priors, in the order of the priors):
  a transit (limb-darkening priors name the dataset), an eclipse (an ``fp``
  prior applies), or both, with juliet's Cowan & Agol or sinusoidal phase
  curve; planets add up as ``M = 1 + sum(planet - 1)``;
* dilution and flux offset: ``(M D + 1 - D) / (1 + D mflux)``;
* linear model ``+ sum theta_i x_i``; non-linear function (exponential ramp
  and/or yours), multiplicative or additive;
* Gaussian likelihood with the jitter ``sigma_w`` (ppm) added to the errors.

Parameter vectors where juliet rejects the model (``b > 1 + p``, ``cos i >=
1``, eccentricity above ``ecclim``, periods not increasing with the planet
number) get a log-likelihood of -inf (juliet: -1e101).

With GP priors on a dataset, its residuals get juliet's GP (celerite or
george kernels, in tinygp; :mod:`.gp`) instead of the white-noise
likelihood. Radial velocities: offsets, jitters, Keplerian orbits (radvel's
conventions) and the ``rv_slope`` / ``rv_quad`` trends. ``light_travel_delay``
(with ``stellar_radius``) places eclipses and phase curves at the times
delayed by the light travel across the orbit, as juliet does.

juliet models jstark does not use (TTVs, catwoman, Lambertian and kelp
phase curves, global models, GPs and linear models on RVs) raise
``NotImplementedError``.
"""
import re

import numpy as np

import jax
import jax.numpy as jnp

from . import gp as _gp, orbit, planet
from .priors import PriorSet

__all__ = ['JaxModel']

_LOG2PI = np.log(2. * np.pi)
# variance of padded points in GP likelihoods of batched fits (relative flux^2): they decouple
_PAD_VARIANCE = 1e20
_UNSUPPORTED = (('dt', 'TTVs (dt_ priors)'), ('T', 'TTVs (T_ priors)'), ('p1', 'catwoman transits'),
                ('p2', 'catwoman transits'), ('phi', 'catwoman transits'),
                ('aglambert', 'Lambertian phase curves'), ('singlescat', 'kelp phase curves'),
                ('cml11', 'kelp phase curves'), ('agkelp', 'kelp phase curves'))


def _matches(key, prefix, inst):
    """juliet: parameter ``key`` of type ``prefix`` names instrument ``inst``."""
    return key[:len(prefix)] == prefix and inst in key.split('_')


def _planet_suffix(priors, prefix, inst, planet_tag):
    """juliet's ``<prefix>_iname[planet][instrument]`` ('' for a parameter shared by all)."""
    out = None
    for k in priors:
        if k[:len(prefix)] != prefix:
            continue
        vec = k.split('_')
        if vec[0] != prefix.rstrip('_') or len(vec) < 2 or vec[1] != planet_tag:
            continue
        if len(vec) > 2:
            if inst in vec:
                out = '_' + '_'.join(vec[2:])
        else:
            out = ''
    return out


def _dataset_suffix(priors, prefix, inst, skip=1):
    """juliet's ``<prefix>_iname[instrument]``: the last matching name, without its first ``skip`` parts."""
    out = None
    for k in priors:
        if k[:len(prefix)] == prefix and inst in k.split('_'):
            out = '_'.join(k.split('_')[skip:])
    return out


class JaxModel:
    """Light-curve model and log-likelihood of every dataset of a fit.

    Parameters
    ----------
    data : dict
        ``{dataset: {'t', 'y', 'e', 'X', 'ramp_x', 'N' (optional)}}`` as in
        ``BroadbandFit.data`` (normalised flux; standardised linear
        regressors ``X``).
    priors : dict
        Complete juliet priors (``BroadbandFit.priors``).
    ld_laws : str
        juliet's limb-darkening string (``'quadratic'`` or
        ``'V1-quadratic,V2-squareroot'``).
    ramps : dict or None
        ``{dataset: number of exponential terms}``.
    non_linear : dict or None
        ``{dataset: {'function': f, 'multiplicative': bool}}``; ``f(x,
        parameters)`` must be written with ``jax.numpy``.
    supersample : dict or None
        ``{dataset: (n, exposure time in days)}``.
    aliases : dict or None
        Group names of parameters, given to the non-linear functions.
    pl, pu, ecclim, light_travel_delay
        juliet.fit options (radius-ratio limits of the r1/r2
        parametrisation, maximum eccentricity).
    """

    def __init__(self, data, priors, ld_laws='quadratic', ramps=None, non_linear=None,
                 supersample=None, aliases=None, pl=0., pu=1., ecclim=1., light_travel_delay=False,
                 stellar_radius=None, rvs=None, ta=None):
        self.light_travel_delay = bool(light_travel_delay)
        if self.light_travel_delay and stellar_radius is None:
            raise ValueError('light_travel_delay needs stellar_radius (R*/Rsun), as in juliet.')
        self.stellar_radius = None if stellar_radius is None else float(stellar_radius)
        self.names = list(data)
        self.priors = dict(priors)
        self.prior_set = PriorSet(self.priors)
        self.aliases = dict(aliases or {})
        self.pl, self.pu, self.ecclim = float(pl), float(pu), float(ecclim)
        self.ramps = {n: int((ramps or {}).get(n) or 0) for n in self.names}
        self.non_linear = dict(non_linear or {})
        self.supersample = dict(supersample or {})
        self._check_supported()
        self.planets = self._planet_numbering()
        laws = self._ld_laws(ld_laws)
        self._single = laws.pop('__single__', False)
        self.config = {n: self._dataset_config(n, laws.get(n)) for n in self.names}
        self.data = {n: self._dataset_arrays(n, d) for n, d in data.items()}
        self._rv_setup(rvs or {}, ta)
        self._loglike = jax.jit(self._log_likelihood)
        self._logdens = jax.jit(self._log_density)

    @classmethod
    def from_fit(cls, fit, data=None, **options):
        """The model of a prepared :class:`~jstark.fitting.BroadbandFit` (``options``: juliet.fit
        options such as ``pl``, ``pu``, ``ecclim``; ``data`` replaces ``fit.data``)."""
        nl = {n: dict(function=v['function'], multiplicative=v['multiplicative'])
              for n, v in fit.non_linear.items()}
        return cls(fit.data if data is None else data, fit.priors, ld_laws=fit._ld_string(),
                   ramps=fit.ramp, non_linear=nl, supersample=fit.supersample, aliases=fit.aliases,
                   rvs=getattr(fit, 'rvs', None), **options)

    # ------------------------------------------------------------------
    # Structure (static)
    # ------------------------------------------------------------------
    def _check_supported(self):
        for k in self.priors:
            head = k.split('_')[0]
            for prefix, what in _UNSUPPORTED:
                if head == prefix and (prefix not in ('p1', 'p2') or re.fullmatch(r'p\d+', k.split('_')[1] if '_' in k else '')):
                    raise NotImplementedError(f'{what} ({k}) are not available in the JAX models.')
            if k.split('_')[-1] in ('lc', 'rv') and head in ('GP', 'sigma'):
                raise NotImplementedError(f'Global juliet models ({k}) are not available in the JAX models.')

    def _planet_numbering(self, heads=('r1', 'b')):
        """Planets as juliet numbers them: the numbers of the ``b_p<n>`` / ``r1_p<n>`` priors
        (``K_p<n>`` for RVs), sorted."""
        out = []
        for k in self.priors:
            vec = k.split('_')
            if vec[0] in heads and len(vec) > 1 and re.fullmatch(r'p\d+', vec[1]):
                out.append(int(vec[1][1:]))
        return sorted(out)

    def _ld_laws(self, ld_laws):
        items = [s.strip() for s in str(ld_laws).split(',') if s.strip()]
        if len(items) == 1 and '-' not in items[0]:
            return {n: items[0].lower() for n in self.names} | {'__single__': True}
        out = {}
        for s in items:
            inst, law = s.split('-')
            out[inst.split()[0]] = law.split()[0].lower()
        return out

    def _limb_darkening(self, inst, law_given, single):
        """(law, parametrisation, parameter names) of one dataset, as juliet decides them."""
        c1, c2, par, suffix = False, False, 'kipping2013', None
        for k in self.priors:
            head = k[:2]
            if head in ('q1', 'u1', 'c1') and inst in k.split('_')[1:]:
                c1 = True
                par = {'u1': 'normal', 'c1': 'normal-nonlinear'}.get(head, par)
            if head in ('q2', 'u2') and inst in k.split('_')[1:]:
                c2 = True
        if not c1:
            if c2:
                raise ValueError(f'Dataset {inst}: q2/u2 prior without q1/u1.')
            return None, None, []
        suffix = _dataset_suffix(self.priors, 'q1', inst) or _dataset_suffix(self.priors, 'u1', inst) \
            or _dataset_suffix(self.priors, 'c1', inst)
        if par == 'normal-nonlinear':
            law, par = 'nonlinear', 'normal'
        elif c2:
            law = law_given
        else:
            law = 'linear' if single else law_given
        law = {'power-2': 'power2'}.get(law, law)
        if law not in planet.LD_LAWS:
            raise ValueError(f'Dataset {inst}: unknown limb-darkening law {law!r}.')
        if law == 'nonlinear':
            names = [f'c{i}_{suffix}' for i in range(1, 5)]
        elif law == 'linear':
            names = [('q1_' if par == 'kipping2013' else 'u1_') + suffix]
        else:
            q = ('q1', 'q2') if par == 'kipping2013' else ('u1', 'u2')
            names = [f'{q[0]}_{suffix}', f'{q[1]}_{suffix}']
        if par == 'kipping2013' and law not in ('quadratic', 'squareroot', 'logarithmic', 'linear'):
            raise ValueError(f'Dataset {inst}: juliet has no q1/q2 form of the {law} law; give u1/u2 priors.')
        return law, par, names

    def _dataset_config(self, inst, law_given):
        P = self.priors
        law_given = law_given or 'quadratic'
        transit = any(k[:2] in ('q1', 'u1', 'c1') and inst in k.split('_') for k in P)

        def applies(prefix):
            return any(k[:len(prefix)] == prefix and (inst in k.split('_') or len(k.split('_')) == 2)
                       for k in P)

        eclipse = applies('fp')
        sine, ca = applies('phaseoffset'), applies('C1')
        cfg = dict(transit=transit and not eclipse, eclipse=eclipse and not transit,
                   traneclipse=transit and eclipse, sine=sine, ca=ca)
        law, par, ld_names = self._limb_darkening(inst, law_given, self._single)
        cfg.update(law=law, ldpar=par, ld=ld_names)
        planets = []
        if transit or eclipse:
            for i in self.planets:
                tag = f'p{i}'
                pp = dict(n=i, P=f'P_{tag}', t0=f't0_{tag}')
                pp['efficient'] = f'r1_{tag}' in P
                if pp['efficient']:
                    pp['r1'], pp['r2'] = f'r1_{tag}', f'r2_{tag}'
                else:
                    sp = _planet_suffix(P, 'p_', inst, tag)
                    if sp is None:
                        raise ValueError(f'Dataset {inst}: no radius-ratio prior p_{tag} applies to it.')
                    pp['p'], pp['b'] = f'p_{tag}{sp}', f'b_{tag}'
                pp['a'] = 'rho' if 'rho' in P else f'a_{tag}'
                if f'ecosomega_{tag}' in P:
                    pp['ecc'] = ('ecosomega', f'ecosomega_{tag}', f'esinomega_{tag}')
                elif f'secosomega_{tag}' in P:
                    pp['ecc'] = ('secosomega', f'secosomega_{tag}', f'sesinomega_{tag}')
                else:
                    pp['ecc'] = ('ecc', f'ecc_{tag}', f'omega_{tag}')
                if cfg['eclipse'] or cfg['traneclipse']:
                    sf = _planet_suffix(P, 'fp', inst, tag)
                    pp['fp'] = f'fp_{tag}{sf}' if sf is not None else None   # juliet: 100 ppm then
                    pp['t_secondary'] = f't_secondary_{tag}'
                    if cfg['ca']:
                        pp['ca'] = [f'{c}_{tag}{sf or ""}' for c in ('C1', 'D1', 'C2', 'D2')]
                    if cfg['sine']:
                        so = _planet_suffix(P, 'phaseoffset', inst, tag)
                        pp['phaseoffset'] = f'phaseoffset_{tag}{so or ""}'
                planets.append(pp)
        cfg['planets'] = planets
        cfg['mdilution'] = 'mdilution_' + (_dataset_suffix(P, 'mdilution', inst) or inst)
        cfg['mflux'] = 'mflux_' + (_dataset_suffix(P, 'mflux', inst) or inst)
        cfg['sigma_w'] = 'sigma_w_' + (_dataset_suffix(P, 'sigma_w', inst, skip=2) or inst)
        nth = 0
        thetas = {}
        for k in P:
            vec = k.split('_')
            if k[:5] == 'theta' and inst in vec and vec[0][5:].isdigit():
                thetas[int(vec[0][5:])] = k
                nth = max(nth, int(vec[0][5:]) + 1)
        cfg['theta'] = [thetas.get(i) for i in range(nth)]
        cfg['ramp'] = self._ramp_names(inst) if self.ramps.get(inst) else []
        cfg['non_linear'] = self.non_linear.get(inst)
        # a ramp alone is multiplicative; with your function, the function decides
        cfg['multiplicative'] = True if cfg['non_linear'] is None else bool(cfg['non_linear']['multiplicative'])
        cfg['supersample'] = self.supersample.get(inst)
        cfg['gp'], cfg['gp_params'] = None, {}
        for name in [cfg['mdilution'], cfg['mflux'], cfg['sigma_w']] + cfg['ld'] + cfg['ramp']:
            if name not in P:
                raise ValueError(f'Dataset {inst}: no prior {name}.')
        return cfg

    def _ramp_names(self, inst):
        from ..fit import ExpRamp
        return ExpRamp(inst, self.ramps[inst], self.priors).names()

    def _dataset_arrays(self, inst, d):
        out = {k: jnp.asarray(d[k], float) for k in ('t', 'y', 'e')}
        cfg = self.config[inst]
        if d.get('G') is not None:
            G = np.asarray(d['G'], float)
            kern, params = _gp.find_kernel(self.priors, inst, nx=G.shape[1] if G.ndim == 2 else 1)
            if kern is None:
                raise ValueError(f'Dataset {inst}: a GP regressor but no GP_ priors.')
            _gp.check_sorted(kern, G, inst)
            cfg['gp'], cfg['gp_params'] = kern, params
            out['G'] = jnp.asarray(G)
        elif _gp.find_kernel(self.priors, inst)[0] is not None:
            raise ValueError(f'Dataset {inst}: GP priors but no GP regressor (gp=...).')
        X = np.asarray(d.get('X', np.zeros((len(d['t']), 0))), float)
        if X.shape[1] != len(cfg['theta']) or any(v is None for v in cfg['theta']):
            raise ValueError(f'Dataset {inst}: {X.shape[1]} linear regressors but theta priors '
                             f'{cfg["theta"]}.')
        out['X'] = jnp.asarray(X)
        out['ramp_x'] = jnp.asarray(d.get('ramp_x', np.asarray(d['t']) - np.min(d['t'])), float)
        if 'N' in d and d['N'] is not None:
            out['N'] = jnp.asarray(d['N'], float)
        ss = cfg['supersample']
        if ss:
            n, exp = ss
            off = np.linspace(-exp / 2., exp / 2., int(n))
            out['t_super'] = jnp.asarray(np.asarray(d['t'], float)[:, None] + off[None, :])
        return out

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def _geometry(self, pv, pp):
        P, t0 = pv[pp['P']], pv[pp['t0']]
        if pp['efficient']:
            b, p = orbit.reverse_bp(pv[pp['r1']], pv[pp['r2']], self.pl, self.pu)
        else:
            b, p = pv[pp['b']], pv[pp['p']]
        a = orbit.a_from_rho(pv['rho'], P) if pp['a'] == 'rho' else pv[pp['a']]
        kind, n1, n2 = pp['ecc']
        if kind == 'ecc':
            ecc, w = pv[n1], pv[n2]
        else:
            c, s = pv[n1], pv[n2]
            e = jnp.sqrt(c ** 2 + s ** 2)
            ecc = e if kind == 'ecosomega' else e ** 2
            w = jnp.degrees(jnp.arctan2(s, c))
        return P, t0, a, b, p, ecc, w

    def geometry(self, pv, inst, n):
        """Orbit of planet number ``n`` in the model of dataset ``inst``: ``P``, ``t0``,
        ``p``, ``a``, ``b``, ``inc`` (deg), ``ecc``, ``w`` (deg), ``t_secondary`` (fitted,
        or from ``t0`` and the orbit as batman computes it)."""
        pp = next((q for q in self.config[inst]['planets'] if q['n'] == n), None)
        if pp is None:
            raise KeyError(f'Dataset {inst} has no model of planet p{n}.')
        P, t0, a, b, p, ecc, w = self._geometry(pv, pp)
        wr = jnp.radians(w)
        cosi = jnp.clip((b / a) * (1. + ecc * jnp.sin(wr)) / (1. - ecc ** 2), -1., 1.)
        if 't_secondary' in pp and not self.light_travel_delay and pp['t_secondary'] in pv:
            tsec = pv[pp['t_secondary']]
        else:
            tsec = orbit.t_secondary_from_t0(t0, P, ecc, wr)
        return dict(P=P, t0=t0, p=p, a=a, b=b, inc=jnp.degrees(jnp.arccos(cosi)), ecc=ecc, w=w,
                    t_secondary=tsec)

    def _planet(self, pv, inst, t, tt):
        """``(M, valid)``: the sum of the planet models of a dataset before dilution
        (``1 + sum_i (planet_i - 1)``; 1 without planets) at times ``t`` (``tt``: the
        supersampled times, ``(n, nsuper)``, or ``t``), and False where juliet rejects
        the parameters."""
        cfg = self.config[inst]
        M = jnp.ones_like(t)
        valid = jnp.asarray(True)
        if cfg['planets']:
            u = None
            if cfg['law'] is not None and (cfg['transit'] or cfg['traneclipse']):
                u = planet.ld_coefficients(cfg['law'], [pv[k] for k in cfg['ld']], cfg['ldpar'])
            last_P = None
            for pp in cfg['planets']:
                P, t0, a, b, p, ecc, w = self._geometry(pv, pp)
                if last_P is not None:
                    valid = valid & (P > last_P)
                last_P = P
                wr = jnp.radians(w)
                cosi = (b / a) * (1. + ecc * jnp.sin(wr)) / (1. - ecc ** 2)
                valid = valid & (ecc <= self.ecclim) & ~((b > 1. + p) | (cosi >= 1.))
                cosi = jnp.clip(cosi, -1., 1.)

                def mean_ss(x):
                    return x.mean(axis=-1) if x.ndim == 2 else x

                if cfg['transit'] or cfg['traneclipse']:
                    z, _ = orbit.sky_separation(tt, t0, P, a, cosi, ecc, wr, secondary=False)
                    tr = mean_ss(planet.transit_flux(cfg['law'], u, z.ravel(), p).reshape(z.shape))
                te = t                            # times of the eclipse model (and phase curves)
                if cfg['eclipse'] or cfg['traneclipse']:
                    fp = pv[pp['fp']] if pp['fp'] is not None else 100e-6
                    tte = tt
                    if self.light_travel_delay:
                        # juliet: t_secondary from t0 (batman), eclipse at the delayed times
                        tsec = orbit.t_secondary_from_t0(t0, P, ecc, wr)
                        t0e = t0
                        te = orbit.light_travel_times(t, t0, P, a, jnp.sqrt(1. - cosi ** 2), ecc, wr,
                                                      self.stellar_radius)
                        tte = te if tt.ndim == 1 else te[:, None] + (tt - t[:, None])
                    else:
                        tsec = pv[pp['t_secondary']]
                        t0e = orbit.t0_from_t_secondary(tsec, P, ecc, wr)
                    z2, _ = orbit.sky_separation(tte, t0e, P, a, cosi, ecc, wr, secondary=True)
                    vis = mean_ss(planet.eclipse_visible(z2.ravel(), p).reshape(z2.shape))
                if cfg['transit']:
                    lc = tr
                elif cfg['eclipse']:
                    lc = 1. + fp * vis
                else:
                    if cfg['sine'] or cfg['ca']:
                        pc = jnp.zeros_like(t)
                        if cfg['sine']:
                            pc = pc + planet.sine_phase_curve(te, t0e, P, fp, pv[pp['phaseoffset']])
                        if cfg['ca']:
                            C1, D1, C2, D2 = (pv[k] for k in pp['ca'])
                            pc = pc + planet.cowan_agol(te, tsec, P, fp, C1, D1, C2, D2)
                        lc = tr * (1. + pc * vis)
                    else:
                        lc = tr * (1. + fp * vis - fp)
                M = M + lc - 1.
        return M, valid

    def planet_model(self, pv, inst, t):
        """Planet model of a dataset (before dilution, as ``components(...)['planet']``) at
        any times ``t`` (supersampled as the dataset)."""
        t = jnp.asarray(t, float)
        ss = self.config[inst]['supersample']
        if ss:
            off = jnp.linspace(-ss[1] / 2., ss[1] / 2., int(ss[0]))
            return self._planet(pv, inst, t, t[:, None] + off[None, :])[0]
        return self._planet(pv, inst, t, t)[0]

    def components(self, pv, inst, data=None):
        """Model of one dataset for the parameter dict ``pv``.

        Returns a dict with ``planet`` (``M``, the sum of the planet models
        before dilution, 1 without planets), ``linear`` (the linear model,
        zeros without one), ``model`` (the full deterministic model,
        ``(M_diluted + linear) * non-linear`` or ``+ non-linear``),
        ``variance`` and ``valid`` (False where juliet rejects the
        parameters).
        """
        cfg = self.config[inst]
        d = self.data[inst] if data is None else data
        planet_only, valid = self._planet(pv, inst, d['t'], d.get('t_super', d['t']))
        D, mf = pv[cfg['mdilution']], pv[cfg['mflux']]
        M = (planet_only * D + (1. - D)) / (1. + D * mf)
        linear = jnp.zeros_like(d['t'])
        if cfg['theta']:
            th = jnp.stack([pv[k] for k in cfg['theta']])
            linear = d['X'] @ th
        model = M + linear
        nl = self._non_linear(pv, inst, d)
        if nl is not None:
            model = model * nl if cfg['multiplicative'] else model + nl
        var = d['e'] ** 2 + (pv[cfg['sigma_w']] * 1e-6) ** 2
        return dict(planet=planet_only, linear=linear, model=model, variance=var, valid=valid)

    def _non_linear(self, pv, inst, d):
        cfg = self.config[inst]
        ramp = None
        if cfg['ramp']:
            r = jnp.ones_like(d['ramp_x'])
            n = cfg['ramp']
            for k in range(len(n) // 2):
                r = r + pv[n[2 * k]] * jnp.exp(-d['ramp_x'] / pv[n[2 * k + 1]])
            ramp = r
        nl = cfg['non_linear']
        if nl is None:
            return ramp
        q = dict(pv, **{a: pv[k] for a, k in self.aliases.items() if k in pv}) if self.aliases else pv
        f = jnp.asarray(nl['function'](d['N'], q))
        return f if ramp is None else ramp * f

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------
    def params(self, x, fixed=None):
        """Parameter dict (free values ``x`` and the fixed ones; ``fixed``: ``{name: value}``
        replacing the model's fixed values)."""
        return self.prior_set.as_dict(x, fixed)

    def _gp(self, pv, inst, d, variance):
        """tinygp GaussianProcess of a dataset (None without a GP)."""
        cfg = self.config[inst]
        if cfg['gp'] is None:
            return None
        from tinygp import GaussianProcess
        p = {v: pv[k] for v, k in cfg['gp_params'].items()}
        kern = _gp.build_kernel(cfg['gp'], p)
        X = _gp.scale_input(cfg['gp'], p, d['G'])
        return GaussianProcess(kern, X, diag=variance)

    def gp_prediction(self, pv, inst, data=None, comp=None):
        """GP mean at the data given the residuals of the deterministic model (juliet's
        ``GP.predict(residuals, X)``); zeros without a GP."""
        d = self.data[inst] if data is None else data
        c = self.components(pv, inst, d) if comp is None else comp
        gp = self._gp(pv, inst, d, c['variance'])
        if gp is None:
            return jnp.zeros_like(d['t'])
        # K (K + D)^-1 r = r - D (K + D)^-1 r (tinygp 0.3.1 cannot condition sums of quasisep kernels)
        r = d['y'] - c['model']
        a = gp.solver.solve_triangular(gp.solver.solve_triangular(r), transpose=True)
        return r - c['variance'] * a

    def gp_predict(self, pv, inst, X, r, variance, X_pred):
        """GP mean and standard deviation at inputs ``X_pred`` given residuals ``r`` at
        inputs ``X`` with white-noise variances ``variance`` (juliet's ``GP.compute(X,
        err); GP.predict(r, X_pred, return_var=True)``) for the kernel of dataset ``inst``
        at the parameters ``pv``. ``X`` must be sorted for the celerite kernels."""
        from tinygp import GaussianProcess
        cfg = self.config[inst]
        p = {v: pv[k] for v, k in cfg['gp_params'].items()}
        kern = _gp.build_kernel(cfg['gp'], p)
        Xs = _gp.scale_input(cfg['gp'], p, jnp.asarray(X, float))
        Xp = _gp.scale_input(cfg['gp'], p, jnp.asarray(X_pred, float))
        gp = GaussianProcess(kern, Xs, diag=variance)
        alpha = gp.solver.solve_triangular(gp.solver.solve_triangular(r), transpose=True)
        Ks = kern(Xp, Xs)                                   # (m, N), dense
        v = gp.solver.solve_triangular(Ks.T)
        kxx = jax.vmap(lambda a: kern(a[None], a[None])[0, 0])(Xp)
        return Ks @ alpha, jnp.sqrt(jnp.clip(kxx - jnp.sum(v ** 2, axis=0), 0., None))

    def _log_likelihood(self, x, data=None, rv_data=None, fixed=None):
        """Log-likelihood of the free values ``x``. ``data``: arrays of every dataset (default
        the model's); a ``mask`` array (1: point, 0: padding) leaves padded points out exactly
        (batched fits of datasets of different lengths). ``fixed``: values of the fixed
        parameters replacing the model's (batched fits)."""
        pv = self.params(x, fixed)
        data = self.data if data is None else data
        ll, valid = 0., jnp.asarray(True)
        for n in self.names:
            d = data[n]
            c = self.components(pv, n, d)
            r = d['y'] - c['model']
            var = c['variance']
            m = d.get('mask')
            if m is not None:
                m = m > 0.5
                r = jnp.where(m, r, 0.)
            if self.config[n]['gp'] is None:
                terms = jnp.log(var) + r ** 2 / var
                if m is None:
                    ll = ll - 0.5 * (r.size * _LOG2PI + jnp.sum(terms))
                else:
                    ll = ll - 0.5 * (jnp.sum(m) * _LOG2PI + jnp.sum(jnp.where(m, terms, 0.)))
            else:
                if m is not None:
                    # padded points: zero residual and a huge variance, so they decouple from the
                    # data; their own log-determinant and normalisation terms are taken off
                    var = jnp.where(m, var, _PAD_VARIANCE)
                gp = self._gp(pv, n, d, var)
                ll = ll + gp.log_probability(r)
                if m is not None:
                    ll = ll + 0.5 * jnp.sum(~m) * (_LOG2PI + np.log(_PAD_VARIANCE))
            valid = valid & c['valid']
        if self.rv_names:
            rd = self.rv_data if rv_data is None else rv_data
            for n in self.rv_names:
                c = self.rv_components(pv, n, rd[n])
                r = rd[n]['rv'] - c['model']
                ll = ll - 0.5 * (r.size * _LOG2PI + jnp.sum(jnp.log(c['variance']) + r ** 2 / c['variance']))
                valid = valid & c['valid']
        ll = jnp.where(valid, ll, -jnp.inf)
        return jnp.where(jnp.isnan(ll), -jnp.inf, ll)

    # ------------------------------------------------------------------
    # Radial velocities
    # ------------------------------------------------------------------
    def _rv_setup(self, rvs, ta):
        self.rv_names = list(rvs)
        self.rv_data = {n: {k: jnp.asarray(v[k], float) for k in ('t', 'rv', 'err')} for n, v in rvs.items()}
        self.rv_planets = self._planet_numbering(('K',)) if rvs else []
        P = self.priors
        if rvs and not self.rv_planets:
            raise ValueError('Radial velocities need K_p<n> priors.')
        for k in P:
            vec = k.split('_')
            if vec[0] in ('GP', 'theta') and any(n in vec for n in self.rv_names):
                raise NotImplementedError(f'GPs and linear models on radial velocities ({k}) are not '
                                          'available in jstark.')
        self.rv_config = []
        for i in self.rv_planets:
            tag = f'p{i}'
            pp = dict(n=i, P=f'P_{tag}', t0=f't0_{tag}', K=f'K_{tag}')
            if f'ecosomega_{tag}' in P:
                pp['ecc'] = ('ecosomega', f'ecosomega_{tag}', f'esinomega_{tag}')
            elif f'secosomega_{tag}' in P:
                pp['ecc'] = ('secosomega', f'secosomega_{tag}', f'sesinomega_{tag}')
            else:
                pp['ecc'] = ('ecc', f'ecc_{tag}', f'omega_{tag}')
            for k in (pp['P'], pp['t0'], pp['K']) + pp['ecc'][1:]:
                if k not in P:
                    raise ValueError(f'Radial velocities: no prior {k}.')
            self.rv_config.append(pp)
        self.rv_trend = 'quad' if 'rv_quad' in P and 'rv_slope' in P else ('line' if 'rv_slope' in P else None)
        if self.rv_trend and ta is None:
            tr = np.concatenate([np.asarray(v['t'], float) for v in rvs.values()])
            ta = 0.5 * (tr.min() + tr.max())
        self.ta = None if ta is None else float(ta)
        for n in self.rv_names:
            for k in (f'mu_{n}', f'sigma_w_{n}'):
                if k not in P:
                    raise ValueError(f'Radial velocities of {n}: no prior {k}.')

    def rv_components(self, pv, inst, data=None):
        """RV model of one instrument: ``model`` (offset + trend + Keplerians), ``variance``,
        ``valid`` (juliet: periods increasing with the planet number, e <= ecclim)."""
        d = self.rv_data[inst] if data is None else data
        t = d['t']
        m = jnp.zeros_like(t)
        valid, last = jnp.asarray(True), None
        for pp in self.rv_config:
            P, t0, K = pv[pp['P']], pv[pp['t0']], pv[pp['K']]
            if last is not None:
                valid = valid & (P > last)
            last = P
            kind, n1, n2 = pp['ecc']
            if kind == 'ecc':
                ecc, w = pv[n1], jnp.radians(pv[n2])
            else:
                c, s_ = pv[n1], pv[n2]
                e = jnp.sqrt(c ** 2 + s_ ** 2)
                ecc = e if kind == 'ecosomega' else e ** 2
                w = jnp.arctan2(s_, c)
            valid = valid & (ecc <= self.ecclim)
            m = m + orbit.rv_keplerian(t, t0, P, K, ecc, w)
        if self.rv_trend:
            dt = t - self.ta
            m = m + pv['rv_intercept'] + dt * pv['rv_slope']
            if self.rv_trend == 'quad':
                m = m + dt ** 2 * pv['rv_quad']
        m = m + pv[f'mu_{inst}']
        var = d['err'] ** 2 + pv[f'sigma_w_{inst}'] ** 2
        return dict(model=m, variance=var, valid=valid)

    def log_likelihood(self, x, data=None):
        """Log-likelihood of the free parameter values ``x`` (juliet's order: ``prior_set.free``)."""
        return self._loglike(jnp.asarray(x, float), self.data if data is None else data)

    def _log_density(self, z, data=None):
        lp, x = self.prior_set.log_density(z)
        return lp + self._log_likelihood(x, data)

    def log_density(self, z, data=None):
        """Log posterior density of the unconstrained parameters ``z`` (for samplers)."""
        return self._logdens(jnp.asarray(z, float), self.data if data is None else data)

    def evaluate(self, x, inst):
        """``components`` of one dataset for free parameter values ``x`` (NumPy arrays)."""
        c = self.components(self.params(jnp.asarray(x, float)), inst)
        return {k: np.asarray(v) for k, v in c.items()}
