"""Priors for juliet fits of one or several datasets.

You give the astrophysical priors (planet, orbit and, when you want them,
limb darkening) in juliet's naming. :func:`build_priors` adds what each
dataset needs and you did not give:

* ``mdilution_<ds>``: fixed to 1;
* ``mflux_<ds>``: normal(0, 0.1) (relative flux offset);
* ``sigma_w_<ds>``: log-uniform(0.1, 1e5) ppm (extra white noise);
* ``theta<i>_<ds>``: uniform(-1, 1) for each linear-regressor column
  (regressors are standardised, so ``theta`` is the flux change for a
  one-sigma change of the regressor);
* GP hyperparameters for the chosen kernel, when the dataset has a GP;
* ``ramp[k]amp_<ds>`` uniform(-0.1, 0.1) and ``ramp[k]tau_<ds>`` log-uniform
  (2 cadences, duration) for each exponential ramp term;
* ``C2``/``D2`` fixed to 0 when you give a Cowan & Agol phase curve
  (``C1``, ``D1``) without them: the first-order model (juliet always
  evaluates the second-order terms).

Phase curves (juliet's Cowan & Agol 2008 model, :func:`check_phase_curves`):
juliet applies them only to datasets that also get a transit model
(limb-darkening priors), and looks ``C1``-``D2`` up with the instrument
suffix of the ``fp`` prior, so ``C1_p1_MIRI`` needs ``fp_p1_MIRI``.

Limb darkening is **never** added. In juliet a dataset gets a transit model
only if a ``q1``/``u1``/``c1`` prior names it, so eclipse-only data need
none; :func:`model_report` lists what each dataset will be fitted with.

Prior formats accepted wherever priors are passed: juliet's dict
``{name: {'distribution': ..., 'hyperparameters': ...}}``, the short form
``{name: (distribution, hyperparameters)}``, or the path of a juliet prior
file.
"""
import warnings

import numpy as np

__all__ = ['normalise_priors', 'build_priors', 'model_report', 'GP_KERNELS', 'covers',
           'priors_to_text', 'check_phase_curves']

# juliet kernel variables (celerite kernels take a 1D regressor)
GP_KERNELS = {
    'matern': ('sigma', 'rho'),               # CeleriteMaternKernel
    'exp': ('sigma', 'timescale'),            # CeleriteExpKernel
    'matern-exp': ('sigma', 'timescale', 'rho'),
    'sho': ('S0', 'Q', 'omega0'),             # CeleriteSHOKernel
    'qp': ('B', 'C', 'L', 'Prot'),            # CeleriteQPKernel (no default priors)
}

DEFAULT_NUISANCE = {
    'mdilution': ('fixed', 1.0),
    'mflux': ('normal', [0.0, 0.1]),
    'sigma_w': ('loguniform', [0.1, 1e5]),
    'theta': ('uniform', [-1.0, 1.0]),
}


def _one(v):
    if isinstance(v, dict):
        return {'distribution': str(v['distribution']).lower(),
                'hyperparameters': v['hyperparameters']}
    if isinstance(v, (tuple, list)) and len(v) == 2 and isinstance(v[0], str):
        return {'distribution': v[0].lower(), 'hyperparameters': v[1]}
    raise ValueError(f'Prior {v!r} not understood; use (distribution, hyperparameters).')


def normalise_priors(priors):
    """Convert priors (dict in either form, or a prior-file path) to juliet's dict form."""
    if priors is None:
        return {}
    if isinstance(priors, str):
        import juliet
        out = juliet.utils.readpriors(priors)[0]
        return {k: {'distribution': v['distribution'].lower(),
                    'hyperparameters': v['hyperparameters']} for k, v in out.items()}
    return {k: _one(v) for k, v in priors.items()}


def covers(priors, prefix, name):
    """True if a prior ``<prefix>_..._<name>_...`` exists (``prefix`` may contain ``_``)."""
    n = len(prefix.split('_'))
    for k in priors:
        parts = k.split('_')
        if '_'.join(parts[:n]) == prefix and name in parts[n:]:
            return True
    return False


def _global_or_covers(priors, prefix, name, nplanet_parts=1):
    """Planet parameter (e.g. ``p_p1``) that applies to all datasets or names ``name``."""
    for k in priors:
        parts = k.split('_')
        if parts[0] != prefix:
            continue
        rest = parts[1 + nplanet_parts:]
        if not rest or name in rest:
            return True
    return False


def _gp_defaults(kernel, x):
    """Broad default priors for a kernel given its 1D regressor ``x``."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    dx = np.nanmedian(np.abs(np.diff(np.sort(x))))
    dx = dx if dx > 0 else np.ptp(x) / max(x.size, 1)
    span = np.ptp(x)
    amp = ('loguniform', [1e-6, 1e-2])          # relative flux
    if kernel == 'matern':
        return {'sigma': amp, 'rho': ('loguniform', [dx, 10. * span])}
    if kernel == 'exp':
        return {'sigma': amp, 'timescale': ('loguniform', [dx, 10. * span])}
    if kernel == 'matern-exp':
        return {'sigma': amp, 'timescale': ('loguniform', [dx, 10. * span]),
                'rho': ('loguniform', [dx, 10. * span])}
    if kernel == 'sho':
        return {'S0': ('loguniform', [1e-16, 1e-4]), 'Q': ('fixed', 1. / np.sqrt(2.)),
                'omega0': ('loguniform', [2. * np.pi / (10. * span), 2. * np.pi / dx])}
    raise ValueError(f'No default priors for GP kernel {kernel!r}; give the GP_ priors yourself.')


def _ramp_defaults(k, x):
    """Priors of the ``k``-th exponential ramp term (``x``: time since the first integration, days)."""
    x = np.asarray(x, float)
    dt = np.nanmedian(np.diff(x)) if x.size > 1 else 1e-3
    sfx = '' if k == 1 else str(k)
    return {f'ramp{sfx}amp': ('uniform', [-0.1, 0.1]),
            f'ramp{sfx}tau': ('loguniform', [2. * dt, max(np.ptp(x), 4. * dt)])}


def build_priors(priors, datasets, nuisance=None, verbose=True):
    """Complete a set of priors for the given datasets.

    Parameters
    ----------
    priors : dict, str or None
        Your priors (see the module docstring). They always take
        precedence.
    datasets : dict
        ``{name: info}`` with ``info`` a dict with keys ``nlin`` (number of
        linear-regressor columns), ``gp_kernel`` (kernel name or None),
        ``gp_x`` (the GP regressor, used to set default hyperparameter
        ranges), ``ramp`` (number of exponential ramp terms) and ``ramp_x``
        (time since the first integration).
    nuisance : dict or None
        Overrides of the default nuisance priors, keyed ``mdilution``,
        ``mflux``, ``sigma_w`` and ``theta`` (applied to every dataset), or
        a full parameter name (e.g. ``'sigma_w_V1'``).

    Returns
    -------
    priors : dict
        juliet priors.
    added : list of str
        Names that were added automatically.
    """
    out = normalise_priors(priors)
    nuis = {k: _one(v) for k, v in DEFAULT_NUISANCE.items()}
    explicit = {}
    for k, v in (nuisance or {}).items():
        if k in nuis:
            nuis[k] = _one(v)
        else:
            explicit[k] = _one(v)
    added = []

    def add(name, prior):
        if name not in out:
            out[name] = dict(explicit[name]) if name in explicit else _one(prior)
            added.append(name)

    for ds, info in datasets.items():
        for p in ('mdilution', 'mflux', 'sigma_w'):
            if not covers(out, p, ds):
                add(f'{p}_{ds}', nuis[p])
        for i in range(int(info.get('nlin', 0))):
            if not covers(out, f'theta{i}', ds):
                add(f'theta{i}_{ds}', nuis['theta'])
        tx = info.get('ramp_x')
        for k in range(1, int(info.get('ramp', 0)) + 1):
            for name, prior in _ramp_defaults(k, tx).items():
                if not covers(out, name, ds):
                    add(f'{name}_{ds}', prior)
        kern = info.get('gp_kernel')
        if kern:
            if kern not in GP_KERNELS:
                raise ValueError(f'GP kernel {kern!r} not known; use one of {list(GP_KERNELS)}.')
            have = [v for v in GP_KERNELS[kern] if covers(out, f'GP_{v}', ds)]
            if len(have) < len(GP_KERNELS[kern]):
                x = info.get('gp_x')
                x = np.asarray(x)[:, 0] if np.ndim(x) == 2 else x
                d = _gp_defaults(kern, x)
                for v in GP_KERNELS[kern]:
                    if v not in have:
                        add(f'GP_{v}_{ds}', d[v])
    for k in [k for k in out if k.split('_')[0] == 'C1']:
        sfx = k[len('C1'):]
        for c in ('C2', 'D2'):
            add(c + sfx, ('fixed', 0.0))
    # Parameters that name no fitted dataset are a typo (juliet would fail later)
    if verbose and added:
        print(f'Added {len(added)} priors: ' + ', '.join(added))
    return out, added


def model_report(priors, names):
    """What juliet will fit for each dataset, from the priors alone.

    Returns ``{name: {'transit': bool, 'eclipse': bool, 'phasecurve': bool,
    'ld': bool}}`` and warns when a radius-ratio prior applies to a dataset
    without limb-darkening priors (juliet then fits no transit for it, which
    is right for eclipse-only data and a mistake otherwise).
    """
    rep = {}
    for ds in names:
        ld = any(covers(priors, c, ds) for c in ('q1', 'u1', 'c1'))
        rp = _global_or_covers(priors, 'p', ds)
        fp = _global_or_covers(priors, 'fp', ds)
        pc = any(_global_or_covers(priors, c, ds) for c in ('phaseoffset', 'C1', 'D1'))
        rep[ds] = dict(transit=ld, eclipse=fp, phasecurve=pc, ld=ld)
        if rp and not ld:
            warnings.warn(f'Dataset {ds}: an Rp/R* prior applies but no limb-darkening prior names '
                          f'{ds}, so juliet will not fit a transit to it. Add q1_{ds}/q2_{ds} '
                          '(or a shared q1_..._' + ds + ') if it contains a transit.')
        if not (ld or fp or pc):
            warnings.warn(f'Dataset {ds}: no transit, eclipse or phase-curve model (systematics only).')
    check_phase_curves(priors, names, rep)
    return rep


def _applies(key, prefix, ds):
    """Planet parameter ``key`` of type ``prefix`` applies to dataset ``ds``."""
    parts = key.split('_')
    return parts[0] == prefix and (len(parts) == 2 or ds in parts[2:])


def check_phase_curves(priors, names, report=None):
    """Raise if juliet would ignore or fail on the phase-curve priors.

    * A phase curve (``C1``/``D1`` or ``phaseoffset``) on a dataset without
      limb-darkening priors: juliet evaluates phase curves only together
      with a transit, and would silently fit a plain eclipse.
    * A phase curve needs an eclipse depth ``fp``; for Cowan & Agol the
      ``C1``, ``D1``, ``C2``, ``D2`` names must carry the same instrument
      suffix as that ``fp`` prior (juliet looks them up that way).
    """
    report = report or {}
    for ds in names:
        pcs = [k for k in priors if any(_applies(k, c, ds) for c in ('C1', 'D1', 'phaseoffset'))]
        if not pcs:
            continue
        ld = report.get(ds, {}).get('ld')
        if ld is None:
            ld = any(covers(priors, c, ds) for c in ('q1', 'u1', 'c1'))
        if not ld:
            raise ValueError(f'Dataset {ds}: phase-curve priors {pcs} but no limb-darkening prior '
                             f'names {ds}. juliet computes phase curves only with a transit model '
                             f'and would silently fit a plain eclipse; add q1/q2 for {ds} (fixed '
                             'values are fine if the transit is not covered).')
        fps = [k for k in priors if _applies(k, 'fp', ds)]
        if not fps:
            raise ValueError(f'Dataset {ds}: phase-curve priors {pcs} need an eclipse depth '
                             f'prior (fp_p1 or fp_p1_{ds}).')
        for pl in sorted({k.split('_')[1] for k in pcs if k.split('_')[0] in ('C1', 'D1')}):
            fpk = [k for k in fps if k.split('_')[1] == pl]
            if not fpk:
                raise ValueError(f'Dataset {ds}: Cowan & Agol priors for {pl} but no fp_{pl}.')
            sfx = fpk[0][len('fp'):]
            missing = [c + sfx for c in ('C1', 'D1') if c + sfx not in priors]
            if missing:
                raise ValueError(f'Dataset {ds}: juliet reads the Cowan & Agol terms with the '
                                 f'suffix of {fpk[0]}, so it needs {missing} (got '
                                 f'{[k for k in pcs if k.split("_")[1] == pl]}).')


def priors_to_text(priors):
    """juliet prior-file text (one line per parameter)."""
    lines = []
    for k, v in priors.items():
        h = v['hyperparameters']
        hs = ','.join(repr(float(x)) for x in np.atleast_1d(h))
        lines.append(f'{k:<28s} {v["distribution"]:<14s} {hs}')
    return '\n'.join(lines) + '\n'
