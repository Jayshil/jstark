"""Priors for juliet fits of one or several datasets.

You give the astrophysical priors (planet, orbit and, when you want them,
limb darkening) in juliet's naming. :func:`build_priors` adds what each
dataset needs and you did not give:

* ``mdilution_<ds>``: fixed to 1;
* ``mflux_<ds>``: normal(0, 0.1) (relative flux offset);
* ``sigma_w_<ds>``: log-uniform(0.1, 1e4) ppm (extra white noise);
* ``theta<i>_<ds>``: uniform(-1, 1) for each linear-regressor column
  (regressors are standardised, so ``theta`` is the flux change for a
  one-sigma change of the regressor);
* GP hyperparameters for the chosen kernel, when the dataset has a GP;
* ``ramp[k]amp_<ds>`` uniform(-0.1, 0.1) and ``ramp[k]tau_<ds>`` log-uniform
  (2 cadences, duration) for each exponential ramp term;
* for radial velocities, ``mu_<instrument>`` uniform (the range of the data,
  extended by five times its width on both sides) and ``sigma_w_<instrument>``
  log-uniform (1% of the median error to ten times the scatter), in the unit
  of the velocities; ``rv_intercept`` fixed to 0 when you give a trend;
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
import re
import warnings

import numpy as np

__all__ = ['normalise_priors', 'build_priors', 'model_report', 'GP_KERNELS', 'covers',
           'priors_to_text', 'check_phase_curves', 'check_groups', 'expand_group_priors',
           'expand_group_options', 'stray_priors']

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
    'sigma_w': ('loguniform', [0.1, 1e4]),
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


def check_groups(groups, names):
    """Validate dataset groups; returns ``{group: [dataset, ...]}``.

    ``names`` are the datasets of the fit. A group name follows the rules of
    dataset names and must differ from all of them; groups may overlap.
    """
    from .lightcurve import check_name
    out = {}
    for g, members in (groups or {}).items():
        check_name(g)
        if g in names:
            raise ValueError(f'Group {g!r} has the name of a dataset.')
        members = [members] if isinstance(members, str) else list(members)
        unknown = [m for m in members if m not in names]
        if unknown:
            raise ValueError(f'Group {g!r}: unknown dataset(s) {unknown}; datasets are {list(names)}.')
        if not members or len(set(members)) < len(members):
            raise ValueError(f'Group {g!r} is empty or names a dataset twice.')
        out[g] = members
    return out


def expand_group_priors(priors, groups, names):
    """Replace group names in parameter names by their datasets.

    A group is one *shared* parameter: ``q1_TESS`` becomes
    ``q1_TESS51_TESS91``. Names without a group are kept as they are.

    Returns
    -------
    priors : dict
    aliases : dict
        ``{name as given: full juliet name}`` of the expanded parameters.
    """
    out, aliases = {}, {}
    for k, v in priors.items():
        parts = k.split('_')
        if not any(x in groups for x in parts[1:]):
            full = k
        else:
            if parts[0] in ('T', 'dt'):
                raise ValueError(f'Prior {k!r}: transit times belong to one dataset, not to a group.')
            new = parts[:1]
            for x in parts[1:]:
                new += [m for m in groups.get(x, [x]) if m not in new[1:] or m not in names]
            full = '_'.join(new)
            aliases[k] = full
        if full in out:
            raise ValueError(f'Prior {k!r} is given twice (as {full!r}).')
        out[full] = v

    def kind(key):
        return tuple(x for x in key.split('_') if x not in names)

    for k, full in aliases.items():
        ds = {x for x in full.split('_') if x in names}
        for other in out:
            both = ds & set(other.split('_')[1:])
            if other != full and both and kind(other) == kind(full):
                raise ValueError(f'Priors {k!r} (= {full!r}) and {other!r} both apply to {sorted(both)}.')
    return out, aliases


def expand_group_options(value, groups):
    """Per-dataset options with group keys: the value of a group goes to each of its datasets.

    A dataset's own key takes precedence. Anything but a dict is returned
    unchanged.
    """
    if not isinstance(value, dict) or not groups:
        return value
    out = {}
    for k, v in value.items():
        for m in groups.get(k, []):
            if m in out and m not in value:
                raise ValueError(f'Dataset {m} gets a value from more than one group.')
            out[m] = value.get(m, v)
        if k not in groups:
            out[k] = v
    return out


_DATASET_PARAMETER = re.compile(r'(q[1-4]|u[12]|c[1-4]|mdilution|mflux|theta\d+|ramp\d*(amp|tau))')


def stray_priors(priors, names, rv_names=()):
    """Priors that juliet ignores because the datasets they name are not in the fit.

    Only parameter names that juliet (or jstark) ties to datasets are judged
    (limb darkening, offsets, jitters, linear, GP and ramp terms, planet
    parameters with a dataset suffix); parameters of your own non-linear
    functions may have any name.
    """
    names, rv_names = set(names), set(rv_names)
    out = []
    for k in priors:
        known = names
        parts = k.split('_')
        if parts[0] == 'sigma' and parts[1:2] == ['w']:
            ds, known = parts[2:], names | rv_names
        elif parts[0] == 'GP':
            ds = parts[2:]
            if ds in (['lc'], ['rv']):            # juliet's global GPs
                continue
        elif parts[0] == 'mu':
            ds, known = parts[1:], rv_names
        elif _DATASET_PARAMETER.fullmatch(parts[0]):
            ds = parts[1:]
        elif len(parts) > 2 and re.fullmatch(r'p\d+', parts[1]):
            ds = parts[2:]
        elif (parts[0] == 'K' and len(parts) == 2) or k in ('rv_slope', 'rv_quad', 'rv_intercept'):
            if not rv_names:                       # these need RV data
                out.append(k)
            continue
        else:
            continue
        if ds and not known & set(ds):
            out.append(k)
    return out


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


def build_priors(priors, datasets, nuisance=None, verbose=True, rvs=None):
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
    rvs : dict or None
        Radial velocities ``{instrument: {'t', 'rv', 'err'}}``.

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
    if rvs:
        if not any(re.fullmatch(r'K_p\d+', k) for k in out):
            raise ValueError('Radial velocities need a prior on the semi-amplitude (K_p1).')
        if 'rv_quad' in out and 'rv_slope' not in out:
            raise ValueError('rv_quad needs rv_slope (it may be fixed to 0).')
        for n, v in rvs.items():
            lo, hi, err = np.min(v['rv']), np.max(v['rv']), float(np.median(v['err']))
            w = max(hi - lo, 3. * err)
            add(f'mu_{n}', ('uniform', [float(lo - 5. * w), float(hi + 5. * w)]))
            add(f'sigma_w_{n}', ('loguniform', [0.01 * err, float(10. * max(np.std(v['rv']), err))]))
        if 'rv_slope' in out:
            add('rv_intercept', ('fixed', 0.0))        # degenerate with the offsets mu
    for k in [k for k in out if k.split('_')[0] == 'C1']:
        sfx = k[len('C1'):]
        for c in ('C2', 'D2'):
            add(c + sfx, ('fixed', 0.0))
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
