"""Batched NUTS: many fits of the same model structure in one vmapped run.

The fits of the wavelength bins of a :class:`~jstark.fitting.SpectroscopicFit`
have the same model (datasets, parameters, prior types, systematics) and
differ in their data, prior ranges and fixed values. Here they are sampled
together: one compiled log density takes the data, prior hyperparameters and
fixed values of a fit as arguments, and blackjax's NUTS (window adaptation
and sampling, chains vmapped) is mapped over the fits inside one compiled
program, ``batch`` fits vmapped at a time (:func:`default_batch`: all on a
GPU, 1 on a CPU). Each fit keeps its own step size and mass matrix.

* Fits are grouped by :func:`structure` (everything the compiled function
  bakes in); each group is one compiled run.
* Datasets of different lengths (points dropped in some bins) are padded to
  the longest with a ``mask``; padded points leave the likelihood exactly
  (white noise: left out of the sums; GP: zero residual, a huge variance and
  their constant terms taken off).
* Starting points and the whitening (Laplace covariance at the posterior
  maximum) are found per fit as in :func:`~jstark.fitting.jaxfit.run_sampler`,
  with the shared compiled function.
"""
import time

import numpy as np

import jax
import jax.numpy as jnp

from .samplers import check_run, _initial_positions, laplace_covariance, _stage

__all__ = ['structure', 'group_models', 'stack_data', 'default_batch', 'run_batch']

_ELEMENTS = 2e6        # model elements per chunk of the log-likelihood of the draws


def _plain(v):
    """Hashable description of a config value (functions by identity)."""
    if callable(v):
        return ('function', id(v))
    if isinstance(v, dict):
        return tuple(sorted((str(k), _plain(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_plain(x) for x in v)
    if isinstance(v, np.ndarray):
        return ('array', v.shape, v.tobytes())
    return repr(v)


def structure(model):
    """Key of everything a compiled log density of ``model`` bakes in: datasets, their model
    configuration, parameter names and prior types, fixed parameter names, model options and
    the array keys and trailing shapes of the data. Fits with equal keys can be batched."""
    data = tuple((n, tuple((k, np.ndim(v), tuple(np.shape(v)[1:])) for k, v in sorted(model.data[n].items())))
                 for n in model.names)
    ps = model.prior_set
    return (tuple(model.names), _plain(model.config), tuple(ps.free), tuple(ps.kinds),
            tuple(sorted(ps.fixed)), data, model.pl, model.pu, model.ecclim, model.light_travel_delay,
            model.stellar_radius, _plain(model.aliases), tuple(model.rv_names), model.ta)


def group_models(models):
    """Indices of ``models`` grouped by :func:`structure` (order kept)."""
    groups = {}
    for i, m in enumerate(models):
        groups.setdefault(structure(m), []).append(i)
    return list(groups.values())


def _pad(name, key, a, n):
    """Pad array ``a`` (dataset ``name``) along its first axis to ``n`` points."""
    a = np.asarray(a)
    k = n - a.shape[0]
    if k <= 0:
        return a
    if key == 'G' and a.ndim == 1 and a.size > 1:
        # GP regressor: continue past the end in the same direction, far away (the celerite
        # kernels need it sorted; distance decouples the padded points further)
        step = np.sign(a[-1] - a[0]) or 1.
        span = abs(a[-1] - a[0]) or 1.
        return np.concatenate([a, a[-1] + step * span * (1. + np.arange(1, k + 1))])
    return np.concatenate([a, np.repeat(a[-1:], k, axis=0)])


def stack_data(models):
    """Batched arguments of the shared log density of ``models`` (one structure):
    ``(data, hyper, fixed)`` with a leading axis over the fits; datasets padded to the
    longest with ``mask``."""
    t = models[0]
    data = {}
    for n in t.names:
        npts = [int(m.data[n]['t'].shape[0]) for m in models]
        nmax = max(npts)
        d = {}
        for key in t.data[n]:
            d[key] = jnp.asarray(np.stack([_pad(n, key, m.data[n][key], nmax) for m in models]))
        d['mask'] = jnp.asarray(np.stack([np.arange(nmax) < k for k in npts]).astype(float))
        data[n] = d
    hyper = jnp.asarray(np.stack([m.prior_set.hyper for m in models]))
    fixed = {k: jnp.asarray([m.prior_set.fixed[k] for m in models], float) for k in t.prior_set.fixed}
    return data, hyper, fixed


def _take(tree, i):
    return jax.tree_util.tree_map(lambda a: a[i], tree)


def default_batch(nfits):
    """Fits vmapped at once by default: all on a GPU/TPU (lanes run in parallel), 1 on a CPU.

    NUTS trees differ in size from draw to draw, and vmapped chains move in
    lockstep, each iteration costing the largest tree of all lanes. On a CPU
    the lanes are computed one after the other, so vmapping fits only adds
    that cost (8 bins x 4 chains: 1.9 times slower); there the fits run one
    after the other inside one compiled program instead (one compilation)."""
    return nfits if jax.default_backend() not in ('cpu',) else 1


def run_batch(models, sampler='blackjax-nuts', batch=None, **options):
    """Sample the posteriors of several :class:`~jstark.fitting.jaxfit.JaxModel` together.

    Parameters
    ----------
    models : list of JaxModel
        Fits to sample; those of the same :func:`structure` share one
        compiled program (blackjax NUTS, every fit with its own
        adaptation).
    sampler : str
        ``'blackjax-nuts'`` (the only batched sampler).
    batch : int or None
        Fits vmapped at once within a structure (``jax.lax.map`` batch
        size); default :func:`default_batch` (all on a GPU, 1 on a CPU).
    **options
        The options of :func:`~jstark.fitting.jaxfit.run_sampler` for
        blackjax-nuts (``chain_method`` is ignored: chains are vmapped).

    Returns
    -------
    list
        One :class:`~jstark.fitting.jaxfit.JaxResults` per model, or the
        exception that stopped that fit (e.g. no finite starting point).
    """
    name, opts = check_run(sampler, options)
    if name != 'blackjax-nuts':
        raise ValueError(f'Batched fits use blackjax-nuts, not {name}.')
    out = [None] * len(models)
    for g in group_models(models):
        for i, r in zip(g, _run_group([models[i] for i in g], name, opts, batch)):
            out[i] = r
    return out


def _run_group(models, name, opts, batch=None):
    import blackjax
    from .results import JaxResults
    t0 = time.time()
    tpl = models[0]
    ps0 = tpl.prior_set
    data, hyper, fixed = stack_data(models)
    nb, nch = len(models), opts['nchains']
    depth, dense = opts['max_tree_depth'], bool(opts['dense_mass'])

    def logd(z, args):
        d, h, f = args
        lp, x = ps0.log_density(z, h)
        return lp + tpl._log_likelihood(x, d, fixed=f)

    lpj = jax.jit(logd)
    vgj = jax.jit(jax.value_and_grad(lambda z, a: -logd(z, a)))
    hsj = jax.jit(jax.hessian(lambda z, a: -logd(z, a)))
    if opts['progress']:
        print(f'{name}: {nb} fits batched; posterior maxima ({opts["nstart"]} starts each)', flush=True)

    # starting points and whitening, per fit (shared compiled functions)
    ok, z0s, Ls, cs, starts, errors = [], [], [], [], [], {}
    for b, m in enumerate(models):
        a = _take((data, hyper, fixed), b)
        rng = np.random.default_rng(opts['seed'] + b)
        try:
            ld = (lambda q, a=a: logd(q, a))
            z0, cov, centre, start = _initial_positions(
                ld, m.prior_set, opts, rng, lp=lambda q, a=a: lpj(q, a),
                vg=lambda q, a=a: vgj(q, a), hess=lambda q, a=a: hsj(q, a))
            if cov is None:
                best = z0[int(np.argmax([float(lpj(jnp.asarray(q), a)) for q in z0]))]
                cov, start['hessian_positive_definite'] = laplace_covariance(
                    ld, best, hess=lambda q, a=a: hsj(q, a))
                centre = best
        except Exception as exc:                   # this fit only
            errors[b] = exc
            continue
        L = np.linalg.cholesky(cov)
        ok.append(b)
        z0s.append(np.linalg.solve(L, (z0 - centre).T).T)
        Ls.append(L)
        cs.append(centre)
        starts.append(start)
    tstart = time.time() - t0
    results = [errors.get(b) for b in range(nb)]
    if not ok:
        return results
    sel = jnp.asarray(ok)
    args = jax.tree_util.tree_map(lambda a: a[sel], (data, hyper, fixed))
    wargs = (args, jnp.asarray(np.stack(Ls)), jnp.asarray(np.stack(cs)))
    u0 = jnp.asarray(np.stack(z0s))                 # (nfit, nchains, ndim)

    def target(u, wa):
        a, L, c = wa
        return logd(c + L @ u, a)

    def warm(key, u, wa):
        ad = blackjax.window_adaptation(blackjax.nuts, lambda q: target(q, wa),
                                        is_mass_matrix_diagonal=not dense,
                                        target_acceptance_rate=float(opts['target_accept']),
                                        max_num_doublings=depth)
        (state, params), _ = ad.run(key, u, num_steps=opts['nwarmup'])
        return state, params['step_size'], params['inverse_mass_matrix']

    def sample(key, state, step, inv, wa):
        kernel = blackjax.nuts(lambda q: target(q, wa), step_size=step, inverse_mass_matrix=inv,
                               max_num_doublings=depth)

        def one(s, k):
            s, i = kernel.step(k, s)
            return s, (s.position, s.logdensity, i.is_divergent, i.num_integration_steps,
                       i.acceptance_rate)

        _, o = jax.lax.scan(one, state, jax.random.split(key, opts['nsamples']))
        return o

    over_chains_w = jax.vmap(warm, in_axes=(0, 0, None))
    over_chains_s = jax.vmap(sample, in_axes=(0, 0, 0, 0, None))
    key = jax.random.PRNGKey(opts['seed'])
    kw, ks = jax.random.split(key)
    nfit = len(ok)
    bv = default_batch(nfit) if batch is None else max(1, min(int(batch), nfit))
    kws = jax.random.split(kw, nfit * nch).reshape(nfit, nch, -1)
    kss = jax.random.split(ks, nfit * nch).reshape(nfit, nch, -1)

    # fits mapped (bv at a time vmapped, the rest one after the other) inside one program
    @jax.jit
    def warm_all(kws, u0, wargs):
        return jax.lax.map(lambda a: over_chains_w(*a), (kws, u0, wargs), batch_size=bv)

    @jax.jit
    def sample_all(kss, state, step, inv, wargs):
        return jax.lax.map(lambda a: over_chains_s(*a), (kss, state, step, inv, wargs), batch_size=bv)

    t = time.time()
    with _stage(f'warmup, {nfit} fits, {bv} at once', opts, opts['nwarmup']):
        state, step, inv = jax.block_until_ready(warm_all(kws, u0, wargs))
    tw = time.time() - t
    with _stage(f'sampling, {nfit} fits, {bv} at once', opts, opts['nsamples']):
        o = jax.block_until_ready(sample_all(kss, state, step, inv, wargs))
    ts = time.time() - t - tw
    u, logdens, div, nsteps, acc = (np.asarray(v) for v in o)

    # parameter values and log-likelihoods of the draws (shared compiled function, chunked)
    npts = sum(int(v['t'].shape[1]) for v in data.values())
    chunk = max(1, int(_ELEMENTS // max(npts, 1)))

    def values(zs, a):
        x = jax.vmap(lambda q: ps0.constrain(q, a[1])[0])(zs)
        ll = jax.lax.map(lambda q: tpl._log_likelihood(q, a[0], fixed=a[2]), x, batch_size=chunk)
        return x, ll

    valj = jax.jit(values)
    # per fit: the shares of the group's times (they add up over the fits)
    tsec = dict(time_start=tstart / nb, time_warmup=tw / nfit, time_sampling=ts / nfit)
    for k, b in enumerate(ok):
        z = cs[k] + u[k] @ Ls[k].T                  # (nchains, ndraws, ndim)
        x, ll = valj(jnp.asarray(z.reshape(-1, z.shape[-1])), _take(args, k))
        st = dict(starts[k])
        if 'map' in st:
            st['map'] = dict(zip(ps0.free, np.asarray(st['map'], float).tolist()))
        for key_ in ('logdensity_start', 'logdensity_end'):
            if key_ in st:
                st[key_] = np.asarray(st[key_], float).tolist()
        st['batch'] = dict(fits=nfit, vmapped=bv)
        stats = dict(logdensity=logdens[k], diverging=div[k], num_steps=nsteps[k], accept=acc[k],
                     step_size=np.asarray(step[k]))
        timing = {kk[5:]: float(v) for kk, v in tsec.items()}
        results[b] = JaxResults(name, ps0.free, np.asarray(x).reshape(z.shape), np.asarray(ll).reshape(z.shape[:2]),
                                stats, settings=dict(opts, batched=True), start=st, timing=timing)
    return results
