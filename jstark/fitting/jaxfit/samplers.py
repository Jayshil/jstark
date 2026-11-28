"""Samplers of the JAX fits: NUTS from blackjax or numpyro.

Every sampler works on the unconstrained parameters ``z`` of a
:class:`~jstark.fitting.jaxfit.PriorSet` (``JaxModel.log_density``), so the
chains never hit a prior boundary.

Starting points (``init='map'``, default): the maximum of the posterior
from L-BFGS runs started at the prior centre and at ``nstart - 1`` prior
draws; each chain then starts at a draw from the Laplace approximation
around it. The samplers work in coordinates whitened by that approximation
(``u = L^-1 (z - z_map)``, ``L L^T`` the Laplace covariance), where all
scales are about 1. ``init='prior'`` starts the chains at prior draws
instead (whitened with the Hessian at the best of them).
"""
import contextlib
import time
import warnings

import numpy as np

import jax
import jax.numpy as jnp

from ..fit import JAX_SAMPLERS as SAMPLERS, jax_sampler_name as sampler_name

__all__ = ['SAMPLERS', 'OPTIONS', 'NSS_OPTIONS', 'options_of', 'sampler_name', 'check_run',
           'find_start', 'run_sampler']

# options of the JAX samplers and their defaults
# (target_accept 0.9 and a dense mass matrix: transit models curve sharply at the contact
# points, and b, a/R*, Rp/R* and the limb darkening are correlated; on a 600-point transit with a
# ramp, 0.8 and a diagonal mass gave 39 divergences and ESS ~120 for b and a, these ESS ~1200)
OPTIONS = dict(nchains=4, nwarmup=1000, nsamples=1000, seed=0, target_accept=0.9,
               max_tree_depth=10, dense_mass=True, chain_method='vectorized', init='map',
               nstart=8, progress=True)


# options of the nested slice sampler (blackjax-nss); None: chosen from nlive and the dimension
NSS_OPTIONS = dict(nlive=500, num_delete=None, num_inner_steps=None, dlogz=0.01, max_steps=200000,
                   nsamples=None, seed=0, progress=True)


def options_of(name):
    """Options and defaults of a JAX sampler."""
    return NSS_OPTIONS if name == 'blackjax-nss' else OPTIONS


def _check_options(options, name='blackjax-nuts'):
    defaults = options_of(name)
    out = dict(defaults)
    unknown = set(options) - set(defaults)
    if unknown:
        raise TypeError(f'Unknown option(s) of {name}: {sorted(unknown)} (available: {sorted(defaults)}).')
    out.update(options)
    if name == 'blackjax-nss':
        for k in ('nlive', 'num_delete', 'num_inner_steps', 'max_steps', 'nsamples'):
            if out[k] is not None:
                out[k] = int(out[k])
                if out[k] < 1:
                    raise ValueError(f'{k} must be positive.')
        if out['num_delete'] is not None and out['num_delete'] >= out['nlive']:
            raise ValueError('num_delete must be smaller than nlive.')
        if not float(out['dlogz']) > 0:
            raise ValueError('dlogz must be positive.')
        return out
    for k in ('nchains', 'nwarmup', 'nsamples', 'max_tree_depth', 'nstart'):
        out[k] = int(out[k])
        if out[k] < (0 if k == 'nwarmup' else 1):
            raise ValueError(f'{k} must be positive.')
    if not 0 < float(out['target_accept']) < 1:
        raise ValueError('target_accept must be between 0 and 1.')
    if out['chain_method'] not in ('vectorized', 'sequential'):
        raise ValueError("chain_method must be 'vectorized' (all chains at once) or 'sequential'.")
    return out


def check_run(sampler, options):
    """Standard sampler name and complete options; raises for unknown ones."""
    name = sampler_name(sampler)
    if name is None:
        raise ValueError(f'Unknown JAX sampler {sampler!r} (available: {list(SAMPLERS)}).')
    return name, _check_options(options, name)


# ----------------------------------------------------------------------
# Starting points
# ----------------------------------------------------------------------
def find_start(logdens, prior_set, rng, nstart=8, maxiter=3000, vg=None, hess=None):
    """Maximum of ``logdens`` (unconstrained ``z``) from several L-BFGS starts.

    Returns ``(z, cov, info)``: the best point, the covariance of the
    Laplace approximation (None if the Hessian is not positive definite)
    and a dict with the log density of every start before and after.
    ``vg``, ``hess``: compiled value-and-gradient and Hessian of
    ``-logdens`` (batched fits share them; default: compiled here).
    """
    from scipy.optimize import minimize

    if vg is None:
        vg = jax.jit(jax.value_and_grad(lambda z: -logdens(z)))

    def fun(z):
        v, g = vg(jnp.asarray(z))
        v, g = float(v), np.asarray(g, float)
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            return 1e300, np.zeros_like(z)
        return v, g

    starts = [prior_set.unconstrain(prior_set.centre())]
    draws = prior_set.sample(rng, 20 * nstart)
    for x in draws:
        if len(starts) >= nstart:
            break
        starts.append(prior_set.unconstrain(x))
    good, before = [], []
    for z in starts:
        v = fun(z)[0]
        if v < 1e300:
            good.append(z)
            before.append(-v)
    if not good:
        raise RuntimeError('The log-likelihood is -inf at the prior centre and at all prior '
                           'draws tried: check the priors (e.g. b > 1 + p everywhere, or a '
                           'non-linear function that returns NaN).')
    best, after = None, []
    for z in good:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = minimize(fun, z, jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter))
        after.append(-float(r.fun))
        if best is None or r.fun < best.fun:
            best = r
    z = np.asarray(best.x, float)
    cov, pd = laplace_covariance(logdens, z, hess=hess)
    return z, cov, dict(logdensity_start=np.asarray(before), logdensity_end=np.asarray(after),
                        logdensity_max=-float(best.fun), hessian_positive_definite=pd)


def laplace_covariance(logdens, z, max_sd=10., hess=None):
    """Covariance of the Laplace approximation of ``logdens`` at ``z`` (unconstrained).

    The eigenvalues of the Hessian of ``-logdens`` are replaced by their
    absolute values and floored at ``1 / max_sd**2``, so the result is
    always positive definite (flat or curved-the-wrong-way directions get a
    width of at most ``max_sd``). Returns ``(cov, positive_definite)``.
    ``hess``: compiled Hessian of ``-logdens`` (default: compiled here).
    """
    if hess is None:
        hess = jax.jit(jax.hessian(lambda q: -logdens(q)))
    H = np.asarray(hess(jnp.asarray(z)), float)
    if not np.all(np.isfinite(H)):
        return np.eye(len(z)), False
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    pd = bool(w.min() > 0)
    w = np.maximum(np.abs(w), 1. / max_sd ** 2)
    cov = (V / w) @ V.T
    return 0.5 * (cov + cov.T), pd


def _initial_positions(logdens, prior_set, opts, rng, lp=None, vg=None, hess=None):
    """(nchains, ndim) starting points, the Laplace covariance (or None), its centre and a record.
    ``lp``, ``vg``, ``hess``: compiled log density, value-and-gradient and Hessian (batched fits)."""
    n = opts['nchains']
    lp = jax.jit(logdens) if lp is None else lp
    init = opts['init']
    info = {}
    if isinstance(init, str) and init == 'map':
        zmap, cov, info = find_start(logdens, prior_set, rng, nstart=opts['nstart'], vg=vg, hess=hess)
        out = []
        L = np.linalg.cholesky(cov)
        for _ in range(n):
            for _try in range(100):
                dz = L @ rng.standard_normal(zmap.size)
                if np.isfinite(float(lp(jnp.asarray(zmap + dz)))):
                    break
            else:
                dz = 1e-4 * rng.standard_normal(zmap.size)
            out.append(zmap + dz)
        info['map'] = prior_set.constrain(jnp.asarray(zmap))[0]
        return np.asarray(out), cov, zmap, info
    if isinstance(init, str) and init == 'prior':
        out = []
        for x in prior_set.sample(rng, 200 * n):
            z = prior_set.unconstrain(x)
            if np.isfinite(float(lp(jnp.asarray(z)))):
                out.append(z)
            if len(out) == n:
                return np.asarray(out), None, None, info
        raise RuntimeError('Fewer than nchains prior draws with a finite log-likelihood.')
    # values: a dict {name: value} or an array (ndim,) / (nchains, ndim) of parameter values
    if isinstance(init, dict):
        missing = [k for k in prior_set.free if k not in init]
        if missing:
            raise ValueError(f'init: no value for {missing}.')
        x = np.array([init[k] for k in prior_set.free], float)
    else:
        x = np.asarray(init, float)
    single = x.ndim == 1
    x = np.broadcast_to(x, (n, prior_set.ndim)) if single else x
    if x.shape != (n, prior_set.ndim):
        raise ValueError(f'init: shape {x.shape}, expected ({n}, {prior_set.ndim}) or ({prior_set.ndim},).')
    z = np.array([prior_set.unconstrain(xi) for xi in x])
    if single and n > 1:                       # chains that start apart make R-hat meaningful
        z = z + 1e-3 * rng.standard_normal(z.shape)
    return z, None, None, info


# ----------------------------------------------------------------------
# Samplers
# ----------------------------------------------------------------------
def _stage(label, opts, steps):
    """blackjax's progress bar if its extra (jaxtap) is installed, else one line per stage."""
    if not opts['progress']:
        return contextlib.nullcontext()
    try:
        import jaxtap  # noqa: F401
    except ImportError:
        print(f'blackjax-nuts: {label} ({opts["nchains"]} chains x {steps} steps)', flush=True)
        return contextlib.nullcontext()
    import blackjax
    return blackjax.progress_bar(label=label)


def _blackjax_nuts(logdens, z0, imm, opts, key):
    import blackjax
    depth, dense = opts['max_tree_depth'], bool(opts['dense_mass'])
    nwarm, ndraw = opts['nwarmup'], opts['nsamples']
    imm0 = None if imm is None else jnp.asarray(imm)

    def warm(key, z):
        wa = blackjax.window_adaptation(blackjax.nuts, logdens, is_mass_matrix_diagonal=not dense,
                                        initial_inverse_mass_matrix=imm0,
                                        target_acceptance_rate=float(opts['target_accept']),
                                        max_num_doublings=depth)
        (state, params), _ = wa.run(key, z, num_steps=nwarm)
        return state, params['step_size'], params['inverse_mass_matrix']

    def sample(key, state, step, inv_mass):
        kernel = blackjax.nuts(logdens, step_size=step, inverse_mass_matrix=inv_mass,
                               max_num_doublings=depth)

        def one(s, k):
            s, i = kernel.step(k, s)
            return s, (s.position, s.logdensity, i.is_divergent, i.num_integration_steps,
                       i.acceptance_rate)

        _, out = jax.lax.scan(one, state, jax.random.split(key, ndraw))
        return out

    kw, ks = jax.random.split(key)
    kws, kss = jax.random.split(kw, len(z0)), jax.random.split(ks, len(z0))
    t = time.time()
    if opts['chain_method'] == 'vectorized':
        with _stage('warmup', opts, nwarm):
            state, step, inv = jax.block_until_ready(jax.jit(jax.vmap(warm))(kws, jnp.asarray(z0)))
        tw = time.time() - t
        with _stage('sampling', opts, ndraw):
            out = jax.block_until_ready(jax.jit(jax.vmap(sample))(kss, state, step, inv))
    else:
        jw, js = jax.jit(warm), jax.jit(sample)
        with _stage('warmup', opts, nwarm):
            res = jax.block_until_ready([jw(kws[i], jnp.asarray(z0[i])) for i in range(len(z0))])
        tw = time.time() - t
        with _stage('sampling', opts, ndraw):
            outs = jax.block_until_ready([js(kss[i], s_, st, iv) for i, (s_, st, iv) in enumerate(res)])
        step = jnp.stack([r[1] for r in res])
        out = tuple(jnp.stack([o[j] for o in outs]) for j in range(5))
    ts = time.time() - t - tw
    z, logd, div, nsteps, acc = (np.asarray(v) for v in out)
    return dict(z=z, logdensity=logd, diverging=div, num_steps=nsteps, accept=acc,
                step_size=np.asarray(step), time_warmup=tw, time_sampling=ts)


def _numpyro_nuts(logdens, z0, imm, opts, key):
    from numpyro.infer import MCMC, NUTS
    kernel = NUTS(potential_fn=lambda z: -logdens(z), target_accept_prob=float(opts['target_accept']),
                  max_tree_depth=opts['max_tree_depth'], dense_mass=bool(opts['dense_mass']),
                  inverse_mass_matrix=None if imm is None else jnp.asarray(imm))
    mcmc = MCMC(kernel, num_warmup=opts['nwarmup'], num_samples=opts['nsamples'],
                num_chains=len(z0), chain_method=opts['chain_method'],
                progress_bar=bool(opts['progress']))
    t = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*There are not enough devices.*')
        mcmc.run(key, init_params=jnp.asarray(z0) if len(z0) > 1 else jnp.asarray(z0[0]),
                 extra_fields=('diverging', 'num_steps', 'accept_prob', 'potential_energy'))
    z = np.asarray(mcmc.get_samples(group_by_chain=True))     # waits for the run (JAX is asynchronous)
    e = {k: np.asarray(v) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
    tt = time.time() - t
    step = np.atleast_1d(np.asarray(mcmc.last_state.adapt_state.step_size))
    return dict(z=z, logdensity=-e['potential_energy'], diverging=e['diverging'],
                num_steps=e['num_steps'], accept=e['accept_prob'], step_size=step,
                time_warmup=np.nan, time_sampling=tt)


def run_sampler(model, sampler, **options):
    """Sample the posterior of a :class:`~jstark.fitting.jaxfit.JaxModel`.

    Parameters
    ----------
    model : JaxModel
    sampler : str
        ``'blackjax-nuts'``, ``'numpyro-nuts'`` (NUTS) or ``'blackjax-nss'``
        (nested slice sampling: posterior and evidence ``lnZ``).
    **options
        ``nchains`` (4), ``nwarmup`` (1000), ``nsamples`` (1000 per chain),
        ``seed`` (0), ``target_accept`` (0.9), ``max_tree_depth`` (10),
        ``dense_mass`` (True), ``chain_method`` (``'vectorized'``: all
        chains in one vmapped computation; ``'sequential'``), ``init``
        (``'map'``, ``'prior'``, a dict ``{name: value}`` or an array of
        parameter values), ``nstart`` (8 optimiser starts), ``progress``.
        ``blackjax-nss``: ``nlive`` (500), ``num_delete`` (nlive / 10
        replaced per step), ``num_inner_steps`` (5 x dimension slice steps
        per new point), ``dlogz`` (0.01), ``max_steps``, ``nsamples``
        (posterior draws; default the effective sample size), ``seed``,
        ``progress``.

    Returns
    -------
    :class:`~jstark.fitting.jaxfit.JaxResults`
    """
    from .results import JaxResults
    name, opts = check_run(sampler, options)
    if name == 'blackjax-nss':
        return _blackjax_nss(model, opts)
    rng = np.random.default_rng(opts['seed'])
    data = model.data

    def logdens(z):
        return model._log_density(z, data)

    t = time.time()
    lp = jax.jit(logdens)
    try:
        float(lp(jnp.asarray(model.prior_set.unconstrain(model.prior_set.centre()))))
    except jax.errors.TracerArrayConversionError as exc:
        raise TypeError('A non-linear function converts JAX arrays to NumPy: write it with '
                        'jax.numpy (import jax.numpy as jnp; jnp.exp, jnp.sin, ...) for the JAX '
                        f'samplers ({exc}).') from exc
    if opts['progress'] and isinstance(opts['init'], str) and opts['init'] == 'map':
        print(f'{name}: posterior maximum from {opts["nstart"]} starts', flush=True)
    z0, cov, centre, start = _initial_positions(logdens, model.prior_set, opts, rng)
    tstart = time.time() - t
    if cov is None:          # init='prior' or given values: scales from the best starting point
        best = z0[int(np.argmax([float(lp(jnp.asarray(q))) for q in z0]))]
        cov, start['hessian_positive_definite'] = laplace_covariance(logdens, best)
        centre = best
    # Sample u = L^-1 (z - centre), whitened by the Laplace approximation (L L^T = cov): every
    # scale is ~1 and the correlations at the maximum are removed. The warmup's mass-matrix
    # regularisation assumes unit scales (it adds ~1e-3 to the variances), so without this
    # parameters with small posterior widths get far too large a mass matrix and tiny steps.
    # An affine map: the posterior is unchanged (constant Jacobian).
    L, c = np.linalg.cholesky(cov), centre
    Lj, cj = jnp.asarray(L), jnp.asarray(c)

    def target(u):
        return logdens(cj + Lj @ u)

    u0 = np.linalg.solve(L, (z0 - c).T).T
    key = jax.random.PRNGKey(opts['seed'])
    run = _blackjax_nuts if name == 'blackjax-nuts' else _numpyro_nuts
    out = run(target, u0, None, opts, key)
    out['z'] = c + out['z'] @ L.T
    out['time_start'] = tstart
    return JaxResults.from_chains(model, name, out, settings=opts, start=start)


# ----------------------------------------------------------------------
# Nested slice sampling (evidence)
# ----------------------------------------------------------------------
def _blackjax_nss(model, opts):
    """blackjax nested slice sampling (Yallup, Kroupa & Handley 2026) on the unconstrained ``z``.

    The prior there is the prior density of ``z`` (prior at ``x(z)`` times
    the Jacobian), normalised, so ``lnZ`` is the evidence of the juliet
    priors, comparable with dynesty's. Live points start at prior draws;
    the run stops when the live points could add less than ``dlogz`` to
    ``lnZ`` (dynesty's criterion). Posterior draws are resampled from the
    weighted dead points (``nsamples``, default the effective sample size).
    """
    import blackjax
    from blackjax.ns import utils as nsu
    from jax.scipy.special import logsumexp
    from .results import JaxResults

    ps, data = model.prior_set, model.data
    nlive, ndim = opts['nlive'], ps.ndim
    ndel = opts['num_delete'] or max(1, nlive // 10)
    ninner = opts['num_inner_steps'] or max(5, 5 * ndim)

    def logprior(z):
        return ps.log_density(z)[0]

    def loglike(z):
        return model._log_likelihood(ps.constrain(z)[0], data)

    rng = np.random.default_rng(opts['seed'])
    t = time.time()
    z0 = np.array([ps.unconstrain(x) for x in ps.sample(rng, nlive)])
    ll0 = np.asarray(jax.jit(jax.vmap(loglike))(jnp.asarray(z0)))
    if not np.any(np.isfinite(ll0)):
        raise RuntimeError('The log-likelihood is -inf at all prior draws: check the priors.')
    algo = blackjax.nss(logprior_fn=logprior, loglikelihood_fn=loglike, num_inner_steps=ninner,
                        num_delete=ndel)
    key = jax.random.PRNGKey(opts['seed'])
    key, k0 = jax.random.split(key)
    state = algo.init(jnp.asarray(z0), rng_key=k0)
    step = jax.jit(algo.step)
    if opts['progress']:
        print(f'blackjax-nss: {nlive} live points, {ndel} replaced per step, {ninner} slice steps '
              f'each, until dlogz < {opts["dlogz"]}', flush=True)
    dead, nstep, dlogz = [], 0, np.inf
    report = time.time()
    while nstep < opts['max_steps']:
        key, k = jax.random.split(key)
        state, info = step(k, state)
        dead.append(info)
        nstep += 1
        I = state.integrator
        dlogz = float(jnp.logaddexp(I.logZ, I.logZ_live) - I.logZ)
        if dlogz < opts['dlogz']:
            break
        if opts['progress'] and time.time() - report > 30.:
            print(f'  step {nstep}: lnZ {float(I.logZ):.2f}, dlogz {dlogz:.3g}', flush=True)
            report = time.time()
    converged = dlogz < opts['dlogz']
    fin = nsu.finalise(state, dead)
    k1, k2, k3 = jax.random.split(key, 3)
    lnz = np.asarray(logsumexp(nsu.log_weights(k1, fin, shape=200), axis=0))
    ess = float(nsu.ess(k2, fin))
    n = opts['nsamples'] or max(1, int(ess))
    draws = nsu.sample(k3, fin, shape=n)
    z = np.asarray(draws.position)[None]
    tt = time.time() - t
    out = dict(z=z, logdensity=np.asarray(draws.loglikelihood)[None], diverging=np.zeros((1, n), bool),
               num_steps=np.zeros((1, n), int), accept=np.ones((1, n)), step_size=np.zeros(1),
               time_sampling=tt)
    ndead = int(np.asarray(fin.particles.loglikelihood).size)
    settings = dict(opts, num_delete=ndel, num_inner_steps=ninner)
    start = dict(steps=nstep, ndead=ndead, ess=ess, dlogz=dlogz, converged=bool(converged))
    res = JaxResults.from_chains(model, 'blackjax-nss', out, settings=settings, start=start)
    res.lnZ, res.lnZerr = float(np.mean(lnz)), float(np.std(lnz))
    return res
