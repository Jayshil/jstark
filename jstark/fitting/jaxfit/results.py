"""Posterior samples of a JAX fit, their convergence diagnostics and their files.

:class:`JaxResults` plays the role of juliet's ``fit`` object for the result
methods of :class:`~jstark.fitting.BroadbandFit`: ``posteriors`` is the
same dict (``{'posterior_samples': {name: array, 'loglike': ...}}``, with
``lnZ`` for nested sampling).
"""
import json
import os

import numpy as np

import jax
import jax.numpy as jnp

__all__ = ['JaxResults', 'POSTERIOR_FILE', 'DIAGNOSTICS_FILE']

POSTERIOR_FILE = 'jstark_jax_posteriors.npz'
DIAGNOSTICS_FILE = 'jstark_jax_diagnostics.ecsv'
# warning limits (Vehtari et al. 2021)
RHAT_MAX, ESS_MIN = 1.01, 400


def _batched(f, xs, batch):
    return jax.lax.map(f, xs, batch_size=max(1, min(int(batch), xs.shape[0])))


class JaxResults:
    """Posterior draws of a JAX fit (chains of NUTS; one resampled set for nested sampling).

    Attributes
    ----------
    sampler : str
    names : list of str
        Free parameters (juliet's order).
    samples : ndarray
        ``(nchains, ndraws, ndim)`` parameter values.
    loglike : ndarray
        ``(nchains, ndraws)`` log-likelihood of every draw.
    stats : dict
        Per-draw sampler statistics (``logdensity``, ``diverging``,
        ``num_steps``, ``accept``) and ``step_size`` per chain.
    settings, start, timing : dict
        Sampler options, starting point record (``map``: the maximum a
        posteriori parameters) and run times (s).
    """

    def __init__(self, sampler, names, samples, loglike, stats, settings=None, start=None,
                 timing=None, lnZ=None, lnZerr=None):
        self.sampler, self.names = sampler, list(names)
        self.samples = np.asarray(samples, float)
        self.loglike = np.asarray(loglike, float)
        self.stats = {k: np.asarray(v) for k, v in stats.items()}
        self.settings, self.start, self.timing = dict(settings or {}), dict(start or {}), dict(timing or {})
        self.lnZ, self.lnZerr = lnZ, lnZerr

    @classmethod
    def from_chains(cls, model, sampler, out, settings=None, start=None):
        """Results from the unconstrained chains ``out['z']`` of a sampler run."""
        z = jnp.asarray(out['z'])
        nch, ndraw, ndim = z.shape
        flat = z.reshape(nch * ndraw, ndim)
        x = jax.jit(jax.vmap(lambda q: model.prior_set.constrain(q)[0]))(flat)
        npts = sum(int(d['t'].size) for d in model.data.values())
        ll = jax.jit(lambda xs: _batched(lambda q: model._log_likelihood(q, model.data), xs,
                                         2e6 // max(npts, 1)))(x)
        stats = {k: out[k] for k in ('logdensity', 'diverging', 'num_steps', 'accept', 'step_size')}
        timing = {k[5:]: float(out[k]) for k in out if k.startswith('time_')}
        st = dict(start or {})
        if 'map' in st:
            st['map'] = dict(zip(model.prior_set.free, np.asarray(st['map'], float).tolist()))
        for k in ('logdensity_start', 'logdensity_end'):
            if k in st:
                st[k] = np.asarray(st[k], float).tolist()
        return cls(sampler, model.prior_set.free, np.asarray(x).reshape(nch, ndraw, ndim),
                   np.asarray(ll).reshape(nch, ndraw), stats, settings=settings, start=st,
                   timing=timing)

    # ------------------------------------------------------------------
    @property
    def nested(self):
        """True for nested sampling (resampled draws, an evidence; no chains)."""
        return self.sampler == 'blackjax-nss'

    @property
    def nchains(self):
        return self.samples.shape[0]

    @property
    def ndraws(self):
        return self.samples.shape[1]

    @property
    def posteriors(self):
        """juliet's posterior dict: ``posterior_samples`` (flattened chains), ``lnZ``."""
        post = {n: self.samples[:, :, i].ravel() for i, n in enumerate(self.names)}
        post['loglike'] = self.loglike.ravel()
        out = {'posterior_samples': post}
        if self.lnZ is not None:
            out['lnZ'], out['lnZerr'] = float(self.lnZ), float(self.lnZerr)
        return out

    def chains(self, name):
        """``(nchains, ndraws)`` samples of one free parameter."""
        return self.samples[:, :, self.names.index(name)]

    # ------------------------------------------------------------------
    def diagnostics(self):
        """Convergence of every free parameter: mean, sd, R-hat, bulk and tail ESS.

        R-hat is the rank-normalised split R-hat and the ESS the bulk and
        tail effective sample sizes of Vehtari et al. (2021); aim for
        R-hat < 1.01 and ESS > 400. The table meta holds the number of
        divergent transitions, of draws at the maximum tree depth, the mean
        acceptance rate, the step sizes and the run times.
        """
        from astropy.table import Table
        if self.nested:
            tab = Table(dict(name=self.names, mean=self.samples.mean(axis=(0, 1)),
                             sd=self.samples.std(axis=(0, 1))))
            st = self.start
            tab.meta.update(sampler=self.sampler, lnZ=self.lnZ, lnZerr=self.lnZerr,
                            ndraws=self.ndraws, ess=st.get('ess'), ndead=st.get('ndead'),
                            steps=st.get('steps'), dlogz=st.get('dlogz'), converged=st.get('converged'),
                            nlive=self.settings.get('nlive'),
                            **{f'time_{k}': v for k, v in self.timing.items()})
            return tab
        from blackjax import diagnostics as bd
        s = jnp.asarray(self.samples)
        rhat = np.asarray(jax.vmap(bd.rhat, in_axes=2)(s))
        bulk = np.asarray(jax.vmap(bd.ess_bulk, in_axes=2)(s))
        tail = np.asarray(jax.vmap(bd.ess_tail, in_axes=2)(s))
        tab = Table(dict(name=self.names, mean=self.samples.mean(axis=(0, 1)),
                         sd=self.samples.std(axis=(0, 1)), rhat=rhat, ess_bulk=bulk, ess_tail=tail))
        depth = int(self.settings.get('max_tree_depth', 10))
        tab.meta.update(sampler=self.sampler, nchains=self.nchains, ndraws=self.ndraws,
                        divergences=int(np.sum(self.stats['diverging'])),
                        max_tree_depth_hits=int(np.sum(self.stats['num_steps'] >= 2 ** depth - 1)),
                        accept=float(np.mean(self.stats['accept'])),
                        step_size=np.atleast_1d(self.stats['step_size']).tolist(),
                        mean_steps=float(np.mean(self.stats['num_steps'])),
                        **{f'time_{k}': v for k, v in self.timing.items()})
        return tab

    def report(self, tab=None):
        """One-paragraph account of the run, with warnings where convergence is doubtful."""
        tab = self.diagnostics() if tab is None else tab
        m = tab.meta
        tt = sum(v for v in self.timing.values() if np.isfinite(v))
        if self.nested:
            lines = [f'{self.sampler}: lnZ = {m["lnZ"]:.3f} +- {m["lnZerr"]:.3f}; {m["nlive"]} live points, '
                     f'{m["ndead"]} dead points in {m["steps"]} steps, {tt:.1f} s; ESS {m["ess"]:.0f}']
            if not m['converged']:
                lines.append(f'WARNING: stopped at max_steps with dlogz = {m["dlogz"]:.3g}: the evidence '
                             'and posterior are incomplete; raise max_steps.')
            if not m['ess'] >= ESS_MIN:
                lines.append(f'WARNING: effective sample size {m["ess"]:.0f} < {ESS_MIN}: raise nlive.')
            return '\n'.join(lines)
        lines = [f'{self.sampler}: {m["nchains"]} chains x {m["ndraws"]} draws in {tt:.1f} s; '
                 f'max R-hat {np.max(tab["rhat"]):.4f}, min ESS bulk {np.min(tab["ess_bulk"]):.0f} / '
                 f'tail {np.min(tab["ess_tail"]):.0f}; {m["divergences"]} divergences; '
                 f'acceptance {m["accept"]:.2f}; {m["mean_steps"]:.0f} steps per draw']
        bad = [str(n) for n, r in zip(tab['name'], tab['rhat']) if not r < RHAT_MAX]
        if bad:
            lines.append(f'WARNING: R-hat >= {RHAT_MAX} for {bad}: the chains disagree (multimodal or '
                         'not converged); run longer (nwarmup, nsamples) or check the priors.')
        low = [str(n) for n, a, b in zip(tab['name'], tab['ess_bulk'], tab['ess_tail'])
               if not min(a, b) >= ESS_MIN]
        if low:
            lines.append(f'WARNING: ESS < {ESS_MIN} for {low}: increase nsamples.')
        if m['divergences']:
            lines.append(f'WARNING: {m["divergences"]} divergent transitions: the posterior has regions '
                         'the step size cannot follow; raise target_accept (e.g. 0.95) or '
                         'reparametrise.')
        if m['max_tree_depth_hits']:
            lines.append(f'WARNING: {m["max_tree_depth_hits"]} draws reached the maximum tree depth; '
                         'raise max_tree_depth or try dense_mass=True.')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    def save(self, folder):
        """Write ``jstark_jax_posteriors.npz`` and ``jstark_jax_diagnostics.ecsv``."""
        meta = dict(sampler=self.sampler, settings=self.settings, start=self.start,
                    timing=self.timing, lnZ=self.lnZ, lnZerr=self.lnZerr)
        np.savez(os.path.join(folder, POSTERIOR_FILE), samples=self.samples, loglike=self.loglike,
                 names=np.asarray(self.names, dtype=str),
                 meta=np.asarray(json.dumps(meta, default=_jsonable)),
                 **{f'stats__{k}': v for k, v in self.stats.items()})
        tab = self.diagnostics()
        tab.write(os.path.join(folder, DIAGNOSTICS_FILE), overwrite=True)
        return tab

    @classmethod
    def load(cls, folder):
        with np.load(os.path.join(folder, POSTERIOR_FILE)) as f:
            meta = json.loads(str(f['meta']))
            stats = {k[len('stats__'):]: f[k] for k in f.files if k.startswith('stats__')}
            return cls(meta['sampler'], [str(n) for n in f['names']], f['samples'], f['loglike'],
                       stats, settings=meta.get('settings'), start=meta.get('start'),
                       timing=meta.get('timing'), lnZ=meta.get('lnZ'), lnZerr=meta.get('lnZerr'))

    @staticmethod
    def exists(folder):
        return os.path.isfile(os.path.join(folder, POSTERIOR_FILE))


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.generic,)):
        return v.item()
    return str(v)
