"""Speed benchmark of the JAX fits (for GPUs; on a CPU for comparison).

Synthetic data (a transit with an exponential ramp, two linear regressors
and white noise; made with batman), so nothing has to be downloaded. Three
tests:

``gradient``
    One log density and gradient of a :class:`JaxModel` (the cost of a
    NUTS step), for light curves of increasing length, alone and vmapped
    over many parameter vectors (parallel chains or bins).
``fit``
    A complete :meth:`~jstark.fitting.BroadbandFit.fit` (white light
    curve) with each NUTS sampler: optimiser start, warmup and sampling
    time, R-hat and ESS. ``--dynesty`` adds juliet's dynamic dynesty on
    the same fit for reference.
``spectrum``
    :meth:`~jstark.fitting.SpectroscopicFit.run` of ``nbins`` bins with
    ``blackjax-nuts`` for several ``batch`` sizes (bins vmapped at once).

Run on the machine to test (JAX picks the GPU when one is installed, e.g.
``pip install "jax[cuda12]"``)::

    python -m jstark.fitting.jaxfit.benchmark                  # all tests
    python -m jstark.fitting.jaxfit.benchmark --quick          # ~minutes, smaller sizes
    python -m jstark.fitting.jaxfit.benchmark --tests spectrum --nbins 256 --batches 1,32,256
    JAX_PLATFORMS=cpu python -m jstark.fitting.jaxfit.benchmark   # the same on the CPU

Results are printed and saved as JSON (``--out``, default
``<outdir>/benchmark_<backend>.json``) with the device, the package
versions and every timing. The fits are written below ``--outdir``
(default ``./jstark_jax_benchmark``) and re-run on every call
(``overwrite=True``).

The models run in double precision (needed for BJD times and ppm-level
light curves): consumer GPUs are much slower in float64 than data-centre
GPUs (A100, H100), so note the device in the output.
"""
import argparse
import json
import os
import platform
import time
import warnings

import numpy as np

import jax
import jax.numpy as jnp

T0 = 2460000.0
PER = 3.0
TRUTH = dict(p=0.1, a=10., b=0.4, u=(0.3, 0.2))
NOISE = 3e-4                      # per point (relative)
SHORT = dict(nwarmup=500, nsamples=500, progress=False)    # --quick sampler settings


# ----------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------
def synthetic(n, nch=1, seed=0, span=0.3):
    """``LightCurves`` of ``nch`` channels and ``n`` integrations over ``span`` days around a
    transit, with depths rising by 1% per channel, a ramp and two linear regressors."""
    import batman
    from .. import LightCurves
    rng = np.random.default_rng(seed)
    t = T0 + np.linspace(-0.5 * span, 0.5 * span, n)
    pm = batman.TransitParams()
    pm.t0, pm.per, pm.a, pm.ecc, pm.w = T0, PER, TRUTH['a'], 0., 90.
    pm.inc = np.degrees(np.arccos(TRUTH['b'] / TRUTH['a']))
    pm.u, pm.limb_dark = list(TRUTH['u']), 'quadratic'
    x, y = np.cumsum(rng.normal(0, 0.05, n)), rng.normal(0, 1, n)
    syst = (1. + 2e-4 * (x - x.mean()) - 1e-4 * y) * (1. + 1e-3 * np.exp(-(t - t[0]) / 0.01))
    flux = np.empty((n, nch))
    for j in range(nch):
        pm.rp = TRUTH['p'] * (1. + 0.01 * j)
        tm = batman.TransitModel(pm, t)
        flux[:, j] = 100. * (tm.light_curve(pm) * syst + rng.normal(0, NOISE, n))
    return LightCurves(t, flux, np.full_like(flux, 100. * NOISE),
                       wavelength=np.linspace(1., 2., nch) if nch > 1 else np.array([1.5]),
                       regressors={'x': x, 'y': y})


def white_priors():
    return {'P_p1': ('fixed', PER), 't0_p1': ('normal', [T0, 0.002]), 'p_p1': ('uniform', [0.05, 0.15]),
            'b_p1': ('uniform', [0., 0.9]), 'a_p1': ('uniform', [5., 15.]), 'ecc_p1': ('fixed', 0.),
            'omega_p1': ('fixed', 90.), 'q1_V1': ('uniform', [0., 1.]), 'q2_V1': ('uniform', [0., 1.])}


def bin_priors(info=None):
    """Orbit fixed, as in a spectroscopic fit after the white light curve."""
    return {'P_p1': ('fixed', PER), 't0_p1': ('fixed', T0), 'b_p1': ('fixed', TRUTH['b']),
            'a_p1': ('fixed', TRUTH['a']), 'ecc_p1': ('fixed', 0.), 'omega_p1': ('fixed', 90.),
            'p_p1': ('uniform', [0.05, 0.15]), 'q1_V1': ('uniform', [0., 1.]), 'q2_V1': ('uniform', [0., 1.])}


def white_fit(n, folder, seed=0):
    from .. import BroadbandFit
    return BroadbandFit({'V1': synthetic(n, seed=seed)}, priors=white_priors(), out_folder=folder,
                        linear=['x', 'y'], ramp=1, verbose=False)


# ----------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------
def _block(out):
    return jax.tree_util.tree_map(lambda a: a.block_until_ready(), out)


def _time(f, *args, repeat=5, min_time=0.5):
    """Compile time and median run time (s) of ``f(*args)``."""
    t0 = time.perf_counter()
    _block(f(*args))
    compile_time = time.perf_counter() - t0
    runs = []
    while len(runs) < repeat or sum(runs) < min_time:
        t0 = time.perf_counter()
        _block(f(*args))
        runs.append(time.perf_counter() - t0)
        if len(runs) >= 200:
            break
    return compile_time, float(np.median(runs))


def bench_gradient(sizes, widths, outdir, max_elements=5e7):
    """Log density + gradient per call, for ``n`` points and ``k`` parameter vectors at once."""
    from . import JaxModel
    rows = []
    for n in sizes:
        fit = white_fit(n, os.path.join(outdir, f'gradient_{n}'))
        m = JaxModel.from_fit(fit)
        ps = m.prior_set
        rng = np.random.default_rng(1)
        z = jnp.asarray(np.array([ps.unconstrain(x) for x in ps.sample(rng, max(widths))]))
        vg = jax.value_and_grad(m.log_density)
        for k in widths:
            if k * n > max_elements:                         # memory of the vmapped intermediates
                continue
            f = jax.jit(vg) if k == 1 else jax.jit(jax.vmap(vg))
            c, t = _time(f, z[0] if k == 1 else z[:k])
            rows.append(dict(points=int(n), width=int(k), ndim=int(ps.ndim), compile_s=c, call_s=t,
                             per_eval_us=1e6 * t / k))
            print(f'gradient: {n:7d} points x {k:5d} vmapped: {1e3 * t:9.3f} ms per call, '
                  f'{1e6 * t / k:9.2f} us per evaluation (compile {c:.1f} s)', flush=True)
    return rows


def _nuts_row(name, res, wall, extra=None):
    tab = res.diagnostics()
    row = dict(sampler=name, wall_s=wall, **{f'{k}_s': float(v) for k, v in res.timing.items()},
               nchains=int(res.nchains), ndraws=int(res.ndraws),
               rhat_max=float(np.max(tab['rhat'])), ess_bulk_min=float(np.min(tab['ess_bulk'])),
               ess_tail_min=float(np.min(tab['ess_tail'])), divergences=int(tab.meta['divergences']),
               steps_per_draw=float(tab.meta['mean_steps']))
    row['ess_per_s'] = row['ess_bulk_min'] / wall
    row.update(extra or {})
    return row


def bench_fit(n, samplers, outdir, quick=False, dynesty=False, nthreads=None):
    rows = []
    for s in samplers:
        fit = white_fit(n, os.path.join(outdir, f'fit_{s}'))
        t0 = time.perf_counter()
        res = fit.fit(sampler=s, overwrite=True, **(SHORT if quick else {'progress': False}))
        wall = time.perf_counter() - t0
        row = _nuts_row(s, res, wall, dict(points=int(n), p_p1=float(np.median(fit.values('p_p1')))))
        rows.append(row)
        print(f'fit: {s:13s} {n} points: {wall:7.1f} s (start {row.get("start_s", np.nan):.1f}, warmup '
              f'{row.get("warmup_s", np.nan):.1f}, sampling {row.get("sampling_s", np.nan):.1f}); '
              f'R-hat {row["rhat_max"]:.4f}, min ESS {row["ess_bulk_min"]:.0f} '
              f'({row["ess_per_s"]:.1f} per s); {row["divergences"]} divergences', flush=True)
    if dynesty:
        fit = white_fit(n, os.path.join(outdir, 'fit_dynamic_dynesty'))
        nthreads = nthreads or min(8, os.cpu_count() or 1)
        t0 = time.perf_counter()
        res = fit.fit(sampler='dynamic_dynesty', overwrite=True, nthreads=nthreads)
        wall = time.perf_counter() - t0
        rows.append(dict(sampler='dynamic_dynesty', wall_s=wall, nthreads=int(nthreads), points=int(n),
                         lnZ=float(res.posteriors['lnZ']), p_p1=float(np.median(fit.values('p_p1')))))
        print(f'fit: dynamic_dynesty ({nthreads} processes) {n} points: {wall:7.1f} s', flush=True)
    return rows


def bench_spectrum(n, nbins, batches, outdir, quick=False):
    from .. import SpectroscopicFit
    lc = synthetic(n, nch=nbins, seed=2)
    rows = []
    for b in batches:
        b = nbins if b in ('all', None) else min(int(b), nbins)
        sf = SpectroscopicFit({'V1': lc}, priors=bin_priors, out_folder=os.path.join(outdir, f'spectrum_b{b}'),
                              linear=['x', 'y'], ramp=1, verbose=False)
        t0 = time.perf_counter()
        spec = sf.run(sampler='blackjax-nuts', overwrite=True, batch=b,
                      **(SHORT if quick else {'progress': False}))
        wall = time.perf_counter() - t0
        f0 = sf.bin_result(int(spec['index'][0]) if 'index' in spec.colnames else 0)
        tm = f0.results.timing
        rhat = max(float(np.max(sf.bin_result(i).results.diagnostics()['rhat'])) for i in range(nbins))
        rows.append(dict(points=int(n), nbins=int(nbins), batch=int(b), wall_s=wall, per_bin_s=wall / nbins,
                         failed=len(sf.failed), rhat_max=rhat,
                         **{f'{k}_per_bin_s': float(v) for k, v in tm.items()}))
        print(f'spectrum: {nbins} bins x {n} points, batch {b:4d}: {wall:8.1f} s ({wall / nbins:.2f} s per bin; '
              f'warmup {tm.get("warmup", np.nan):.2f}, sampling {tm.get("sampling", np.nan):.2f} s per bin); '
              f'max R-hat {rhat:.4f}; failed {len(sf.failed)}', flush=True)
    return rows


# ----------------------------------------------------------------------
def environment():
    import importlib.metadata as md
    vers = {}
    for p in ('jax', 'jaxlib', 'jaxoplanet', 'blackjax', 'numpyro', 'tinygp', 'juliet', 'dynesty', 'jwstark'):
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    devs = jax.devices()
    return dict(backend=jax.default_backend(), devices=[f'{d.platform}:{d.device_kind}' for d in devs],
                ndevices=len(devs), x64=bool(jax.config.jax_enable_x64), cpu_count=os.cpu_count(),
                machine=platform.machine(), python=platform.python_version(), versions=vers,
                date=time.strftime('%Y-%m-%d %H:%M:%S'))


def _ints(s):
    return [int(v) for v in s.split(',') if v]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tests', default='gradient,fit,spectrum', help='comma-separated: gradient, fit, spectrum')
    ap.add_argument('--quick', action='store_true', help='small sizes and 500 + 500 NUTS steps')
    ap.add_argument('--sizes', help='gradient: light-curve lengths (default 1000,10000,100000)')
    ap.add_argument('--widths', help='gradient: parameter vectors vmapped at once (default 1,16,128,1024)')
    ap.add_argument('--points', type=int, help='fit: white light curve length (default 5000)')
    ap.add_argument('--samplers', default='blackjax-nuts,numpyro-nuts', help='fit: JAX samplers')
    ap.add_argument('--dynesty', action='store_true', help='fit: also juliet dynamic dynesty (CPU)')
    ap.add_argument('--nthreads', type=int, help='fit: dynesty processes (default min(8, cores))')
    ap.add_argument('--nbins', type=int, help='spectrum: bins (default 64)')
    ap.add_argument('--bin-points', type=int, help='spectrum: points per bin (default 1000)')
    ap.add_argument('--batches', help="spectrum: batch sizes, 'all' for all bins (default 1,8,all)")
    ap.add_argument('--outdir', default='jstark_jax_benchmark', help='folder of the fits and the JSON')
    ap.add_argument('--out', help='JSON file (default <outdir>/benchmark_<backend>.json)')
    a = ap.parse_args(argv)

    from . import JaxModel  # noqa: F401  (switches on float64 before anything is traced)
    q = a.quick
    sizes = _ints(a.sizes) if a.sizes else ([1000, 10000] if q else [1000, 10000, 100000])
    widths = _ints(a.widths) if a.widths else ([1, 16, 128] if q else [1, 16, 128, 1024])
    points = a.points or (1000 if q else 5000)
    nbins = a.nbins or (8 if q else 64)
    bin_points = a.bin_points or (500 if q else 1000)
    batches = [b if b == 'all' else int(b) for b in (a.batches or ('1,8' if q else '1,8,all')).split(',')]
    tests = [t.strip() for t in a.tests.split(',') if t.strip()]
    unknown = set(tests) - {'gradient', 'fit', 'spectrum'}
    if unknown:
        ap.error(f'unknown tests {sorted(unknown)}')
    os.makedirs(a.outdir, exist_ok=True)
    env = environment()
    print(f'jstark JAX benchmark: backend {env["backend"]}, devices {env["devices"]}, '
          f'jax {env["versions"]["jax"]}', flush=True)
    if env['backend'] == 'cpu' and 'spectrum' in tests and 'all' in batches:
        print('note: on a CPU vmapped bins run in lockstep (slower than batch=1)', flush=True)
    out = dict(environment=env, settings=dict(quick=q, sizes=sizes, widths=widths, points=points,
                                              nbins=nbins, bin_points=bin_points, batches=batches))
    path = a.out or os.path.join(a.outdir, f'benchmark_{env["backend"]}.json')

    def save():
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if 'gradient' in tests:
            out['gradient'] = bench_gradient(sizes, widths, a.outdir)
            save()
        if 'fit' in tests:
            out['fit'] = bench_fit(points, [s for s in a.samplers.split(',') if s], a.outdir, q,
                                   a.dynesty, a.nthreads)
            save()
        if 'spectrum' in tests:
            out['spectrum'] = bench_spectrum(bin_points, nbins, batches, a.outdir, q)
            save()
    print(f'saved {path}', flush=True)
    return out


if __name__ == '__main__':
    main()
