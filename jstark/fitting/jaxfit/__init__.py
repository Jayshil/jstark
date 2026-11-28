"""Light-curve models and likelihoods in JAX (optional backend of jstark fits).

Needs ``jax`` and ``jaxoplanet``. Importing this module switches JAX to
double precision (``jax_enable_x64``): full BJD times and ppm-level light
curves need it.

The models reproduce juliet's (batman's) models exactly, so a fit with a JAX
sampler fits the same model as a fit with dynesty:

* :class:`PriorSet`: juliet's prior distributions, with transforms to an
  unconstrained space for gradient-based samplers.
* :class:`JaxModel`: the light-curve model and log-likelihood of every
  dataset of a :class:`~jstark.fitting.BroadbandFit`.
* :func:`run_sampler`: NUTS from blackjax (``'blackjax-nuts'``) or numpyro
  (``'numpyro-nuts'``), blackjax nested slice sampling (``'blackjax-nss'``,
  with the evidence); :class:`JaxResults` holds the draws.
* :mod:`.gp`: juliet's GP kernels in tinygp.
* :class:`JaxEvaluator`: posterior models for the result methods.

You do not need these directly: ``BroadbandFit.fit(sampler='blackjax-nuts')``
uses them.
"""
import jax

jax.config.update('jax_enable_x64', True)

from .priors import PriorSet  # noqa: E402
from .model import JaxModel  # noqa: E402
from .results import JaxResults  # noqa: E402
from .samplers import run_sampler, OPTIONS, NSS_OPTIONS  # noqa: E402
from .evaluate import JaxEvaluator  # noqa: E402

__all__ = ['PriorSet', 'JaxModel', 'JaxResults', 'JaxEvaluator', 'run_sampler', 'OPTIONS', 'NSS_OPTIONS']
