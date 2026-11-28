"""Plots of light-curve fits: full models, phase-folded light curves, GP components,
noise versus bin size, corner plots and radial velocities.

The models come from the evaluator of the fit (:mod:`jstark.fitting.evaluators`):
juliet's models for juliet fits, the JAX model for JAX fits. ``julietPlots`` was
moved here from ``jstark/jstark.py``.
"""
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gd
import juliet

from ..plotting import styled, INK2, MUTED, PALETTE, WIDTH
from .utils import t14, pipe_mad, lcbin, fake_allan_deviation, corner_plot

__all__ = ['julietPlots']


def _bin_data(x, y, n_bin, yerr=None):
    """``juliet.utils.bin_data`` that survives few points.

    juliet's version loops with step ``n_bin`` (0 raises) and averages
    ``x[i:i + n_bin - 1]`` (empty for ``n_bin = 1``). Empty input (e.g. no
    data in the transit window of a phase curve that misses the transit)
    gives empty arrays.
    """
    x = np.asarray(x)
    if x.size == 0:
        return np.array([]), np.array([]), np.array([])
    return juliet.utils.bin_data(x=x, y=y, n_bin=max(2, int(n_bin)), yerr=yerr)


def _robust_ylim(ax, v, k=5.):
    """y-limits of median +- k robust sigma of ``v``; left alone if ``v`` has no finite values."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return
    m, s = np.median(v), pipe_mad(v)
    if np.isfinite(s) and s > 0:
        ax.set_ylim(m - k * s, m + k * s)


class julietPlots(object):
    """Plots of a light-curve fit: full models, phase-folded light curves, GP
    components, noise versus bin size, corner plots and radial velocities.

    The models come from an evaluator (:mod:`jstark.fitting.evaluators`):
    juliet's models for juliet fits, the JAX model for JAX fits
    (:meth:`BroadbandFit.plots` picks it). Without one, the class loads a
    juliet input/output folder (``input_folder``) or takes juliet's
    ``dataset`` and ``res``.

    Attributes
    ----------
    input_folder : str, optional
        Folder of the fit (where figures are saved).
    dataset : juliet.load object, optional
        juliet's dataset (juliet fits only).
    res : juliet.fit or JaxResults, optional
        Results of the fit.
    N : int
        Number of posterior samples drawn when evaluating sampled models.
    evaluator : optional
        Model interface of the fit (default: juliet's, from ``dataset`` and ``res``).
    **kwargs : dict, optional
        Any additional keywords provided to juliet.fit object.

    Main methods
    ------------
    full_model_lc(instruments=None, save=False, nrandom=50, quantile_models=True)
        Plot observed light curves with the full fitted model and residuals.

    phase_folded_lc(phmin=0.8, instruments=None, highres=False, ...)
        Plot phase folder light curve with best-fitted planetary model for one or more instruments.

    plot_gp(instruments=None, highres=False, one_plot=False, ...)
        Plot GP components and binned residuals for one or more instruments.

    plot_fake_allan_deviation(instruments=None, binmax=10, method='pipe', timeunit=None)
        Plot "allan deviation" plot, which is noise as a function of binning, for one or more instruments

    plot_corner(planet_only=False, save=True)
        Plot corner plots for fitted posterior samples, can use planet-only or all parameters.

    plot_rv(nrandom=300, save=False)
        Radial velocities and the phase-folded orbits (BroadbandFit with ``rvs``).
    """
    def __init__(self, input_folder=None, dataset=None, res=None, N=1000, evaluator=None, **kwargs):
        if evaluator is None:
            if (dataset is None) and (input_folder is not None):
                dataset = juliet.load(input_folder=input_folder)
                res = dataset.fit(**kwargs)
            from .evaluators import JulietEvaluator
            evaluator = JulietEvaluator(dataset, res, fit_kwargs=kwargs)
        self.ev = evaluator
        self.dataset = getattr(evaluator, 'dataset', None)
        self.res = evaluator.results
        self.fit_kwargs = kwargs

        # Location of the input folder (falls back to juliet's output folder)
        self.input_folder = input_folder if input_folder is not None else getattr(self.dataset, 'out_folder', None)

        self.names = list(evaluator.names)
        self.priors = evaluator.priors
        self.times, self.flux, self.errs = {}, {}, {}
        # linear and GP regressors of the datasets that have them
        self._lm, self._gp = {}, {}
        for n in self.names:
            self.times[n], self.flux[n], self.errs[n] = evaluator.data(n)
            reg = evaluator.regressors(n)
            if reg['linear'] is not None:
                self._lm[n] = reg['linear']
            if reg['gp'] is not None:
                self._gp[n] = reg['gp']

        # Number of samples
        self.nsamps = N

        # Models of all datasets: lc_models[name] is the evaluator's dict; models_all_ins[name] is
        # (samples, median, upper 68%, lower 68%, {'lm': median linear model})
        self.lc_models, self.models_all_ins = {}, {}
        # Sorting indices for all instruments (sometimes GP regressors are not time -- in that case, times
        # are sorted according to GP regressors, which will mess-up the time array -- we need to sort it again)
        self.idx_time_sort = {}
        for n in self.names:
            m = evaluator.lc_models(n, nsamples=self.nsamps)
            self.lc_models[n] = m
            comp = {} if m['linear'] is None else {'lm': m['linear']}
            self.models_all_ins[n] = (m['samples'], m['median'], m['upper'], m['lower'], comp)
            self.idx_time_sort[n] = np.argsort(self.times[n])

    def _gpx(self, instrument):
        """First (or only) column of the GP regressor of an instrument."""
        x = np.asarray(self._gp[instrument])
        return x if x.ndim == 1 else x[:, 0]

    #: The :class:`BroadbandFit` the plots belong to (for :meth:`plot_rv`)
    fit = None
    #: Dataset groups of the fit (:class:`BroadbandFit`), accepted wherever instruments are given
    groups = {}

    def _instruments(self, instruments):
        """List of instrument names (None: all; a single name or a group is accepted)."""
        if instruments is None:
            return list(self.names)
        if isinstance(instruments, str):
            instruments = [instruments]
        return [m for i in instruments for m in self.groups.get(i, [i])]

    def _samples(self, prefix, instrument, default):
        """Posterior samples of an instrument parameter (``prefix_..._instrument``), its fixed
        value, or ``default``."""
        post = self.ev.posteriors
        n = len(prefix.split('_'))
        for k in post:
            parts = k.split('_')
            if '_'.join(parts[:n]) == prefix and instrument in parts[n:]:
                return np.asarray(post[k])
        for k, v in self.priors.items():
            parts = k.split('_')
            if '_'.join(parts[:n]) == prefix and instrument in parts[n:] and v['distribution'] == 'fixed':
                return np.atleast_1d(float(np.atleast_1d(v['hyperparameters'])[0]))
        return np.atleast_1d(float(default))

    def _nlm(self, instrument):
        """Non-linear model of an instrument at the posterior medians: ``(values,
        multiplicative)``, or ``(None, False)`` if there is none."""
        return self.ev.non_linear(instrument)

    def _pmed(self, name):
        """Posterior median of a parameter, else its fixed prior value (KeyError if neither)."""
        from .evaluators import median_or_fixed
        return median_or_fixed(self.ev.posteriors, self.priors, name)

    def _period_t0(self, planet, instrument):
        """Period and time of transit of a planet (posterior medians, fixed values, or the
        evaluator's orbit)."""
        out = []
        for k in ('P', 't0'):
            try:
                out.append(self._pmed(f'{k}_{planet}'))
            except KeyError:
                out.append(self.ev.orbit(planet, instrument)[k])
        return tuple(out)

    @styled
    def full_model_lc(self, instruments=None, save=False, nrandom=50, quantile_models=True):
        """Plot the full fitted light-curve model for one or more instruments.

        This method creates a figure per instrument showing the observed
        light curve with the fitted (full) model overplotted and a
        residuals panel beneath.

        Parameters
        ----------
        instruments : list or None
            List of instrument names to plot. If ``None``, all instruments
            in the current juliet dataset plotted.
        save : bool, optional
            If ``True``, each figure is saved to
            ``<input_folder>/full_model_<instrument>.png``. Default ``False``.
        nrandom : int, optional
            Number of random posterior-sample models to draw when
            ``quantile_models`` is ``False``. Default is 50.
        quantile_models : bool, optional
            If ``True``, plot the 68%% credible interval as a filled band;
            if ``False``, plot ``nrandom`` random posterior models instead.

        Returns
        -------
        all_fig : list
            List of matplotlib Figure objects created (one per instrument).
        all_axs1 : list
            List of top-panel Axes (data + model) for each figure.
        all_axs2 : list
            List of bottom-panel Axes (residuals) for each figure.

        Notes
        -----
        The routine expects that ``self.models_all_ins`` was populated by
        a prior call (see the class constructor) and that the dataset
        arrays (times, data, errors) are in ``self.times``, ``self.flux``, ``self.errs``.
        """

        # Computing the model for all instruments (None: all instruments)
        instruments = self._instruments(instruments)
        
        all_fig, all_axs1, all_axs2 = [], [], []
        for i in range(len(instruments)):
            # Data
            tim_ins = self.times[instruments[i]][self.idx_time_sort[instruments[i]]]
            fl_ins = self.flux[instruments[i]][self.idx_time_sort[instruments[i]]]
            fle_ins = self.errs[instruments[i]][self.idx_time_sort[instruments[i]]]

            ## Median model
            full_model = self.models_all_ins[instruments[i]][1][self.idx_time_sort[instruments[i]]]

            ## Quantile models
            up_68CI = self.models_all_ins[instruments[i]][2][self.idx_time_sort[instruments[i]]]
            lo_68CI = self.models_all_ins[instruments[i]][3][self.idx_time_sort[instruments[i]]]

            ## Random models
            sample_models = self.models_all_ins[instruments[i]][0][:, self.idx_time_sort[instruments[i]]]
            idx = np.random.choice( np.arange(sample_models.shape[0]), size=nrandom )
            random_models = sample_models[idx, :]

            # And figure
            fig = plt.figure(figsize=(9, 5.4))
            gs = gd.GridSpec(2,1, height_ratios=[2,1])

            # Top panel
            ax1 = plt.subplot(gs[0])
            ax1.errorbar(tim_ins, fl_ins, yerr=fle_ins, fmt='.', elinewidth=0.5, markersize=1.5, color='#86b6ef')#, alpha=0.3)
            ax1.plot(tim_ins, full_model, c='#184f95', lw=2, zorder=50)
            if quantile_models:
                ax1.fill_between(x=tim_ins, y1=lo_68CI, y2=up_68CI, color='#eb6834', alpha=0.5, zorder=25)
            else:
                for rand in range(nrandom):
                    ax1.plot(tim_ins, random_models[rand,:], color='#eb6834', alpha=0.5, lw=1., zorder=25)

            ax1.set_ylabel('Relative Flux')
            ax1.set_title('Full fitted model for instrument ' + str(instruments[i]))
            ax1.set_xlim(np.min(tim_ins), np.max(tim_ins))
            ax1.xaxis.set_major_formatter(plt.NullFormatter())

            # Bottom panel
            ax2 = plt.subplot(gs[1])
            ax2.errorbar(tim_ins, (fl_ins-full_model)*1e6, yerr=fle_ins*1e6, fmt='.', elinewidth=0.5, markersize=1.5, color='#86b6ef')#, alpha=0.3)
            ax2.axhline(y=0.0, c='black', ls='--', zorder=100)
            ax2.set_ylabel('Residuals [ppm]')
            ax2.set_xlabel(r'Time [BJD$_\mathrm{TDB}$]')
            ax2.set_xlim(np.min(tim_ins), np.max(tim_ins))

            all_fig.append(fig)
            all_axs1.append(ax1)
            all_axs2.append(ax2)

            if save:
                plt.savefig(os.path.join(self.input_folder, 'full_model_' + instruments[i] + '.png'), dpi=250)
        
        return all_fig, all_axs1, all_axs2
    
    @styled
    def plot_rv(self, nrandom=300, save=False):
        """Radial velocities: time series, and phase-folded orbit of every planet.

        Offsets are subtracted; the folded panels also have the trend and the
        other planets subtracted (all at the posterior medians). Errors
        include the jitter. Lines are the median model, bands its 68%
        interval. Phase 0 is the transit (conjunction).

        Parameters
        ----------
        nrandom : int
            Posterior samples for the models.
        save : bool
            Save the figure as ``<input_folder>/rv.png``.

        Returns
        -------
        fig, axes : matplotlib figure and the array of axes (first row data
            and models, second row residuals; first column the time series).
        """
        if self.fit is None or not self.fit.rv_names:
            raise ValueError('No radial velocities: use BroadbandFit(..., rvs=...).plots().plot_rv().')
        fit = self.fit
        post = self.ev.posteriors
        nall = max(np.size(v) for k, v in post.items() if k != 'unnamed' and np.ndim(v) == 1)
        idx = np.random.default_rng(0).choice(nall, size=min(int(nrandom), nall), replace=False)

        def get(name, default=0.):
            v = fit.values(name)
            v = np.full(1, default) if v is None else np.asarray(v, float)
            return v[idx] if v.size == nall else np.full(idx.size, v[0])

        planets = sorted(k.split('_')[1] for k in self.priors if re.fullmatch(r'K_p\d+', k))
        par = {}
        for pl in planets:
            e, w = fit._eccentricity(pl)
            e, w = (np.full(idx.size, d) if x is None else (x[idx] if np.size(x) == nall
                    else np.full(idx.size, np.ravel(x)[0])) for x, d in ((e, 0.), (w, 90.)))
            par[pl] = [get(f'P_{pl}'), get(f't0_{pl}'), e, np.radians(w), get(f'K_{pl}')]

        def orbit(pl, t):
            P, t0, e, w, K = par[pl]
            return self.ev.keplerian(t, P, t0, e, w, K)

        ta = self.ev.rv_ta

        def trend(t):
            d = t[None, :] - ta
            return get('rv_intercept')[:, None] + get('rv_slope')[:, None] * d \
                + get('rv_quad')[:, None] * d ** 2

        fig, axes = plt.subplots(2, 1 + len(planets), figsize=(WIDTH, 4.6), sharex='col', sharey='row',
                                 gridspec_kw=dict(height_ratios=[2, 1], hspace=0.08, wspace=0.1),
                                 squeeze=False)
        data = {n: fit.rv_residuals(n) for n in fit.rv_names}
        allt = np.concatenate([r['t'] for r in data.values()])
        tref = np.floor(allt.min())
        pad = 0.02 * max(np.ptp(allt), 1.)
        per = min(np.median(par[pl][0]) for pl in planets)
        ngrid = int(np.clip(30 * (np.ptp(allt) + 2 * pad) / per, 500, 6000))
        tg = np.linspace(allt.min() - pad, allt.max() + pad, ngrid)
        mod = np.median(trend(tg) + sum(orbit(pl, tg) for pl in planets), axis=0)
        axes[0, 0].plot(tg - tref, mod, color=MUTED,
                        lw=float(np.clip(1500. / ngrid, 0.5, 1.5)), alpha=0.8, zorder=1)
        for j, pl in enumerate(planets, start=1):
            ph = np.linspace(-0.5, 0.5, 301)
            P, t0 = np.median(par[pl][0]), np.median(par[pl][1])
            Pi, ti, e, w, K = par[pl]
            m = self.ev.keplerian(ti[:, None] + ph[None, :] * Pi[:, None], Pi, ti, e, w, K)
            lo, md, hi = np.percentile(m, [15.865, 50., 84.135], axis=0)
            axes[0, j].fill_between(ph, lo, hi, color=MUTED, alpha=0.35, lw=0, zorder=1)
            axes[0, j].plot(ph, md, color=INK2, lw=1.5, zorder=2)
            axes[1, j].set_xlabel('Orbital phase')
            axes[1, j].set_xticks([-0.4, -0.2, 0., 0.2, 0.4])
            axes[0, j].set_xlim(-0.53, 0.53)
            axes[0, j].text(0.97, 0.96, f'Planet {pl[1:]}', transform=axes[0, j].transAxes,
                            ha='right', va='top')
        for i, (n, r) in enumerate(data.items()):
            mu = float(np.median(fit.values(f'mu_{n}')))
            err = np.hypot(r['err'], float(np.median(fit.values(f'sigma_w_{n}'))))
            kw = dict(fmt='osD^vPX*'[i % 8], color=PALETTE[i % len(PALETTE)], ms=5, elinewidth=1,
                      mec='white', mew=0.6, zorder=5)
            axes[0, 0].errorbar(r['t'] - tref, r['rv'] - mu, err, label=n, **kw)
            axes[1, 0].errorbar(r['t'] - tref, r['resid'], err, **kw)
            for j, pl in enumerate(planets, start=1):
                P, t0 = np.median(par[pl][0]), np.median(par[pl][1])
                x = (r['t'] - t0 + 0.5 * P) % P / P - 0.5
                axes[0, j].errorbar(x, r['resid'] + np.median(orbit(pl, r['t']), axis=0), err, **kw)
                axes[1, j].errorbar(x, r['resid'], err, **kw)
        for ax in axes[1]:
            ax.axhline(0., color=MUTED, ls='--', lw=1, zorder=1)
        axes[0, 0].set_ylabel('Radial velocity - offset')
        axes[1, 0].set_ylabel('Residual')
        axes[1, 0].set_xlabel(f'Time [BJD - {tref:.0f}]')
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center',
                   bbox_to_anchor=(0.5, 0.89), ncols=min(len(data), 5))
        if save:
            fig.savefig(os.path.join(self.input_folder, 'rv.png'), dpi=250)
        return fig, axes

    def detrend_data(self, phmin, instruments, planet='p1'):
        """This function will generate detrended data from the raw data. It will also generate phases."""
        instruments = self._instruments(instruments)
        
        # First, create detrended dataset and models
        ## For detrended dataset
        self.phase, self.detrended_data, self.detrended_errs = {}, {}, {}
        self.ph_minimum, self.ph_maximum = 1e10, 1e-10

        ## For normalising the data, we will compute F1 and F2,
        ## We can use the same F1 and F2 to normalise the model as well
        ## So, let's save them!
        self.F1, self.F2 = {}, {}

        for i in range(len(instruments)):
            # Period and t0
            per, t0 = self._period_t0(planet, instruments[i])

            # GP model (GP prediction at the data) and linear model, to detrend the data
            gp_model = self.lc_models[instruments[i]]['gp'] if instruments[i] in self._gp else 0.

            if instruments[i] in self._lm.keys():
                linear_model = self.models_all_ins[instruments[i]][-1]['lm']
            else:
                linear_model = 0.
            
            # -----------------------------------------

            # -----------------------------------------
            #  Finally, we want to compute F1 and F2
            #  So we can normalise the detrended data
            #  The full model is defined as (courtesy of the original juliet code),
            #  y = [ T * ( D / (1 + D*M) ) ] + [ (1 - D) / (1 + D*M)]
            #  => y = T * F1 + F2
            #  => T = (y - F2) / F1
            # -----------------------------------------

            mflx = self._samples('mflux', instruments[i], 0.)
            mdil = self._samples('mdilution', instruments[i], 1.)
            
            F1 = np.nanmedian( mdil / ( 1 + (mdil * mflx) ) )
            F2 = np.nanmedian( (1 - mdil) / (1 + (mdil * mflx) ) )

            ## Saving F1 and F2
            self.F1[instruments[i]], self.F2[instruments[i]] = F1, F2

            sigw = np.nanmedian(self._samples('sigma_w', instruments[i], 0.) * 1e-6)

            ## Here we create detrended data by subtracting GP (if any) and linear (if any) models from the raw data
            ## (and removing a juliet non-linear model, if any: the full model is (T*F1 + F2 + LM) * NLM + GP)
            nlm, mult = self._nlm(instruments[i])
            data_i = self.flux[instruments[i]] - gp_model
            if nlm is not None:
                data_i = data_i / nlm if mult else data_i - nlm
            self.detrended_data[instruments[i]] = ( data_i - linear_model - F2) / F1 
            self.detrended_errs[instruments[i]] = np.sqrt( self.errs[instruments[i]]**2 + sigw**2 )

            phases_instrument = juliet.utils.get_phases(self.times[instruments[i]], P=per, t0=t0, phmin=phmin)
            self.phase[instruments[i]] = phases_instrument
            
            # Computing the minimum and maximum of the orbital phases
            if np.max(phases_instrument) > self.ph_maximum:
                self.ph_maximum = np.max(phases_instrument)
            if np.min(phases_instrument) < self.ph_minimum:
                self.ph_minimum = np.min(phases_instrument)

    
    def detrend_model(self, instruments, phmin, highres, planet='p1'):
        """Planet-only models (no dilution, offsets, systematics or GP; uncertainties of the
        planet parameters only) at the data, or with ``highres`` at 10000 times over the
        phases of the data (``self.planet_only_models``: samples, median, upper, lower)."""
        instruments = self._instruments(instruments)
        self.times_model, self.phases_model = {}, {}
        self.planet_only_models = {}
        per, t0 = self._period_t0(planet, instruments[0])
        if highres:
            phs = np.linspace(self.ph_minimum, self.ph_maximum, 10000)
            tim = t0 + phs * per
        for ins in instruments:
            if highres:
                m = self.ev.planet_models(ins, nsamples=self.nsamps, t=tim)
                self.times_model[ins], self.phases_model[ins] = tim, phs
            else:
                m = self.ev.planet_models(ins, nsamples=self.nsamps)
                self.times_model[ins] = self.times[ins]
                self.phases_model[ins] = juliet.utils.get_phases(t=self.times[ins], P=per, t0=t0, phmin=phmin)
            self.planet_only_models[ins] = (m['samples'], m['median'], m['upper'], m['lower'])

    @styled
    def phase_folded_lc(self, phmin=0.8, instruments=None, planet='p1', highres=False, nrandom=50, quantile_models=True, one_plot=None, figsize=(9, 5.4), pycheops_binning=False, nos_bin_tra=20, nos_bin_pc=30, window=1.):
        """Plot phase-folded light curves and models for the fitted dataset.

        This method computes detrended data and planetary models by
        calling ``detrend_data`` and ``detrend_model`` internally, detects
        whether the fit contains transits, eclipses or phase-curves, and
        then produces matplotlib figures showing the data, model, and
        residuals. It can produce either one figure per instrument or a single
        combined figure with all instruments plotted together.

        Parameters
        ----------
        phmin : float, optional
            Minimum phase value used for generating orbital phase (default
            ``0.8``).
        instruments : list or None, optional
            List of instrument names to plot. If ``None``, all instruments
            in the juliet dataset (``self.names``) are used. One plot
            is produced if all instruments are provided; otherwise, one plot
            per instrument is created.
        planet : str, optional
            Planet identifier used in the juliet fit (e.g., 'p1', 'p2', etc.) to select 
            the appropriate planet model. Default is 'p1'.
        highres : bool, optional
            If ``True``, compute and plot high time-resolution planet-only
            models. Default ``False``.
        nrandom : int, optional
            Number of random posterior-sample models to draw when
            ``quantile_models`` is ``False``. Default ``50``.
        quantile_models : bool, optional
            If ``True``, display the 68%% credible interval as a filled
            band; if ``False``, overplot ``nrandom`` random posterior
            samples. Default ``True``.
        one_plot : bool or None, optional
            If ``True``, produce a single combined plot for all instruments
            (regardless of the number of instruments provided). If
            ``False``, produce one plot per instrument (unless all
            instruments are provided). If ``None``, the behaviour is as
            just described. Default ``None``.
        figsize : tuple, optional
            Figure size passed to matplotlib when creating figures. Default
            is ``(16/1.5, 9/1.5)``.
        pycheops_binning : bool, optional
            If ``True``, use binning as produced by ``pycheops`` the
            binned datapoints. If ``False``, it will use default ``juliet``
            binning. Default ``False``.
        nos_bin_tra : int, optional
            Number of total binned data points in transit plot. Default is 20.
        nos_bin_pc : int, optional
            Number of total binned data points in phase curve plot. Default is 30.
        window : float, optional
            Half-width of the transit/eclipse panels in units of the transit
            duration T14 (binning uses ``window + 0.1``). Default ``1.0``.

        Returns
        -------
        If a single combined plot is produced the function returns the
        combined ``fig`` and the primary axes used. If multiple figures are
        generated, the function returns lists: ``figs_all``, ``axs1_all``,
        ``axs2_all``, ``axs3_all``, ``axs4_all`` corresponding to created figures and
        their axes panels.

        """
        # -------------------------------------------
        #         Do we want one plot?
        # -------------------------------------------
        # If instruments == None, then we will make one common plot for _all_ instruments
        # Also, if all instruments are provided, this will make one common plot
        # Otherwise, one plot per instrument
        # However, sometimes we want to force one plot, so we have the option one_plot
        instruments = self._instruments(instruments)
        if one_plot is None:
            one_plot = len(instruments) == len(self.names)

        # -------------------------------------------
        #               Compute t14
        # -------------------------------------------
        
        # Let's compute the transit/occultation duration for the planet (that can help setting the xlimits of the plots)
        ## We need several other parameters for that first

        ### Orbit (posterior medians or fixed values; derived ones from the model)
        orb = self.ev.orbit(planet, instruments[0])
        per, rprs, ar, b = orb['P'], orb['p'], orb['a'], orb['b']
        t14_dur = t14(per=per, ar=ar, rprs=rprs, b=b, ecc=orb['ecc'], omega=orb['w'])
        t14_phs = t14_dur / per

        ### Phase of the eclipse centre (0.5 unless an eclipse time was fitted, e.g. eccentric orbits)
        ecl_ph = 0.5
        try:
            tsec = self._pmed('t_secondary_' + planet)
            t0_ref = self._pmed('t0_' + planet)
            ecl_ph = ((tsec - t0_ref) / per) % 1.
        except KeyError:
            # light_travel_delay=True: the eclipse time follows from the orbit
            if orb.get('eclipse_phase') is not None and np.isfinite(orb['eclipse_phase']):
                ecl_ph = orb['eclipse_phase']

        # -------------------------------------------
        #     Computing detrended data and model
        # -------------------------------------------
        self.detrend_data(phmin=phmin, instruments=instruments, planet=planet)
        self.detrend_model(instruments=instruments, phmin=phmin, highres=highres, planet=planet)

        # -------------------------------------------
        #           Type of fitting
        # -------------------------------------------
        # We can find the type of fitting just by looking at the phase range covered
        phasecurve, transit, eclipse = False, False, False

        for ins in range(len(instruments)):
            kind = self.ev.fit_type(instruments[ins])
            if kind == 'phasecurve':
                phasecurve = True
            elif kind == 'eclipse':
                eclipse = True
            else:
                transit = True
        
        ## For phase curve fitting, eclipse fitting is by default true, so, we need to manually turn it off
        if phasecurve and eclipse:
            eclipse = False
        
        ## Sometimes, even for just eclipse fitting we might have turned full transit+eclipse fitting, so double check
        ## if there is indeed phase curve fitting in the data
        if phasecurve:
            diff_phs = self.ph_maximum - self.ph_minimum
            if diff_phs<0.3:
                phasecurve = False
                eclipse = True

        # -------------------------------------------
        #         Starting figure code:
        #      one plot: do plt.figure() here
        #   Depending on phase curve/transit/occ
        #   We need either 4 panels, or 2 panels
        # -------------------------------------------
        if one_plot:
            if not phasecurve:
                fig = plt.figure(figsize=figsize)
                gs = gd.GridSpec(2,1, height_ratios=[2,1])
                ax1 = plt.subplot(gs[0])  # Top panel
                ax2 = plt.subplot(gs[1])  # Bottom panel
                ax3, ax4 = None, None
            else:
                fig = plt.figure(figsize=figsize)
                gs = gd.GridSpec(2, 2, height_ratios=[2,1], width_ratios=[1,3])#, wspace=0.1)
                ax1 = plt.subplot(gs[0,0])  # Upper-left (for transit data)
                ax2 = plt.subplot(gs[1,0])  # Bottom-left (for transit residuals)

                ax3 = plt.subplot(gs[0,1])  # Upper-right (for phase curve)
                ax4 = plt.subplot(gs[1,1])  # Bottom-right (for phase curve residuals)
            
            # If there is one plot, then we also need to 
            # collect all data to compute binned data
            bin_phs, bin_fl, bin_fle, bin_res = np.array([]), np.array([]), np.array([]), np.array([])
            # Also, in case we don't have highres model, we want to save all models before plotting
            all_models_for_plotting = np.array([])
            all_u68_mods_plotting, all_l68_mods_plotting = np.array([]), np.array([])

        # ------------------------------------------------------------------------
        # Creating a list of all fig, axes to return at the end of the function
        # if one_plot==False
        if not one_plot:
            figs_all, axs1_all, axs2_all, axs3_all, axs4_all = [], [], [], [], []
        
        for i in range(len(instruments)):

            # -------------------------------------------
            #         Starting figure code:
            #    If not one plot: do plt.figure() 
            #           inside the loop
            #   Depending on phase curve/transit/occ
            #   We need either 4 panels, or 2 panels
            # -------------------------------------------

            if not one_plot:
                if not phasecurve:
                    fig = plt.figure(figsize=figsize)
                    gs = gd.GridSpec(2,1, height_ratios=[2,1])
                    ax1 = plt.subplot(gs[0])
                    ax2 = plt.subplot(gs[1])
                    ax3, ax4 = None, None
                else:
                    fig = plt.figure(figsize=figsize)
                    gs = gd.GridSpec(2, 2, height_ratios=[2,1], width_ratios=[1,3])#, wspace=0.1)
                    ax1 = plt.subplot(gs[0,0])  # Upper-left (for transit data)
                    ax2 = plt.subplot(gs[1,0])  # Bottom-left (for transit residuals)

                    ax3 = plt.subplot(gs[0,1])  # Upper-right (for phase curve)
                    ax4 = plt.subplot(gs[1,1])  # Bottom-right (for phase curve residuals)
            
            # -------------------------------------------
            #      Collecting the required data
            # -------------------------------------------
            # Detrended data
            ## Sorting according to phase
            idx_phs = np.argsort(self.phase[instruments[i]])

            phs = self.phase[instruments[i]][idx_phs]
            detrend_data = self.detrended_data[instruments[i]][idx_phs]
            detrend_errs = self.detrended_errs[instruments[i]][idx_phs]

            # Detrended model
            ## Sorting according to phase
            idx_phs_mod = np.argsort( self.phases_model[instruments[i]] )

            phs_model = self.phases_model[instruments[i]][idx_phs_mod]
            detrend_model = self.planet_only_models[instruments[i]][1][idx_phs_mod]

            ## Quantile models
            up_68CI = self.planet_only_models[instruments[i]][2][idx_phs_mod]
            lo_68CI = self.planet_only_models[instruments[i]][3][idx_phs_mod]

            ## Random models
            sample_models = self.planet_only_models[instruments[i]][0][:, idx_phs_mod]
            idx = np.random.choice( np.arange(sample_models.shape[0]), size=nrandom )
            random_models = sample_models[idx, :]

            # Residuals
            residuals = ( self.flux[instruments[i]] - self.models_all_ins[instruments[i]][1] )[idx_phs]

            # If we have phase curve fitting, we will use ppm units for the y-axis
            if phasecurve or eclipse:
                detrend_data, detrend_errs = (detrend_data-1.)*1e6, detrend_errs*1e6
                detrend_model = (detrend_model-1.)*1e6
                random_models = (random_models-1.)*1e6
                lo_68CI, up_68CI = (lo_68CI-1.)*1e6, (up_68CI-1.)*1e6

            if one_plot:
                # -------------------------------------------
                #       Saving the data for binning 
                #            outside the loop
                # -------------------------------------------
                bin_phs = np.hstack(( bin_phs, phs ))
                bin_fl, bin_fle = np.hstack(( bin_fl, detrend_data )), np.hstack(( bin_fle, detrend_errs ))
                bin_res = np.hstack(( bin_res, residuals ))
                all_models_for_plotting = np.hstack(( all_models_for_plotting, detrend_model ))
                all_l68_mods_plotting = np.hstack(( all_l68_mods_plotting, lo_68CI ))
                all_u68_mods_plotting = np.hstack(( all_u68_mods_plotting, up_68CI ))

            # ----------------------------------------------------------
            #     The actual plotting code starts from here
            # ----------------------------------------------------------

            # -------------------------------------------
            #         Top left: transit data
            # -------------------------------------------
            if phasecurve:
                ppt = 1e-3
            else:
                ppt = 1.
            
            ax1.errorbar(phs, detrend_data*ppt, fmt='.', elinewidth=0.5, markersize=1.5, alpha=0.5, color='#86b6ef', zorder=1)
            if (not one_plot) or highres:
                ## if we are making one-plot, we can plot the detrended model out of loop
                ax1.plot(phs_model, detrend_model*ppt, color='#184f95', lw=2.5, zorder=50)
            if quantile_models:
                if (not one_plot) or highres:
                    ## Again, if we are making one plot, we can plot the detrended quantile models out of loop
                    ax1.fill_between(phs_model, y1=lo_68CI*ppt, y2=up_68CI*ppt, color='#eb6834', alpha=0.15, zorder=25)
            else:
                for rand in range(nrandom):
                    ax1.plot(phs_model, random_models[rand,:]*ppt, color='#eb6834', alpha=0.5, lw=1., zorder=25)
            
            # Limits
            ## Maximum of the transit model would always be 1. (or, 0 in case we have phase curve model, and we decided to use ppm)
            if transit or phasecurve:
                ax1.set_xlim(-window*t14_phs, window*t14_phs)
            else:
                ax1.set_xlim(ecl_ph-window*t14_phs, ecl_ph+window*t14_phs)
            
            if transit:
                ax1.set_ylim([np.min(detrend_model)-0.2*np.ptp(detrend_model), 1+0.2*np.ptp(detrend_model)])
            elif phasecurve:
                ax1.set_ylim([np.min(detrend_model*ppt)-0.2*np.ptp(detrend_model*ppt), 0.+0.2*np.ptp(detrend_model*ppt)])
            else:
                ax1.set_ylim([1-0.3*np.ptp(detrend_model), np.max(detrend_model)+0.3*np.ptp(detrend_model)])

            # -------------------------------------------
            #        Bottom left: transit residuals
            # -------------------------------------------
            ax2.errorbar(phs, residuals*ppt*1e6, fmt='.', elinewidth=0.5, markersize=1.5, alpha=0.5, color='#86b6ef', zorder=1)
            ax2.axhline(y=0., ls='--', color='k', zorder=50)

            # Limits (x-limit same as the top axis; for y-lim, we will set 3-sigma from median)
            if transit or phasecurve:
                ax2.set_xlim(-window*t14_phs, window*t14_phs)
            else:
                ax2.set_xlim(ecl_ph-window*t14_phs, ecl_ph+window*t14_phs)

            # -------------------------------------------
            #        Binned data: For TRANSIT
            # -------------------------------------------
            ## We do binned data here: ONLY if we don't have one plot
            if not one_plot:
                if transit or phasecurve:
                    idx_transits = ( phs > -(window+0.1)*t14_phs ) & ( phs < (window+0.1)*t14_phs )
                else:
                    idx_transits = ( phs > ecl_ph-(window+0.1)*t14_phs ) & ( phs < ecl_ph+(window+0.1)*t14_phs )
                
                if not pycheops_binning:
                    nbin = int( np.sum(idx_transits) / nos_bin_tra )

                    bin_phs_tra, bin_fl_tra, bin_fle_tra = _bin_data(x=phs[idx_transits], y=detrend_data[idx_transits]*ppt, n_bin=nbin, yerr=detrend_errs[idx_transits]*ppt)
                    if phasecurve or eclipse:
                        _, bin_res_tra, bin_reserr_tra = _bin_data(x=phs[idx_transits], y=residuals[idx_transits]*ppt*1e6, n_bin=nbin, yerr=detrend_errs[idx_transits]*ppt)
                    else:
                        # When transit only, we haven't multiplied the errorbars with 1e6 yet; so we need to do it again
                        _, bin_res_tra, bin_reserr_tra = _bin_data(x=phs[idx_transits], y=residuals[idx_transits]*1e6, n_bin=nbin, yerr=detrend_errs[idx_transits]*1e6)
                else:
                    binwid = np.ptp( phs[idx_transits] ) / nos_bin_tra

                    bin_phs_tra, bin_fl_tra, bin_fle_tra, _ = lcbin(time=phs[idx_transits], flux=detrend_data[idx_transits]*ppt, binwidth=binwid)
                    if phasecurve or eclipse:
                        _, bin_res_tra, bin_reserr_tra, _ = lcbin(time=phs[idx_transits], flux=residuals[idx_transits]*ppt*1e6, binwidth=binwid)
                    else:
                        # When transit only, we haven't multiplied the errorbars with 1e6 yet; so we need to do it again
                        _, bin_res_tra, bin_reserr_tra, _ = lcbin(time=phs[idx_transits], flux=residuals[idx_transits]*1e6, binwidth=binwid)

                # Plotting them
                ax1.errorbar(bin_phs_tra, bin_fl_tra, yerr=bin_fle_tra, fmt='o', c='#184f95', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
                ax2.errorbar(bin_phs_tra, bin_res_tra, yerr=bin_reserr_tra, fmt='o', c='#184f95', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)

                # Setting y-lim on residuals based on binned residuals
                _robust_ylim(ax2, bin_res_tra)

            # -------------------------------------------
            #         If phase curve modelling,
            #        then we also need right panel
            # -------------------------------------------
            if phasecurve:
                # -------------------------------------------
                #         Top right: phase curve
                # -------------------------------------------
                ax3.errorbar(phs, detrend_data, fmt='.', elinewidth=0.5, markersize=1.5, alpha=0.5, color='#86b6ef', zorder=1)
                if (not one_plot) or highres:
                    ## If we are making one plot, then we can take this outside of the loop
                    ax3.plot(phs_model, detrend_model, color='#184f95', lw=2.5, zorder=50)
                ax3.axhline(0.0, color='k', ls='--', lw=1.5, zorder=45)
                
                if quantile_models:
                    if (not one_plot) or highres:
                        ## If we are making one plot, we can take this outside of the loop
                        ax3.fill_between(phs_model, y1=lo_68CI, y2=up_68CI, color='#eb6834', alpha=0.15, zorder=25)
                else:
                    for rand in range(nrandom):
                        ax3.plot(phs_model, random_models[rand,:], color='#eb6834', alpha=0.5, lw=1., zorder=25)
                
                # Limits
                ax3.set_xlim(phmin-1., phmin)
                ## For y-lim; we have baseline of 0, we can leave half occultation depth/phase amplitude below and above
                ylen = np.ptp( detrend_model[np.abs(phs_model) > 0.1] )
                ax3.set_ylim([-0.4*ylen, 1.5*ylen])

                # -------------------------------------------
                #     Bottom right: phase curve residuals
                # -------------------------------------------
                ax4.errorbar(phs, residuals*1e6, fmt='.', elinewidth=0.5, markersize=1.5, alpha=0.5, color='#86b6ef', zorder=1)
                ax4.axhline(y=0., ls='--', color='k', zorder=10)

                # Limits
                ax4.set_xlim(phmin-1., phmin)

                # -------------------------------------------
                #        Binned data: For PHASE CURVE
                # -------------------------------------------
                ## We do binned data here: ONLY if we don't have one plot
                if not one_plot:
                    if not pycheops_binning:
                        bin_phs_pc, bin_fl_pc, bin_fle_pc = _bin_data(x=phs, y=detrend_data, n_bin=int( len(phs)/nos_bin_pc ), yerr=detrend_errs)
                        _, bin_res_pc, bin_reserr_pc = _bin_data(x=phs, y=residuals*1e6, n_bin=int( len(phs)/nos_bin_pc ), yerr=detrend_errs)
                    else:
                        bin_phs_pc, bin_fl_pc, bin_fle_pc, _ = lcbin(time=phs, flux=detrend_data, binwidth=1/nos_bin_pc)
                        _, bin_res_pc, bin_reserr_pc, _ = lcbin(time=phs, flux=residuals*1e6, binwidth=1/nos_bin_pc)

                    # Plotting them
                    ax3.errorbar(bin_phs_pc, bin_fl_pc, yerr=bin_fle_pc, fmt='o', c='#184f95', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
                    ax4.errorbar(bin_phs_pc, bin_res_pc, yerr=bin_reserr_pc, fmt='o', c='#184f95', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)

                    _robust_ylim(ax4, bin_res_pc)

            if not one_plot:
                # ---------------------------------------------------
                #           Figure labels etc.
                # ---------------------------------------------------
                # ------------------------
                #.   Transits
                # ------------------------
                ## First, removing the x-axis labels of top panels
                ax1.xaxis.set_major_formatter(plt.NullFormatter())
                # x-and y-labels
                if not phasecurve:
                    if not eclipse:
                        ax1.set_ylabel('Normalised flux')
                    else:
                        ax1.set_ylabel('Normalised flux [ppm]')
                    ax2.set_ylabel('Residuals [ppm]')
                else:
                    ax1.set_ylabel('Normalised flux [ppt]')
                    ax2.set_ylabel('Residuals [ppt]')
                
                ax2.set_xlabel('Orbital phase')

                # ------------------------
                #.   Transits
                # ------------------------
                if phasecurve:
                    ## Removing x-axis labels of top panels
                    ax3.xaxis.set_major_formatter(plt.NullFormatter())

                    ## Now, we will switch the y-axis labels to right side for phase curve panels
                    ax3.yaxis.tick_right()
                    ax3.tick_params(labelright=True)
                    ax3.yaxis.set_label_position('right')
                    
                    ax4.yaxis.tick_right()
                    ax4.tick_params(labelright=True)
                    ax4.yaxis.set_label_position('right')

                    ## x-and y-labels
                    ax3.set_ylabel('Normalised flux [ppm]', rotation=270, labelpad=25)
                    
                    ax4.set_xlabel('Orbital phase')
                    ax4.set_ylabel('Residuals [ppm]', rotation=270, labelpad=25)

                # Saving figs and axes if not one_plot
                figs_all.append(fig)
                axs1_all.append(ax1)
                axs2_all.append(ax2)
                axs3_all.append(ax3)
                axs4_all.append(ax4)

        # -------------------------------------------------------------
        #            Now, we plot the binned data when 
        #                    we have one plot
        # -------------------------------------------------------------
        if one_plot:# and (not highres):
            ## This ppt will convert transit data to PPT, when phase curve is plotted
            if phasecurve:
                ppt = 1e-3
            else:
                ppt = 1.
            
            # -------------------------------------------
            #        Binned data: For Transits
            # -------------------------------------------
            idx_phs_bin = np.argsort(bin_phs)
            bin_phs, bin_fl, bin_fle, bin_res = bin_phs[idx_phs_bin], bin_fl[idx_phs_bin], bin_fle[idx_phs_bin], bin_res[idx_phs_bin]


            # Now, we want to decide nbin (number of data points to bin)
            # We would like to have different binning for transit and phase curve -- so first selecting nbin for transit/occultation
            if transit or phasecurve:
                idx_transits = ( bin_phs > -(window+0.1)*t14_phs ) & ( bin_phs < (window+0.1)*t14_phs )
            else:
                idx_transits = ( bin_phs > ecl_ph-(window+0.1)*t14_phs ) & ( bin_phs < ecl_ph+(window+0.1)*t14_phs )
            
            if not pycheops_binning:
                nbin = int( np.sum(idx_transits) / nos_bin_tra )

                # Performing binning
                bin_phs_tra, bin_fl_tra, bin_fle_tra = _bin_data(x=bin_phs[idx_transits], y=bin_fl[idx_transits]*ppt, n_bin=nbin, yerr=bin_fle[idx_transits]*ppt)
                if phasecurve or eclipse:
                    _, bin_res_tra, bin_reserr_tra = _bin_data(x=bin_phs[idx_transits], y=bin_res[idx_transits]*ppt*1e6, n_bin=nbin, yerr=bin_fle[idx_transits]*ppt)
                else:
                    # When transit only, we haven't multiplied the errorbars with 1e6 yet; so we need to do it again
                    _, bin_res_tra, bin_reserr_tra = _bin_data(x=bin_phs[idx_transits], y=bin_res[idx_transits]*1e6, n_bin=nbin, yerr=bin_fle[idx_transits]*1e6)
            else:
                binwid = np.ptp( bin_phs[idx_transits] ) / nos_bin_tra

                # Performing binning
                bin_phs_tra, bin_fl_tra, bin_fle_tra, _ = lcbin(time=bin_phs[idx_transits], flux=bin_fl[idx_transits]*ppt, binwidth=binwid)
                if phasecurve or eclipse:
                    _, bin_res_tra, bin_reserr_tra, _ = lcbin(time=bin_phs[idx_transits], flux=bin_res[idx_transits]*ppt*1e6, binwidth=binwid)
                else:
                    # When transit only, we haven't multiplied the errorbars with 1e6 yet; so we need to do it again
                    _, bin_res_tra, bin_reserr_tra, _ = lcbin(time=bin_phs[idx_transits], flux=bin_res[idx_transits]*1e6, binwidth=binwid)

            # Plotting them
            ax1.errorbar(bin_phs_tra, bin_fl_tra, yerr=bin_fle_tra, fmt='o', c='#184f95', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
            ax2.errorbar(bin_phs_tra, bin_res_tra, yerr=bin_reserr_tra, fmt='o', c='#184f95', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)

            # Setting y-lim on residuals based on binned residuals
            _robust_ylim(ax2, bin_res_tra)
            
            # -------------------------------------------
            #        Binned data: For PHASE CURVE
            # -------------------------------------------
            if phasecurve:
                if not pycheops_binning:
                    bin_phs_pc, bin_fl_pc, bin_fle_pc = _bin_data(x=bin_phs, y=bin_fl, n_bin=int( len(bin_phs)/nos_bin_pc ), yerr=bin_fle)
                    _, bin_res_pc, bin_reserr_pc = _bin_data(x=bin_phs, y=bin_res*1e6, n_bin=int( len(bin_phs)/nos_bin_pc ), yerr=bin_fle)
                else:
                    bin_phs_pc, bin_fl_pc, bin_fle_pc, _ = lcbin(time=bin_phs, flux=bin_fl, binwidth=1/nos_bin_pc)
                    _, bin_res_pc, bin_reserr_pc, _ = lcbin(time=bin_phs, flux=bin_res*1e6, binwidth=1/nos_bin_pc)

                # Plotting them
                ax3.errorbar(bin_phs_pc, bin_fl_pc, yerr=bin_fle_pc, fmt='o', c='#184f95', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
                ax4.errorbar(bin_phs_pc, bin_res_pc, yerr=bin_reserr_pc, fmt='o', c='#184f95', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)

                _robust_ylim(ax4, bin_res_pc)

            
            # -----------------------------------------------------------------------
            #
            #               Now, we want to plot the planetary model
            #                        if highres = False
            #
            # -----------------------------------------------------------------------

            if not highres:
                # First, collecting all models
                all_models_for_plotting = all_models_for_plotting[idx_phs_bin]
                all_l68_mods_plotting, all_u68_mods_plotting = all_l68_mods_plotting[idx_phs_bin], all_u68_mods_plotting[idx_phs_bin]

                # ---------------------------------------------------
                #           For transit/occultation
                # ---------------------------------------------------
                
                ## ----------- We can now plot the models (detrended and quantile models)
                ax1.plot(bin_phs, all_models_for_plotting*ppt, color='#184f95', lw=2.5, zorder=50)
                if quantile_models:
                    ax1.fill_between(bin_phs, y1=all_l68_mods_plotting*ppt, y2=all_u68_mods_plotting*ppt, color='#eb6834', alpha=0.5, zorder=25)

                # ---------------------------------------------------
                #           For phase curve
                # ---------------------------------------------------
                if phasecurve:
                    ## ----------- We can now plot the models (detrended and quantile models)
                    ax3.plot(bin_phs, all_models_for_plotting, color='#184f95', lw=2.5, zorder=50)
                    if quantile_models:
                        ax3.fill_between(bin_phs, y1=all_l68_mods_plotting, y2=all_u68_mods_plotting, color='#eb6834', alpha=0.5, zorder=25)

        if one_plot:
            # ---------------------------------------------------
            #           Figure labels etc.
            # ---------------------------------------------------
            # ------------------------
            #.   Transits
            # ------------------------
            ## First, removing the x-axis labels of top panels
            ax1.xaxis.set_major_formatter(plt.NullFormatter())
            # x-and y-labels
            if not phasecurve:
                if not eclipse:
                    ax1.set_ylabel('Normalised flux')
                else:
                    ax1.set_ylabel('Normalised flux [ppm]')
                ax2.set_ylabel('Residuals [ppm]')
            else:
                ax1.set_ylabel('Normalised flux [ppt]')
                ax2.set_ylabel('Residuals [ppt]')
            
            ax2.set_xlabel('Orbital phase')
            

            # ------------------------
            #.   Phase curves
            # ------------------------
            if phasecurve:
                ## Removing x-axis labels of top panels
                ax3.xaxis.set_major_formatter(plt.NullFormatter())

                ## Now, we will switch the y-axis labels to right side for phase curve panels
                ax3.yaxis.tick_right()
                ax3.tick_params(labelright=True)
                ax3.yaxis.set_label_position('right')
                
                ax4.yaxis.tick_right()
                ax4.tick_params(labelright=True)
                ax4.yaxis.set_label_position('right')

                ## x-and y-labels
                ax3.set_ylabel('Normalised flux [ppm]', rotation=270, labelpad=25)
                
                ax4.set_xlabel('Orbital phase')
                ax4.set_ylabel('Residuals [ppm]', rotation=270, labelpad=25)

        ## Depending on one_plot, we return the appropriate objects
        if not one_plot:
            return figs_all, axs1_all, axs2_all, axs3_all, axs4_all
        else:
            return fig, ax1, ax2, ax3, ax4

    @styled
    def plot_fake_allan_deviation(self, instruments=None, binmax=10, method='pipe', timeunit=None):
        """Compute and plot per-instrument noise-vs-bin-size curves.

        This is a thin wrapper around ``utils.fake_allan_deviation`` that
        computes residuals for each instrument (data minus the full model)
        and returns the figures, axes and numeric results for further use.

        Parameters
        ----------
        instruments : list or None
            List of instrument names to include. If ``None``, all
            instruments in the juliet dataset (``self.names``)
            are processed.
        binmax : int, optional
            Passed to ``utils.fake_allan_deviation`` to control the maximum
            number of bins (default ``10``).
        method : {'pipe','std','rms','astropy'}, optional
            Method used to estimate the scatter of the binned residuals:
            - 'pipe' : use ``pipe_mad`` (median absolute differences estimator)
            - 'std'  : use ``np.nanstd``
            - 'rms'  : use the root-mean-square (``rms`` helper)
            - 'astropy' : use ``astropy.stats.mad_std`` (robust MAD-based std)
        timeunit : str or None, optional
            If provided, forwarded to ``utils.fake_allan_deviation`` to
            force the secondary x-axis time unit; otherwise the helper
            chooses an appropriate unit automatically.

        Returns
        -------
        figs_all : list
            List of matplotlib Figure objects (one per instrument).
        axs_all : list
            List of matplotlib Axes objects corresponding to the figures.
        binsize_all : list
            List of binsize arrays returned for each instrument.
        noise_all : list
            List of noise arrays (ppm) computed for each instrument.
        white_noise_all : list
            List of white-noise expectation arrays (ppm) for each
            instrument.
        """

        # Let's first gather the names of all instruments (we will always do one plot per instrument)
        instruments = self._instruments(instruments)

        ## All stuff we can save
        figs_all, axs_all = [], []
        binsize_all, noise_all, white_noise_all = [], [], []

        for ins in range(len(instruments)):
            # Getting the data
            tim_ins = self.times[instruments[ins]][self.idx_time_sort[instruments[ins]]]
            fl_ins = self.flux[instruments[ins]][self.idx_time_sort[instruments[ins]]]

            ## Median model
            full_model = self.models_all_ins[instruments[ins]][1][self.idx_time_sort[instruments[ins]]]

            residuals = fl_ins - full_model

            # And, now, let's plot the Allan deviation
            fig, ax, binsize, noise, white_noise = fake_allan_deviation(times=tim_ins, residuals=residuals, binmax=binmax, method=method, timeunit=timeunit, plot=True)

            figs_all.append(fig)
            axs_all.append(ax)
            binsize_all.append(binsize)
            noise_all.append(noise)
            white_noise_all.append(white_noise)

        return figs_all, axs_all, binsize_all, noise_all, white_noise_all
    
    @styled
    def plot_corner(self, planet_only=False, param_list=None, save=True):
        """Create a corner plot of selected posterior parameters.


        Parameters
        ----------
        planet_only : bool, optional
            If ``True``, include only planetary parameters in the corner
            plot. If ``False`` (default), include instrumental and noise
            parameters as well when available.
        param_list : list of str or None, optional
            If provided, a list of parameter names to include in the corner
            plot. If ``None`` (default), all parameters are included (subject
            to the ``planet_only`` filter).
        save : bool, optional
            If ``True`` (default), save the generated figure to
            ``<input_folder>/corner_plot.png``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure produced by ``utils.corner_plot`` containing the
            marginal and pairwise plots for the selected parameters.

        """

        # Access the posteriors
        post_samps = self.ev.posteriors

        # Arrays for posteriors and labels
        posteriors, labels, titles = [], [], []
        for i in post_samps.keys():
            if param_list is not None and i not in param_list:
                continue
            par = i.split('_')
            if ( 'p1' in par ) or ( 'p2' in par ) or ( 'p3' in par ) or ( 'p4' in par ) or ( 'q1' in par ) or ( 'q2' in par ) or ( 'rho' in par ):
                ## Adding posteriors and labels ( this parameters are added no matter planet_only is True/False)
                if i[0:3] == 'P_p':
                    ## Period parameter: the uncertainty is generally very small, so we can scale it to make it plottable
                    per_exponent_part = np.ceil( np.log10( np.nanstd( post_samps[i] ) ) )
                    per_median = np.nanmedian( post_samps[i] )
                    per = ( post_samps[i] - per_median ) / ( 10**per_exponent_part )

                    ## Saving the posteriors
                    posteriors.append( per )
                    labels.append( i + ' [(P $-$ ' + str(np.around(per_median,5)) + ') / $10^{' + str(int(per_exponent_part)) + '}$ d]' )
                    titles.append( par[0] )
                
                elif i[0:2] == 't0':
                    ## Same for t0, we don't need to write out the whole thing
                    t02 = np.floor( np.nanmedian(post_samps[i]) )

                    posteriors.append( post_samps[i] - t02 )
                    labels.append( i + ' [t0 $-$ ' + str(int(t02)) + ' d]' )
                    titles.append( par[0] )
                
                elif (i[0:3] == 'p_p') or (i[0:4] == 'p1_p') or (i[0:4] == 'p2_p'):
                    ## For Rp/R*, if there are more than 1 instruments, we will label them as 'instrument_name et al.'
                    posteriors.append( post_samps[i] )
                    titles.append( par[0] )
                    if len(i.split('_')) > 3:
                        labels.append('_'.join(i.split('_')[0:3]) + '_et al.')
                    else:
                        labels.append(i)
                
                elif (i[0:2] == 'q1') or (i[0:2] == 'q2'):
                    ## Same for limb-darkening coefficients, if there are more than 1 instruments, we will label them as 'instrument_name et al.'
                    posteriors.append( post_samps[i] )
                    titles.append( par[0] )
                    if len(i.split('_')) > 2:
                        labels.append('_'.join(i.split('_')[0:2]) + '_et al.')
                    else:
                        labels.append(i)
                
                elif ( i[0:2] == 'fp' ) or ( i[0:4] == 'C1_p' ) or ( i[0:2] == 'C2' ) or ( i[0:2] == 'D1' ) or ( i[0:2] == 'D2' ):
                    ## For flux parameters, we will label them as 'instrument_name et al.'
                    posteriors.append( post_samps[i] * 1e6 )
                    titles.append( par[0] )
                    if len(i.split('_')) > 3:
                        labels.append('_'.join(i.split('_')[0:3]) + '_et al. [ppm]')
                    else:
                        labels.append(i + ' [ppm]')

                ## If there are 'sesinomega_p1' and 'secosomega_p1' parameters, we will convert them to 'e_p1' and 'omega_p1'
                elif ( i[0:10] == 'secosomega' ):
                    pl_num = i.split('_')[-1]
                    sesinomega = post_samps['sesinomega_' + pl_num]
                    secosomega = post_samps['secosomega_' + pl_num]

                    e_p = sesinomega**2 + secosomega**2
                    omega_p = np.arctan2( sesinomega, secosomega ) * 180. / np.pi

                    ## First, add sesinomega
                    posteriors.append( post_samps[i] )
                    labels.append( i )
                    titles.append( par[0] )

                    ## And now, e and w
                    posteriors.append( e_p )
                    labels.append( 'e_' + pl_num )
                    titles.append( 'e' )

                    posteriors.append( omega_p )
                    labels.append( r'$\omega$_' + pl_num + ' [deg]' )
                    titles.append( r'$\omega$' )
                
                else:
                    ## Else
                    posteriors.append( post_samps[i] )
                    if len(i.split('_')) > 3:
                        labels.append('_'.join(i.split('_')[0:3]) + '_et al.')
                    else:
                        labels.append(i)
                    titles.append( par[0] )

            else:
                ## Rest of the parameters are only added if planet_only is False
                if not planet_only:
                    if ( i[0:7] != 'unnamed' ) and ( i[0:7] != 'loglike' ):
                        posteriors.append( post_samps[i] )
                        if len(i.split('_')) > 3:
                            labels.append('_'.join(i.split('_')[0:3]) + '_et al.')
                        else:
                            labels.append(i)
                        titles.append( par[0] )
        
        # ------------------------------------------
        #     Now, we can make the corner plot
        # ------------------------------------------

        ## Reducing fontsize specially for this plot
        from matplotlib import rcParams
        rcParams['font.size'] = 10.
        rcParams['axes.labelsize'] = 'medium'
        rcParams['xtick.labelsize'] = 'medium'
        rcParams['ytick.labelsize'] = 'medium'
        rcParams['axes.titlesize'] = 'large'

        fig = corner_plot(samples=posteriors, labels=labels, titles=titles)
        if save:
            plt.savefig(os.path.join(self.input_folder, 'corner_plot.png'), dpi=250)

        return fig
    
    @styled
    def plot_gp(self, instruments=None, highres=False, one_plot=False, figsize=(9, 5.4), pycheops_binning=False, xlabel='GP regressor', robust_errors=False, plot_errorbars=False, parameter_vector=None, nos_bin=20):
        """Plot the Gaussian Process component(s) from the fitted `juliet` model.

        This routine computes and plots the GP model (and binned data)
        for one or more instruments present in the current `juliet` dataset.

        Parameters
        ----------
        instruments : list or None
            List of instrument names to plot. If ``None``, all instruments for
            which GP regressors exist in the fit are
            plotted.
        highres : bool, optional
            If ``True``, compute a high-resolution GP prediction (calls
            ``compute_gp_model``). Default is ``False``.
        one_plot : bool, optional
            If ``True``, combine all instruments into a single figure and
            return a single ``(fig, axs)``. If ``False``, return lists of
            figures and axes (one per instrument). Default ``False``.
        figsize : tuple, optional
            Figure size passed to ``matplotlib``. Default ``(16/1.5, 9/1.5)``.
        pycheops_binning : bool, optional
            Use ``pycheops`` produced binning when set to ``True``.
            Default ``False``.
        xlabel : str, optional
            X-axis label for the GP regressor. Default ``'GP regressor'``.
        robust_errors : bool, optional
            If ``True``, compute GP uncertainties using
            ``compute_gp_model`` instead of using ``juliet`` precomputed errors.
        plot_errorbars : bool, optional
            If ``True``, plot error bars on the GP data points. Default ``False``.
        parameter_vector : dict, array-like or None, optional
            GP parameters of the predictions of ``compute_gp_model``
            (default: the posterior medians): ``{prior name: value}``
            replacing some medians. For juliet fits the GP object's own
            celerite/george vector is also accepted, used as given (its
            jitter term should then be negligible: the errors include
            ``sigma_w``).
        nos_bin : int, optional
            Approximate number of bins to use when computing binned data per
            instrument. Default ``20``.

        Returns
        -------
        (fig, axs) or (fig_all, axs_all)
            If ``one_plot`` is ``True``, returns a single tuple ``(fig, axs)``
            for the combined figure. Otherwise returns ``(fig_all, axs_all)``
            where each is a list containing one Figure/Axis per instrument.

        """
        # -------------------------------------------
        #         List of all instruments
        # -------------------------------------------
        ## If instruments=None, we will generate one plot per instruments for all instruments for which GP fitting is done
        if instruments is not None:
            instruments = self._instruments(instruments)
            ## Let's check if the provided instruments indeed have GP fitting
            for ins in range(len(instruments)):
                if instruments[ins] not in self._gp.keys():
                    raise ValueError('GP fitting is not found for instrument ' + instruments[ins])
        else:
            instruments = [i for i in self._gp.keys()]
        
        # -------------------------------------------
        #     Computing detrended data and model
        # -------------------------------------------
        self.detrend_data(phmin=0.8, instruments=instruments)
        self.detrend_model(instruments=instruments, phmin=0.8, highres=False)

        # -------------------------------------------
        #      Let's now compute the residuals 
        #        needed to plot the GP model
        #   The total juliet model is:
        #      y = T*F1 + F2 + GP + LM
        #     => GP = y - T*F1 - F2 - LM
        # -------------------------------------------
        gp_model_regressor, self.gp_data = {}, {}
        gp_med_model, gp_up68, gp_lo68 = {}, {}, {}
        for i in range(len(instruments)):
            # We note here that every array should be sorted according to GP regressors (e.g., time or roll angle)
            # We will keep it that way -- everything sorted according GP regressors

            ## -- We also need to compute linear model, if there is any
            if instruments[i] in self._lm.keys():
                linear_model = self.models_all_ins[instruments[i]][-1]['lm']
            else:
                linear_model = 0.
            
            # -------------------------------------------
            #   GP data: i.e., data fitted to GP model
            # -------------------------------------------
            nlm, mult = self._nlm(instruments[i])
            deterministic = ( self.F1[instruments[i]] * self.planet_only_models[instruments[i]][1] ) + self.F2[instruments[i]] + linear_model
            if nlm is not None:
                deterministic = deterministic * nlm if mult else deterministic + nlm
            self.gp_data[instruments[i]] = self.flux[instruments[i]] - deterministic
            
            # -------------------------------------------
            #  And GP model -- we need to compute them
            # -------------------------------------------
            
            if not highres:
                gp_model_regressor[instruments[i]] = np.copy( self._gpx(instruments[i]) )
                gp_med_model[instruments[i]] = self.lc_models[instruments[i]]['gp']
                if not robust_errors:
                    gp_up68[instruments[i]], gp_lo68[instruments[i]] = self.lc_models[instruments[i]]['gp_upper'], self.lc_models[instruments[i]]['gp_lower']
                else:
                    ## In case of robust errors, we need to compute errors from ``compute_gp_model`` function
                    gp_quantiles = self.compute_gp_model(instrument=instruments[i], highres=False, parameter_vector=parameter_vector)
                    gp_up68[instruments[i]], gp_lo68[instruments[i]] = gp_quantiles[1,:], gp_quantiles[2,:]

            else:
                # High-resolution GP regressor per instrument
                predict_time = np.linspace( np.min(self._gpx(instruments[i])),\
                                            np.max(self._gpx(instruments[i])), 10000 )
                
                # ------ For highres models, we always need to compute models from ``compute_gp_model`` function
                gp_quantiles = self.compute_gp_model(instrument=instruments[i], highres=True, pred_time=predict_time, parameter_vector=parameter_vector)

                gp_model_regressor[instruments[i]] = predict_time

                gp_med_model[instruments[i]] = gp_quantiles[0,:]
                gp_up68[instruments[i]], gp_lo68[instruments[i]] = gp_quantiles[1,:], gp_quantiles[2,:]
        
        # Now, I think, we have everything -- so, let's plot!
        # -------------------------------------------
        #         Starting figure code:
        #      one plot: do plt.subplots() here
        # -------------------------------------------
        if one_plot:
            fig, axs = plt.subplots(figsize=figsize)
            
            # If there is one plot, then we also need to collect all data to compute binned data
            bin_gp_reg, bin_resids, bin_errors = np.array([]), np.array([]), np.array([])
            
            # Also, in case we don't have highres model, we want to save all models before plotting
            all_regressors_for_plotting, all_models_for_plotting = np.array([]), np.array([])
            all_u68_mods_plotting, all_l68_mods_plotting = np.array([]), np.array([])
        else:
            # We need a list to save all fig and axs objects if we are planning more than one plot
            fig_all, axs_all = [], []

        for i in range(len(instruments)):
            # -------------------------------------------
            #    If not one plot, we need to create
            #    fig and axs object inside for loop
            # -------------------------------------------
            if not one_plot:
                fig, axs = plt.subplots(figsize=figsize)

            # Plotting the "raw" data
            errorbars = self.detrended_errs[instruments[i]]*1e6 if plot_errorbars else None
            alp = 0.3 if plot_errorbars else 0.8
            axs.errorbar(self._gpx(instruments[i]), self.gp_data[instruments[i]]*1e6, yerr=errorbars, fmt='.', elinewidth=0.3, markersize=1.5, alpha=alp, color='#86b6ef', zorder=1)
            
            # ------------------------------------------
            #    Now, we can plot the bin data and
            #    quantile models here, if there is
            #  no one plot, else, just save everything
            #    to one big array, to plot outside
            #     the for loop. We can also do
            #        the label thing here.
            # ------------------------------------------
            if not one_plot:
                # First bin-data
                if not pycheops_binning:
                    nbin = int( len( self.gp_data[instruments[i]] ) / nos_bin )
                    bin_tim, bin_fl, bin_fle = _bin_data(x=self._gpx(instruments[i]), y=self.gp_data[instruments[i]]*1e6, yerr=self.detrended_errs[instruments[i]]*1e6, n_bin=nbin)

                else:
                    binwid = np.ptp( self._gpx(instruments[i]) ) / nos_bin
                    bin_tim, bin_fl, bin_fle, _ = lcbin(time=self._gpx(instruments[i]), flux=self.gp_data[instruments[i]]*1e6, binwidth=binwid)
                axs.errorbar(bin_tim, bin_fl, yerr=bin_fle, fmt='o', c='#184f95', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
                    
                # And now models
                ## If we don't need highres model, we can simply plot what we have
                ## Else, we need to create highres model first
                axs.plot(gp_model_regressor[instruments[i]], gp_med_model[instruments[i]]*1e6, color='#184f95', lw=1.5, zorder=50)
                
                # If we are making one plot per instrument (i.e., one_plot=False), we can simply plot quantile models right here
                axs.fill_between(gp_model_regressor[instruments[i]], y1=gp_lo68[instruments[i]]*1e6, y2=gp_up68[instruments[i]]*1e6, color='#eb6834', alpha=0.5, zorder=25)

                # Labels, limits
                axs.set_xlabel(xlabel)
                axs.set_ylabel('Relative flux [ppm]')

                axs.set_xlim([ np.min(self._gpx(instruments[i])), np.max(self._gpx(instruments[i])) ])
                axs.set_ylim([ np.nanmedian(bin_fl)-10*pipe_mad(bin_fl), np.nanmedian(bin_fl)+10*pipe_mad(bin_fl) ])

                # Saving fig, axs to the list
                fig_all.append(fig)
                axs_all.append(axs)
            else:
                # Saving data to the big list
                bin_gp_reg, bin_resids = np.hstack( (bin_gp_reg, self._gpx(instruments[i])) ), np.hstack( (bin_resids, self.gp_data[instruments[i]]*1e6) )
                bin_errors = np.hstack( (bin_errors, self.detrended_errs[instruments[i]]*1e6) )
                # Saving models to the list
                all_regressors_for_plotting = np.hstack( (all_regressors_for_plotting, gp_model_regressor[instruments[i]]) )
                all_models_for_plotting = np.hstack( (all_models_for_plotting, gp_med_model[instruments[i]]*1e6) )

                ## Let's first save the quantile models
                all_l68_mods_plotting = np.hstack( (all_l68_mods_plotting, gp_lo68[instruments[i]]*1e6) )
                all_u68_mods_plotting = np.hstack( (all_u68_mods_plotting, gp_up68[instruments[i]]*1e6) )
        
        # ------------------------------------------
        #    If one plot, we can now plot the 
        #    bin data, quantile models, labels 
        #    etc. now, outside of the for loop
        # ------------------------------------------
        if one_plot:
            # First, bin data
            if not pycheops_binning:
                nbin = int( len( bin_gp_reg ) / nos_bin / len(instruments) )
                bin_tim, bin_fl, bin_fle = _bin_data(x=bin_gp_reg, y=bin_resids, yerr=bin_errors, n_bin=nbin)

            else:
                binwid = np.ptp( bin_gp_reg ) / nos_bin / len(instruments)
                bin_tim, bin_fl, bin_fle, _ = lcbin(time=bin_gp_reg, flux=bin_resids, binwidth=binwid)
            axs.errorbar(bin_tim, bin_fl, yerr=bin_fle, fmt='o', c='#184f95', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)

            if highres:
                # ------------------------------------------
                #     One plot _and_ highres requires a 
                #          special treatment
                # ------------------------------------------
                ## -- Okay, we need one plot, in high resolution, i.e., we need to call a function to do this.
                idx_gp_reg_sort = np.argsort(bin_gp_reg)
                pred_time = np.linspace(np.min(bin_gp_reg[idx_gp_reg_sort]), np.max(bin_gp_reg[idx_gp_reg_sort]), 10000)
                gp_quantiles = self.compute_gp_model(instrument=None, highres=True, times=bin_gp_reg[idx_gp_reg_sort], pred_time=pred_time,\
                                                     resids=bin_resids[idx_gp_reg_sort]/1e6, errors=bin_errors[idx_gp_reg_sort]/1e6,\
                                                     parameter_vector=parameter_vector)
                
                axs.plot(pred_time, gp_quantiles[0,:]*1e6, color='#184f95', lw=1.5, zorder=50)
                axs.fill_between(pred_time, y1=gp_quantiles[2,:]*1e6, y2=gp_quantiles[1,:]*1e6, color='#eb6834', alpha=0.5, zorder=25)

            else:
                ## First, sorting according to regressors, and then plotting the full model
                idx_reg_sort = np.argsort(all_regressors_for_plotting)
                axs.plot(all_regressors_for_plotting[idx_reg_sort], all_models_for_plotting[idx_reg_sort], color='#184f95', lw=1.5, zorder=50)
                
                # We can plot qunatile models here, if highres=False
                axs.fill_between(all_regressors_for_plotting[idx_reg_sort], y1=all_l68_mods_plotting[idx_reg_sort], y2=all_u68_mods_plotting[idx_reg_sort], color='#eb6834', alpha=0.5, zorder=25)
            
            # Labels, limits
            axs.set_xlabel(xlabel)
            axs.set_ylabel('Relative flux [ppm]')

            axs.set_xlim([ np.min(bin_gp_reg), np.max(bin_gp_reg) ])
            axs.set_ylim([ np.nanmedian(bin_fl)-10*pipe_mad(bin_fl), np.nanmedian(bin_fl)+10*pipe_mad(bin_fl) ])

        if one_plot:
            return fig, axs
        else:
            return fig_all, axs_all
        
    def compute_gp_model(self, instrument=None, highres=None, times=None, pred_time=None, resids=None, errors=None, parameter_vector=None):
        """GP mean and 68% band (``(3, n)``: mean, mean + std, mean - std) conditioned on the GP
        data, from the evaluator's GP at the posterior medians (juliet: its GP object, ``GP.compute``
        and ``GP.predict``; JAX: the tinygp kernel). The errors are the data errors with
        ``sigma_w``.

        ``parameter_vector``: ``{prior name: value}`` replacing posterior medians (juliet fits
        also take the GP object's own parameter vector)."""
        
        # If we compute GP model this way, then the errors will always be robust because we will be using gp.predict

        # If time is None: that means that we need model for an instrument (not one model for all instruments)
        #
        # times is not None, means that we need one model for more than one instruments
        # This model is always highres
        # This also assumes that the GP model is the same for all instruments
        
        if times is None:
            # When time is None, means that we haven't provided any data, so we need to extract it from self.
            ## GP regressor of the dataset
            times = self._gpx(instrument)
            ## Similarly, extracting GP data and errors from self
            resids, errors = self.gp_data[instrument], self.detrended_errs[instrument]
            
            if highres:
                # Model for one instrument, but still highres model
                pred_time = np.linspace(np.min(times), np.max(times), 10000)
            else:
                # Not highres model
                pred_time = np.copy(times)
        
        else:
            # We don't need any data in this case, since everything is provided, 
            # but we still need a name of an instrument to access GP object
            instrument = [i for i in self._gp.keys()][0]

        gp_mean, gp_std = self.ev.gp_predict(instrument, times, resids, errors, pred_time, params=parameter_vector)

        gp_quantiles = np.zeros( (3, len(gp_mean)) )
        gp_quantiles[0, :] = gp_mean
        gp_quantiles[1, :], gp_quantiles[2, :] = gp_mean + gp_std, gp_mean - gp_std

        return gp_quantiles
