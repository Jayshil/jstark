import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.stats import mad_std
from scipy.interpolate import interp1d
import matplotlib
import multiprocessing
from pathlib import Path
from glob import glob
from tqdm import tqdm
#from .plotstyles import *
import pickle
import juliet
from .utils import *
import warnings
import os

try:
    from photutils.aperture import CircularAnnulus, CircularAperture, ApertureStats
    from photutils.aperture import aperture_photometry as aphot
except:
    print('Warning: photutils is not installed! Will not be able to use photutils based photometry.')

try:
    from dace_query.cheops import Cheops
except:
    print('It looks like DACE API is not installed! Will not be able to download CHEOPS data products.')


class julietPlots(object):
    """Plotting helper for results produced by a `juliet` analysis.

    This class loads a `juliet` input/output folder, computes models from the
    fitted posterior and provides a collection of convenience plotting and
    data-processing methods for light-curve visualisation. Typical usage is
    to instantiate the class with the `input_folder` used for a juliet fit
    and then call plotting helpers such as :meth:`full_model_lc`,
    :meth:`plot_gp` or :meth:`detrend_data`.

    Attributes
    ----------
    input_folder : str, optional
        Path to the juliet input/output folder provided at construction.
        The user need to provide either input_folder or dataset and res
    dataset : juliet.load object, optioanl
        The object returned by ``juliet.load`` for the given input folder.
    res : juliet.fit, optional
        The result of calling ``dataset.fit(...)``.
    N : int
        Number of posterior samples drawn when evaluating sampled models.
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
    
    """
    def __init__(self, input_folder=None, dataset=None, res=None, N=1000, **kwargs):
        # First order of business: loading the juliet folder
        # We can optinally provide datasets and res directly

        if ( dataset is None ) and ( input_folder is not None ):
            self.dataset = juliet.load(input_folder=input_folder)
            self.res = self.dataset.fit(**kwargs)
        else:
            self.dataset = dataset
            self.res = res

        # Saving those kwargs
        self.fit_kwargs = kwargs

        # Location of the input folder
        self.input_folder = input_folder

        # Number of samples
        self.nsamps = N

        # Computing all models, for all instruments
        self.models_all_ins = {}
        self.all_mods_ins = {}
        # Sorting indices for all instruments (sometimes GP regressors are not time -- in that case, times 
        # are sorted according to GP regressors, which will mess-up the time array -- we need to sort it again)
        self.idx_time_sort = {}
        for i in range(len(self.dataset.inames_lc)):
            # Evaluating the models
            ## This will return 5 elements: model samples, median model, upper_CI, lower_CI, and components
            self.models_all_ins[self.dataset.inames_lc[i]] =\
                self.res.lc.evaluate(self.dataset.inames_lc[i], nsamples=self.nsamps, return_err=True,\
                                     return_components=True, return_samples=True)

            # Saving all models
            self.all_mods_ins[self.dataset.inames_lc[i]] = self.res.lc.model[self.dataset.inames_lc[i]]

            # Sorting array
            self.idx_time_sort[self.dataset.inames_lc[i]] = np.argsort( self.dataset.times_lc[self.dataset.inames_lc[i]] )

    def highres_planet_models_workaround(self, instrument, times_highres, GP_reg_highres, lin_reg_highres):
        """
        The usual way of generating high time-resolution models
        is not working with juliet when light_travel_delay=True for
        occultations and phase curve modelling. This is a workaround for this."""
        
        # Creating dictionaries to save high-res times, fluxes, and errors
        tim_highres, fl_highres, fle_highres = {}, {}, {}
        ## Actual high-res time array
        tim_highres[instrument] = times_highres
        ## For flux and flux error highres array, we will simply use array with ones
        fl_highres[instrument], fle_highres[instrument] = np.ones(len(times_highres)), np.ones(len(times_highres))

        # GP and linear regressors would be arrays, so creating dictionaries to save them
        if GP_reg_highres is not None:
            gp_pars = {}
            gp_pars[instrument] = GP_reg_highres
        else:
            gp_pars = None
        
        if lin_reg_highres is not None:
            lin_pars = {}
            lin_pars[instrument] = lin_reg_highres
        else:
            lin_pars = None

        ## Again, running juliet.load and juliet.fit
        data_highres = juliet.load(priors=self.input_folder + '/priors.dat', t_lc=tim_highres, y_lc=fl_highres, yerr_lc=fle_highres,\
                                   GP_regressors_lc=gp_pars, linear_regressors_lc=lin_pars, out_folder=self.input_folder)
        res_highres = data_highres.fit(**self.fit_kwargs)

        post_samps_new = res_highres.posteriors['posterior_samples'].copy()
        for i in post_samps_new.keys():
            if ( i[0:5] == 'mflux' ) or ( i[0:9] == 'mdilution' ) or ( i[0:5] == 'theta' ) or ( i[0:2] == 'GP' ):
                post_samps_new[i] = np.zeros(len(post_samps_new[i]))

        planet_only_models_highres = \
            res_highres.lc.evaluate(instrument, nsamples=self.nsamps, return_err=True, parameter_values=post_samps_new,\
                                    return_components=True, return_samples=True, evaluate_transit=True)

        return planet_only_models_highres
        

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
        arrays (times, data, errors) are available in ``self.dataset``.
        """

        # Computing the model for all instruments
        if instruments is None:
            # If instruments is None, then all instruments are selected
            instruments = self.dataset.inames_lc
        
        all_fig, all_axs1, all_axs2 = [], [], []
        for i in range(len(instruments)):
            # Data
            tim_ins = self.dataset.times_lc[instruments[i]][self.idx_time_sort[instruments[i]]]
            fl_ins = self.dataset.data_lc[instruments[i]][self.idx_time_sort[instruments[i]]]
            fle_ins = self.dataset.errors_lc[instruments[i]][self.idx_time_sort[instruments[i]]]

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
            fig = plt.figure()
            gs = gd.GridSpec(2,1, height_ratios=[2,1])

            # Top panel
            ax1 = plt.subplot(gs[0])
            ax1.errorbar(tim_ins, fl_ins, yerr=fle_ins, fmt='.', elinewidth=0.5, markersize=1.5, color='dodgerblue')#, alpha=0.3)
            ax1.plot(tim_ins, full_model, c='navy', lw=2, zorder=50)
            if quantile_models:
                ax1.fill_between(x=tim_ins, y1=lo_68CI, y2=up_68CI, color='orangered', alpha=0.5, zorder=25)
            else:
                for rand in range(nrandom):
                    ax1.plot(tim_ins, random_models[rand,:], color='orangered', alpha=0.5, lw=1., zorder=25)

            ax1.set_ylabel('Relative Flux')
            ax1.set_title('Full fitted model for instrument ' + str(instruments[i]))
            ax1.set_xlim(np.min(tim_ins), np.max(tim_ins))
            ax1.xaxis.set_major_formatter(plt.NullFormatter())

            # Bottom panel
            ax2 = plt.subplot(gs[1])
            ax2.errorbar(tim_ins, (fl_ins-full_model)*1e6, yerr=fle_ins*1e6, fmt='.', elinewidth=0.5, markersize=1.5, color='dodgerblue')#, alpha=0.3)
            ax2.axhline(y=0.0, c='black', ls='--', zorder=100)
            ax2.set_ylabel('Residuals [ppm]')
            ax2.set_xlabel('Time [BJD]')
            ax2.set_xlim(np.min(tim_ins), np.max(tim_ins))

            all_fig.append(fig)
            all_axs1.append(ax1)
            all_axs2.append(ax2)

            if save:
                plt.savefig(self.input_folder + '/full_model_' + instruments[i] + '.png', dpi=250)
        
        return all_fig, all_axs1, all_axs2
    
    def detrend_data(self, phmin, instruments, planet='p1'):
        """This function will generate detrended data from the raw data. It will also generate phases."""
        
        # First, create detrended dataset and models
        ## For detrended dataset
        self.phase, self.detrended_data, self.detrended_errs = {}, {}, {}
        self.ph_minimum, self.ph_maximum = 1e10, 1e-10

        ## For normalising the data, we will compute F1 and F2,
        ## We can use the same F1 and F2 to normalise the model as well
        ## So, let's save them!
        self.F1, self.F2 = {}, {}

        for i in range(len(instruments)):
            # -----------------------------------------
            #  Our first task is to compute
            #  period and t0:
            # -----------------------------------------
            try:
                per = np.nanmedian(self.res.posteriors['posterior_samples']['P_' + planet])
            except:
                per = self.all_mods_ins[instruments[i]]['params'].per
            
            try:
                t0 = np.nanmedian(self.res.posteriors['posterior_samples']['t0_' + planet])
            except:
                t0 = self.all_mods_ins[instruments[i]]['params'].t0
            # -----------------------------------------

            # -----------------------------------------
            #  Next we would like to compute GP
            #  and linear models, so that we can 
            #  detrend the data
            # -----------------------------------------
            # GP model
            if self.dataset.GP_lc_arguments != None:
                if instruments[i] in self.dataset.GP_lc_arguments.keys():
                    gp_model = self.all_mods_ins[instruments[i]]['GP']
                else:
                    gp_model = 0.
            else:
                gp_model = 0.

            if instruments[i] in self.dataset.lm_lc_arguments.keys():
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

            mflx = self.res.posteriors['posterior_samples']['mflux_' + instruments[i]]
            try:
                mdil = self.res.posteriors['posterior_samples']['mdilution_' + instruments[i]]
            except:
                mdil = np.ones( len(mflx) )
            
            F1 = np.nanmedian( mdil / ( 1 + (mdil * mflx) ) )
            F2 = np.nanmedian( (1 - mdil) / (1 + (mdil * mflx) ) )

            ## Saving F1 and F2
            self.F1[instruments[i]], self.F2[instruments[i]] = F1, F2

            sigw = np.nanmedian(self.res.posteriors['posterior_samples']['sigma_w_' + instruments[i]] * 1e-6)

            ## Here we create detrended data by subtracting GP (if any) and linear (if any) models from the raw data
            self.detrended_data[instruments[i]] = ( self.dataset.data_lc[instruments[i]] - gp_model - linear_model - F2) / F1 
            self.detrended_errs[instruments[i]] = np.sqrt( self.dataset.errors_lc[instruments[i]]**2 + sigw**2 )

            phases_instrument = juliet.utils.get_phases(self.dataset.times_lc[instruments[i]], P=per, t0=t0, phmin=phmin)
            self.phase[instruments[i]] = phases_instrument
            
            # Computing the minimum and maximum of the orbital phases
            if np.max(phases_instrument) > self.ph_maximum:
                self.ph_maximum = np.max(phases_instrument)
            if np.min(phases_instrument) < self.ph_minimum:
                self.ph_minimum = np.min(phases_instrument)

    
    def detrend_model(self, instruments, phmin, highres, planet='p1'):
        # If we want to create high-resolution model (highres=True), then we will use the same times for all instruments
        # If not, then we will use different times for different instruments
        self.times_model, self.phases_model = {}, {}
        self.planet_only_models = {} 
        
        # -----------------------------------------
        #  Our first task is to compute
        #  period and t0:
        # -----------------------------------------
        try:
            per = np.nanmedian(self.res.posteriors['posterior_samples']['P_' + planet])
        except:
            per = self.all_mods_ins[instruments[0]]['params'].per
        
        try:
            t0 = np.nanmedian(self.res.posteriors['posterior_samples']['t0_' + planet])
        except:
            t0 = self.all_mods_ins[instruments[0]]['params'].t0

        # We also have to create a new posterior array since when we compute detrended model,
        # we only want uncertainties from planetary parameters to be propagated to the model uncertainties
        # So, we will make posteriors of mflux, theta, and GP to zero, such that any uncertainties
        # in them would not propagate to model uncertainties.

        post_samps_new = self.res.posteriors['posterior_samples'].copy()
        for i in post_samps_new.keys():
            if ( i[0:5] == 'mflux' ) or ( i[0:9] == 'mdilution' ) or ( i[0:5] == 'theta' ) or ( i[0:2] == 'GP' ):
                post_samps_new[i] = np.zeros(len(post_samps_new[i]))
        
        if highres:
            ## First selecting the times
            ## This this is a high time resolution scenario so we will create a dummy time array
            dummy_phs_highres = np.linspace(self.ph_minimum, self.ph_maximum, 10000)
            dummy_time_highres = t0 + (dummy_phs_highres * per)

            # Now loop over all instruments to create high time-resolution planet-only models
            for ins in range(len(instruments)):
                
                ## First we have to check if there are GP or linear model fitted to the data
                ## Because, we have to create high-resolution GP and linear regressors as well.
                ## We will do this by interpolating the original GP and linear regressors

                ### High resolution GP regressors
                ### (we have to remember that everything is sorted according to GP regressors)
                ### (while dummy time is sorted according to times)
                ### (we will perform interpolation to arrays sorted according to time)
                if self.dataset.GP_lc_arguments != None:
                    if instruments[ins] in self.dataset.GP_lc_arguments.keys():
                        #! Try/except because it is possible that not all instruments have GP model
                        try:
                            inter_gp = interp1d(x=self.dataset.times_lc[instruments[ins]][self.idx_time_sort[instruments[ins]]],\
                                                y=self.dataset.GP_lc_arguments[instruments[ins]][:,0][self.idx_time_sort[instruments[ins]]],\
                                                kind='cubic')
                            gp_reg_highres = inter_gp(x=dummy_time_highres)

                            #### Now, we can sort time, phase, gp regressors according to GP regressors
                            idx_gp = np.argsort( gp_reg_highres )
                            dummy_time_highres, dummy_phs_highres = dummy_time_highres[idx_gp], dummy_phs_highres[idx_gp]
                            gp_reg_highres = gp_reg_highres[idx_gp]
                        except:
                            gp_reg_highres = None
                    else:
                        gp_reg_highres = None
                else:
                    gp_reg_highres = None

                ### High resolution linear regressors
                if instruments[ins] in self.dataset.lm_lc_arguments.keys():
                    #! Again, try/except because not all instruments might have a linear model
                    ## (Here, we don't need to sort according to time -- because the dummy_times should have now
                    ##  sorted according to GP regressors, if applicable)
                    try:
                        inter_lin = interp1d(x=self.dataset.times_lc[instruments[ins]],\
                                             y=self.dataset.lm_lc_arguments[instruments[ins]],\
                                             kind='cubic', axis=0)
                        lin_reg_highres = inter_lin(x=dummy_time_highres)
                    except:
                        lin_reg_highres = None
                else:
                    lin_reg_highres = None

                try:
                    # This high-res time models will not work for occultations/phase curves, when light_travel_delay=True
                    self.planet_only_models[instruments[ins]] = \
                        self.res.lc.evaluate(instruments[ins], nsamples=self.nsamps, return_err=True, parameter_values=post_samps_new,\
                                            return_components=True, return_samples=True, evaluate_transit=True,\
                                            t=dummy_time_highres, GPregressors=gp_reg_highres, LMregressors=lin_reg_highres)
                except:
                    # In case when it doesn't work, I have invented a workaround
                    self.planet_only_models[instruments[ins]] = \
                        self.highres_planet_models_workaround(instrument=instruments[ins], times_highres=dummy_time_highres,\
                                                              GP_reg_highres=gp_reg_highres, lin_reg_highres=lin_reg_highres)
                ## Saving times and phases for model
                self.times_model[instruments[ins]] = dummy_time_highres
                self.phases_model[instruments[ins]] = dummy_phs_highres
        else:
            for ins in range(len(instruments)):
                # This should be fairly trivial since we don't need to compute the models at higher time resolution
                self.planet_only_models[instruments[ins]] = \
                    self.res.lc.evaluate(instruments[ins], nsamples=self.nsamps, return_err=True, parameter_values=post_samps_new,\
                                            return_components=True, return_samples=True, evaluate_transit=True)
                
                ## Saving times and phases for model
                self.times_model[instruments[ins]] = self.dataset.times_lc[instruments[ins]]
                self.phases_model[instruments[ins]] = juliet.utils.get_phases(t=self.dataset.times_lc[instruments[ins]], P=per, t0=t0, phmin=phmin)

    
    def phase_folded_lc(self, phmin=0.8, instruments=None, planet='p1', highres=False, nrandom=50, quantile_models=True, one_plot=None, figsize=(16/1.5, 9/1.5), pycheops_binning=False, nos_bin_tra=20, nos_bin_pc=30):
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
            in the juliet dataset (``self.dataset.inames_lc``) are used. One plot
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

        Returns
        -------
        If a single combined plot is produced the function returns the
        combined ``fig`` and the primary axes used. If multiple figures are
        generated, the function returns lists: ``figs_all, axs1_all,
        axs2_all, axs3_all, axs4_all`` corresponding to created figures and
        their axes panels.

        """
        # -------------------------------------------
        #         Do we want one plot?
        # -------------------------------------------
        # If instruments == None, then we will make one common plot for _all_ instruments
        # Also, if all instruments are provided, this will make one common plot
        # Otherwise, one plot per instrument
        # However, sometimes we want to force one plot, so we have the option one_plot
        if one_plot is None:
            one_plot = True
            if instruments != None:
                if len(instruments) != len(self.dataset.inames_lc):
                    one_plot = False
            else:
                instruments = self.dataset.inames_lc

        # -------------------------------------------
        #               Compute t14
        # -------------------------------------------
        
        # Let's compute the transit/occultation duration for the planet (that can help setting the xlimits of the plots)
        ## We need several other parameters for that first

        ### Orbital period
        try:
            per = np.nanmedian(self.res.posteriors['posterior_samples']['P_' + planet])
        except:
            per = self.all_mods_ins[instruments[0]]['params'].per
        
        ### Rp/R*
        try:
            rprs = np.nanmedian(self.res.posteriors['posterior_samples']['p_' + planet])
        except:
            rprs = self.all_mods_ins[instruments[0]]['params'].rp

        ### a/R*
        if 'rho' in self.res.model_parameters:
            ### This would mean that we have fitted for rho. We can convert rho to a/*
            try:
                rho = np.nanmedian(self.res.posteriors['posterior_samples']['rho'])
                ar = rho_to_ar(rho, per)
            except:
                ar = self.all_mods_ins[instruments[0]]['params'].a
        else:
            ### That means that we directly fit for a/R*
            try:
                ar = np.nanmedian(self.res.posteriors['posterior_samples']['a_' + planet])
            except:
                ar = self.all_mods_ins[instruments[0]]['params'].a
        
        ### b
        try:
            b = np.nanmedian(self.res.posteriors['posterior_samples']['b_' + planet])
        except:
            inc = self.all_mods_ins[instruments[0]]['params'].inc
            b = inc_to_b(inc=inc, ar=ar, ecc=self.all_mods_ins[instruments[0]]['params'].ecc,\
                               omega=self.all_mods_ins[instruments[0]]['params'].w)
        t14_dur = t14(per=per, ar=ar, rprs=rprs, b=b, ecc=self.all_mods_ins[instruments[0]]['params'].ecc,\
                        omega=self.all_mods_ins[instruments[0]]['params'].w)
        t14_phs = t14_dur / per

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
            if self.dataset.lc_options[instruments[ins]]['TranEclFit']:
                phasecurve = True
            elif self.dataset.lc_options[instruments[ins]]['EclipseFit']:
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
            residuals = ( self.dataset.data_lc[instruments[i]] - self.models_all_ins[instruments[i]][1] )[idx_phs]

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
            
            ax1.errorbar(phs, detrend_data*ppt, fmt='.', elinewidth=0.5, markersize=1.5, alpha=0.5, color='dodgerblue', zorder=1)
            if (not one_plot) or highres:
                ## if we are making one-plot, we can plot the detrended model out of loop
                ax1.plot(phs_model, detrend_model*ppt, color='navy', lw=2.5, zorder=50)
            if quantile_models:
                if (not one_plot) or highres:
                    ## Again, if we are making one plot, we can plot the detrended quantile models out of loop
                    ax1.fill_between(phs_model, y1=lo_68CI*ppt, y2=up_68CI*ppt, color='orangered', alpha=0.15, zorder=25)
            else:
                for rand in range(nrandom):
                    ax1.plot(phs_model, random_models[rand,:]*ppt, color='orangered', alpha=0.5, lw=1., zorder=25)
            
            # Limits
            ## Maximum of the transit model would always be 1. (or, 0 in case we have phase curve model, and we decided to use ppm)
            if transit or phasecurve:
                ax1.set_xlim(-0.7*t14_phs, 0.7*t14_phs)
            else:
                ax1.set_xlim(0.5-0.7*t14_phs, 0.5+0.7*t14_phs)
            
            if transit:
                ax1.set_ylim([np.min(detrend_model)-0.2*np.ptp(detrend_model), 1+0.2*np.ptp(detrend_model)])
            elif phasecurve:
                ax1.set_ylim([np.min(detrend_model*ppt)-0.2*np.ptp(detrend_model*ppt), 0.+0.2*np.ptp(detrend_model*ppt)])
            else:
                ax1.set_ylim([1-0.3*np.ptp(detrend_model), np.max(detrend_model)+0.3*np.ptp(detrend_model)])

            # -------------------------------------------
            #        Bottom left: transit residuals
            # -------------------------------------------
            ax2.errorbar(phs, residuals*ppt*1e6, fmt='.', elinewidth=0.5, markersize=1.5, alpha=0.5, color='dodgerblue', zorder=1)
            ax2.axhline(y=0., ls='--', color='k', zorder=50)

            # Limits (x-limit same as the top axis; for y-lim, we will set 3-sigma from median)
            if transit or phasecurve:
                ax2.set_xlim(-0.7*t14_phs, 0.7*t14_phs)
            else:
                ax2.set_xlim(0.5-0.7*t14_phs, 0.5+0.7*t14_phs)

            # -------------------------------------------
            #        Binned data: For TRANSIT
            # -------------------------------------------
            ## We do binned data here: ONLY if we don't have one plot
            if not one_plot:
                if transit or phasecurve:
                    idx_transits = ( phs > -0.8*t14_phs ) & ( phs < 0.8*t14_phs )
                else:
                    idx_transits = ( phs > 0.5-0.8*t14_phs ) & ( phs < 0.5+0.8*t14_phs )
                
                if not pycheops_binning:
                    nbin = int( np.sum(idx_transits) / nos_bin_tra )

                    bin_phs_tra, bin_fl_tra, bin_fle_tra = juliet.utils.bin_data(x=phs[idx_transits], y=detrend_data[idx_transits]*ppt, n_bin=nbin, yerr=detrend_errs[idx_transits]*ppt)
                    if phasecurve or eclipse:
                        _, bin_res_tra, bin_reserr_tra = juliet.utils.bin_data(x=phs[idx_transits], y=residuals[idx_transits]*ppt*1e6, n_bin=nbin, yerr=detrend_errs[idx_transits]*ppt)
                    else:
                        # When transit only, we haven't multiplied the errorbars with 1e6 yet; so we need to do it again
                        _, bin_res_tra, bin_reserr_tra = juliet.utils.bin_data(x=phs[idx_transits], y=residuals[idx_transits]*1e6, n_bin=nbin, yerr=detrend_errs[idx_transits]*1e6)
                else:
                    binwid = np.ptp( phs[idx_transits] ) / nos_bin_tra

                    bin_phs_tra, bin_fl_tra, bin_fle_tra, _ = lcbin(time=phs[idx_transits], flux=detrend_data[idx_transits]*ppt, binwidth=binwid)
                    if phasecurve or eclipse:
                        _, bin_res_tra, bin_reserr_tra, _ = lcbin(time=phs[idx_transits], flux=residuals[idx_transits]*ppt*1e6, binwidth=binwid)
                    else:
                        # When transit only, we haven't multiplied the errorbars with 1e6 yet; so we need to do it again
                        _, bin_res_tra, bin_reserr_tra, _ = lcbin(time=phs[idx_transits], flux=residuals[idx_transits]*1e6, binwidth=binwid)

                # Plotting them
                ax1.errorbar(bin_phs_tra, bin_fl_tra, yerr=bin_fle_tra, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
                ax2.errorbar(bin_phs_tra, bin_res_tra, yerr=bin_reserr_tra, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)

                # Setting y-lim on residuals based on binned residuals
                ax2.set_ylim(np.nanmedian(bin_res_tra)-5*pipe_mad(bin_res_tra), np.nanmedian(bin_res_tra)+5*pipe_mad(bin_res_tra))

            # -------------------------------------------
            #         If phase curve modelling,
            #        then we also need right panel
            # -------------------------------------------
            if phasecurve:
                # -------------------------------------------
                #         Top right: phase curve
                # -------------------------------------------
                ax3.errorbar(phs, detrend_data, fmt='.', elinewidth=0.5, markersize=1.5, alpha=0.5, color='dodgerblue', zorder=1)
                if (not one_plot) or highres:
                    ## If we are making one plot, then we can take this outside of the loop
                    ax3.plot(phs_model, detrend_model, color='navy', lw=2.5, zorder=50)
                ax3.axhline(0.0, color='k', ls='--', lw=1.5, zorder=45)
                
                if quantile_models:
                    if (not one_plot) or highres:
                        ## If we are making one plot, we can take this outside of the loop
                        ax3.fill_between(phs_model, y1=lo_68CI, y2=up_68CI, color='orangered', alpha=0.15, zorder=25)
                else:
                    for rand in range(nrandom):
                        ax3.plot(phs_model, random_models[rand,:], color='orangered', alpha=0.5, lw=1., zorder=25)
                
                # Limits
                ax3.set_xlim(phmin-1., phmin)
                ## For y-lim; we have baseline of 0, we can leave half occultation depth/phase amplitude below and above
                ylen = np.ptp( detrend_model[np.abs(phs_model) > 0.1] )
                ax3.set_ylim([-0.4*ylen, 1.5*ylen])

                # -------------------------------------------
                #     Bottom right: phase curve residuals
                # -------------------------------------------
                ax4.errorbar(phs, residuals*1e6, fmt='.', elinewidth=0.5, markersize=1.5, alpha=0.5, color='dodgerblue', zorder=1)
                ax4.axhline(y=0., ls='--', color='k', zorder=10)

                # Limits
                ax4.set_xlim(phmin-1., phmin)

                # -------------------------------------------
                #        Binned data: For PHASE CURVE
                # -------------------------------------------
                ## We do binned data here: ONLY if we don't have one plot
                if not one_plot:
                    if not pycheops_binning:
                        bin_phs_pc, bin_fl_pc, bin_fle_pc = juliet.utils.bin_data(x=phs, y=detrend_data, n_bin=int( len(phs)/nos_bin_pc ), yerr=detrend_errs)
                        _, bin_res_pc, bin_reserr_pc = juliet.utils.bin_data(x=phs, y=residuals*1e6, n_bin=int( len(phs)/nos_bin_pc ), yerr=detrend_errs)
                    else:
                        bin_phs_pc, bin_fl_pc, bin_fle_pc, _ = lcbin(time=phs, flux=detrend_data, binwidth=1/nos_bin_pc)
                        _, bin_res_pc, bin_reserr_pc, _ = lcbin(time=phs, flux=residuals*1e6, binwidth=1/nos_bin_pc)

                    # Plotting them
                    ax3.errorbar(bin_phs_pc, bin_fl_pc, yerr=bin_fle_pc, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
                    ax4.errorbar(bin_phs_pc, bin_res_pc, yerr=bin_reserr_pc, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)

                    ax4.set_ylim(np.nanmedian(bin_res_pc)-5*pipe_mad(bin_res_pc), np.nanmedian(bin_res_pc)+5*pipe_mad(bin_res_pc))

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
                idx_transits = ( bin_phs > -0.8*t14_phs ) & ( bin_phs < 0.8*t14_phs )
            else:
                idx_transits = ( bin_phs > 0.5-0.8*t14_phs ) & ( bin_phs < 0.5+0.8*t14_phs )
            
            if not pycheops_binning:
                nbin = int( np.sum(idx_transits) / nos_bin_tra )

                # Performing binning
                bin_phs_tra, bin_fl_tra, bin_fle_tra = juliet.utils.bin_data(x=bin_phs[idx_transits], y=bin_fl[idx_transits]*ppt, n_bin=nbin, yerr=bin_fle[idx_transits]*ppt)
                if phasecurve or eclipse:
                    _, bin_res_tra, bin_reserr_tra = juliet.utils.bin_data(x=bin_phs[idx_transits], y=bin_res[idx_transits]*ppt*1e6, n_bin=nbin, yerr=bin_fle[idx_transits]*ppt)
                else:
                    # When transit only, we haven't multiplied the errorbars with 1e6 yet; so we need to do it again
                    _, bin_res_tra, bin_reserr_tra = juliet.utils.bin_data(x=bin_phs[idx_transits], y=bin_res[idx_transits]*1e6, n_bin=nbin, yerr=bin_fle[idx_transits]*1e6)
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
            ax1.errorbar(bin_phs_tra, bin_fl_tra, yerr=bin_fle_tra, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
            ax2.errorbar(bin_phs_tra, bin_res_tra, yerr=bin_reserr_tra, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)

            # Setting y-lim on residuals based on binned residuals
            ax2.set_ylim(np.nanmedian(bin_res_tra)-5*pipe_mad(bin_res_tra), np.nanmedian(bin_res_tra)+5*pipe_mad(bin_res_tra))
            
            # -------------------------------------------
            #        Binned data: For PHASE CURVE
            # -------------------------------------------
            if phasecurve:
                if not pycheops_binning:
                    bin_phs_pc, bin_fl_pc, bin_fle_pc = juliet.utils.bin_data(x=bin_phs, y=bin_fl, n_bin=int( len(bin_phs)/nos_bin_pc ), yerr=bin_fle)
                    _, bin_res_pc, bin_reserr_pc = juliet.utils.bin_data(x=bin_phs, y=bin_res*1e6, n_bin=int( len(bin_phs)/nos_bin_pc ), yerr=bin_fle)
                else:
                    bin_phs_pc, bin_fl_pc, bin_fle_pc, _ = lcbin(time=bin_phs, flux=bin_fl, binwidth=1/nos_bin_pc)
                    _, bin_res_pc, bin_reserr_pc, _ = lcbin(time=bin_phs, flux=bin_res*1e6, binwidth=1/nos_bin_pc)

                # Plotting them
                ax3.errorbar(bin_phs_pc, bin_fl_pc, yerr=bin_fle_pc, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
                ax4.errorbar(bin_phs_pc, bin_res_pc, yerr=bin_reserr_pc, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)

                ax4.set_ylim(np.nanmedian(bin_res_pc)-5*pipe_mad(bin_res_pc), np.nanmedian(bin_res_pc)+5*pipe_mad(bin_res_pc))

            
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
                ax1.plot(bin_phs, all_models_for_plotting*ppt, color='navy', lw=2.5, zorder=50)
                if quantile_models:
                    ax1.fill_between(bin_phs, y1=all_l68_mods_plotting*ppt, y2=all_u68_mods_plotting*ppt, color='orangered', alpha=0.5, zorder=25)

                # ---------------------------------------------------
                #           For phase curve
                # ---------------------------------------------------
                if phasecurve:
                    ## ----------- We can now plot the models (detrended and quantile models)
                    ax3.plot(bin_phs, all_models_for_plotting, color='navy', lw=2.5, zorder=50)
                    if quantile_models:
                        ax3.fill_between(bin_phs, y1=all_l68_mods_plotting, y2=all_u68_mods_plotting, color='orangered', alpha=0.5, zorder=25)

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

    def plot_fake_allan_deviation(self, instruments=None, binmax=10, method='pipe', timeunit=None):
        """Compute and plot per-instrument noise-vs-bin-size curves.

        This is a thin wrapper around ``utils.fake_allan_deviation`` that
        computes residuals for each instrument (data minus the full model)
        and returns the figures, axes and numeric results for further use.

        Parameters
        ----------
        instruments : list or None
            List of instrument names to include. If ``None``, all
            instruments in the juliet dataset (``self.dataset.inames_lc``)
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
        if instruments != None:
            if len(instruments) != len(self.dataset.inames_lc):
                one_plot = False
        else:
            instruments = self.dataset.inames_lc

        ## All stuff we can save
        figs_all, axs_all = [], []
        binsize_all, noise_all, white_noise_all = [], [], []

        for ins in range(len(instruments)):
            # Getting the data
            tim_ins = self.dataset.times_lc[instruments[ins]][self.idx_time_sort[instruments[ins]]]
            fl_ins = self.dataset.data_lc[instruments[ins]][self.idx_time_sort[instruments[ins]]]

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
        post_samps = self.res.posteriors['posterior_samples']

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
            plt.savefig(self.input_folder + '/corner_plot.png', dpi=250)

        return fig
    
    def plot_gp(self, instruments=None, highres=False, one_plot=False, figsize=(16/1.5, 9/1.5), pycheops_binning=False, xlabel='GP regressor', robust_errors=False, plot_errorbars=False, parameter_vector=None, nos_bin=20):
        """Plot the Gaussian Process component(s) from the fitted `juliet` model.

        This routine computes and plots the GP model (and binned data)
        for one or more instruments present in the current `juliet` dataset.

        Parameters
        ----------
        instruments : list or None
            List of instrument names to plot. If ``None``, all instruments for
            which GP regressors exist in ``self.dataset.GP_lc_arguments`` are
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
        parameter_vector : array-like or None, optional
            If provided, passed to the GP parameter setter before computing
            predictions (useful for evaluating GP at different parameter
            values).
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
        if instruments != None:
            ## Let's check if the provided instruments indeed have GP fitting
            for ins in range(len(instruments)):
                if instruments[ins] not in self.dataset.GP_lc_arguments.keys():
                    raise ValueError('GP fitting is not found for instrument ' + instruments[ins])
        else:
            instruments = [i for i in self.dataset.GP_lc_arguments.keys()]
        
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
            if instruments[i] in self.dataset.lm_lc_arguments.keys():
                linear_model = self.models_all_ins[instruments[i]][-1]['lm']
            else:
                linear_model = 0.
            
            # -------------------------------------------
            #   GP data: i.e., data fitted to GP model
            # -------------------------------------------
            self.gp_data[instruments[i]] = self.dataset.data_lc[instruments[i]] - ( self.F1[instruments[i]] * self.planet_only_models[instruments[i]][1] ) -\
                                           self.F2[instruments[i]] - linear_model
            
            # -------------------------------------------
            #  And GP model -- we need to compute them
            # -------------------------------------------
            
            if not highres:
                gp_model_regressor[instruments[i]] = np.copy( self.dataset.GP_lc_arguments[instruments[i]][:, 0] )
                gp_med_model[instruments[i]] = self.all_mods_ins[instruments[i]]['GP']
                if not robust_errors:
                    gp_up68[instruments[i]], gp_lo68[instruments[i]] = self.all_mods_ins[instruments[i]]['GP_uerror'], self.all_mods_ins[instruments[i]]['GP_lerror']
                else:
                    ## In case of robust errors, we need to compute errors from ``compute_gp_model`` function
                    gp_quantiles = self.compute_gp_model(instrument=instruments[i], highres=False, parameter_vector=parameter_vector)
                    gp_up68[instruments[i]], gp_lo68[instruments[i]] = gp_quantiles[1,:], gp_quantiles[2,:]

            else:
                # High-resolution GP regressor per instrument
                predict_time = np.linspace( np.min(self.dataset.GP_lc_arguments[instruments[i]][:, 0]),\
                                            np.max(self.dataset.GP_lc_arguments[instruments[i]][:, 0]), 10000 )
                
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
            axs.errorbar(self.dataset.GP_lc_arguments[instruments[i]][:,0], self.gp_data[instruments[i]]*1e6, yerr=errorbars, fmt='.', elinewidth=0.3, markersize=1.5, alpha=alp, color='dodgerblue', zorder=1)
            
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
                    bin_tim, bin_fl, bin_fle = juliet.utils.bin_data(x=self.dataset.GP_lc_arguments[instruments[i]][:,0], y=self.gp_data[instruments[i]]*1e6, yerr=self.detrended_errs[instruments[i]]*1e6, n_bin=nbin)

                else:
                    binwid = np.ptp( self.dataset.GP_lc_arguments[instruments[i]][:,0] ) / nos_bin
                    bin_tim, bin_fl, bin_fle, _ = lcbin(time=self.dataset.GP_lc_arguments[instruments[i]][:,0], flux=self.gp_data[instruments[i]]*1e6, binwidth=binwid)
                axs.errorbar(bin_tim, bin_fl, yerr=bin_fle, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
                    
                # And now models
                ## If we don't need highres model, we can simply plot what we have
                ## Else, we need to create highres model first
                axs.plot(gp_model_regressor[instruments[i]], gp_med_model[instruments[i]]*1e6, color='navy', lw=1.5, zorder=50)
                
                # If we are making one plot per instrument (i.e., one_plot=False), we can simply plot quantile models right here
                axs.fill_between(gp_model_regressor[instruments[i]], y1=gp_lo68[instruments[i]]*1e6, y2=gp_up68[instruments[i]]*1e6, color='orangered', alpha=0.5, zorder=25)

                # Labels, limits
                axs.set_xlabel(xlabel)
                axs.set_ylabel('Relative flux [ppm]')

                axs.set_xlim([ np.min(self.dataset.GP_lc_arguments[instruments[i]][:,0]), np.max(self.dataset.GP_lc_arguments[instruments[i]][:,0]) ])
                axs.set_ylim([ np.nanmedian(bin_fl)-10*pipe_mad(bin_fl), np.nanmedian(bin_fl)+10*pipe_mad(bin_fl) ])

                # Saving fig, axs to the list
                fig_all.append(fig)
                axs_all.append(axs)
            else:
                # Saving data to the big list
                bin_gp_reg, bin_resids = np.hstack( (bin_gp_reg, self.dataset.GP_lc_arguments[instruments[i]][:,0]) ), np.hstack( (bin_resids, self.gp_data[instruments[i]]*1e6) )
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
                bin_tim, bin_fl, bin_fle = juliet.utils.bin_data(x=bin_gp_reg, y=bin_resids, yerr=bin_errors, n_bin=nbin)

            else:
                binwid = np.ptp( bin_gp_reg ) / nos_bin / len(instruments)
                bin_tim, bin_fl, bin_fle, _ = lcbin(time=bin_gp_reg, flux=bin_resids, binwidth=binwid)
            axs.errorbar(bin_tim, bin_fl, yerr=bin_fle, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)

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
                
                axs.plot(pred_time, gp_quantiles[0,:]*1e6, color='navy', lw=1.5, zorder=50)
                axs.fill_between(pred_time, y1=gp_quantiles[2,:]*1e6, y2=gp_quantiles[1,:]*1e6, color='orangered', alpha=0.5, zorder=25)

            else:
                ## First, sorting according to regressors, and then plotting the full model
                idx_reg_sort = np.argsort(all_regressors_for_plotting)
                axs.plot(all_regressors_for_plotting[idx_reg_sort], all_models_for_plotting[idx_reg_sort], color='navy', lw=1.5, zorder=50)
                
                # We can plot qunatile models here, if highres=False
                axs.fill_between(all_regressors_for_plotting[idx_reg_sort], y1=all_l68_mods_plotting[idx_reg_sort], y2=all_u68_mods_plotting[idx_reg_sort], color='orangered', alpha=0.5, zorder=25)
            
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
        """This function computes the GP models. It accesses the GP object from the juliet files, and then do GP.compute and GP.predict."""
        
        # If we compute GP model this way, then the errors will always be robust because we will be using gp.predict

        # If time is None: that means that we need model for an instrument (not one model for all instruments)
        #
        # times is not None, means that we need one model for more than one instruments
        # This model is always highres
        # This also assumes that the GP model is the same for all instruments
        
        if times is None:
            # When time is None, means that we haven't provided any data, so we need to extract it from self.
            ## Extracting times (GP regressors) from self.dataset object
            times = self.dataset.GP_lc_arguments[instrument][:,0]
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
            instrument = [i for i in self.dataset.GP_lc_arguments.keys()][0]

        if parameter_vector is not None:
            self.dataset.lc_options[instrument]['noise_model'].GP.set_parameter_vector(parameter_vector)
        
        # GP.compute
        self.dataset.lc_options[instrument]['noise_model'].GP.compute(t=times, yerr=errors)
        # And GP prediction
        gp_mean, gp_var = self.dataset.lc_options[instrument]['noise_model'].GP.predict(y=resids, t=pred_time, return_var=True)
        gp_std = np.sqrt(gp_var)

        gp_quantiles = np.zeros( (3, len(gp_mean)) )
        gp_quantiles[0, :] = gp_mean
        gp_quantiles[1, :], gp_quantiles[2, :] = gp_mean + gp_std, gp_mean - gp_std

        return gp_quantiles
    
class SpectroscopicLC(object):
    """Analyze and fit multi-wavelength time-series spectroscopic data.

    This class provides tools for handling spectroscopic light curves where
    flux is measured as a function of both time and wavelength. It supports
    binning spectral data into channels, fitting individual wavelength bins
    using ``juliet``, parallel processing of multiple channels, and
    visualization of spectral transit/eclipse signatures and systematics.

    Parameters
    ----------
    times : ndarray
        1D array of time stamps (in BJD or similar) with shape (N_frames,).
    lc : ndarray
        2D array of flux measurements with shape (N_frames, N_wavelengths).
        Each row is a time frame; each column is a wavelength bin.
    lc_errs : ndarray
        2D array of flux uncertainties with the same shape as ``lc``.
    wavelengths : ndarray
        1D array of wavelength values (in microns or other units) corresponding
        to each column of ``lc``. Shape is (N_wavelengths,).
    gp_pars : dict or None, optional
        Dictionary mapping instrument names to arrays of Gaussian Process (GP)
        regressor values (e.g., time, roll angle). Used when fitting with GP
        systematics. Default is ``None``.
    lin_pars : dict or None, optional
        Dictionary mapping instrument names to 2D arrays of linear regressor
        values (shape: N_frames x N_regressors). Used for linear detrending.
        Default is ``None``.
    priors : callable or None, optional
        A callable that accepts a channel name (string) and returns a juliet-format priors 
        for that channel. Required for fitting
        via :meth:`analyse_lc_parallel`. Default is ``None``.
    pout : str, optional
        Output directory for saving results, intermediate data, and pickled
        light curves. Default is the current working directory.

    Attributes
    ----------
    times : ndarray
        Input time array.
    lc : ndarray
        Input 2D light-curve array.
    lc_errs : ndarray
        Input 2D error array.
    wavelengths : ndarray
        Input wavelength array.
    gp_pars, lin_pars : dict or None
        Input systematics parameters.
    priors : callable or None
        Input priors function.
    pout : str
        Output directory.
    col_st, col_end : ndarray
        Start and end column indices for each binned channel (populated by
        :meth:`col_spec`).
    spectral_lcs_dict : dict
        Dictionary containing binned spectral light curves and metadata,
        populated by :meth:`generating_lightcurves`. Keys include 'lc', 'err',
        'wave', and 'wave_bin'.
    wav, wav_bin : ndarray
        Wavelength array and half-widths for each bin (populated by
        :meth:`plot_parameter_spectrum`).
    pars_med, pars_up, pars_lo : ndarray
        Median and upper/lower credible interval bounds on fitted parameters
        as a function of wavelength (populated by :meth:`plot_parameter_spectrum`).
    qua_white : tuple
        Quantiles (median, upper, lower) of the white-light parameter
        (populated by :meth:`plot_parameter_spectrum`).

    Main methods
    ------------
    generating_lightcurves(ch_nos=None)
        Extract and optionally bin spectral light curves into channels.
    bin_lightcurve(unbinned_lc, unbinned_lc_err)
        Inverse-variance weighted binning of light curves along wavelength direction.
    white_light_lc()
        Compute the white-light (all-wavelength) light curve.
    analyse_lc_parallel(nthreads, ch_nos=None, juliet_fit_kwargs={})
        End-to-end parallel analysis: bin, organize, and fit all channels.
    plot_parameter_spectrum(parameter, bins=None, post_size=10000, ...)
        Plot fitted parameter values as a function of wavelength.
    plot2Ddata(cmap='plasma')
        2D heatmap of spectral light curves vs time.
    plot2D_data_model_resids(cmap='plasma', detrend=False, ...)
        Side-by-side 2D heatmaps of data, model, and residuals.
    joint_fake_allan_deviation(binmax=10, method='pipe', timeunit=None)
        Combined noise vs binning plot across all channels.

    Notes
    -----
    - All fitting is performed using the ``juliet`` package (Espinoza et al. 2019).
    - Parallel processing uses Python's ``multiprocessing.Pool``.
    - Results are saved as pickled dictionaries and ASCII text files to ``pout``.
    - Methods check for existing output files and skip re-computation if files
      are already present (useful for resuming interrupted analyses).

    Examples
    --------
    Create and analyze a spectroscopic dataset:

    >>> import numpy as np
    >>> times = np.linspace(2459000, 2459001, 100)
    >>> wavelengths = np.linspace(1.1, 1.7, 50)
    >>> lc = np.random.normal(1.0, 0.01, (100, 50))
    >>> lc_errs = np.full_like(lc, 0.005)

    >>> def get_priors(ch_name):
    ...     return juliet.utils.priors()

    >>> spec = SpectroscopicLC(times, lc, lc_errs, wavelengths,
    ...                        priors=get_priors, pout='./results')
    >>> spec.analyse_lc_parallel(nthreads=4, ch_nos=10)
    >>> fig, ax = spec.plot_parameter_spectrum('p_p1', bins=10, plot_white=True)
    """
    def __init__(self, times, lc, lc_errs, wavelengths, gp_pars=None, lin_pars=None, priors=None, pout=os.getcwd()):
        # Dimensions of lc should be (nframes, ncols)
        self.times = times
        self.lc = lc
        self.lc_errs = lc_errs
        self.wavelengths = wavelengths
        self.gp_pars = gp_pars
        self.lin_pars = lin_pars
        self.priors = priors
        self.pout = pout

    def col_spec(self, ch_nos):
        """Given a number of total number of channels, this function gives an array containing
        start and end column for each channel"""
        if ch_nos != 1:
            col_in_1_ch = round(self.lc.shape[1]/ch_nos)
            self.col_st = np.arange(0, self.lc.shape[1]-col_in_1_ch, col_in_1_ch, dtype=int)
            self.col_end = np.arange(0+col_in_1_ch, self.lc.shape[1], col_in_1_ch, dtype=int)
        else:
            self.col_st, self.col_end = np.array([0]), np.array([self.lc.shape[1]])
        
        if self.col_end[-1] != self.lc.shape[1]:
            self.col_st = np.hstack((self.col_st, self.col_end[-1]))
            self.col_end = np.hstack((self.col_end, self.lc.shape[1]))

    def generating_lightcurves(self, ch_nos=None):
        """Extract and optionally bin spectral light curves into channels.

        This method either bins the original light curve into ``ch_nos`` channels
        (using inverse-variance weighting) or uses the original wavelength resolution.
        Results are saved as a pickled dictionary and stored in
        ``self.spectral_lcs_dict``.

        If a pickle file for the requested binning already exists, it is loaded
        instead of recomputing.

        Parameters
        ----------
        ch_nos : int or None, optional
            Number of channels to bin the light curve into. If ``None``, use
            native wavelength resolution (no binning). Default is ``None``.

        Returns
        -------
        None
            Results are stored in ``self.spectral_lcs_dict`` with keys:

            - 'lc' : ndarray
                Binned light curves (shape: N_frames x N_channels).
            - 'err' : ndarray
                Binned flux errors (same shape as 'lc').
            - 'wave' : ndarray
                Central wavelength of each bin.
            - 'wave_bin' : ndarray
                Half-width (or full width) of each wavelength bin.

        Notes
        -----
        The output is pickled to
        ``pout/spectroscopic_lc_ch_<ch_nos>.pkl`` for future loading.
        """
        if ch_nos == None:
            fname = self.pout + '/spectroscopic_lc_ch_' + str(self.lc.shape[1]) + '.pkl'
        else:
            fname = self.pout + '/spectroscopic_lc_ch_' + str(ch_nos) + '.pkl'

        if not Path(fname).exists():
            # Columns
            if ch_nos != None:
                self.col_spec(ch_nos)
                # Creating spectral lc array
                spec_lc, spec_err_lc = np.zeros( ( self.lc.shape[0], len(self.col_st) ) ), np.zeros( ( self.lc.shape[0], len(self.col_st) ) )
                wavs, wav_bin_size = np.zeros( len(self.col_st) ), np.zeros( len(self.col_st) )
                for i in range(len(self.col_st)):
                    spec_lc[:,i], spec_err_lc[:,i] = self.bin_lightcurve( self.lc[:, self.col_st[i]:self.col_end[i]], \
                                                                          self.lc_errs[:, self.col_st[i]:self.col_end[i]] )
                    if self.col_end[i] != self.lc.shape[1]:
                        wavs[i] = ( self.wavelengths[self.col_st[i]] + self.wavelengths[self.col_end[i]] )/2
                        wav_bin_size[i] = np.abs( self.wavelengths[self.col_st[i]] - self.wavelengths[self.col_end[i]] )
                    else:
                        wavs[i] = ( self.wavelengths[self.col_st[i]] + self.wavelengths[self.col_end[i]-1] )/2
                        wav_bin_size[i] = np.abs( self.wavelengths[self.col_st[i]] - self.wavelengths[self.col_end[i]-1] )
            else:
                spec_lc, spec_err_lc = self.lc, self.lc_errs
                wavs, wav_bin_size = self.wavelengths, np.append(np.diff(self.wavelengths), np.diff(self.wavelengths)[-1])

            # Save the light curves
            self.spectral_lcs_dict = {}
            self.spectral_lcs_dict['lc'], self.spectral_lcs_dict['err'] = spec_lc, spec_err_lc
            self.spectral_lcs_dict['wave'], self.spectral_lcs_dict['wave_bin'] = wavs, wav_bin_size

            if ch_nos == None:
                pickle.dump(self.spectral_lcs_dict, open(self.pout + '/spectroscopic_lc_ch_' + str(self.lc.shape[1]) + '.pkl','wb'))
            else:
                pickle.dump(self.spectral_lcs_dict, open(self.pout + '/spectroscopic_lc_ch_' + str(ch_nos) + '.pkl','wb'))
        else:
            print('>>>> --- The spectroscopic lightcurves already exists...')
            print('         Loading them...')
            self.spectral_lcs_dict = pickle.load( open(fname, 'rb') )

    
    def bin_lightcurve(self, unbinned_lc, unbinned_lc_err):
        """Perform inverse-variance weighted binning of a light curve.

        Each output bin is the weighted average of input values, where weights
        are the inverse squares of the input uncertainties. Output uncertainties
        are computed by propagating input uncertainties through the weighted
        average formula.

        Parameters
        ----------
        unbinned_lc : ndarray
            2D array of unbinned flux values (shape: N_frames x N_wavelengths).
        unbinned_lc_err : ndarray
            2D array of flux uncertainties (same shape as ``unbinned_lc``).

        Returns
        -------
        bin_lc : ndarray
            1D array of binned flux values (shape: N_frames,).
        bin_lc_err : ndarray
            1D array of binned flux uncertainties (same shape as ``bin_lc``).
        """
        # shape of both arrays are: (nframes, ncols)
        weights = 1 / unbinned_lc_err**2
        bin_lc = np.nansum(unbinned_lc * weights, axis=1) / np.nansum(weights, axis=1)
        bin_lc_err = ( 1 / np.nansum(weights, axis=1) ) * np.sqrt( np.nansum( (weights**2) * (unbinned_lc_err**2), axis=1 ) )

        return bin_lc, bin_lc_err
    
    def white_light_lc(self):
        """Compute the white-light (all-wavelength) light curve.

        Bins the entire 2D light curve (all wavelengths combined) into a single
        1D time-series using inverse-variance weighting.

        Returns
        -------
        white_lc : ndarray
            1D array of white-light flux values.
        white_lc_err : ndarray
            1D array of white-light flux uncertainties.
        """
        return self.bin_lightcurve(unbinned_lc=self.lc, unbinned_lc_err=self.lc_errs)
    
    # MultiProcessing helper function (does the actual fitting)
    def fit_lc(self, lightcurves, ch_name):
        """Inteernal helper function for multiprocessing.
        This function fits lightcurve for one spectroscopic channel"""
        print('---------------------------------')
        print('Working on Channel: ' + ch_name)
        print('')
        # Output folder
        pout = self.pout + '/' + ch_name
        f15 = Path(pout + '/_dynesty_DNS_posteriors.pkl')
        f16 = Path(pout + '/model_resids.dat')
        f17 = Path(pout + '/posteriors.dat')
        if f15.exists() and f16.exists() and f17.exists():
            print('>>>> --- The result files already exists...')
            print('         Continuing to the next channel...')
            res = np.zeros(10)
        else:
            # Extracting the data
            tim9, fl9, fle9 = lightcurves['times'], lightcurves['lc'], lightcurves['err']
            
            # Removing Nan values
            tim7, fl7, fle7 = tim9[~np.isnan(fl9)], fl9[~np.isnan(fl9)], fle9[~np.isnan(fl9)]

            # Outlier removal
            #msk2 = utl.outlier_removal(tim7, fl7, fle7, clip=10, msk1=False)
            #tim7, fl7, fle7 = tim7[msk2], fl7[msk2], fle7[msk2]

            # Normalizing the lightcurve
            tim7, fl7, fle7 = tim7, fl7/np.median(fl7), fle7/np.median(fl7)

            # Making data such that juliet can understand
            tim, fl, fle = {}, {}, {}
            tim[ch_name], fl[ch_name], fle[ch_name] = tim7, fl7, fle7


            if type( lightcurves['lins'] ) == type( None ):
                lin_pars = None
            else:
                lin_pars = {}
                lin_pars[ch_name] = lightcurves['lins']

            if type( lightcurves['gp'] ) == type( None ):
                gp_pars = None
            else:
                gp_pars = {}
                gp_pars[ch_name] = lightcurves['gp']

            # Fitting
            dataset = juliet.load(priors=self.priors(ch_name), t_lc=tim, y_lc=fl, yerr_lc=fle, linear_regressors_lc=lin_pars, GP_regressors_lc=gp_pars, out_folder=pout)
            res = dataset.fit(sampler = 'dynamic_dynesty', **self.juliet_fit_kwargs)#, nthreads=8)

            # Some plots
            model = res.lc.evaluate(ch_name)
            residuals = fl[ch_name]-model

            data12 = np.vstack((model, residuals))
            np.savetxt(pout + '/model_resids.dat', np.transpose(data12))

            print('>>>> --- Done!!')
        return res
    
    # Function that does the multiprocessing
    def multi_fit_lcs(self, lightcurves, nthreads=4):
        input_data = [(lightcurves[lc], lc) for lc in lightcurves]
            
        with multiprocessing.Pool(nthreads) as p:
            result_list = p.starmap(self.fit_lc, input_data)
                    
        return np.array(result_list)
    
    def analyse_lc_parallel(self, nthreads, ch_nos=None, juliet_fit_kwargs={}):
        """End-to-end parallel analysis: bin, organize, and fit lightcurves from all channels.

        This high-level wrapper orchestrates the full analysis pipeline:
        1. Bins the spectral light curve into channels via :meth:`generating_lightcurves`
        2. Organizes the data into a format suitable for fitting
        3. Calls :meth:`multi_fit_lcs` to fit all channels in parallel

        Parameters
        ----------
        nthreads : int
            Number of parallel threads to use for fitting.
        ch_nos : int or None, optional
            Number of wavelength channels to bin into. If ``None``, use native
            wavelength resolution. Default is ``None``.
        juliet_fit_kwargs : dict, optional
            Keyword arguments to forward to ``juliet.dataset.fit(...)``,
            such as ``sampler='dynamic_dynesty'``, ``nthreads=8``, etc.
            Default is an empty dictionary.

        Returns
        -------
        None
            Results are written to ``pout`` and its subdirectories.

        Notes
        -----
        Requires that ``self.priors`` is a callable that returns a priors
        for each channel name.
        """
        # Setting juliet_fit_kwargs
        self.juliet_fit_kwargs = juliet_fit_kwargs

        # Generating the data
        ## Bin the data, if necessary
        self.generating_lightcurves(ch_nos=ch_nos)

        # Storing all lightcurves in a big dictionary
        all_lightcurve_data = {}
        for i in range( self.spectral_lcs_dict['lc'].shape[1] ):
            # Storing all lightcurves in a big dictionary
            all_lightcurve_data['CH' + str(i)] = {}
            
            ## And now storing the actual lightcurve data
            all_lightcurve_data['CH' + str(i)]['times'] = self.times
            all_lightcurve_data['CH' + str(i)]['lc'] = self.spectral_lcs_dict['lc'][:,i]
            all_lightcurve_data['CH' + str(i)]['err'] = self.spectral_lcs_dict['err'][:,i]

            ## Storing linear and GP parameters, if provided (None, otherwise)
            all_lightcurve_data['CH' + str(i)]['lins'] = self.lin_pars
            all_lightcurve_data['CH' + str(i)]['gp'] = self.gp_pars
        
        # Now, analysing them
        _ = self.multi_fit_lcs(all_lightcurve_data, nthreads=nthreads)

    def plot_parameter_spectrum(self, parameter, bins=None, post_size=10000, bin_method='mean', ppm=False, plot_white=False):
        """Plot a fitted parameter as a function of wavelength (a spectrum).

        Loads posterior samples for a parameter from all fitted channels,
        optionally re-bins them in the wavelength direction, and plots
        median with credible intervals as a function of wavelength.

        Parameters
        ----------
        parameter : str
            Name of the parameter to plot (e.g., 'p_p1', 'fp_p1', 'C1_p1').
            Should match the parameter naming convention used by juliet.
        bins : int or None, optional
            Number of wavelength bins to use for the output spectrum. If ``None``
            or equal to the native number of channels, no re-binning is performed.
            Default is ``None``.
        post_size : int, optional
            Number of posterior samples to draw from each channel. Default is 10000.
        bin_method : {'mean', 'median', 'weighted_average'}, optional
            Method for combining posteriors across wavelength bins.
            Default is ``'mean'``.
        ppm : bool, optional
            If ``True``, scale parameter values to ppm (multiply by 1e6).
            Useful for small parameters like occultation depths. Default is ``False``.
        plot_white : bool, optional
            If ``True``, overplot the white-light (all-wavelength) parameter
            value as a horizontal band with credible interval. Default is ``False``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        axs : matplotlib.axes.Axes
            The plot axes.

        Notes
        -----
        Posterior samples are loaded from files named like
        ``pout/CH<i>/*_posteriors.pkl``.
        """
        ## First, let's put all posteriros in a big 2D array
        all_posteriors = []
        for i in range( self.spectral_lcs_dict['lc'].shape[1] ):
            fname = glob( self.pout + '/CH' + str(i) + '/*_posteriors.pkl' )[0]
            post = pickle.load( open(fname, 'rb') )
            all_posteriors.append( np.random.choice( post['posterior_samples'][parameter + '_CH' + str(i)], size=post_size ) )
        posteriors = np.transpose( np.vstack( all_posteriors ) )

        ## Calculating the white light parameter value
        if bin_method == 'mean':
            white_post = np.mean( posteriors, axis=1 )
        elif bin_method == 'median':
            white_post = np.median( posteriors, axis=1 )
        elif bin_method == 'weighted_average':
            white_post = np.average( posteriors, axis=1, weights=1/np.std(posteriors, axis=0)**2 )
        else:
            raise ValueError('>>> --- bin_method not recognized. Please use mean, median or weighted_average.')
        
        ### Quantiles
        self.qua_white = juliet.utils.get_quantiles( white_post )

        if ( bins == self.spectral_lcs_dict['lc'].shape[1] ) or ( bins == None ):
            ## Wavelengths
            self.wav, self.wav_bin = self.spectral_lcs_dict['wave'], self.spectral_lcs_dict['wave_bin']/2
            
            ## If bin size is the same as the column size, then we don't perform any binning
            self.pars_med, self.pars_up, self.pars_lo = np.zeros( self.spectral_lcs_dict['lc'].shape[1] ), np.zeros( self.spectral_lcs_dict['lc'].shape[1] ), np.zeros( self.spectral_lcs_dict['lc'].shape[1] )
            for i in range( self.spectral_lcs_dict['lc'].shape[1] ):
                ## And, appending parameter values
                qua = juliet.utils.get_quantiles( posteriors[:,i] )
                self.pars_med[i], self.pars_up[i], self.pars_lo[i] = qua[0], qua[1] - qua[0], qua[0] - qua[2]

        else:
            ## Wavelengths
            native_wav, native_wav_bin = self.spectral_lcs_dict['wave'], self.spectral_lcs_dict['wave_bin']

            # Column location, for low resolution eclipse spectrum
            col_in_1_ch = round( len(native_wav) / bins )
            col_st = np.arange(0, len(native_wav)-col_in_1_ch, col_in_1_ch, dtype=int)
            col_end = np.arange(0+col_in_1_ch, len(native_wav), col_in_1_ch, dtype=int)
            if col_end[-1] != len(native_wav):
                col_st = np.hstack(( col_st, col_end[-1] ))
                col_end = np.hstack(( col_end, len(native_wav) ))

            # For binning the spectrum
            self.pars_med, self.pars_up, self.pars_lo = np.zeros( len(col_st) ), np.zeros( len(col_st) ), np.zeros( len(col_st) )
            self.wav, self.wav_bin = np.zeros( len(col_st) ), np.zeros( len(col_end) )
            # And performing actual binning
            for i in range(len(col_st)):
                ## Calculating the median of the posteriors in the bin
                if bin_method == 'mean':
                    fp12 = np.mean( posteriors[:,col_st[i]:col_end[i]], axis=1 )
                elif bin_method == 'median':
                    fp12 = np.median( posteriors[:,col_st[i]:col_end[i]], axis=1 )
                elif bin_method == 'weighted_average':
                    fp12 = np.average( posteriors[:,col_st[i]:col_end[i]], axis=1, weights=1/np.std( posteriors[:,col_st[i]:col_end[i]], axis=0 )**2 )
                else:
                    raise ValueError('>>> --- bin_method not recognized. Please use mean, median or weighted_average.')
                
                qua_fp12 = juliet.utils.get_quantiles(fp12)
                self.pars_med[i], self.pars_up[i], self.pars_lo[i] = qua_fp12[0], qua_fp12[1]-qua_fp12[0], qua_fp12[0]-qua_fp12[2]
                # For wavelength bins
                if col_end[i] != len(native_wav):
                    self.wav[i] = ( native_wav[col_st[i]] + native_wav[col_end[i]] ) / 2
                    self.wav_bin[i] = np.abs( native_wav[col_st[i]] - native_wav[col_end[i]] ) / 2
                else:
                    self.wav[i] = ( native_wav[col_st[i]] + native_wav[col_end[i]-1] ) / 2
                    self.wav_bin[i] = np.abs( native_wav[col_st[i]] - native_wav[col_end[i]-1] ) / 2
        

        # ---------------------------------------
        #.    Now, plotting the spectrum
        # ---------------------------------------
        if ppm:
            self.pars_med *= 1e6
            self.pars_up *= 1e6
            self.pars_lo *= 1e6
            self.qua_white = ( self.qua_white[0]*1e6, self.qua_white[1]*1e6, self.qua_white[2]*1e6 )

        fig, axs = plt.subplots()
        axs.errorbar(self.wav, self.pars_med, yerr=[self.pars_lo, self.pars_up], xerr=self.wav_bin, fmt='o', c='orangered', mfc='white', elinewidth=1.5, capthick=1.5, capsize=3.)

        if plot_white:
            wav12 = np.linspace( min(self.wav)-max(self.wav_bin)*2, max(self.wav)+max(self.wav_bin)*2, 1000 )
            axs.axhline(self.qua_white[0], color='cornflowerblue', ls='-', zorder=10)
            axs.fill_between(wav12, self.qua_white[2], self.qua_white[1], color='cornflowerblue', alpha=0.3)

        axs.set_xlim( min(self.wav)-max(self.wav_bin)*1.5, max(self.wav)+max(self.wav_bin)*1.5 )
        
        axs.set_xlabel('Wavelength')
        axs.set_ylabel(parameter)
        axs.set_title(f'Spectral {parameter} spectrum')
        
        return fig, axs
    
    def plot2Ddata(self, cmap='plasma'):
        """Create a 2D heatmap of spectral light curves (flux vs time) vs wavelength.

        Parameters
        ----------
        cmap : str, optional
            Matplotlib colormap name. Default is ``'plasma'``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        ax1 : matplotlib.axes.Axes
            The plot axes.
        im1 : matplotlib.image.AxesImage
            The image object.
        cbar : matplotlib.colorbar.Colorbar
            The colorbar object.
        """
        # Adjusting the times to time since beginning in hours
        tim = ( self.times - self.times[0] ) * 24

        ## Preparing the data
        norm_lcs = np.copy( self.lc )

        fig, ax1 = plt.subplots( figsize=(7, 7) )
        # Data
        im1 = ax1.imshow(norm_lcs.T, interpolation='none', cmap=cmap, aspect = 'auto', extent=[tim[0], tim[-1], self.wavelengths[-1], self.wavelengths[0]])
        im1.set_clim( [ np.nanmedian(norm_lcs) - 5*mad_std(norm_lcs), np.nanmedian(norm_lcs) + 5*mad_std(norm_lcs) ] )
        plt.ylabel(r'Wavelength [$\mu$m]')
        plt.xlabel('Time from beginning [h]')

        # Colorbar:
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = fig.colorbar(im1, shrink = 0.08, cax=cax)
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel('Relative flux', rotation=270)#, fontsize = 15)

        return fig, ax1, im1, cbar
    
    def plot2D_data_model_resids(self, cmap='plasma', detrend=False, binwidth=None, ppm=False, **kwargs):
        """Create side-by-side 2D heatmaps of data, model, and residuals.

        Loads the fitted models and residuals from all channels and creates
        three synchronized heatmaps showing the full spectro-temporal evolution
        of the data, fit quality, and systematics.

        Parameters
        ----------
        cmap : str, optional
            Matplotlib colormap name. Default is ``'plasma'``.
        detrend : bool, optional
            If ``True``, load and plot detrended data (subtracting GP and
            linear systematics). Requires additional ``juliet`` processing
            and may be slow. If ``False``, plot raw data. Default is ``False``.
        binwidth : float or None, optional
            Time binwidth (in days) for additional temporal binning of the
            data/model/residuals. If ``None``, no binning is performed.
            Default is ``None``.
        ppm : bool, optional
            If ``True``, express data and model in ppm relative to unity.
            Residuals are always in ppm. Default is ``False``.
        **kwargs : dict
            Forwarded to the ``julietPlots`` constructor when ``detrend=True``
            (e.g., ``N=1000`` for the number of posterior samples).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure with three subplots.
        axs : ndarray
            Array of three Axes objects (data, model, residuals).
        im_list : list
            List of three AxesImage objects (useful for colorbars).
        """
        # binwidth = None means that you should not perform any binning (in time)
        # We don't perform any binning in wavelength direction -- please do that before fitting the light curves
        # Detrend = True detrends the data and plot detrended model instead.
        data = np.zeros( ( len(self.spectral_lcs_dict['wave']), len(self.times) ) )
        models = np.zeros( data.shape )
        resids = np.zeros( data.shape )

        ## Loading the wavelengths
        waves = self.spectral_lcs_dict['wave']

        min_mod, max_mod = 1e10, -1e10
        for c in range( self.spectral_lcs_dict['lc'].shape[1] ):
            if not detrend:
                ## First, we need data
                tim, fl = np.loadtxt( self.pout + '/CH' + str(c) + '/lc.dat', usecols=(0,1), unpack=True )
                ## And the model and residuals
                model, resid = np.loadtxt( self.pout + '/CH' + str(c) + '/model_resids.dat', usecols=(0,1), unpack=True )
            else:
                ## Well, if detrend = True, then we need to detrend the data (that can take a while)
                data = julietPlots(input_folder=self.pout + '/CH' + str(c), **kwargs)
                ## We now need to detrend the data and the calculate the detrended model
                data.detrend_data(phmin=0.8, instruments='CH' + str(c))
                data.detrend_model(phmin=0.8, instruments='CH' + str(c), highres=False)

                # Time data
                ## Array that sort any array according to time
                idx_time = np.argsort( data.dataset.times_lc['CH' + str(c)] )
                tim = data.dataset.times_lc['CH' + str(c)][idx_time]
                resid = ( data.dataset.data_lc['CH' + str(c)] - data.models_all_ins['CH' + str(c)][1] )[idx_time]

                ## Detrended dataset
                fl = data.detrended_data['CH' + str(c)][idx_time]

                ## Detrended model
                model = data.planet_only_models['CH' + str(c)][1][idx_time]

            if ppm:
                fl, model = ( fl - 1. ) * 1e6, ( model - 1. ) * 1e6
            ## Residuals are always in ppm
            resid = resid * 1e6

            if binwidth is not None:
                _, fl, _, _ = lcbin(time=tim, flux=fl, binwidth=binwidth)
                _, model, _, _ = lcbin(time=tim, flux=model, binwidth=binwidth)
                _, resid, _, _ = lcbin(time=tim, flux=resid, binwidth=binwidth)

            # And putting them in a big array
            data[c, :], models[c, :], resids[c, :] = fl, model, resid
            
            ## Calculating the minimum and maximum values of the model
            if np.nanmin(model) < min_mod:
                min_mod = np.nanmin(model)
            if np.nanmax(model) > max_mod:
                max_mod = np.nanmax(model)

        # And, finally, the plotting
        # Convering time to hours
        tim = ( tim - tim[0] ) * 24
        
        # Colorbars
        cmin1, cmax1 = min_mod-0.005*min_mod, max_mod+0.005*max_mod
        norm1 = matplotlib.colors.Normalize(vmin=cmin1, vmax=cmax1, clip=True)
        mapper1 = cm.ScalarMappable(norm=norm1, cmap=cmap)
        mapper1.set_array([])

        cmin2, cmax2 = np.nanmedian(resids)-3*mad_std(resids), np.nanmedian(resids)+3*mad_std(resids)
        norm2 = matplotlib.colors.Normalize(vmin=cmin2, vmax=cmax2, clip=True)
        mapper2 = cm.ScalarMappable(norm=norm2, cmap=cmap)
        mapper2.set_array([])

        # And plotting it
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5), sharex=True, sharey=True)

        ## First box: plotting the data 
        im1 = axs[0].imshow(data, aspect='auto', cmap=cmap, interpolation='none', extent=[tim[0], tim[-1], waves[-1], waves[0]])
        im1.set_clim([cmin1, cmax1])
        
        ## Second box: plotting the model
        im2 = axs[1].imshow(models, aspect='auto', cmap=cmap, interpolation='none', extent=[tim[0], tim[-1], waves[-1], waves[0]])
        im2.set_clim([cmin1, cmax1])

        ## First colorbar
        cbar1 = plt.colorbar(mapper1, ax=np.array([axs[0],axs[1]]))#, location='top')
        if ppm:
            cbar1.set_label('Relative flux [ppm]', rotation=270, labelpad=20)
        else:
            cbar1.set_label('Relative flux', rotation=270, labelpad=20)
        cbar1.set_ticks(ticks=np.linspace(cmin1, cmax1, 7))
        cbar1.set_ticklabels(ticklabels=[ "{:.3f}".format( i ) for i in np.linspace(cmin1, cmax1, 7) ] )

        im3 = axs[2].imshow(resids, aspect='auto', cmap=cmap, interpolation='none', extent=[tim[0], tim[-1], waves[-1], waves[0]])
        im3.set_clim([cmin2, cmax2])
        cbar2 = plt.colorbar(mapper2, ax=axs[2])#, location='top')
        cbar2.set_label('Residuals [ppm]', rotation=270, labelpad=20)

        cbar2.set_ticks( ticks=np.linspace(cmin2, cmax2, 7) )
        cbar2.set_ticklabels(ticklabels=[ "{:d}".format( int(i) ) for i in np.linspace(cmin2, cmax2, 7) ] )

        fig.supxlabel('Time since beginning [hr]', fontsize=15)
        axs[0].set_ylabel(r'Wavelength [$\mu$m]')

        if detrend:
            axs[0].set_title('Detrended data')
        else:
            axs[0].set_title('Data')
        axs[1].set_title('Model')
        axs[2].set_title('Residuals')

        return fig, axs, [im1, im2, im3]
    
    def joint_fake_allan_deviation(self, binmax=10, method='pipe', timeunit=None):
        """Plot combined noise-vs-binning curves across all spectroscopic channels.

        Computes the "Allan deviation" (actually the MAD-based noise as a
        function of binning) for each channel's residuals and overlays all
        curves on a single plot. Also shows the white-noise expectation curve.

        Parameters
        ----------
        binmax : int, optional
            Maximum number of bins to test. Default is 10.
        method : {'pipe', 'std', 'rms', 'astropy'}, optional
            Noise estimator to use (passed to :func:`utils.fake_allan_deviation`).
            Default is ``'pipe'`` (MAD-based).
        timeunit : {'d', 'hr', 'min'} or None, optional
            Time unit for the secondary x-axis. If ``None``, chosen automatically
            based on data span. Default is ``None``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        axs : matplotlib.axes.Axes
            The plot axes (with primary x-axis for bin size and secondary
            x-axis for time period).
        """
        # Creating a figure outside of the loop
        fig, axs = plt.subplots()
        
        ## Looping over all channels
        for c in tqdm(range( self.spectral_lcs_dict['lc'].shape[1] )):
            times = np.loadtxt( self.pout + '/CH' + str(c) + '/lc.dat', usecols=0, unpack=True )
            resids = np.loadtxt( self.pout + '/CH' + str(c) + '/model_resids.dat', usecols=1, unpack=True )
            _, _, binsize, noise, white_noise_expec = fake_allan_deviation(times=times, residuals=resids, binmax=binmax, method=method, timeunit=timeunit, plot=False)

            ## And plotting it over the main plot
            ## First plotting the computed noise
            if c == 0:
                label1, label2 = 'Computed noise', 'White-noise expectation'
            else:
                label1, label2 = None, None
            axs.plot(binsize, noise, color='dodgerblue', lw=0.7, label=label1, zorder=10)
            ## Now plotting the white-noise expectation
            axs.plot(binsize, white_noise_expec, color='orangered', lw=1., label=label2, zorder=20)

        # Converting binsize to time units
        time_binsize = binsize * np.nanmedian( np.diff(times) )

        # First, let's estimate the time units we need from times array:
        if timeunit is None:
            if np.ptp(time_binsize) >= 1.:
                ## If times duration is greater than 2 hours, we use days
                time_unit_multiplication_factor = 1.     # Units are already in days
                time_unit_label = 'Time period [d]'
            elif ( np.ptp(time_binsize) > 5/24 ) and ( np.ptp(time_binsize) < 1. ):
                ## If times duration is greater than 2 hours, but less than 2 days, we use hours
                time_unit_multiplication_factor = 24.    # Converting days to hours
                time_unit_label = 'Time period [hr]'
            else:
                ## If times duration is less than 2 hours, we use minutes
                time_unit_multiplication_factor = 24 * 60
                time_unit_label = 'Time period [min]'
        else:
            if timeunit == 'd':
                time_unit_multiplication_factor = 1.
                time_unit_label = 'Time period [d]'
            elif timeunit == 'hr':
                time_unit_multiplication_factor = 24.
                time_unit_label = 'Time period [hr]'
            elif timeunit == 'min':
                time_unit_multiplication_factor = 24 * 60
                time_unit_label = 'Time period [min]'
            else:
                raise ValueError("timeunit should be one of 'd', 'hr', or 'min'.")

        # The following two functions will convert binsize to time (in minutes) and vice versa
        def bin2time(binsize):
            return binsize * np.nanmedian(np.diff(times)) * time_unit_multiplication_factor

        def time2bin(bintime):
            return bintime / ( np.nanmedian(np.diff(times)) * time_unit_multiplication_factor )

        ## Adding secondary x-axis for time
        secax = axs.secondary_xaxis('top', functions=(bin2time, time2bin))
        secax.set_xlabel(time_unit_label, labelpad=10)

        axs.set_xlim([ np.min(binsize), np.max(binsize) ])

        axs.set_xscale('log')
        axs.set_yscale('log')

        axs.set_xlabel('Bin size [Number of points]')
        axs.set_ylabel('Noise estimate [ppm]')

        axs.legend()

        return fig, axs