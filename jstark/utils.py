import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import mad_std
from scipy.optimize import minimize
from scipy.integrate import simpson
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from astropy.timeseries import LombScargle
import astropy.constants as con
from tqdm import tqdm
import astropy.units as u
from pathlib import Path
import multiprocessing
import corner
import os

def t14(per, ar, rprs, b, ecc=0, omega=90, transit=True):
    """
    To compute transit/eclipse duration from Period, a/R*, Rp/R*, b, eccentricity, and omega

    Parameters:
    -----------
    per : float, or numpy.ndarray
        Orbital period (in days) of the planet
    aR : float, or numpy.ndarray
        Scaled semi-major axis, a/R*
    rprs : float, or numpy.ndarray
        Planet-to-star radius ratio, Rp/R*
    bb : float, or numpy.ndarray
        Impact parameter
    ecc: float, or numpy.ndarray
        Eccentricity of the orbit
    omega: float, or numpy.ndarray
        Argument of periastron (deg)
    transit: bool
        Whether to compute transit or occultation duration
    -----------
    return
    -----------
    t14 : float, or numpy.ndarray
        Transit/occultation duration, in days
    """

    inc_rad = np.radians( b_to_inc(b=b, ar=ar, ecc=ecc, omega=omega) )
    omega = np.radians(omega)

    # Computing inclination from impact parameter
    term1 = np.sqrt( (1 + rprs)**2 -  b**2  )
    t14 = ( per / np.pi ) * np.arcsin( term1 / ar / np.sin( inc_rad ) )

    # Factor accounting for eccentric orbit
    if transit:
        fac = np.sqrt( 1 - ecc**2 ) / ( 1 + (ecc*np.sin(omega)) )
    else:
        fac = np.sqrt( 1 - ecc**2 ) / ( 1 - (ecc*np.sin(omega)) )
    
    return t14 * fac

def b_to_inc(b, ar, ecc=0., omega=90., transit=True):
    """Computing inclination from impact parameter"""
    # First converting all angles to radians
    omega = np.radians(omega)
    if transit:
        cosi = b * ( 1 + ( ecc * np.sin(omega) ) ) / ar / (1 - ecc**2)
    else:
        cosi = b * ( 1 - ( ecc * np.sin(omega) ) ) / ar / (1 - ecc**2)
    return np.rad2deg( np.arccos( cosi ) )

def inc_to_b(inc, ar, ecc=0., omega=90., transit=True):
    """To compute impact parameter from inclination"""
    # First convert all angles to radians
    inc, omega = np.radians(inc), np.radians(omega)
    if transit:
        bb = ar * np.cos(inc) * (1 - ecc**2) / ( 1 + ( ecc * np.sin(omega) ) )
    else:
        bb = ar * np.cos(inc) * (1 - ecc**2) / ( 1 - ( ecc * np.sin(omega) ) )
    return bb

def rho_to_ar(rho, per):
    """To compute a/R* from rho"""
    G = con.G.value
    a = ((rho * G * ((per * 24. * 3600.)**2)) / (3. * np.pi))**(1. / 3.)
    return a

def pipe_mad(x, axis=0):
    return np.nanmedian(np.abs(np.diff(x, axis=axis)), axis=axis)

def lcbin(time, flux, binwidth=0.06859, nmin=4, time0=None,
        robust=False, tmid=False):
    """
    ------ A function from pycheops -------
    Calculate average flux and error in time bins of equal width.
    The default bin width is equivalent to one CHEOPS orbit in units of days.
    To avoid binning data on either side of the gaps in the light curve due to
    the CHEOPS orbit, the algorithm searches for the largest gap in the data
    shorter than binwidth and places the bin edges so that they fall at the
    centre of this gap. This behaviour can be avoided by setting a value for
    the parameter time0.
    The time values for the output bins can be either the average time value
    of the input points or, if tmid is True, the centre of the time bin.
    If robust is True, the output bin values are the median of the flux values
    of the bin and the standard error is estimated from their mean absolute
    deviation. Otherwise, the mean and standard deviation are used.
    The output values are as follows.
    * t_bin - average time of binned data points or centre of time bin.
    * f_bin - mean or median of the input flux values.
    * e_bin - standard error of flux points in the bin.
    * n_bin - number of flux points in the bin.
    :param time: time
    :param flux: flux (or other quantity to be time-binned)
    :param binwidth:  bin width in the same units as time
    :param nmin: minimum number of points for output bins
    :param time0: time value at the lower edge of one bin
    :param robust: use median and robust estimate of standard deviation
    :param tmid: return centre of time bins instead of mean time value
    :returns: t_bin, f_bin, e_bin, n_bin
    """
    if time0 is None:
        tgap = (time[1:]+time[:-1])/2
        gap = time[1:]-time[:-1]
        j = gap < binwidth
        gap = gap[j]
        tgap = tgap[j]
        time0 = tgap[np.argmax(gap)]
        time0 = time0 - binwidth*np.ceil((time0-min(time))/binwidth)

    n = int(1+np.ceil(np.ptp(time)/binwidth))
    r = (time0,time0+n*binwidth)
    n_in_bin,bin_edges = np.histogram(time,bins=n,range=r)
    bin_indices = np.digitize(time,bin_edges)

    t_bin = np.zeros(n)
    f_bin = np.zeros(n)
    e_bin = np.zeros(n)
    n_bin = np.zeros(n, dtype=int)

    for i,n in enumerate(n_in_bin):
        if n >= nmin:
            j = bin_indices == i+1
            n_bin[i] = n
            if tmid:
                t_bin[i] = (bin_edges[i]+bin_edges[i+1])/2
            else:
                t_bin[i] = np.nanmean(time[j])
            if robust:
                f_bin[i] = np.nanmedian(flux[j])
                e_bin[i] = 1.25*np.nanmean(abs(flux[j] - f_bin[i]))/np.sqrt(n)
            else:
                f_bin[i] = np.nanmean(flux[j])
                e_bin[i] = np.std(flux[j])/np.sqrt(n-1)

    j = (n_bin >= nmin)
    return t_bin[j], f_bin[j], e_bin[j], n_bin[j]

def clip_outliers(flux, clip=5, width=11, verbose=True):
    """
    Another function from pycheops. Courtesy of P. Maxted.
    Remove outliers from the light curve.

    Data more than clip*mad from a smoothed version of the light curve are
    removed where mad is the mean absolute deviation from the
    median-smoothed light curve.

    :param clip: tolerance on clipping
    :param width: width of window for median-smoothing filter

    :returns: time, flux, flux_err

    """
    # medfilt pads the array to be filtered with zeros, so edge behaviour
    # is better if we filter flux-1 rather than flux.
    d = abs(medfilt(flux-1, width)+1-flux)
    mad = d.mean()
    ok = d < clip*mad

    if verbose:
        print('\nRejected {} points more than {:0.1f} x MAD = {:0.0f} '
                'ppm from the median'.format(sum(~ok),clip,1e6*mad*clip))

    return ok

def rms(x):
    return np.sqrt( np.nanmean( (x - np.nanmean(x))**2 ) )

def fake_allan_deviation(times, residuals, binmax=10, method='pipe', timeunit=None, plot=True):
    """Estimate and optionally plot a noise-vs-bin-size curve (a proxy for Allan deviation).

    The function computes the scatter of binned residuals for a sequence
    of integer bin sizes and returns both the computed noise (in ppm) and
    a white-noise expectation for comparison. When ``plot=True`` a
    log-log figure is produced with a secondary x-axis that maps bin size
    to a time-unit (days/hours/minutes) according to the total span of
    ``times``.

    Parameters
    ----------
    times : array-like
        Time stamps of the observations (assumed in days).
    residuals : array-like
        Residual fluxes (same length as ``times``).
    binmax : int, optional
        Parameter controlling the maximum number of bins. The function
        will evaluate binsizes from 1..int(len(times)/binmax). Default 10.
    method : {'pipe', 'std', 'rms', 'astropy'}, optional
        Method used to estimate the scatter of the binned residuals:
        - 'pipe' : use ``pipe_mad`` (median absolute differences estimator)
        - 'std'  : use ``np.nanstd``
        - 'rms'  : use the root-mean-square (``rms`` helper)
        - 'astropy' : use ``astropy.stats.mad_std`` (robust MAD-based std)
    timeunit : {'d', 'hr', 'min'}, optional
        If provided, forces the time unit used in the secondary x-axis
        to days ('d'), hours ('hr'), or minutes ('min'). By default the
        function selects the most appropriate unit based on the total
        span of ``times``.
    plot : bool, optional
        If ``True`` (default), create and display a matplotlib figure with
        the computed noise curve and the white-noise expectation.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The created figure if ``plot=True``, otherwise ``None``.
    axs : matplotlib.axes.Axes or None
        The axes object for the figure if ``plot=True``, otherwise ``None``.
    binsize : ndarray
        Integer array of evaluated bin sizes.
    noise_ppm : ndarray
        Computed scatter of binned residuals converted to parts-per-million
        (ppm), corresponding to ``binsize``.
    white_noise_expec : ndarray
        White-noise expectation (ppm) computed as
        ``noise_func(residuals) / sqrt(binsize)`` (with a small-sample
        correction applied). Values for ``binsize==0`` will be ``inf`` or
        undefined.
    """

    # Computing bin-size
    binsize = np.arange(1, int(len(times)/binmax), 1)
    # Array to save noise
    noise = np.zeros( len(binsize) )
    # Number of data points in each bin
    nbin = np.zeros( len(binsize) )

    # Choosing the method to compute noise
    if method == 'pipe':
        noise_func = pipe_mad
    elif method == 'std':
        noise_func = np.nanstd
    elif method == 'rms':
        noise_func = rms
    elif method == 'astropy':
        noise_func = lambda x: mad_std(x, ignore_nan=True)
    else:
        raise ValueError("Method should be one of 'pipe', 'std', 'rms', or 'astropy'.")

    # Computing binning for each binsize, and then computing the stddev of the binned residuals
    for i in range(len(binsize)):
        if binsize[i] != 1:
            _, binned_flux, _, _ = lcbin(time=times, flux=residuals, nmin=1, binwidth=binsize[i] * np.nanmedian(np.diff(times)) )
        else:
            binned_flux= np.copy( residuals )
    
        noise[i] = noise_func(binned_flux)
        nbin[i] = int( np.floor( len(residuals) / binsize[i] ) )

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
    
    # Computing the white-noise expectation
    white_noise_expec = noise_func(residuals)/np.sqrt(binsize) * 1e6 * np.sqrt(nbin / (nbin - 1.))
    
    if plot:
        # ----------------------------------
        #  And generating the plot
        # ----------------------------------
        fig, axs = plt.subplots()
        
        ## First plotting the computed noise
        axs.plot(binsize, noise * 1e6, color='dodgerblue', lw=1., label='Computed noise', zorder=10)
        ## Now plotting the white-noise expectation
        axs.plot(binsize, white_noise_expec, color='orangered', label='White-noise expectation')

        ## Adding secondary x-axis for time
        secax = axs.secondary_xaxis('top', functions=(bin2time, time2bin))
        secax.set_xlabel(time_unit_label, labelpad=10)

        axs.set_xscale('log')
        axs.set_yscale('log')

        axs.set_xlabel('Bin size [Number of points]')
        axs.set_ylabel('Noise estimate [ppm]')

        axs.legend()
    else:
        fig, axs = None, None

    return fig, axs, binsize, noise*1e6, white_noise_expec


def corner_plot(samples, labels, **kwargs):
    """Generate a corner plot from MCMC samples.

    Parameters
    ----------
    samples : ndarray
        2D array of shape (n_samples, n_parameters) containing the MCMC samples.
    labels : list of str
        List of parameter names for labeling the axes.
    figsize : tuple, optional
        Size of the figure to create. Default is (9, 9).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated corner plot figure.
    """

    samples = np.transpose( np.vstack( samples ) )

    fig = corner.corner(samples, labels=labels,
                        show_titles=True, title_fmt=".2f",
                        quantiles=[0.16,0.5,0.84], **kwargs)

    return fig

def make_psd(times, flux, nos_freq_points=100000, plot=True, plot_max_freq=True, timeunit=None):
    """Compute a Lomb-Scargle power spectral density (PSD) for a light curve.

    The routine converts ``times`` (assumed in days) to seconds and the
    input ``flux`` to parts-per-million (ppm), computes a dense frequency
    grid, evaluates a Lomb-Scargle PSD on that grid and (optionally)
    produces a log-log plot with a secondary axis showing period in an
    appropriate time unit.

    Parameters
    ----------
    times : array-like
        Time stamps of the observations (in days).
    flux : array-like
        Relative flux measurements (unitless, e.g. normalised to 1.0). The
        function converts these to ppm internally.
    nos_freq_points : int, optional
        Number of frequency points to evaluate the PSD on. Default is 100,000.
    plot : bool, optional
        If ``True`` (default) create and return a matplotlib figure and
        axes containing the PSD plot. If ``False``, no plot is created and
        ``fig`` and ``axs`` in the return tuple will be ``None``.
    plot_max_freq : bool, optional
        If ``True`` (default) mark the frequency with maximum power on the
        plot with a vertical dashed line.
    timeunit : {'d','hr','min'} or None, optional
        Force the secondary x-axis unit for the period conversion. If
        ``None`` (default) the function selects days/hours/minutes
        automatically based on the frequency grid span.

    Returns
    -------
    freq_grid : astropy.units.Quantity (Hz)
        Frequency grid used to evaluate the PSD.
    psd1 : astropy.units.Quantity
        Lomb-Scargle power spectral density evaluated on ``freq_grid``
        (units: ppm^2 Hz^-1).
    fig : matplotlib.figure.Figure or None
        Figure with the PSD plot when ``plot=True``, otherwise ``None``.
    axs : matplotlib.axes.Axes or None
        Axes for the PSD plot when ``plot=True``, otherwise ``None``.
    per_max_pow : astropy.units.Quantity
        Period corresponding to the maximum power in the PSD (converted
        to days).

    Notes
    -----
    The function uses ``astropy.timeseries.LombScargle`` for the PSD and
    multiplies the input flux by 1e6 to report power in ppm^2 Hz^-1.
    """

    # Computing time in units of days to seconds
    times = times * 24. * 60. * 60. * u.s
    # Converting flux to ppm
    fl1 = 1e6 * flux * u.dimensionless_unscaled

    ## Frequency grid
    min_freq, max_freq = 1 / np.ptp(times), 1 / np.nanmedian(np.diff(times)) * 0.5
    freq_grid = np.linspace(min_freq, max_freq, nos_freq_points)

    ## And the Lomb-Scargle periodogram
    psd1 = LombScargle(times, fl1, normalization='psd').power(frequency=freq_grid)
    
    ## Computing the period corresponding to the highest power
    idx_max_pow = np.argmax(psd1.value)
    freq_max_pow = freq_grid[idx_max_pow]
    per_max_pow = ( 1/freq_max_pow ).to(u.day)

    # First, let's estimate the time units we need from times array:
    time_grid = (1 / freq_grid ).to(u.d).value
    if timeunit is None:
        if np.ptp(time_grid) >= 1.:
            ## If times duration is greater than 2 hours, we use days
            time_unit = u.d
            time_unit_label = 'Time [d]'
        elif ( np.ptp(time_grid) > 5/24 ) and ( np.ptp(time_grid) < 1. ):
            ## If times duration is greater than 2 hours, but less than 2 days, we use hours
            time_unit = u.hr    # Converting days to hours
            time_unit_label = 'Time [hr]'
        else:
            ## If times duration is less than 2 hours, we use minutes
            time_unit = u.min
            time_unit_label = 'Time [min]'
    else:
        if timeunit == 'd':
            time_unit = u.d
            time_unit_label = 'Time [d]'
        elif timeunit == 'hr':
            time_unit = u.hr
            time_unit_label = 'Time [hr]'
        elif timeunit == 'min':
            time_unit = u.min
            time_unit_label = 'Time [min]'
        else:
            raise ValueError("timeunit should be one of 'd', 'hr', or 'min'.")

    if plot:
        # Making the plot:
        fig, axs = plt.subplots()

        ## Un-binned data
        axs.plot(freq_grid, psd1, alpha=1., color='orangered', lw=1., zorder=10)
        if plot_max_freq:
            axs.axvline(freq_max_pow.value, ls='--', c='cornflowerblue', lw=1., zorder=5)

        # Define properties to define upper axis as well:
        def freq2tim(x):
            x = np.where(x != 0, x, np.nan) * u.Hz
            return (1/x).to(time_unit).value
        
        def tim2freq(x):
            x = np.where(x != 0, x, np.nan) * u.s
            return (1/x).to(u.Hz).value

        ax2 = axs.secondary_xaxis("top", functions=(freq2tim, tim2freq))
        ax2.tick_params(axis='both', which='major')
        ax2.set_xlabel(time_unit_label, labelpad=10)

        axs.set_xscale('log')
        axs.set_yscale('log')
        
        axs.set_xlabel(r'Frequency [Hz]')
        axs.set_ylabel(r'Power [ppm$^2$ Hz$^{-1}$]')
        
        axs.set_xlim([np.min(freq_grid.value), np.max(freq_grid.value)])
    else:
        fig, axs = None, None

    return freq_grid, psd1, fig, axs, per_max_pow

def psd_fitting_kernel(times, flux, kernel, init_values, bounds=None, nos_freq_points=100000, plot=True, timeunit=None, **kwargs):
    """Fit a kernel PSD to the Lomb-Scargle periodogram of a light curve.

    The function computes the Lomb-Scargle PSD of the input light curve
    and fits the provided kernel PSD to it using ``scipy.optimize.minimize``.
    The kernel should be a function that takes frequency and kernel parameters
    as input and returns the corresponding PSD value. The fitting is done by
    minimizing the squared difference between the Lomb-Scargle PSD and the
    kernel PSD evaluated on the same frequency grid.

    Parameters
    ----------
    times : array-like
        Time stamps of the observations (in days).
    flux : array-like
        Relative flux measurements (unitless, e.g. normalised to 1.0).
    kernel : callable
        A celerite.terms term that takes parameters as input and returns kernel instance.
    init_values : list or array-like
        Initial guess for the kernel parameters to be fitted.
    bounds : list of tuples, optional
        Bounds for the kernel parameters in the form [(min1, max1), (min2, max2), ...].
        If ``None`` (default), no bounds are applied.
    nos_freq_points : int, optional
        Number of frequency points to evaluate the PSD on. Default is 100,000.
    plot : bool, optional
        If ``True`` (default), create and return a matplotlib figure and axes containing
        the PSD plot with the Lomb-Scargle PSD, the fitted kernel PSD, and the
        initial guess kernel PSD for comparison. If ``False``, no plot is created
        and ``fig`` and ``axs`` in the return tuple will be ``None``.
    timeunit : {'d','hr','min'} or None, optional
        Force the secondary x-axis unit for the period conversion. If ``None`` (default)
        the function selects days/hours/minutes automatically based on the frequency grid span.
    **kwargs
        Additional keyword arguments to pass to ``scipy.optimize.minimize`` for the optimization process (e.g. method, options, etc.).
    
    Returns
    -------
    best_fit_params : ndarray
        Best-fit kernel parameters found by the optimization.
    kernel_psd_fit : ndarray
        Kernel PSD evaluated on the frequency grid using the best-fit parameters.
    freq_grid : astropy.units.Quantity (Hz)
        Frequency grid used to evaluate the PSD.
    psd_ls : astropy.units.Quantity
        Lomb-Scargle power spectral density evaluated on ``freq_grid`` (units: ppm^2 Hz^-1).
    fig : matplotlib.figure.Figure or None
        Figure with the PSD plot when ``plot=True``, otherwise ``None``.
    axs : matplotlib.axes.Axes or None
        Axes for the PSD plot when ``plot=True``, otherwise ``None``.
    """

    # First, let's compute the Lomb-Scargle PSD using the previously defined function
    freq_grid, psd_ls, fig, axs, _ = make_psd(times=times, flux=flux * 1e-6, nos_freq_points=nos_freq_points, plot=plot, timeunit=timeunit)
    freq_grid = freq_grid.to(u.d**-1)
    psd_ls = psd_ls / len(times)

    # Now we define the objective function for optimization (squared difference between PSDs)
    def objective(params):
        kernel_instance = kernel(*params)
        kernel_psd = kernel_instance.get_psd(2*np.pi*freq_grid.value) / 2 / np.pi
        return np.sum( (psd_ls.value - kernel_psd)**2 )

    # Performing the optimization to find best-fit parameters
    result = minimize(objective, x0=init_values, bounds=bounds, **kwargs)
    best_fit_params = result.x

    # Evaluating the kernel PSD with the best-fit parameters
    kernel_instance = kernel(*best_fit_params)
    kernel_psd_fit = kernel_instance.get_psd(2*np.pi*freq_grid.value) / 2 / np.pi

    if plot:
        # Plotting the fitted kernel PSD on top of the Lomb-Scargle PSD
        axs.plot(freq_grid.to(u.s**-1), kernel_psd_fit * len(times), color='navy', lw=1., label='Fitted Kernel PSD', zorder=20)

        # Also plotting the kernel PSD with initial guess parameters for comparison
        kernel_instance_init = kernel(*init_values)
        kernel_psd_init = kernel_instance_init.get_psd(2*np.pi*freq_grid.value) / 2 / np.pi
        axs.plot(freq_grid.to(u.s**-1), kernel_psd_init * len(times), color='gray', lw=1., ls='--', label='Initial Guess Kernel PSD')
        
        axs.legend()

    return best_fit_params, kernel_psd_fit, freq_grid, psd_ls, fig, axs


def standarize_data(input_data):
    """
    Standarize the dataset.
    The function originally written by N. Espinoza
    for TESS data analysis.
    """
    output_data = np.copy(input_data)
    averages = np.nanmedian(input_data,axis=1)
    for i in range(len(averages)):
        sigma = mad_std(output_data[i,:])
        output_data[i,:] = output_data[i,:] - averages[i]
        output_data[i,:] = output_data[i,:]/sigma
    return output_data

def classic_PCA(Input_Data, standarize = True):
    """  
    classic_PCA function
    Description
    This function performs the classic Principal Component Analysis on a given dataset.
    The function originally written by N. Espinoza
    for TESS data analysis.
    """
    if standarize:
        Data = standarize_data(Input_Data)
    else:
        Data = np.copy(Input_Data)
    eigenvectors_cols,eigenvalues,eigenvectors_rows = np.linalg.svd(np.cov(Data))
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx[::-1]]
    eigenvectors_cols = eigenvectors_cols[:,idx[::-1]]
    eigenvectors_rows = eigenvectors_rows[idx[::-1],:]
    # Return: V matrix, eigenvalues and the principal components.
    return eigenvectors_rows,eigenvalues,np.dot(eigenvectors_rows,Data)

def generate_times_with_gaps(times, efficiency):
    """This function takes the times array with a regular cadence and generate
    another times array with data gaps in it with a given efficiency to mimick
    CHEOPS observations. Efficiency here is the fraction of the actual time spent 
    on the star.
    
    Parameters
    ----------
    times : numpy.ndarray
        Regularly spaced time array
    efficiency : float
        Efficiency of the observations in per cent.
        
    Returns
    -------
    tim : numpy.ndarray
        Time array with gaps in it.
    roll : numpy.ndarray
        The corresponding roll angle array.
    """
    # Generating gappy times
    ## Computing total number of orbits in the data
    cheops_orbit_time_day = 98.77 / (60 * 24)
    orbit_nos = np.ptp(times) / cheops_orbit_time_day

    ## Generating roll numbers
    roll = np.linspace(0, orbit_nos*360, len(times)) + np.random.randint(0,360)
    roll = roll % 360
    idx_rollsort = np.argsort(roll)
    roll_rollsort = roll[idx_rollsort]
    tim_rollsort = times[idx_rollsort]

    ## Selecting the range of roll-angles to discard
    st_roll = np.random.choice(roll_rollsort)    # Starting roll number of discarded roll angles
    loc_st_roll = np.where(roll_rollsort == st_roll)[0][0]
    nos_discarded_pts = int((1 - (0.01 * efficiency)) * len(times)) # Total number of discarded points
    idx_discarded = np.ones(len(times), dtype=bool)

    if int(loc_st_roll+nos_discarded_pts) > len(times):
        # Roll over the starting roll numbers
        idx_discarded[loc_st_roll:] = False
        dis_pt = len(times) - loc_st_roll
        idx_discarded[0:nos_discarded_pts-dis_pt] = False
    else:
        idx_discarded[loc_st_roll:loc_st_roll + nos_discarded_pts] = False

    # Let's actually discard the points now
    tim = tim_rollsort[idx_discarded]
    roll = roll_rollsort[idx_discarded]

    # And sort them
    idx_timsort = np.argsort(tim)
    tim, roll = tim[idx_timsort], roll[idx_timsort]

    return tim, roll

def planck_func(lam, temp):
    """
    Given the wavelength and temperature
    this function will compute the specific
    intensity using the Planck's law
    """
    
    coeff1 = 2 * con.h * con.c * con.c / lam**5
    expo = np.exp( con.h * con.c / lam / con.k_B / temp ) - 1
    planck = (coeff1/expo).to(u.W / u.m**2 / u.micron)
    
    return planck

class BrightnessTemperatureCalculator:
    """Compute brightness temperature(s) from eclipse depths and a bandpass.

    Parameters
    ----------
    fp : float or array-like
        Eclipse depths (planet/star flux ratio). If array-like, temperature distribution
        will be computed (multiprocessing is used in this case).
    rprs : float or array-like
        Planet-to-star radius ratio (Rp/R*). If scalar and fp is array-like,
        the same rprs is applied to all elements.
    bandpass : dict
        Bandpass dict with keys 'WAVE' (astropy Quantity) and 'RESPONSE' (array).
    nthreads : int, optional
        Number of worker processes for multiprocessing. Default is cpu count.
    method : str, optional
        Minimizer method passed to scipy.optimize.minimize. Default 'Nelder-Mead'.
    teff_star : float or None
        Stellar effective temperature in K (used if stellar_spec not provided).
    stellar_spec : dict or None
        Stellar spectrum dict with keys 'WAVE' and 'FLUX' (astropy Quantities).
    pout : str, optional
        Output directory used to cache results. Default current working dir.

    Methods
    -------
    compute()
        Run the calculation and return temperature(s) in Kelvin (float or 1D ndarray).
    """

    def __init__(self, fp, rprs, bandpass, nthreads=multiprocessing.cpu_count(),
                 method='Nelder-Mead', teff_star=None, stellar_spec=None, pout=os.getcwd()):
        self.fp = fp
        self.rprs = rprs
        self.bandpass = bandpass
        self.nthreads = nthreads
        self.method = method
        self.teff_star = teff_star
        self.stellar_spec = stellar_spec
        self.pout = pout

        # prepare stellar spectrum
        if (self.stellar_spec is not None) and (self.teff_star is None):
            self.wav_star = self.stellar_spec['WAVE'].to(u.micron)
            self.fl_star = self.stellar_spec['FLUX'].to(u.J / u.s / u.m**2 / u.micron)
        elif (self.teff_star is not None) and (self.stellar_spec is None):
            self.wav_star = self.bandpass['WAVE'].to(u.micron)
            self.fl_star = planck_func(self.wav_star, self.teff_star)
        else:
            raise ValueError('Provide either stellar_spec or teff_star.')

        # bandpass
        self.wav_instrument = self.bandpass['WAVE'].to(u.micron)
        self.response_instrument = self.bandpass['RESPONSE']

        # build transmission evaluated on stellar wavelengths
        spln2 = interp1d(x=self.wav_instrument.value, y=self.response_instrument,
                         bounds_error=False, fill_value=0.0)
        trans_fun = spln2(self.wav_star.value)
        # guard against division by zero if response all zeros
        if np.max(trans_fun) > 0:
            trans_fun = trans_fun / np.max(trans_fun)
        self.trans_fun = trans_fun

    def forward_model(self, temp_pl):
        """Forward model to compute planet/star flux ratio for given planet brightness temperature.

        Parameters
        ----------
        temp_pl : float or array-like
            Planet brightness temperature(s) in Kelvin.

        Returns
        -------
        float or ndarray
            Planet/star flux ratio. Returns a scalar for scalar inputs, otherwise an
            array with the broadcasted input shape.
        """
        temp_arr = np.asarray(temp_pl, dtype=float)
        rprs_arr = np.asarray(self.rprs, dtype=float)
        temp_b, rprs_b = np.broadcast_arrays(temp_arr, rprs_arr)

        star_den = simpson(y=self.fl_star.value * self.trans_fun, x=self.wav_star.value)
        fp_pl = np.empty(temp_b.shape, dtype=float)

        for idx in np.ndindex(temp_b.shape):
            planet_bb = planck_func(self.wav_star, temp_b[idx] * u.K)
            planet_den = simpson(y=planet_bb.value * self.trans_fun, x=self.wav_star.value)
            fp_pl[idx] = planet_den * (rprs_b[idx]**2) / star_den

        return float(fp_pl) if fp_pl.ndim == 0 else fp_pl

    def _solve_single(self, ecl_dep, rprs_ratio):
        """Compute brightness temperature for a single eclipse depth."""
        # scaled fp for integration comparison
        if ecl_dep > 0:
            fp_pl = ecl_dep * simpson(y=self.fl_star.value * self.trans_fun, x=self.wav_star.value) / (rprs_ratio**2)

            def func_to_minimize_new(x):
                planet_bb = planck_func(self.wav_star, x * u.K)
                planet_den = simpson(y=planet_bb.value * self.trans_fun, x=self.wav_star.value)
                chi2 = (fp_pl - planet_den)**2
                return chi2

            soln = minimize(fun=func_to_minimize_new, x0=1000.0, method=self.method)
            temp_pl = float(np.atleast_1d(soln.x)[0])
        else:
            temp_pl = 0.
        # return scalar Kelvin
        return temp_pl

    def compute(self):
        """Compute and return brightness temperature(s) in Kelvin.

        Returns
        -------
        temp_pl : float or ndarray
            Brightness temperature (K) for scalar input or 1D ndarray for array input.
        """
        fp = self.fp
        rprs = self.rprs

        # scalar path
        if isinstance(fp, (float, np.floating)) or (np.asarray(fp).size == 1):
            return self._solve_single(float(fp), float(rprs) if np.isscalar(rprs) else float(np.asarray(rprs).item()))

        # array path
        fp_arr = np.asarray(fp, dtype=float)
        if np.isscalar(rprs):
            rprs_arr = np.full(fp_arr.shape, float(rprs))
        else:
            rprs_arr = np.asarray(rprs, dtype=float)

        cache_path = Path(self.pout) / 'Brightness_temp.npy'
        if cache_path.exists():
            print('>>> --- Loading...')
            temp_pl = np.load(str(cache_path))
            return temp_pl

        # prepare inputs for multiprocessing
        inputs = [(float(fp_arr[i]), float(rprs_arr[i])) for i in range(len(fp_arr))]

        # use Pool with bound method; user's environment set_start_method('fork') so this is fine
        with multiprocessing.Pool(self.nthreads) as p:
            result_list = p.starmap(self._solve_single, tqdm(inputs, total=len(fp)), chunksize=max(1, self.nthreads))
        temp_pl = np.array(result_list, dtype=float)

        # save cache
        np.save(str(cache_path), temp_pl)
        return temp_pl
    
def trapz2d(z, x, y):
    """
    Helper function to perform 2D trapezoidal integration.
    Integrates a regularly spaced 2D grid using the composite trapezium rule.

    Source: https://github.com/tiagopereira/python_tips/blob/master/code/trapz2d.py
    My sourse: I have copied from `kelp`: https://github.com/bmorris3/kelp/blob/219a922849634d9e982cd7bd05910596dea2ef6e/kelp/core.py#L15C1-L44C48

    Parameters
    ----------
    z : `~numpy.ndarray`
        2D array
    x : `~numpy.ndarray`
        grid values for x (1D array)
    y : `~numpy.ndarray`
        grid values for y (1D array)

    Returns
    -------
    t : `~numpy.ndarray`
        Trapezoidal approximation to the integral under z
    """
    m = z.shape[0] - 1
    n = z.shape[1] - 1
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    s1 = z[0, 0, :] + z[m, 0, :] + z[0, n, :] + z[m, n, :]
    s2 = (np.sum(z[1:m, 0, :], axis=0) + np.sum(z[1:m, n, :], axis=0) +
        np.sum(z[0, 1:n, :], axis=0) + np.sum(z[m, 1:n, :], axis=0))
    s3 = np.sum(np.sum(z[1:m, 1:n, :], axis=0), axis=0)
    return dx * dy * (s1 + 2 * s2 + 4 * s3) / 4

def autocorrelation_function(chain):
    """Compute the autocorrelation function for a given chain and lags.

    Parameters
    ----------
    chain : ndarray
        Array containing the chain values.

    Returns
    -------
    rhos : ndarray
        Autocorrelation values for each lag.
    """
    mu, variance = np.mean(chain), np.var(chain)
    N = len(chain)
    
    rhos = np.zeros( len(chain) )
    for i in range(N):
        X0 = chain[:N-i] - mu
        Xk = chain[i:] - mu
        rhoi = np.sum(X0*Xk)/variance
        rhos[i] = rhoi/(N-i)
    return rhos