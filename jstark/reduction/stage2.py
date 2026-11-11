"""Stage 2: wavelength solution (and optional Stage 2 calibration steps).

For time-series spectroscopy only the wavelength solution is needed from the
official Stage 2 (``assign_wcs``). :class:`Stage2` evaluates the WCS on the
*full* subarray pixel grid so that the data geometry produced by Stage 1 is
preserved for jstark's own spectral extraction:

* NIRSpec BOTS: slit WCS (``S1600A1``) via ``nrs_wcs_set_input``.
* NIRCam grism TSO: ``assign_wcs`` + ``extract_2d`` (which accounts for
  source offsets), mapped back onto the full subarray.
* NIRISS SOSS: PASTASOSS wavelength maps for all orders (these depend on the
  pupil-wheel position ``PWCPOS``); the trace positions are kept as well.
* MIRI LRS: ``assign_wcs`` slitless WCS.
* MIRI imaging: no wavelength map.

Optional data-modifying steps (``flat_field``, ``photom``, ...) can be run
on every segment with :meth:`Stage2.run`; they are not needed for relative
time-series measurements and are skipped by default.
"""
import logging
import os
import warnings

import numpy as np
from astropy.io import fits

from .instruments import identify_mode, get_mode_config
from .io import find_exposures, select_exposure, output_filename, StepLog

__all__ = ['Stage2', 'wavelength_map']


def _lightweight_model(fname):
    """A 2D ImageModel with the metadata of a rateints file (first integration only)."""
    from stdatamodels.jwst import datamodels
    with datamodels.open(fname) as cube:
        data = np.array(cube.data[0], dtype=np.float32)
        im = datamodels.ImageModel(data=data, dq=np.array(cube.dq[0]), err=np.array(cube.err[0]))
        im.update(cube)
        im.meta.wcsinfo = cube.meta.wcsinfo
    return im


def _call(step_name, model, **kwargs):
    from ._compat import apply_all
    apply_all()
    from jwst.pipeline import Spec2Pipeline
    cls = Spec2Pipeline.step_defs[step_name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return cls.call(model, configure_log=False, **kwargs)


def _grid(shape):
    y, x = np.mgrid[:shape[0], :shape[1]]
    return x.astype(float), y.astype(float)


def _nirspec_wavelength(model):
    from jwst.assign_wcs import nirspec
    m = _call('assign_wcs', model)
    slit = m.meta.instrument.fixed_slit or 'S1600A1'
    wcs = nirspec.nrs_wcs_set_input(m, slit)
    x, y = _grid(m.data.shape)
    out = wcs(x, y, with_bounding_box=False)
    lam = np.asarray(out[2], dtype=float)
    lam_bb = np.asarray(wcs(x, y)[2], dtype=float)
    return lam, {'slit': slit, 'in_slit': np.isfinite(lam_bb)}


def _nircam_wavelength(model, order=1):
    m = _call('assign_wcs', model)
    ny = m.data.shape[0]
    sub = _call('extract_2d', m, extract_orders=[order], tsgrism_extract_height=ny)
    slit = sub.slits[0] if hasattr(sub, 'slits') and len(sub.slits) else sub
    wl = np.asarray(slit.wavelength, dtype=float)
    lam = np.full(m.data.shape, np.nan)
    # extract_2d cut-out position in the subarray (1-indexed xstart/ystart)
    x0 = int(slit.xstart) - 1 if slit.xstart is not None else 0
    y0 = int(slit.ystart) - 1 if slit.ystart is not None else 0
    h, w = wl.shape
    lam[y0:y0 + h, x0:x0 + w] = wl
    return lam, {'order': order, 'cutout': (y0, x0, h, w),
                 'source_ypos': float(getattr(slit, 'source_ypos', np.nan) or np.nan)}


def _soss_wavelength(model):
    from jwst.extract_1d.soss_extract.pastasoss import get_soss_wavemaps
    pwcpos = model.meta.instrument.pupil_position
    subarray = model.meta.subarray.name
    wavemaps, traces = get_soss_wavemaps(pwcpos, subarray=subarray, spectraces=True)
    wavemaps = np.asarray(wavemaps, dtype=float)
    ny, nx = model.data.shape
    if wavemaps.shape[1:] != (ny, nx) and wavemaps.shape[1:] == (nx, ny):
        wavemaps = np.swapaxes(wavemaps, 1, 2)
    wavemaps = np.where(wavemaps > 0, wavemaps, np.nan)
    return wavemaps, {'pwcpos': pwcpos, 'traces': traces}


def _miri_lrs_wavelength(model):
    m = _call('assign_wcs', model)
    x, y = _grid(m.data.shape)
    lam = np.asarray(m.meta.wcs(x, y, with_bounding_box=False)[2], dtype=float)
    lam_bb = np.asarray(m.meta.wcs(x, y)[2], dtype=float)
    return lam, {'in_bbox': np.isfinite(lam_bb)}


def wavelength_map(fname, mode=None):
    """Compute the 2D wavelength map (micron) of a rateints/rate file.

    Parameters
    ----------
    fname : str
        A ``rateints`` (or ``rate``) file of the exposure.
    mode : str or None
        jstark mode; identified from the header if ``None``.

    Returns
    -------
    wavelength : ndarray or None
        ``(ny, nx)``, or ``(norder, ny, nx)`` for NIRISS/SOSS. ``None`` for
        imaging modes.
    info : dict
        Mode-specific extra information (e.g. SOSS trace positions).
    """
    mode = mode or identify_mode(fits.getheader(fname, 0))
    inst = mode.split('/')[0]
    model = _lightweight_model(fname)
    if inst == 'NIRSpec':
        return _nirspec_wavelength(model)
    if inst == 'NIRCam':
        return _nircam_wavelength(model)
    if mode == 'NIRISS/SOSS':
        return _soss_wavelength(model)
    if mode == 'MIRI/LRS':
        return _miri_lrs_wavelength(model)
    return None, {}


class Stage2:
    """Wavelength solution and optional Stage 2 calibrations for a TSO exposure.

    Parameters
    ----------
    inputs : str or list
        Directory with the Stage 1 ``*_rateints.fits`` files, a glob
        pattern or a list of files.
    outdir : str or None
        Output directory (default: the input directory).
    mode : str or None
        jstark mode (identified from the header if ``None``).
    exposure : str or None
        Exposure selection if several are present.
    suffix : str
        Input file suffix (default ``'rateints'``).

    Attributes
    ----------
    wavelength : ndarray or None
        Wavelength map (after :meth:`compute_wavelength`).
    wavelength_info : dict
        Extra information (e.g. SOSS trace positions, NIRSpec in-slit mask).
    """

    def __init__(self, inputs, outdir=None, mode=None, exposure=None, suffix='rateints',
                 verbose=True):
        exps = find_exposures(inputs, suffix=suffix)
        self.exposure, self.files = select_exposure(exps, exposure, verbose=verbose)
        self.header = fits.getheader(self.files[0], 0)
        self.mode = mode or identify_mode(self.header)
        self.config = get_mode_config(self.mode)
        self.outdir = outdir or os.path.dirname(os.path.abspath(self.files[0]))
        os.makedirs(self.outdir, exist_ok=True)
        self.verbose = verbose
        self.log = StepLog(os.path.join(self.outdir, f'{self.exposure}_stage2_log.json'))
        self.wavelength = None
        self.wavelength_info = {}
        for name in ('stpipe', 'jwst', 'stcal'):
            logging.getLogger(name).setLevel(logging.WARNING)

    def compute_wavelength(self, segment=0):
        """Compute (and store) the wavelength map from one segment."""
        self.wavelength, self.wavelength_info = wavelength_map(self.files[segment], self.mode)
        self.log.add('wavelength_map', segment=segment)
        if self.verbose and self.wavelength is not None:
            print(f'Wavelength map {self.wavelength.shape}: '
                  f'{np.nanmin(self.wavelength):.3f}-{np.nanmax(self.wavelength):.3f} micron')
        return self.wavelength

    def save_wavelength(self, fname=None, overwrite=True):
        """Save the wavelength map (extension ``WAVELENGTH``) to a FITS file."""
        if self.wavelength is None:
            self.compute_wavelength()
        if fname is None:
            fname = os.path.join(self.outdir, f'{self.exposure}_wavelength.fits')
        hdr = fits.Header()
        hdr['JSMODE'] = self.mode
        hdr['EXPOSURE'] = self.exposure
        hdus = [fits.PrimaryHDU(header=hdr),
                fits.ImageHDU(np.asarray(self.wavelength, np.float32), name='WAVELENGTH')]
        info = self.wavelength_info
        for key in ('in_slit', 'in_bbox'):
            if key in info:
                hdus.append(fits.ImageHDU(info[key].astype(np.uint8), name=key.upper()))
        if 'traces' in info and info['traces'] is not None:
            for k, tr in enumerate(info['traces']):
                tr = np.asarray(tr, dtype=float)
                hdus.append(fits.ImageHDU(tr, name=f'TRACE_O{k + 1}'))
        if 'pwcpos' in info:
            hdus[0].header['PWCPOS'] = info['pwcpos']
        fits.HDUList(hdus).writeto(fname, overwrite=overwrite)
        if self.verbose:
            print(f'Saved wavelength map to {fname}')
        return fname

    def run(self, steps=('assign_wcs',), segments=None, save_wavelength=True, overwrite=True):
        """Run Stage 2.

        Parameters
        ----------
        steps : sequence of str
            Stage 2 steps. ``'assign_wcs'`` only computes the wavelength map
            (no per-segment products are written). Any other ``jwst`` Spec2
            step names (e.g. ``'srctype'``, ``'flat_field'``, ``'photom'``)
            are run, in order, on every segment, writing ``*_calints.fits``.
            Note that for NIRSpec, ``flat_field`` and ``photom`` require
            ``extract_2d``, which crops the data.
        segments : list of int or None
            Segments to process (default: all).

        Returns
        -------
        dict
            ``{'wavelength': file or None, 'calints': [files]}``.
        """
        out = {'wavelength': None, 'calints': []}
        if 'assign_wcs' in steps:
            self.compute_wavelength()
            if save_wavelength and self.wavelength is not None:
                out['wavelength'] = self.save_wavelength(overwrite=overwrite)
        extra = [s for s in steps if s != 'assign_wcs']
        if not extra:
            return out
        from stdatamodels.jwst import datamodels
        segments = range(len(self.files)) if segments is None else segments
        for seg in segments:
            m = datamodels.open(self.files[seg])
            m = _call('assign_wcs', m)
            for s in extra:
                m = _call(s, m)
                self.log.add(s, segment=seg)
            fn = output_filename(self.files[seg], self.outdir, 'calints')
            m.save(fn, overwrite=overwrite)
            out['calints'].append(fn)
            if self.verbose:
                print(f'Segment {seg}: {", ".join(extra)} -> {fn}')
        return out
