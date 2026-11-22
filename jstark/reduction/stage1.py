"""Stage 1: detector-level processing from ``uncal`` ramps to ``rateints``."""
import logging
import os
import time
import warnings

import numpy as np
from astropy.io import fits

from .instruments import StepSpec, identify_mode, get_mode_config
from .io import find_exposures, select_exposure, output_filename, StepLog
from .products import (ProductStore, Check, cleaned_exposures, content_id, differences,
                       loaded_message, plain)
from .masks import (quick_signal_image, build_trace_mask, ensure_min_background)
from .onef import group_background, amp_bounds_from_model
from .jump import temporal_jump
from .bias import self_superbias, superbias_scale_factors, apply_bias

__all__ = ['Stage1']


def _jwst_imports():
    from stdatamodels.jwst import datamodels
    from jwst.pipeline import Detector1Pipeline
    return datamodels, Detector1Pipeline


def set_jwst_log_level(level=logging.WARNING):
    """Set the verbosity of the jwst / stpipe / stcal / CRDS loggers."""
    for name in ('stpipe', 'jwst', 'stcal', 'CRDS', 'crds'):
        logging.getLogger(name).setLevel(level)


class _LazyDiagnostics(dict):
    """``{segment: diagnostics}``; segments loaded from disk are read on first access."""

    def __init__(self):
        super().__init__()
        self.files = {}

    def __missing__(self, seg):
        from .diagnostics import load_stage1_diagnostics
        if seg not in self.files:
            raise KeyError(seg)
        d = load_stage1_diagnostics(self.files[seg])
        self[seg] = {step: {k: (v.item() if v.ndim == 0 else v) for k, v in info.items()}
                     for step, info in d.items() if isinstance(info, dict)}
        return self[seg]


class Stage1:
    """Detector-level processing of a JWST time-series exposure.

    The class wraps every ``jwst`` Detector1 step individually and adds
    custom steps (custom bias, group-level background / 1/f subtraction and
    time-domain jump detection). The reduction is described by an ordered,
    editable *recipe* of steps (see :meth:`show`), whose defaults depend on
    the observing mode (see :mod:`jstark.reduction.instruments`).

    Every segment of the exposure is processed by :meth:`run`. For
    inspection, the pipeline can be stopped after any step and resumed
    later, e.g.::

        s1 = Stage1('/path/to/uncal/dir', outdir='/path/to/output')
        s1.show()                                  # current recipe
        s1.set('group_bkg', method='poly', poly_order=1)
        model = s1.run(segments=[0], stop='refpix')     # stop after refpix
        ...inspect model.data...
        s1.run(segments=[0], start='group_bkg', model=model)   # resume
        s1.run()                                   # everything, all segments

    Segments whose ``rateints`` file already exists (made with the same
    settings) are not processed again: :meth:`run` returns the existing file.
    ``run(overwrite=True)`` processes them again. See
    :mod:`jstark.reduction.products` for how this is decided.

    Parameters
    ----------
    inputs : str or list
        Directory containing the ``*_uncal.fits`` files, a glob pattern, or
        a list of files. All segments of the selected exposure are used.
    outdir : str or None
        Output directory (default: ``<input dir>/jstark_stage1``).
    mode : str or None
        Mode name (e.g. ``'NIRSpec/G395H'``); identified from the header
        when ``None``.
    exposure : str or None
        Sub-string selecting one exposure if several are present in the
        input directory (default: the exposure with the most integrations).
    verbose : bool
        Print progress information.
    jwst_log_level : int
        Logging level for the jwst pipeline (default ``logging.WARNING``).

    Notes
    -----
    Ramp fitting and jump detection can use several cores
    (``maximum_cores``). On macOS / Windows Python starts worker processes
    by re-importing the main script, so when running from a script (not a
    notebook) protect the code with ``if __name__ == '__main__':``.

    Attributes
    ----------
    files : list of str
        Ordered segment files of the exposure.
    config : ModeConfig
        Mode configuration; ``config.recipe`` is the current recipe.
    trace_mask, bkg_mask : ndarray or None
        Source and background masks used by the custom steps (built from
        the first processed segment unless set with :meth:`set_trace_mask`).
    signal_image : ndarray or None
        Quick source image used to build the masks.
    diagnostics : dict
        Per-segment, per-step diagnostic information.
    model : jwst datamodel or None
        The most recent in-memory model.
    log : StepLog
        Log of all executed steps and their parameters.
    """

    def __init__(self, inputs, outdir=None, mode=None, exposure=None, verbose=True,
                 jwst_log_level=logging.WARNING):
        exposures = find_exposures(inputs, suffix='uncal')
        self.exposure, self.files = select_exposure(exposures, exposure, verbose=verbose)
        self.header = fits.getheader(self.files[0], 0)
        self.mode = mode if mode is not None else identify_mode(self.header)
        self.config = get_mode_config(self.mode)
        det = str(self.header.get('DETECTOR', '')).upper()
        grating = str(self.header.get('GRATING', '')).upper()
        if det == 'NRS2' and (grating == 'PRISM' or grating.endswith('M')):
            warnings.warn(f'{self.exposure}: for NIRSpec {grating} the spectral trace falls on NRS1 only; '
                          f'NRS2 contains no source. Pass exposure="nrs1" to select NRS1.')
        if outdir is None:
            outdir = os.path.join(os.path.dirname(os.path.abspath(self.files[0])), 'jstark_stage1')
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.verbose = verbose
        self.log = StepLog(os.path.join(outdir, f'{self.exposure}_stage1_log.json'))
        self.store = ProductStore(outdir, self.exposure)
        self.trace_mask = None
        self.bkg_mask = None
        self.wing_threshold = None    # group_bkg wing_frac threshold (DN/group), set once
        self.signal_image = None
        self.saturation_map = None
        self.auto_firstgroup = None
        self._user_mask = False
        # exposure-wide quantities found automatically (or restored), see _derived_settings
        self._auto_mask = self._auto_wing = self._auto_satmap = None
        self._satmap_info = None
        self._overwrite = False
        self._restored = []
        self.diagnostics = _LazyDiagnostics()
        self.model = None
        self.outputs = {}
        set_jwst_log_level(jwst_log_level)
        # let stpipe install its own (INFO) handlers only if verbose jwst output is wanted
        self._configure_jwst_log = jwst_log_level < logging.WARNING
        if verbose:
            print(f'jstark Stage1: {self.mode} | exposure {self.exposure} | '
                  f'{len(self.files)} segment(s) | output -> {self.outdir}')

    # ------------------------------------------------------------------
    # Recipe handling
    # ------------------------------------------------------------------
    @property
    def recipe(self):
        return self.config.recipe

    @property
    def step_names(self):
        return [s.name for s in self.recipe]

    def show(self):
        """Print the current recipe."""
        print(f'Recipe for {self.mode}:')
        for i, s in enumerate(self.recipe):
            flag = 'SKIP' if s.skip else 'run '
            kind = 'jstark' if s.is_custom else 'jwst  '
            print(f'  {i:2d}. [{flag}] {kind} {s.name:20s} {s.kwargs if s.kwargs else ""}')

    def set(self, step, skip=None, replace=False, **kwargs):
        """Change the parameters of a step in the recipe.

        Parameters
        ----------
        step : str
            Step name.
        skip : bool or None
            Skip (``True``) or run (``False``) the step; unchanged if ``None``.
        replace : bool
            If ``True``, replace all existing keyword arguments of the step;
            otherwise update them.
        **kwargs
            Step parameters.
        """
        spec = self.config.step(step)
        if skip is not None:
            spec.skip = bool(skip)
        if replace:
            spec.kwargs = dict(kwargs)
        else:
            spec.kwargs.update(kwargs)
        # running custom_bias replaces the standard superbias step
        if step == 'custom_bias' and skip is False:
            self.config.step('superbias').skip = True
        if step == 'superbias' and skip is False and 'custom_bias' in self.step_names:
            self.config.step('custom_bias').skip = True
        return spec

    def add_step(self, name, after, skip=False, **kwargs):
        """Insert an additional (jwst) step into the recipe after ``after``."""
        idx = self.step_names.index(after)
        self.recipe.insert(idx + 1, StepSpec(name, dict(kwargs), skip))

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------
    def set_trace_mask(self, trace_mask, bkg_mask=None):
        """Set the source mask (``True`` on the source) used by custom steps.

        If ``bkg_mask`` is not given, it is ``~trace_mask``.
        """
        self.trace_mask = np.asarray(trace_mask, dtype=bool)
        self.bkg_mask = ~self.trace_mask if bkg_mask is None else np.asarray(bkg_mask, bool)
        self._user_mask = True

    def build_trace_mask(self, model, axis=None, **kwargs):
        """Build the source and background masks from a ramp model.

        A quick, high-S/N image (median CDS) is computed, the source is
        masked with :func:`~jstark.reduction.masks.build_trace_mask`, and
        pixels flagged as reference / do-not-use / non-science are excluded
        from the background. At least ``min_pixels`` background pixels per
        column/row are guaranteed.

        Keyword arguments override the mode defaults (``config.trace_mask``).
        """
        opts = dict(self.config.trace_mask)
        opts.update(kwargs)
        min_pixels = opts.pop('min_pixels', 4)
        groupdq = getattr(model, 'groupdq', None)
        img = quick_signal_image(model.data, groupdq)
        valid = self._valid_pixels(model)
        self.signal_image = img
        self.trace_mask = build_trace_mask(img, self.config.dispersion_axis, valid=valid, **opts)
        bkg = ~self.trace_mask & valid & np.isfinite(img)
        axis = axis or self.config.onef_axis or 'column'
        self.bkg_mask = ensure_min_background(bkg, img, axis, min_pixels=min_pixels, valid=valid)
        if self.verbose:
            print(f'  trace mask: {self.trace_mask.mean() * 100:.1f}% of pixels on source, '
                  f'{self.bkg_mask.mean() * 100:.1f}% background')
        return self.trace_mask, self.bkg_mask

    @staticmethod
    def _valid_pixels(model):
        from stdatamodels.jwst.datamodels import dqflags
        bad = (dqflags.pixel['DO_NOT_USE'] | dqflags.pixel['REFERENCE_PIXEL']
               | dqflags.pixel['NON_SCIENCE'] | dqflags.pixel['DEAD'])
        pdq = getattr(model, 'pixeldq', None)
        if pdq is None or pdq.size == 0:
            pdq = model.dq if model.dq.ndim == 2 else model.dq[0]
        return (pdq & bad) == 0

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    def load(self, segment=0):
        """Open segment ``segment`` (0-indexed) as a ``RampModel``."""
        datamodels, _ = _jwst_imports()
        return datamodels.RampModel(self.files[segment])

    def run(self, segments=None, start=None, stop=None, model=None, save_after=None,
            overwrite=False, save_variances=True):
        """Run the recipe on one or more segments (finished segments are loaded).

        Parameters
        ----------
        segments : list of int or None
            0-indexed segments to process (default: all).
        start, stop : str or None
            First / last step to execute (inclusive). Default: the whole recipe.
        model : jwst datamodel, str or None
            Model to start from (only with a single segment and together
            with ``start``): an in-memory model or the path of an
            intermediate product saved with ``save_after``. Otherwise the
            ``uncal`` file is read.
        save_after : list of str or None
            Save the intermediate ramp after these steps
            (``<root>_<step>.fits``). The ``rateints`` product is always saved
            after ``ramp_fit`` / ``gain_scale``.
        overwrite : bool
            ``False`` (default): segments whose ``rateints`` file exists and
            was made with the current settings are not processed again; an
            existing file made with other settings raises an error. The
            exposure-wide quantities measured on the first segment (masks,
            wing threshold, group range) are restored, so a run that was
            interrupted continues as if it had not been. ``True``: process
            everything again. Runs with ``model=``, or that stop before
            ``ramp_fit``, are always executed (``save_after`` files are
            replaced).
        save_variances : bool
            Keep the ``VAR_POISSON`` and ``VAR_RNOISE`` arrays in the
            ``rateints`` files. jstark does not use them (``ERR`` is kept);
            ``False`` makes the files 40% smaller.

        Returns
        -------
        list of str or datamodel
            Paths of the ``rateints`` files if ramp fitting was performed,
            otherwise the in-memory model of the last processed segment.
        """
        segments = list(range(len(self.files))) if segments is None else list(segments)
        names = self.step_names
        i0 = names.index(start) if start is not None else 0
        i1 = names.index(stop) if stop is not None else len(names) - 1
        if model is not None and len(segments) != 1:
            raise ValueError('An input model can only be given for a single segment.')
        if model is None and i0 > 0:
            raise ValueError(f'start="{start}" requires model= (the in-memory model or a file '
                             f'saved with save_after) of the step before "{start}".')
        if isinstance(model, str):
            datamodels, _ = _jwst_imports()
            model = datamodels.RampModel(model)
        save_after = set(save_after or [])
        self._overwrite = overwrite
        if overwrite and len(set(segments)) == len(self.files):
            self._drop_restored()
        steps = self.recipe[i0:i1 + 1]
        makes_rates = any(s.name == 'ramp_fit' and not s.skip for s in steps)
        settings = self._settings(i0, i1, start, model)
        finished = {}
        if makes_rates and model is None and not overwrite:
            finished = self._find_finished(segments, settings)
        outputs = []
        for seg in segments:
            t0 = time.time()
            fn = output_filename(self.files[seg], self.outdir, 'rateints')
            todo = steps
            if seg in finished:
                # only intermediate products that are still missing
                last = [k for k, s in enumerate(steps) if s.name in save_after and not s.skip and not
                        os.path.isfile(output_filename(self.files[seg], self.outdir, s.name))]
                todo = steps[:last[-1] + 1] if last else []
                outputs.append(fn)
                self.outputs[seg] = fn
                if self.verbose:
                    print(f'Segment {seg + 1}/{len(self.files)}: {finished[seg]}')
                if not todo:
                    continue
            elif self.verbose:
                print(f'Segment {seg + 1}/{len(self.files)}: {os.path.basename(self.files[seg])}')
            m = model if model is not None else self.load(seg)
            ramp_done = False
            for spec in todo:
                if spec.skip:
                    continue
                ts = time.time()
                m = self.run_step(spec.name, m, segment=seg, **spec.kwargs)
                if spec.name == 'ramp_fit':
                    ramp_done = True
                if self.verbose:
                    print(f'    {spec.name:20s} done in {time.time() - ts:6.1f} s')
                if spec.name in save_after:
                    m.save(output_filename(self.files[seg], self.outdir, spec.name), overwrite=True)
            if ramp_done and seg not in finished:
                if not save_variances:
                    for attr in ('var_poisson', 'var_rnoise', 'var_flat'):
                        if m.hasattr(attr):
                            delattr(m, attr)
                self.store.begin(fn)
                m.save(fn, overwrite=True)
                outputs.append(fn)
                self.outputs[seg] = fn
                self._save_diagnostics(seg)
                self.store.done(fn, dict(settings, **self._derived_settings()), self._inputs(seg),
                                uid=content_id(m.data, m.err, m.dq), derived=self._derived_state())
                if self.verbose:
                    print(f'  -> {fn}  ({time.time() - t0:.1f} s)')
            self.model = m
        return outputs if outputs else self.model

    # ------------------------------------------------------------------
    # Products of earlier runs
    # ------------------------------------------------------------------
    def _settings(self, i0, i1, start=None, model=None):
        """Settings of a run (recipe steps and mode configuration), for the product records."""
        out = dict(mode=self.mode,
                   steps={s.name: dict(s.kwargs) for s in self.recipe[i0:i1 + 1] if not s.skip},
                   order=[s.name for s in self.recipe[i0:i1 + 1] if not s.skip],
                   trace_mask=dict(self.config.trace_mask), onef_axis=self.config.onef_axis,
                   per_amplifier=self.config.per_amplifier)
        if model is not None or i0 > 0:
            out['resumed_from'] = start or 'a model in memory'
        return plain(out)

    def _inputs(self, seg):
        fn = self.files[seg]
        out = {os.path.basename(fn): f'{os.path.getsize(fn)} bytes'}
        info = self._satmap_expected()
        if info is not None:
            out[os.path.basename(self._satmap_file())] = info[1]
        return out

    def _derived_settings(self):
        """Exposure-wide quantities: ``'auto'`` (found by the pipeline) or set by the user."""
        mask = 'auto'
        if self.trace_mask is not None:
            h = content_id(self.trace_mask, self.bkg_mask)
            mask = 'auto' if h == self._auto_mask else f'user {h}'
        wing = 'auto'
        if self.wing_threshold is not None and self.wing_threshold != self._auto_wing:
            wing = float(self.wing_threshold)
        info = self._satmap_expected()
        return dict(masks=mask, wing_threshold=wing,
                    saturation_map=None if info is None else info[0])

    def _derived_state(self):
        """What the other segments of the exposure need from the first one (saved in the record)."""
        return dict(masks=None if self.trace_mask is None else content_id(self.trace_mask, self.bkg_mask),
                    auto_mask=self._auto_mask, wing_threshold=self.wing_threshold,
                    auto_wing=self._auto_wing, group_range=self.auto_firstgroup)

    def _find_finished(self, segments, settings):
        """Segments of ``segments`` already done: ``{segment: message}``; restores the exposure state.

        Raises :class:`~jstark.reduction.products.ProductMismatchError` if a
        product was made with other settings.
        """
        full = dict(settings, **self._derived_settings())
        cleaned = cleaned_exposures([self.outdir]).get(self.exposure)
        checks = {}
        for seg in range(len(self.files)):
            fn = output_filename(self.files[seg], self.outdir, 'rateints')
            removed = not os.path.isfile(fn) and self._was_cleaned(fn, cleaned)
            c = self.store.check(fn, full, self._inputs(seg), need=('SCI', 'ERR', 'DQ'),
                                 input_files=[self.files[seg]], missing_ok=removed)
            checks[seg] = (c, removed)
        ProductStore.refuse([checks[s][0] for s in segments])
        ok = [s for s in checks if checks[s][0].status in ('done', 'unrecorded')]
        self._restore([(s, checks[s][0]) for s in ok])
        out = {}
        for s in segments:
            if s not in ok:
                continue
            c, removed = checks[s]
            out[s] = loaded_message(c)
            if removed:
                out[s] = (f'{os.path.basename(c.path)} was removed after cleaning; the cleaned segment '
                          f'exists (overwrite=True to process it again)')
            diag = c.path.replace('_rateints.fits', '_stage1diag.npz')
            if os.path.isfile(diag):
                self.diagnostics.files[s] = diag
        return out

    @staticmethod
    def _was_cleaned(fn, cleaned):
        if cleaned is None:
            return False
        name = os.path.basename(fn)
        return any(f == name and c for f, c in zip(cleaned['files'], cleaned['cleaned']))

    def _restore(self, checks):
        """Restore masks, wing threshold and group range from finished segments."""
        states = []
        for seg, c in checks:
            diag = c.path.replace('_rateints.fits', '_stage1diag.npz')
            st = dict((c.record or {}).get('state', {}).get('derived') or {})
            if c.record is None and os.path.isfile(diag):           # made before records existed
                with np.load(diag) as z:
                    if 'trace_mask' in z.files:
                        st['masks'] = st['auto_mask'] = content_id(z['trace_mask'], z['bkg_mask'])
                    if 'group_bkg__wing_threshold' in z.files:
                        st['wing_threshold'] = st['auto_wing'] = float(z['group_bkg__wing_threshold'])
                for e in self.log.entries:
                    if e['step'] == 'find_linear_group_range':
                        st['group_range'] = [e['firstgroup'], e['lastgroup']]
            if st:
                states.append((seg, st, diag))
        if not states:
            return
        seg, st, diag = states[0]
        other = [s for s, t, _ in states[1:] if differences(
            {k: st.get(k) for k in ('masks', 'wing_threshold', 'group_range')},
            {k: t.get(k) for k in ('masks', 'wing_threshold', 'group_range')})]
        if other:
            warnings.warn(f'Segments {[s + 1 for s in other]} were processed with other masks, wing '
                          f'threshold or group range than segment {seg + 1} (processed separately?). '
                          f'Process all segments with overwrite=True to make them consistent.')
        if self.trace_mask is None and st.get('masks') and os.path.isfile(diag):
            with np.load(diag) as z:
                if 'trace_mask' in z.files:
                    self.trace_mask, self.bkg_mask = z['trace_mask'], z['bkg_mask']
                    self._auto_mask = st.get('auto_mask')
                    self._restored.append('mask')
                    if 'signal_image' in z.files:
                        self.signal_image = z['signal_image']
        if self.wing_threshold is None and st.get('wing_threshold') is not None:
            self.wing_threshold, self._auto_wing = st['wing_threshold'], st.get('auto_wing')
            self._restored.append('wing')
        if self.auto_firstgroup is None and st.get('group_range') is not None:
            self.auto_firstgroup = tuple(st['group_range'])
            self._restored.append('groups')

    def _drop_restored(self):
        """Forget quantities restored from disk (not set by the user): everything is made again."""
        if 'mask' in self._restored and self.trace_mask is not None and \
                content_id(self.trace_mask, self.bkg_mask) == self._auto_mask:
            self.trace_mask = self.bkg_mask = self.signal_image = self._auto_mask = None
        if 'wing' in self._restored and self.wing_threshold == self._auto_wing:
            self.wing_threshold = self._auto_wing = None
        if 'groups' in self._restored:
            self.auto_firstgroup = None
        self._restored = []

    def run_step(self, name, model, segment=None, **kwargs):
        """Run a single step (jwst or jstark) on ``model`` and return the result."""
        if name == 'consistent_saturation':
            out = self._consistent_saturation(model, segment=segment, **kwargs)
        elif name == 'custom_bias':
            out = self._custom_bias(model, segment=segment, **kwargs)
        elif name == 'group_bkg':
            out = self._group_bkg(model, segment=segment, **kwargs)
        elif name == 'temporal_jump':
            out = self._temporal_jump(model, segment=segment, **kwargs)
        else:
            if name == 'ramp_fit' and 'auto' in (kwargs.get('firstgroup'), kwargs.get('lastgroup')):
                kwargs = dict(kwargs)
                first, last = self.find_linear_group_range(model, **kwargs.pop('auto_kwargs', {}))
                if kwargs.get('firstgroup') == 'auto':
                    kwargs['firstgroup'] = first
                if kwargs.get('lastgroup') == 'auto':
                    kwargs['lastgroup'] = last
            kwargs = {k: v for k, v in kwargs.items() if k != 'auto_kwargs'}
            if name == 'ramp_fit':
                kwargs = self._check_group_range(model, kwargs)
            out = self._jwst_step(name, model, **kwargs)
        self.log.add(name, segment=segment, **kwargs)
        return out

    def _check_group_range(self, model, kwargs, min_groups=3):
        """Guard ``firstgroup`` / ``lastgroup`` against ramps that are too short.

        jwst ignores the selection (and silently uses all groups) when
        ``firstgroup >= lastgroup``, and leaves pixels without a rate when
        only one usable group remains. If fewer than ``min_groups`` groups
        that are not flagged in every integration would be left, the
        selection is dropped with a warning.
        """
        first, last = kwargs.get('firstgroup'), kwargs.get('lastgroup')
        if first is None and last is None:
            return kwargs
        ngroup = model.data.shape[1]
        lo = 0 if first is None else int(first)
        hi = ngroup - 1 if last is None else min(int(last), ngroup - 1)
        always_flagged = np.all((np.asarray(model.groupdq[:, :, 0, 0]) & 1) > 0, axis=0) \
            if model.groupdq.shape[0] else np.zeros(ngroup, bool)
        usable = [g for g in range(lo, hi + 1) if not always_flagged[g]]
        if len(usable) < min_groups:
            warnings.warn(f'ramp_fit: firstgroup={first}, lastgroup={last} would leave {len(usable)} usable '
                          f'group(s) of {ngroup}; ignoring the group selection and fitting all usable groups.')
            kwargs = {k: v for k, v in kwargs.items() if k not in ('firstgroup', 'lastgroup')}
        return kwargs

    def find_linear_group_range(self, model=None, tol=0.005, nrun=5, recompute=False):
        """Range of groups over which the ramps are linear (e.g. MIRI RSCD / end-of-ramp effects).

        The median (over integrations, excluding the first one) group-to-group
        increase is summed over the source pixels and compared with the
        plateau (median over the middle third of the ramp). The first group
        is the first ``k`` from which ``nrun`` consecutive increases stay
        within ``tol`` of the plateau; the last group is found in the same way
        from the end of the ramp. Groups already flagged (e.g. by ``rscd``,
        ``firstframe``, ``lastframe``) are excluded. The result is computed
        once (on the first segment processed) and reused, so the whole
        exposure uses the same groups.

        Use it with ``s1.set('ramp_fit', firstgroup='auto')`` (and/or
        ``lastgroup='auto'``; tune with ``auto_kwargs={'tol': 0.01}``).
        Caveat: a dark reference (``dark_current``) adds a fixed
        group-to-group pattern of ~1-3 per cent to the summed increases of
        MIRI ramps, which can hide sub-per-cent curvature; a larger ``tol``
        is then needed.

        Returns
        -------
        (int, int)
            Zero-indexed ``(firstgroup, lastgroup)`` for ``ramp_fit``.
        """
        if self.auto_firstgroup is not None and not recompute:
            return self.auto_firstgroup
        if model is None:
            raise ValueError('A ramp model is needed to determine the linear group range.')
        self._ensure_masks(model)
        nint, ng = model.data.shape[:2]
        sel = slice(1, min(nint, 201)) if nint > 2 else slice(None)
        d = np.array(model.data[sel], dtype=np.float32)
        d[(np.asarray(model.groupdq[sel]) & 3) > 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            inc = np.diff(np.nanmedian(d, axis=0), axis=0)              # increase k: group k -> k+1
            tot = np.nansum(np.where(self.trace_mask[None], inc, 0.), axis=(1, 2))
        valid = np.isfinite(tot) & (tot != 0)
        k_valid = np.where(valid)[0]
        if len(k_valid) < 3 * nrun:
            first, last = (int(k_valid[0]), int(k_valid[-1]) + 1) if len(k_valid) else (0, ng - 1)
            self.auto_firstgroup = (first, last)
            return self.auto_firstgroup
        n = len(k_valid)
        plateau = np.median(tot[k_valid[n // 3: 2 * n // 3]])
        ok = valid & (np.abs(tot / plateau - 1) < tol)
        run = np.convolve(ok.astype(int), np.ones(nrun, int), mode='valid') == nrun   # run starting at k
        starts = np.where(run)[0]
        if len(starts) == 0:
            warnings.warn(f'No linear stretch of {nrun} increases within {tol * 100:g}% found; '
                          f'using all unflagged groups.')
            first, last = int(k_valid[0]), int(k_valid[-1]) + 1
        else:
            first = int(starts[0])                         # fit starts at group first
            last = int(starts[-1] + nrun)                  # ... and ends at group last (inclusive)
        self.auto_firstgroup = (first, last)
        if self.verbose:
            print(f'    ramp_fit: linear group range = {first}-{last} of 0-{ng - 1} (increases within '
                  f'{tol * 100:g}% of the plateau; used for the whole exposure)')
        self.log.add('find_linear_group_range', segment=None, tol=tol, firstgroup=first, lastgroup=last)
        return self.auto_firstgroup

    def _jwst_step(self, name, model, **kwargs):
        _, Detector1Pipeline = _jwst_imports()
        cls = Detector1Pipeline.step_defs[name]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = cls.call(model, skip=False, configure_log=self._configure_jwst_log, **kwargs)
        if name == 'ramp_fit':
            rate, rateints = result
            if rateints is None:
                raise RuntimeError('Ramp fitting did not produce integration-level output.')
            del rate
            result = rateints
        if result is not model:
            try:
                model.close()
            except Exception:
                pass
        return result

    def _diag(self, segment, step):
        return self.diagnostics.setdefault(segment, {}).setdefault(step, {})

    # ------------------------------------------------------------------
    # Custom steps
    # ------------------------------------------------------------------
    def _ensure_masks(self, model):
        if self.trace_mask is None or self.bkg_mask is None:
            self.build_trace_mask(model)
            self._auto_mask = content_id(self.trace_mask, self.bkg_mask)

    def _crds_superbias(self, model):
        from jwst.superbias import SuperBiasStep
        from jwst.lib.reffile_utils import get_subarray_model
        datamodels, _ = _jwst_imports()
        fname = SuperBiasStep().get_reference_file(model, 'superbias')
        with datamodels.SuperBiasModel(fname) as ref:
            sub = get_subarray_model(model, ref)
            return np.array(sub.data, dtype=np.float32), np.array(sub.dq)

    def _readnoise(self, model):
        from jwst.ramp_fitting import RampFitStep
        from jwst.lib.reffile_utils import get_subarray_model
        datamodels, _ = _jwst_imports()
        fname = RampFitStep().get_reference_file(model, 'readnoise')
        with datamodels.ReadnoiseModel(fname) as ref:
            sub = get_subarray_model(model, ref)
            return np.array(sub.data, dtype=np.float32)

    @staticmethod
    def _first_saturated_group(sat_counts, nint, min_frac):
        """First group saturated in at least ``min_frac`` of the integrations (ngroup if none)."""
        ngroup = sat_counts.shape[0]
        frac = sat_counts / max(nint, 1)
        hit = frac >= min_frac
        first = np.where(hit.any(axis=0), np.argmax(hit, axis=0), ngroup)
        return first.astype(np.int16)

    def _satmap_file(self):
        return os.path.join(self.outdir, f'{self.exposure}_saturation_map.fits')

    def _satmap_settings(self, segments, min_frac):
        """Settings that define a saturation map."""
        stop = self.step_names.index('saturation')
        segs = 'all' if sorted(segments) == list(range(len(self.files))) else sorted(segments)
        return plain(dict(segments=segs, min_frac=min_frac,
                          steps={s.name: dict(s.kwargs) for s in self.recipe[:stop + 1]
                                 if not s.skip and not s.is_custom}))

    def _nint_total(self):
        n = 0
        for f in self.files:
            h = fits.getheader(f, 0)
            n += int(h['INTEND']) - int(h['INTSTART']) + 1 if 'INTEND' in h else int(h.get('NINTS', 0))
        return n

    def _load_saturation_map(self, segments, min_frac):
        """Load the map on disk if it covers ``segments`` (or the whole exposure). Returns success."""
        fn = self._satmap_file()
        need = ('SATCOUNT', 'STATIC')
        c = None
        for segs in (list(segments), list(range(len(self.files)))):
            want = self._satmap_settings(segs, min_frac)
            c = self.store.check(fn, want, need=need)
            if c.status in ('done', 'missing', 'incomplete'):
                break
            if c.status == 'unrecorded':
                with fits.open(fn) as h:
                    ok = (h[0].header.get('MINFRAC') == min_frac
                          and h[0].header.get('NINT') == self._nint_total())
                want = self._satmap_settings(range(len(self.files)), min_frac)
                if not ok:              # cannot tell which segments it was made from
                    c = Check('incomplete', fn, [], None)
                break
        if c.status in ('missing', 'incomplete'):
            return False
        if c.status == 'different':
            # a map of other segments (a test) is replaced; other settings are not
            if all(d.startswith('segments') for d in c.diffs):
                return False
            ProductStore.refuse(c)
        with fits.open(fn) as h:
            self.saturation_map = np.asarray(h[0].data, np.int16)
            self._saturation_counts = (np.asarray(h['SATCOUNT'].data, np.int32), int(h[0].header['NINT']))
            self._saturation_static = np.asarray(h['STATIC'].data, np.int16)
        self._satmap_info = (want, self.store.uid(fn))
        self._auto_satmap = content_id(self.saturation_map)
        if self.verbose:
            n = int((self.saturation_map < self._saturation_counts[0].shape[0]).sum())
            print(f'  {loaded_message(c)}: {n} pixels saturate')
        return True

    def _satmap_expected(self):
        """``(description, identifier)`` of the exposure-wide saturation map a run uses, or None."""
        spec = next((s for s in self.recipe if s.name == 'consistent_saturation'), None)
        if (spec is None or spec.skip or spec.kwargs.get('saturation_map') is not None
                or spec.kwargs.get('scope', 'exposure') != 'exposure'):
            return None
        if self.saturation_map is not None:
            h = content_id(np.asarray(self.saturation_map, np.int16))
            if h != self._auto_satmap:
                return f'user {h}', None
            return {k: self._satmap_info[0][k] for k in ('segments', 'min_frac')}, self._satmap_info[1]
        want = self._satmap_settings(range(len(self.files)), spec.kwargs.get('min_frac', 0.01))
        fn = self._satmap_file()
        uid = self.store.uid(fn) if self.store.check(fn, want).status == 'done' else None
        return {k: want[k] for k in ('segments', 'min_frac')}, uid

    def compute_saturation_map(self, segments=None, min_frac=0.01, save=True, overwrite=False):
        """Pre-pass: first consistently saturated group of every pixel over the exposure.

        Runs only the recipe steps up to and including ``saturation`` (with
        their current parameters) on every segment and counts, for each pixel
        and group, in how many integrations the group is flagged SATURATED.
        A pixel is considered saturated from group ``g`` on if group ``g`` is
        saturated in at least ``min_frac`` of all integrations (so that
        saturation by single cosmic rays does not count).

        Static bad pixels (``DO_NOT_USE`` in the pixel DQ after ``dq_init``,
        i.e. dead / hot / low-QE pixels of the CRDS mask) that the
        ``saturation`` step flags, and the neighbours that saturate at the
        same group only because the step grows the flag around them
        (``n_pix_grow_sat``), are not saturation by the scene. They are
        kept for ``consistent_saturation`` (Stage 1 processing is the same)
        but left out of the returned map, which is the one saved for
        extraction (``min_groups`` would otherwise blank whole columns);
        they are saved in the ``STATIC`` extension.

        With ``overwrite=False`` a saved map made with the same settings is
        loaded instead, if it was made from ``segments`` or from the whole
        exposure (a map made from other segments only is replaced).

        Returns
        -------
        ndarray
            ``(ny, nx)`` map of the first saturated group by the scene
            (``ngroup`` = never).
        """
        segments = list(range(len(self.files))) if segments is None else list(segments)
        if not overwrite and self._load_saturation_map(segments, min_frac):
            return self.saturation_map
        names = self.step_names
        stop = names.index('saturation')
        counts, nint, static_dq = None, 0, None
        for seg in segments:
            m = self.load(seg)
            for spec in self.recipe[:stop + 1]:
                if spec.skip or spec.is_custom:
                    continue
                m = self._jwst_step(spec.name, m, **spec.kwargs)
                if spec.name == 'dq_init' and static_dq is None:
                    static_dq = (np.asarray(m.pixeldq) & 1) > 0
            sat = (np.asarray(m.groupdq) & 2) > 0
            c = sat.sum(axis=0).astype(np.int32)
            counts = c if counts is None else counts + c
            nint += sat.shape[0]
            m.close()
            if self.verbose:
                print(f'  saturation pre-pass: segment {seg + 1}/{len(self.files)}')
        full = self._first_saturated_group(counts, nint, min_frac)
        ngroup = counts.shape[0]
        grow = int(self.recipe[stop].kwargs.get('n_pix_grow_sat', 1))
        static = self._static_saturation(full, static_dq, ngroup, grow)
        self.saturation_map = np.where(static < ngroup, ngroup, full).astype(np.int16)
        self._saturation_static = static
        self._saturation_counts = (counts, nint)
        self._auto_satmap = content_id(self.saturation_map)
        uid = content_id(self.saturation_map, static)
        self._satmap_info = (self._satmap_settings(segments, min_frac), uid)
        n = int((self.saturation_map < ngroup).sum())
        ns = int((static < ngroup).sum())
        if self.verbose:
            print(f'  saturation map: {n} pixels saturate (in >= {min_frac * 100:g}% of integrations); '
                  f'first saturated group histogram {np.bincount(self.saturation_map.ravel(), minlength=ngroup + 1).tolist()}')
            if ns:
                print(f'  saturation map: {ns} more pixels are static bad pixels (DO_NOT_USE) or '
                      f'flagged only by growth around them; left out (STATIC extension)')
        if save:
            fn = self._satmap_file()
            self.store.begin(fn)
            hdr = fits.Header()
            hdr['MINFRAC'] = (min_frac, 'min fraction of integrations saturated')
            hdr['NINT'] = nint
            hdr['NGROUPS'] = (ngroup, 'groups per integration (map value = never saturated)')
            hdr['NSTATIC'] = (ns, 'static bad / grown pixels left out (STATIC ext.)')
            fits.HDUList([fits.PrimaryHDU(self.saturation_map, header=hdr),
                          fits.ImageHDU(counts, name='SATCOUNT'),
                          fits.ImageHDU(static, name='STATIC')]).writeto(fn, overwrite=True)
            self.store.done(fn, self._satmap_info[0], uid=uid)
        self.log.add('compute_saturation_map', segment=None, min_frac=min_frac, n_pixels=n,
                     n_static=ns)
        return self.saturation_map

    @staticmethod
    def _static_saturation(first, static_dq, ngroup, grow=1):
        """First saturated group of static bad pixels and of their grown neighbours.

        A neighbour within ``grow`` pixels of a saturating static bad pixel
        counts as grown when it first saturates at the same group (its own
        saturation would come earlier or later). Returns a map with
        ``ngroup`` everywhere else.
        """
        out = np.full(first.shape, ngroup, np.int16)
        if static_dq is None:
            return out
        seeds = static_dq & (first < ngroup)
        ny, nx = first.shape
        for y, x in zip(*np.where(seeds)):
            g = first[y, x]
            y0, y1 = max(0, y - grow), min(ny, y + grow + 1)
            x0, x1 = max(0, x - grow), min(nx, x + grow + 1)
            blk = first[y0:y1, x0:x1]
            sel = (blk == g) | static_dq[y0:y1, x0:x1] & (blk < ngroup)
            sub = out[y0:y1, x0:x1]
            sub[sel] = np.minimum(sub[sel], blk[sel])
        return out

    def _consistent_saturation(self, model, segment=None, min_frac=0.01, scope='exposure',
                               saturation_map=None):
        """Use the same set of groups for every pixel in every integration.

        Partially saturated pixels otherwise switch between different numbers
        of fitted groups from one integration to the next, which changes
        their rate by several per cent (residual non-linearity near
        saturation, charge spilling from saturated neighbours), and the
        switching can correlate with the transit itself. Groups from the
        pixel's first consistently saturated group on are flagged SATURATED
        in all integrations.

        Parameters
        ----------
        min_frac : float
            Fraction of integrations in which a group must be saturated.
        scope : {'exposure', 'segment'}
            ``'exposure'`` (default) uses a map built from all segments by
            :meth:`compute_saturation_map` (computed automatically the first
            time); ``'segment'`` uses only the current segment (cheaper, but
            the group set can change at segment boundaries).
        saturation_map : ndarray or str or None
            User-supplied map (first saturated group per pixel) or FITS file.
        """
        gdq = model.groupdq
        nint, ngroup = gdq.shape[:2]
        if saturation_map is not None:
            smap = fits.getdata(saturation_map) if isinstance(saturation_map, str) else saturation_map
        elif scope == 'exposure':
            if self.saturation_map is None:
                if self.verbose:
                    print('  consistent_saturation: building the exposure-wide saturation map')
                self.compute_saturation_map(min_frac=min_frac, overwrite=self._overwrite)
            smap = self.saturation_map
            static = getattr(self, '_saturation_static', None)
            if static is not None:                   # static bad pixels stay consistent too
                smap = np.minimum(smap, static)
        elif scope == 'segment':
            counts = ((np.asarray(gdq) & 2) > 0).sum(axis=0)
            smap = self._first_saturated_group(counts, nint, min_frac)
        else:
            raise ValueError("scope must be 'exposure' or 'segment'.")
        smap = np.asarray(smap)
        g = np.arange(ngroup)[:, None, None]
        flag = g >= smap[None]                                       # (ngroup, ny, nx)
        before = int(((np.asarray(gdq) & 2) > 0).sum())
        gdq |= (flag[None] * np.uint8(2)).astype(gdq.dtype)
        after = int(((np.asarray(gdq) & 2) > 0).sum())
        info = self._diag(segment, 'consistent_saturation')
        info.update({'saturation_map': smap.astype(np.int16), 'n_groups_added': after - before})
        if self.verbose:
            print(f'    consistent_saturation: {int((smap < ngroup).sum())} pixels, '
                  f'{after - before} saturated-group flags added')
        return model

    def _custom_bias(self, model, segment=None, method='scaled', mode='smooth', window=51,
                     bias=None, propagate_dq=True):
        """Custom bias subtraction (replaces the ``superbias`` step).

        Parameters
        ----------
        method : {'scaled', 'self', 'user'}
            See :mod:`jstark.reduction.bias`.
        mode : {'per_int', 'mean', 'smooth'}
            Combination of the scale factors for ``method='scaled'``.
        window : int
            Smoothing window (integrations) for ``mode='smooth'``.
        bias : ndarray or str or None
            User bias image or FITS file (``method='user'``).
        propagate_dq : bool
            OR the CRDS superbias DQ into the pixel DQ (``'scaled'``).
        """
        info = self._diag(segment, 'custom_bias')
        info['method'] = method
        if method == 'scaled':
            sb, sbdq = self._crds_superbias(model)
            # mask for the ratio: use a quick mask from the raw data if needed
            if self.bkg_mask is None:
                tmp = np.array(model.data[:, :, :, :], dtype=np.float32)
                tmp -= sb[None, None]
                img = quick_signal_image(tmp, getattr(model, 'groupdq', None))
                del tmp
                valid = self._valid_pixels(model)
                tm = build_trace_mask(img, self.config.dispersion_axis, valid=valid,
                                      **{k: v for k, v in self.config.trace_mask.items()
                                         if k != 'min_pixels'})
                bkg = ~tm & valid
            else:
                bkg = self.bkg_mask
            sf = superbias_scale_factors(np.asarray(model.data[:, 0]), sb, bkg, mode=mode,
                                         window=window)
            apply_bias(model.data, sb, sf)
            if propagate_dq:
                model.pixeldq |= sbdq.astype(model.pixeldq.dtype)
            info['scale'] = sf
        elif method == 'self':
            nframes = getattr(model.meta.exposure, 'nframes', 1) or 1
            groupgap = getattr(model.meta.exposure, 'groupgap', 0) or 0
            b = self_superbias(model.data, getattr(model, 'groupdq', None), nframes, groupgap)
            # fall back to CRDS superbias where no estimate is possible
            if np.any(~np.isfinite(b)):
                sb, _ = self._crds_superbias(model)
                b = np.where(np.isfinite(b), b, sb)
            apply_bias(model.data, b)
            info['bias'] = b
        elif method == 'user':
            if isinstance(bias, str):
                bias = fits.getdata(bias)
            if bias is None:
                raise ValueError("method='user' needs a bias image.")
            apply_bias(model.data, np.asarray(bias, dtype=np.float32))
        else:
            raise ValueError(f'Unknown custom bias method {method}.')
        model.meta.cal_step.superbias = 'COMPLETE'
        return model

    def _group_bkg(self, model, segment=None, method='median', axis=None, nsigma=3., maxiters=3,
                   poly_order=1, per_amplifier=None, min_pixels=None, chunk=50, keep_profile=True,
                   use_source_pixels=False, readout_order=3, readout_direct=True,
                   wing_max=None, wing_frac=None, wing_min_pixels=100):
        """Group-level background / 1/f subtraction (see :mod:`jstark.reduction.onef`).

        Parameters
        ----------
        method : {'median', 'poly', 'diff', 'readout', 'rowcol'}
        axis : {'column', 'row'} or None
            Stripe direction (mode default if ``None``).
        per_amplifier : bool or None
            Row-wise estimate per amplifier (mode default if ``None``).
        min_pixels : int or None
            Minimum number of background pixels per amplifier-row segment.
        use_source_pixels : bool
            ``method='diff'`` only: use all pixels (including the source)
            with inverse-variance weights.
        readout_order, readout_direct
            ``method='readout'`` only, see
            :func:`~jstark.reduction.onef.group_background`.
        wing_max : float or None
            ``method='diff'`` / ``'readout'``: also measure the offsets on
            trace pixels fainter than this (DN per group).
        wing_frac : float or None
            As ``wing_max``, but relative: the threshold is ``wing_frac`` times
            the peak signal per group of the source. It is measured on the
            first segment processed and reused for all segments
            (``self.wing_threshold``; set it to impose a value).
        wing_min_pixels : int
            With ``wing_max`` / ``wing_frac``: minimum number of wing pixels
            for an amplifier-row without background pixels to be measured
            directly.
        """
        axis = axis or self.config.onef_axis or 'column'
        self._ensure_masks(model)
        per_amp = self.config.per_amplifier if per_amplifier is None else per_amplifier
        amps = amp_bounds_from_model(model) if (per_amp and axis == 'row') else None
        if min_pixels is None:
            min_pixels = self.config.trace_mask.get('min_pixels', 4)
        groupdq = getattr(model, 'groupdq', None)
        if wing_max is None and wing_frac is not None and self.wing_threshold is not None:
            wing_max = self.wing_threshold            # same threshold for every segment
        _, info = group_background(model.data, self.bkg_mask, method=method, axis=axis,
                                   groupdq=groupdq, nsigma=nsigma, maxiters=maxiters,
                                   poly_order=poly_order, amp_bounds=amps, min_pixels=min_pixels,
                                   source_mask=self.trace_mask, chunk=chunk,
                                   use_source_pixels=use_source_pixels,
                                   readout_order=readout_order, readout_direct=readout_direct,
                                   wing_max=wing_max, wing_frac=wing_frac,
                                   wing_min_pixels=wing_min_pixels)
        if wing_frac is not None and self.wing_threshold is None:
            self.wing_threshold = self._auto_wing = info['wing_threshold']
            if self.verbose:
                print(f'  group_bkg: wing threshold {self.wing_threshold:.0f} DN/group '
                      f'({wing_frac:g} x peak {info["wing_peak"]:.0f} DN/group)')
        d = self._diag(segment, 'group_bkg')
        d.update({k: v for k, v in info.items() if k not in ('median_frames', 'weights')})
        if not keep_profile:
            d.pop('profile', None)
        return model

    def _temporal_jump(self, model, segment=None, window=11, threshold=10., mode='auto',
                       two_sided=True, flag_neighbors=False, neighbor_threshold=None,
                       readnoise_floor=True, max_mem_gb=4.):
        """Time-domain jump detection (see :func:`jstark.reduction.jump.temporal_jump`).

        Parameters
        ----------
        readnoise_floor : bool
            Use ``sqrt(2) * readnoise`` (CRDS) as a floor on the noise of a
            group difference.
        """
        floor = None
        if readnoise_floor:
            try:
                floor = np.sqrt(2.) * self._readnoise(model)
            except Exception as e:  # pragma: no cover - CRDS access problems
                warnings.warn(f'Could not get the read noise ({e}); no noise floor used.')
        _, _, info = temporal_jump(model.data, model.groupdq, window=window, threshold=threshold,
                                   mode=mode, two_sided=two_sided, flag_neighbors=flag_neighbors,
                                   neighbor_threshold=neighbor_threshold, sigma_floor=floor,
                                   max_mem_gb=max_mem_gb)
        jm = info.pop('jump_mask')
        info['jump_count_map'] = jm.sum(axis=(0, 1)).astype(np.int32)
        info['jumps_per_group'] = jm.sum(axis=(0, 2, 3)).astype(np.int64)
        del jm
        self._diag(segment, 'temporal_jump').update(info)
        if self.verbose:
            print(f'    temporal_jump: {info["n_jumps"]} jumps ({info["mode"]}), '
                  f'median fraction of pixels hit per integration = '
                  f'{np.median(info["frac_per_int"]) * 100:.3f}%')
        model.meta.cal_step.jump = 'COMPLETE'
        return model

    # ------------------------------------------------------------------
    def _save_diagnostics(self, segment):
        diag = self.diagnostics.get(segment, {})
        out = {}
        for step, info in diag.items():
            for k, v in info.items():
                if isinstance(v, (np.ndarray, float, int, np.floating, np.integer)):
                    out[f'{step}__{k}'] = np.asarray(v)
        if self.trace_mask is not None:
            out['trace_mask'] = self.trace_mask
            out['bkg_mask'] = self.bkg_mask
        if self.signal_image is not None:
            out['signal_image'] = self.signal_image
        if out:
            fn = output_filename(self.files[segment], self.outdir, 'stage1diag').replace('.fits', '.npz')
            np.savez_compressed(fn, **out)
