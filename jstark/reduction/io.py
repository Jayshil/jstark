"""File handling utilities: exposure/segment discovery, naming and logging."""
import json
import os
import re
import time
import warnings
from glob import glob
from pathlib import Path

import numpy as np
from astropy.io import fits

__all__ = ['parse_jwst_filename', 'find_exposures', 'select_exposure', 'output_filename',
           'StepLog', 'read_int_times']

_FNAME_RE = re.compile(
    r'^(?P<root>jw\d{11}_\d{5}_\d{5})(?:-seg(?P<seg>\d{3}))?_(?P<det>[a-z0-9]+)_(?P<suffix>[a-z0-9_]+)\.fits$')


def parse_jwst_filename(fname):
    """Split a JWST file name into its components.

    Returns
    -------
    dict or None
        Dictionary with keys ``root``, ``seg`` (int or None), ``det`` and
        ``suffix``; ``None`` if the name does not follow the JWST convention.
    """
    m = _FNAME_RE.match(os.path.basename(fname))
    if m is None:
        return None
    d = m.groupdict()
    d['seg'] = int(d['seg']) if d['seg'] is not None else None
    return d


def find_exposures(inputs, suffix='uncal'):
    """Group files into exposures, each with an ordered list of segments.

    Parameters
    ----------
    inputs : str, Path or list
        A directory, a glob pattern, a single file, or a list of files.
    suffix : str
        File suffix to look for when ``inputs`` is a directory
        (e.g. ``'uncal'``, ``'rateints'``).

    Returns
    -------
    dict
        ``{exposure_id: [segment files sorted by segment number]}`` where
        ``exposure_id`` is ``<root>_<detector>``.
    """
    if isinstance(inputs, (list, tuple)):
        files = [str(f) for f in inputs]
    else:
        inputs = str(inputs)
        if os.path.isdir(inputs):
            files = glob(os.path.join(inputs, f'*_{suffix}.fits'))
        elif any(c in inputs for c in '*?['):
            files = glob(inputs)
        else:
            files = [inputs]
    if len(files) == 0:
        raise FileNotFoundError(f'No *_{suffix}.fits files found in {inputs}.')

    exposures = {}
    for f in files:
        p = parse_jwst_filename(f)
        key = f"{p['root']}_{p['det']}" if p is not None else Path(f).stem
        exposures.setdefault(key, []).append(f)
    for key, flist in exposures.items():
        flist.sort(key=lambda f: (parse_jwst_filename(f) or {}).get('seg') or 0)
    return dict(sorted(exposures.items()))


def select_exposure(exposures, exposure=None, verbose=True):
    """Select one exposure from the output of :func:`find_exposures`.

    If ``exposure`` is ``None`` and several exposures are present, the one
    with the largest total number of integrations (i.e. the time series) is
    chosen, and the others are reported.
    """
    if exposure is not None:
        matches = [k for k in exposures if exposure in k]
        if len(matches) != 1:
            raise ValueError(f'Exposure "{exposure}" matches {matches}; available: {list(exposures)}')
        return matches[0], exposures[matches[0]]
    if len(exposures) == 1:
        key = next(iter(exposures))
        return key, exposures[key]
    nints = {}
    for key, flist in exposures.items():
        h = fits.getheader(flist[0], 0)
        nints[key] = h.get('NINTS', 0)
    key = max(nints, key=nints.get)
    if verbose:
        others = ', '.join(f'{k} (NINTS={nints[k]})' for k in exposures if k != key)
        print(f'Several exposures found; using {key} (NINTS={nints[key]}). Ignored: {others}. '
              f'Pass exposure=... to choose another one.')
    return key, exposures[key]


def output_filename(infile, outdir, suffix):
    """Build an output file name by replacing the suffix of a JWST file name."""
    p = parse_jwst_filename(infile)
    base = os.path.basename(infile)
    if p is not None:
        seg = f"-seg{p['seg']:03d}" if p['seg'] is not None else ''
        name = f"{p['root']}{seg}_{p['det']}_{suffix}.fits"
    else:
        name = f'{Path(base).stem}_{suffix}.fits'
    return os.path.join(outdir if outdir is not None else os.path.dirname(infile), name)


def read_int_times(fname):
    """Read the ``INT_TIMES`` table of a JWST file.

    Returns
    -------
    dict
        Columns of the table as numpy arrays (e.g. ``int_mid_BJD_TDB``),
        or an empty dict if the table is missing.
    """
    with fits.open(fname) as hdul:
        if 'INT_TIMES' not in hdul:
            return {}
        tab = hdul['INT_TIMES'].data
        return {name: np.array(tab[name]) for name in tab.names}


class StepLog:
    """A small JSON log recording which steps were run with which parameters.

    Parameters
    ----------
    fname : str or None
        Path of the JSON file. If ``None``, the log is only kept in memory.
    """

    def __init__(self, fname=None):
        self.fname = fname
        self.entries = []
        if fname is not None and os.path.isfile(fname):
            try:
                with open(fname) as f:
                    self.entries = json.load(f)
            except (json.JSONDecodeError, OSError):
                warnings.warn(f'Could not read existing log {fname}; starting a new one.')

    def add(self, step, segment=None, **info):
        entry = {'step': step, 'segment': segment,
                 'time': time.strftime('%Y-%m-%dT%H:%M:%S')}
        entry.update({k: _jsonable(v) for k, v in info.items()})
        self.entries.append(entry)
        self.write()

    def write(self):
        if self.fname is None:
            return
        with open(self.fname, 'w') as f:
            json.dump(self.entries, f, indent=1)

    def __repr__(self):
        lines = [f"{e['time']}  seg={e['segment']}  {e['step']}" for e in self.entries]
        return '\n'.join(lines) if lines else '<empty log>'


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return f'<array shape={v.shape} dtype={v.dtype}>' if v.size > 20 else v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)
