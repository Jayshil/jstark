"""Records of the products on disk, so that finished steps can be loaded instead of re-run.

Every output folder holds one ``<exposure>_products.json`` per exposure. For
each product (file) it stores the settings it was made with, an identifier
of its content, the identifiers of the files it was made from, and whether it
was written completely. The ``run`` methods of jstark use it when
``overwrite=False``:

* complete product, same settings and inputs: loaded;
* no product, or one whose writing was interrupted: made;
* different settings or inputs: :class:`ProductMismatchError`, nothing is touched;
* product without a record (made before records existed): loaded with a
  notice, after a check that the file is complete.

Not detected: changes of the jstark code itself or of the CRDS context.
"""
import hashlib
import json
import os
import time
import warnings
from collections import namedtuple
from fnmatch import fnmatch
from glob import glob

import numpy as np
from astropy.io import fits

from .io import find_exposures

__all__ = ['ProductStore', 'ProductMismatchError', 'content_id', 'plain', 'differences',
           'fits_complete', 'cleaned_exposures', 'input_exposures']

# parameters that do not change the result
IGNORED = ('verbose', 'maximum_cores', 'max_mem_gb', 'chunk')

Check = namedtuple('Check', 'status path diffs record')


class ProductMismatchError(RuntimeError):
    """An existing product was made with other settings or inputs."""


def content_id(*arrays):
    """Short hash of the content (bytes, shapes and dtypes) of arrays, ``None`` allowed."""
    h = hashlib.blake2b(digest_size=8)
    for a in arrays:
        if a is None:
            h.update(b'none')
            continue
        a = np.asarray(a)
        if a.dtype.byteorder == '>' or (a.dtype.byteorder == '=' and not np.little_endian):
            a = a.astype(a.dtype.newbyteorder('<'))          # FITS arrays are big-endian
        h.update(f'{a.shape}{a.dtype.kind}{a.dtype.itemsize}'.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def plain(v):
    """JSON-able form of a setting (large arrays are replaced by a hash of their content)."""
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items() if k not in IGNORED}
    if isinstance(v, (set, frozenset)):
        return sorted(plain(x) for x in v)
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 20 else f'array {v.shape} {content_id(v)}'
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return repr(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if callable(v):
        return getattr(v, '__qualname__', repr(type(v)))
    return str(v)


def _flat(v, prefix=''):
    if isinstance(v, dict) and v:
        out = {}
        for k, x in v.items():
            out.update(_flat(x, f'{prefix}.{k}' if prefix else str(k)))
        return out
    return {prefix: v}


_ABSENT = object()


def differences(old, new):
    """``['name: old -> new', ...]`` for two (nested) settings dictionaries."""
    a, b = _flat(plain(old or {})), _flat(plain(new or {}))
    out = []
    for k in list(a) + [k for k in b if k not in a]:
        x, y = a.get(k, _ABSENT), b.get(k, _ABSENT)
        if x is _ABSENT or y is _ABSENT or x != y:
            show = lambda z: '(not set)' if z is _ABSENT else _short(z)
            if k:
                out.append(f'{k}: {show(x)} -> {show(y)}')
    return out


def input_differences(old, new):
    """As :func:`differences`, for inputs (``{name: identifier}``; unknown new ones are not compared)."""
    old, new = old or {}, plain(new or {})
    out = []
    for k in new:
        if k not in old:
            out.append(f'input {k}: added')
        elif new[k] is not None and old[k] != new[k]:
            out.append(f'input {k}: changed')
    return out + [f'input {k}: no longer used' for k in old if k not in new]


def _short(v, n=80):
    s = repr(v)
    return s if len(s) <= n else s[:n - 3] + '...'


def fits_complete(fname, need=()):
    """``True`` if a FITS file is not truncated and has the extensions ``need``."""
    try:
        size = os.path.getsize(fname)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with fits.open(fname, memmap=True) as h:
                names = [hdu.name for hdu in h]
                for hdu in h:
                    info = hdu.fileinfo()
                    if info is not None and info['datLoc'] + info['datSpan'] > size:
                        return False
    except Exception:
        return False
    return all(n in names for n in need)


class ProductStore:
    """Records of the products of one exposure in one folder (``<exposure>_products.json``).

    The file is read before and replaced after every change, so several
    objects (Stage 1, cleaner, extractor, ...) can share it.

    Parameters
    ----------
    outdir : str
    exposure : str
    """

    def __init__(self, outdir, exposure):
        self.outdir, self.exposure = os.path.abspath(outdir), exposure
        self.fname = os.path.join(self.outdir, f'{exposure}_products.json')

    def __repr__(self):
        return '\n'.join(f'{r.get("time", "")}  {r["status"]:8s} {k}'
                         for k, r in self.read().items()) or '<no recorded products>'

    # ------------------------------------------------------------------
    def read(self):
        """All records (``{file name: record}``)."""
        if not os.path.isfile(self.fname):
            return {}
        try:
            with open(self.fname) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            warnings.warn(f'Could not read {self.fname}; existing products are treated as unrecorded.')
            return {}

    def _write(self, recs):
        tmp = f'{self.fname}.{os.getpid()}.tmp'
        with open(tmp, 'w') as f:
            json.dump(recs, f, indent=1)
        os.replace(tmp, self.fname)

    def get(self, path):
        """Record of a product, or ``None``."""
        return self.read().get(os.path.basename(path))

    def begin(self, path):
        """Mark a product as being written (an interrupted run leaves it in this state)."""
        recs = self.read()
        recs[os.path.basename(path)] = dict(status='writing', time=time.strftime('%Y-%m-%dT%H:%M:%S'))
        self._write(recs)

    def done(self, path, settings=None, inputs=None, uid=None, **state):
        """Record a completely written product.

        ``inputs``: ``{file name: identifier}`` (see :meth:`ids`); ``uid``:
        identifier of the content; ``state``: anything needed to resume.
        """
        recs = self.read()
        recs[os.path.basename(path)] = dict(
            status='done', time=time.strftime('%Y-%m-%dT%H:%M:%S'), stamp=time.time(),
            size=os.path.getsize(path), uid=uid, settings=plain(settings or {}),
            inputs=plain(inputs or {}), state=plain(state))
        self._write(recs)

    def forget(self, *paths):
        """Delete the records of products."""
        recs = self.read()
        n = len(recs)
        for p in paths:
            recs.pop(os.path.basename(p), None)
        if len(recs) != n:
            self._write(recs)

    def uid(self, path):
        """Identifier of a product (``None`` if it has no record)."""
        r = self.get(path)
        if r is None:
            return None
        return r.get('uid') if r['status'] == 'done' else 'incomplete'

    @staticmethod
    def ids(paths, exposure):
        """``{file name: identifier}`` of input files (looked up next to the files)."""
        out, stores = {}, {}
        for p in paths:
            d = os.path.dirname(os.path.abspath(p))
            if d not in stores:
                stores[d] = ProductStore(d, exposure).read()
            r = stores[d].get(os.path.basename(p))
            out[os.path.basename(p)] = None if r is None else \
                (r.get('uid') if r['status'] == 'done' else 'incomplete')
        return out

    # ------------------------------------------------------------------
    def check(self, path, settings=None, inputs=None, need=(), input_files=(), missing_ok=False):
        """State of a product on disk.

        Parameters
        ----------
        path : str
        settings : dict
            The settings a new product would be made with.
        inputs : dict
            Identifiers of its input files. Identifiers unknown *now* (no
            record) are not compared.
        need : sequence of str
            FITS extensions a complete product has (unrecorded products).
        input_files : sequence of str
            Paths of the inputs (an unrecorded product older than a recorded
            input counts as different).
        missing_ok : bool
            The file itself may have been removed on purpose.

        Returns
        -------
        Check
            ``status`` is ``'missing'``, ``'incomplete'``, ``'done'``,
            ``'unrecorded'`` or ``'different'`` (with ``diffs``).
        """
        rec = self.get(path)
        exists = os.path.isfile(path)
        if not exists and not missing_ok:
            return Check('missing', path, [], rec)
        if rec is None:
            if exists and not fits_complete(path, need):
                return Check('incomplete', path, [], None)
            if exists:
                stamp = os.path.getmtime(path)
                newer = []
                for f in input_files:
                    r = ProductStore(os.path.dirname(os.path.abspath(f)), self.exposure).get(f)
                    if r is not None and r.get('stamp', 0.) > stamp + 2:
                        newer.append(f'input {os.path.basename(f)} was made again after it')
                if newer:
                    return Check('different', path, newer, None)
            return Check('unrecorded', path, [], None)
        if rec['status'] != 'done':
            return Check('incomplete', path, [], rec)
        if exists and os.path.getsize(path) != rec['size']:
            return Check('different', path, ['the file changed since it was written by jstark '
                                             f'({rec["size"]} -> {os.path.getsize(path)} bytes)'], rec)
        diffs = differences(rec['settings'], settings or {}) if settings is not None else []
        if inputs is not None:
            diffs += input_differences(rec.get('inputs'), inputs)
        return Check('different' if diffs else 'done', path, diffs, rec)

    @staticmethod
    def refuse(checks, hint='Use overwrite=True to make it again, or another output folder.'):
        """Raise :class:`ProductMismatchError` if any of ``checks`` is ``'different'``."""
        bad = [c for c in (checks if isinstance(checks, (list, tuple)) and not isinstance(checks, Check)
                           else [checks]) if c.status == 'different']
        if not bad:
            return
        d = bad[0].diffs
        txt = '\n'.join(f'  {x}' for x in d[:12]) + (f'\n  ... and {len(d) - 12} more' if len(d) > 12 else '')
        more = f' (and {len(bad) - 1} more file(s))' if len(bad) > 1 else ''
        raise ProductMismatchError(f'{bad[0].path}{more} exists, but was made with other settings or '
                                   f'inputs (old -> new):\n{txt}\n{hint}')


def loaded_message(check, compared=False):
    """One line for the user about a loaded product (``compared``: the parameters stored in an
    unrecorded product itself were compared)."""
    name = os.path.basename(check.path)
    if check.record is not None:
        return f'Loaded existing {name} (made {check.record["time"].replace("T", " ")})'
    try:
        date = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(check.path)))
    except OSError:
        date = 'unknown'
    what = 'only the parameters stored in it were checked' if compared else 'so they were not checked'
    return f'Loaded existing {name} (file date {date}; made before jstark recorded settings, {what})'


def cleaned_exposures(folders):
    """Exposures described by the ``*_cleaninfo.fits`` files in ``folders``.

    Returns
    -------
    dict
        ``{exposure: dict(info, files, cleaned, nint, start, header)}``;
        ``files`` are the names of the input (``rateints``) files, which may
        have been removed after cleaning, and ``cleaned`` the paths of the
        cleaned segments ('' if not cleaned or missing).
    """
    out = {}
    for d in dict.fromkeys(os.path.abspath(f) for f in folders if f):
        for fn in sorted(glob(os.path.join(d, '*_cleaninfo.fits'))):
            if os.path.basename(fn).startswith('._'):
                continue
            try:
                with fits.open(fn) as h:
                    seg = h['SEGMENTS'].data
                    files = [str(x).strip() for x in seg['input']]
                    cl = [str(x).strip() for x in seg['cleaned']]
                    nint = [int(n) for n in seg['nint']]
                    hdr = h[0].header.copy()
            except Exception:
                continue
            cl = [os.path.join(d, c) if c and os.path.isfile(os.path.join(d, c)) else '' for c in cl]
            key = hdr.get('EXPOSURE') or os.path.basename(fn).replace('_cleaninfo.fits', '')
            out.setdefault(key, dict(info=fn, files=files, cleaned=cl, nint=nint, header=hdr))
    return out


def input_exposures(inputs, suffix, outdir=None):
    """Exposures found in ``inputs``, including those whose files were removed after cleaning.

    Returns
    -------
    exposures : dict
        As :func:`~jstark.reduction.io.find_exposures`. For exposures in
        ``removed`` the list holds the paths of all segments, some of which
        no longer exist.
    removed : dict
        Exposures (entries of :func:`cleaned_exposures`) with input files
        that no longer exist but whose cleaned segments do.
    """
    folder = None
    if isinstance(inputs, (list, tuple)):
        folder = os.path.dirname(os.path.abspath(str(inputs[0]))) if len(inputs) else None
    else:
        p = os.path.abspath(str(inputs))
        folder = p if os.path.isdir(p) else os.path.dirname(p)
    try:
        exps, err = find_exposures(inputs, suffix=suffix), None
    except FileNotFoundError as e:
        exps, err = {}, e
    removed = {}
    for key, info in cleaned_exposures([outdir, folder]).items():
        files = [os.path.join(folder, f) for f in info['files']]
        if not files or not all(f.endswith(f'_{suffix}.fits') for f in files):
            continue
        gone = [i for i, f in enumerate(files) if not os.path.isfile(f)]
        if not gone or not all(info['cleaned'][i] for i in gone):
            continue
        if isinstance(inputs, (list, tuple)):       # a selection of files is left alone
            wanted = {os.path.basename(str(f)) for f in inputs}
            if not set(info['files']) <= wanted:
                continue
        elif not os.path.isdir(str(inputs)):
            if not all(fnmatch(f, os.path.basename(str(inputs))) for f in info['files']):
                continue
        if all(os.path.basename(f) in info['files'] for f in exps.get(key, [])):
            exps[key] = files
            removed[key] = info
    if not exps:
        raise err
    return dict(sorted(exps.items())), removed


def load_result(fname, key):
    """Result stored by :func:`save_result` for the same settings ``key`` (None if there is none)."""
    try:
        with open(fname) as f:
            entries = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    for e in entries:
        if not (differences(e['key'], {k: v for k, v in key.items() if k != 'inputs'} |
                            {'inputs': e['key'].get('inputs')})
                or input_differences(e['key'].get('inputs'), key.get('inputs'))):
            return e['result']
    return None


def save_result(fname, key, result):
    """Add a result (JSON-able after ``plain``) with its settings to a file of results."""
    try:
        with open(fname) as f:
            entries = json.load(f)
    except (OSError, json.JSONDecodeError):
        entries = []
    entries = [e for e in entries if differences(e['key'], key)]
    res = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in result.items()}
    entries.append(dict(time=time.strftime('%Y-%m-%dT%H:%M:%S'), key=key, result=plain(res)))
    with open(fname + '.tmp', 'w') as f:
        json.dump(entries, f, indent=1)
    os.replace(fname + '.tmp', fname)
