"""Spline PSF models of a spectral trace.

The method follows ``stark`` (A. Brandeker & J. A. Patel,
https://github.com/Jayshil/stark): pixel values normalised by the flux of
their column are modelled with a cubic spline in the spatial offset from the
trace (``LSQUnivariateSpline``, the "1D PSF"), or with a tensor-product spline
in (offset, column) (``LSQBivariateSpline``, the "2D PSF") so that the
profile can change with wavelength. The spectrum is then extracted optimally
with this profile.

Differences from ``stark``:

* The least-squares problem is accumulated chunk by chunk
  (:class:`~jstark.extraction.bspline.NormalSums`), so *all* integrations of
  all segments are used without holding them in memory.
* Optional smoothness penalties on the second differences of the spline
  coefficients (P-splines), separately along the spatial and spectral
  directions. Their strengths are chosen by cross-validation between even
  and odd integrations, computed directly from the accumulated sums. This
  suppresses the oscillations ("waviness") that fine knot spacings
  otherwise produce, which would couple the extracted flux to trace motion.
* Outliers are rejected on normalised residuals ``(data - model) / sigma``
  with a symmetric threshold (``stark`` used ``|r| < median + clip * std`` on
  unweighted residuals, which preferentially rejects the bright core).
* The normalisation is noise-free: median spectrum x per-integration
  white-light factor, instead of each integration's own noisy sum.
* The profile is normalised to unit integral over the PSF window in every
  column, so the flux scale does not depend on the sub-pixel position of the
  trace or on the extraction aperture.
"""
import json

import numpy as np
from astropy.io import fits

from .bspline import (clamped_knots, basis, n_basis, integrals, second_difference,
                      BandedSystem)

__all__ = ['SplinePSF']


class SplinePSF:
    """Tensor-product cubic spline profile ``S(u, x)`` of one trace.

    Parameters
    ----------
    halfwidth : float
        The profile is defined for spatial offsets ``|u| <= halfwidth``.
    ncol : int
        Number of trace columns (the column coordinate is ``0 ... ncol-1``).
    oversample : float
        Knots per pixel along the spatial offset.
    knot_col : int
        Interior knots along the columns; ``0`` gives a column-independent
        (1D) profile.
    """

    def __init__(self, halfwidth, ncol, oversample=2., knot_col=20):
        self.halfwidth = float(halfwidth)
        self.ncol = int(ncol)
        self.oversample = float(oversample)
        self.knot_col = int(knot_col or 0)
        h, s = self.halfwidth, 1. / self.oversample
        self.tu = clamped_knots(-h, h, np.arange(-h + s, h - 0.5 * s, s))
        self.nbu = n_basis(self.tu)
        if self.knot_col > 0:
            self.tx = clamped_knots(0., self.ncol - 1.,
                                    np.linspace(0, self.ncol - 1, self.knot_col + 2)[1:-1])
            self.nbx = n_basis(self.tx)
            self.bx_j0, self.bx = basis(np.arange(self.ncol, dtype=float), self.tx)
        else:
            self.tx = None
            self.nbx = 1
            self.bx_j0 = np.zeros(self.ncol, int)
            self.bx = np.ones((self.ncol, 1))
        self.C = None
        self.smooth = (0., 0.)
        self.fit_info = {}

    # ------------------------------------------------------------------
    @property
    def ncoef(self):
        return self.nbu * self.nbx

    def fit(self, sums, smooth='cv', grid=None, ridge=1e-10, cv='parity', cv_block=None,
            verbose=False):
        """Solve for the spline coefficients from accumulated normal sums.

        Parameters
        ----------
        sums : NormalSums
        smooth : 'cv', float, (float, float) or None
            Penalty strengths ``(lambda_u, lambda_x)`` on the second
            differences of the coefficients along the spatial offset and
            along the columns, in units of the mean diagonal of the normal
            matrix. ``'cv'`` chooses them by cross-validation between even
            and odd integrations; ``0``/``None`` gives the unpenalised
            least-squares fit (as ``stark``).
        grid : sequence or None
            Trial values (relative units) for the cross-validation.
        cv : {'parity', 'columns'}
            ``'parity'`` (default): two-fold cross-validation between the two
            parities of the accumulated sums. The extractor assigns pixels to
            them by **sub-pixel phase** of their offset from the trace
            (fractional part below / above one half; ``psf cv='phase'``,
            default), so the model trained on one half of the phases must
            predict the other: this tests exactly how the profile
            interpolates between sampled phases, which is what matters when
            the trace moves, without penalising real changes of the profile
            along the dispersion. With ``psf cv='ints'`` the parities are
            even / odd integrations (tests only the noise).
            ``'columns'``: alternating blocks of ``cv_block`` columns (tests
            the ability to predict unseen columns, which extraction never
            needs; it over-smooths fast-changing profiles such as
            NIRSpec/PRISM).
        cv_block : int or None
            Block width for ``cv='columns'`` (default ``ncol / 40``, at least 5).
        """
        Mb, r, s = sums.assemble(self.bx_j0, self.bx, self.nbx)
        Mt = BandedSystem(Mb.sum(0))
        rt = r.sum(0).ravel()
        tau = Mt.trace() / Mt.n
        Du, Dx = second_difference(self.nbu), second_difference(self.nbx)

        def system(M, lu, lx):
            S = BandedSystem(M) if not isinstance(M, BandedSystem) else M.copy()
            S.add_u_penalty(tau * lu, Du)
            if self.nbx > 2:
                S.add_x_penalty(tau * lx, Dx)
            S.add_ridge(ridge * tau)
            return S

        info = {}
        if isinstance(smooth, str) and smooth == 'cv':
            grid = np.asarray(grid if grid is not None else 10. ** np.arange(-8, 4.1, 1.))
            gx = grid if self.nbx > 3 else np.array([0.])
            if cv == 'columns':
                blk = cv_block or max(5, self.ncol // 40)
                fold = (np.arange(self.ncol) // blk) % 2
                parts = []
                for k in (0, 1):
                    mb, rk, sk = sums.assemble(self.bx_j0, self.bx, self.nbx, cols=fold == k)
                    parts.append((mb.sum(0), rk.sum(0).ravel(), sk.sum()))
                info['cv'] = f'columns (blocks of {blk})'
            else:
                parts = [(Mb[k], r[k].ravel(), s[k]) for k in (0, 1)]
                info['cv'] = 'parity folds'
            if any(np.sum(p[0]) == 0 for p in parts):
                lu, lx = 1e-4, 1e-4
                info['cv'] = 'skipped (empty fold)'
            else:
                held = [BandedSystem(p[0]) for p in parts]
                score = np.full((len(grid), len(gx)), np.nan)

                def cv_score(i, j):
                    if np.isnan(score[i, j]):
                        tot = 0.
                        for tr, te in ((0, 1), (1, 0)):
                            c = system(parts[tr][0], grid[i], gx[j]).solve(parts[tr][1])
                            tot += parts[te][2] - 2 * c @ parts[te][1] + c @ held[te].matvec(c)
                        score[i, j] = tot
                    return score[i, j]

                # coordinate search over the two penalties (u, then x, until stable)
                i = int(np.argmin(np.abs(np.log10(grid) + 4)))
                j = int(np.argmin(np.abs(np.log10(np.where(gx > 0, gx, 1e-4)) + 4)))
                for _ in range(4):
                    i_new = int(np.argmin([cv_score(k, j) for k in range(len(grid))]))
                    j_new = int(np.argmin([cv_score(i_new, k) for k in range(len(gx))]))
                    if (i_new, j_new) == (i, j):
                        break
                    i, j = i_new, j_new
                lu, lx = grid[i], gx[j]
                info['cv_grid_u'], info['cv_grid_x'] = grid.tolist(), gx.tolist()
                info['cv_score'] = np.where(np.isnan(score), None, score).tolist()
            # each half holds half of the data: keep the same penalty-to-data ratio
            lam = (2 * lu, 2 * lx)
        elif smooth is None or np.isscalar(smooth):
            v = float(smooth or 0.)
            lam = (v, v)
        else:
            lam = tuple(float(v) for v in smooth)
        c = system(Mt, *lam).solve(rt)
        self.C = c.reshape(self.nbx, self.nbu).T.copy()
        self.smooth = lam
        chi2 = float(s.sum() - 2 * c @ rt + c @ Mt.matvec(c))
        npix = int(sums.n.sum())
        self.fit_info = dict(info, smooth=list(lam), chi2=chi2, npix=npix,
                             chi2_red=chi2 / max(npix - self.ncoef, 1))
        self._cache = None
        if verbose:
            print(f'  PSF fit: {npix} pixels, {self.ncoef} coefficients, smoothing '
                  f'(u, x) = ({lam[0]:.1e}, {lam[1]:.1e}), reduced chi2 '
                  f'{self.fit_info["chi2_red"]:.3f}')
        return self

    # ------------------------------------------------------------------
    def column_coeffs(self):
        """1D spline coefficients of every column, ``(ncol, nbu)``, and their norms."""
        if getattr(self, '_cache', None) is None:
            cx = np.zeros((self.ncol, self.nbu))
            for q in range(self.bx.shape[1]):
                cx += self.C[:, self.bx_j0 + q].T * self.bx[:, q:q + 1]
            norm = cx @ integrals(self.tu)
            self._cache = (cx, norm)
        return self._cache

    def __call__(self, u, col, deriv=0, normalise=True):
        """Profile (or its derivative in ``u``) at offsets ``u`` in columns ``col``.

        ``u`` and ``col`` broadcast together; ``col`` holds column indices
        ``0 ... ncol-1``. Points with ``|u| > halfwidth`` give 0.
        """
        u, col = np.broadcast_arrays(np.asarray(u, float), np.asarray(col, int))
        shape = u.shape
        uf, cf = u.ravel(), col.ravel()
        cx, norm = self.column_coeffs()
        j0, B = basis(uf, self.tu, deriv=deriv)
        out = np.zeros(len(uf))
        for p in range(4):
            out += cx[cf, j0 + p] * B[:, p]
        if normalise:
            out /= np.where(norm[cf] != 0, norm[cf], np.inf)
        out[np.abs(uf) > self.halfwidth] = 0.
        return out.reshape(shape)

    def profile(self, col, u=None):
        """Fine-sampled profile of column ``col`` (for plots)."""
        u = np.linspace(-self.halfwidth, self.halfwidth, 1001) if u is None else u
        return u, self(u, np.full(len(u), col))

    def width(self, frac=0.5, npt=4001):
        """Full width at ``frac`` of the maximum of every column's profile."""
        u = np.linspace(-self.halfwidth, self.halfwidth, npt)
        cx, norm = self.column_coeffs()
        j0, B = basis(u, self.tu)
        prof = np.zeros((self.ncol, npt))
        for p in range(4):
            prof += cx[:, j0 + p] * B[:, p]
        out = np.zeros(self.ncol)
        for c in range(self.ncol):
            pr = prof[c]
            k = np.argmax(pr)
            above = pr >= frac * pr[k]
            lo = k
            while lo > 0 and above[lo - 1]:
                lo -= 1
            hi = k
            while hi < npt - 1 and above[hi + 1]:
                hi += 1
            out[c] = u[hi] - u[lo]
        return out

    # ------------------------------------------------------------------
    def to_hdus(self, suffix=''):
        """FITS extensions describing the model."""
        h = fits.Header()
        h['HALFWID'] = self.halfwidth
        h['OVERSAMP'] = self.oversample
        h['KNOTCOL'] = self.knot_col
        h['NCOL'] = self.ncol
        h['SMOOTHU'] = self.smooth[0]
        h['SMOOTHX'] = self.smooth[1]
        h['FITINFO'] = json.dumps({k: v for k, v in self.fit_info.items()
                                   if not k.startswith('cv_')})[:60]
        return [fits.ImageHDU(self.C, header=h, name=f'PSF{suffix}')]

    @classmethod
    def from_hdu(cls, hdu):
        h = hdu.header
        obj = cls(h['HALFWID'], h['NCOL'], h['OVERSAMP'], h['KNOTCOL'])
        obj.C = np.asarray(hdu.data, float)
        obj.smooth = (h.get('SMOOTHU', 0.), h.get('SMOOTHX', 0.))
        obj._cache = None
        return obj


