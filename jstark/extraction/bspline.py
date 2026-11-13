"""Vectorised cubic B-spline basis, penalties and streamed least-squares sums.

The PSF models of :mod:`jstark.extraction` are tensor-product cubic
B-splines. Instead of handing every pixel to
``scipy.interpolate.LSQBivariateSpline`` (which needs all of them in memory),
we accumulate the least-squares *normal equations* chunk by chunk:

.. math::

    \\Big(\\sum_\\mathrm{pix} w\\, B B^T\\Big)\\, c = \\sum_\\mathrm{pix} w\\, B\\, y

Both sides are sums over pixels, so they can be built one segment (or one
chunk of integrations) at a time; the solution is identical to fitting all
pixels at once. Only 4 basis functions are non-zero at any point, so each
pixel touches a 4 x 4 block of the matrix.
"""
import numpy as np

__all__ = ['clamped_knots', 'basis', 'n_basis', 'second_difference', 'integrals',
           'NormalSums', 'BandedSystem']


def clamped_knots(lo, hi, interior, k=3):
    """Full clamped knot vector for ``interior`` knots on ``[lo, hi]``."""
    interior = np.asarray(interior, dtype=float)
    interior = interior[(interior > lo) & (interior < hi)]
    return np.concatenate([[lo] * (k + 1), np.sort(interior), [hi] * (k + 1)])


def n_basis(t, k=3):
    return len(t) - k - 1


def basis(x, t, k=3, deriv=0):
    """Non-zero cubic B-spline basis values at ``x``.

    Parameters
    ----------
    x : ndarray
        Evaluation points; values outside ``[t[0], t[-1]]`` are clipped to the
        boundary (callers mask them).
    t : ndarray
        Clamped knot vector (see :func:`clamped_knots`).
    k : int
        Degree (3 = cubic).
    deriv : int
        0 for the values, 1 for the first derivative.

    Returns
    -------
    j0 : ndarray of int
        Index of the first non-zero basis function for each point; the
        non-zero functions are ``j0 ... j0 + k``.
    vals : ndarray
        ``(len(x), k + 1)`` basis values (or derivatives).
    """
    x = np.clip(np.asarray(x, dtype=float).ravel(), t[0], t[-1])
    nb = n_basis(t, k)
    # interval m with t[m] <= x < t[m+1], k <= m <= nb - 1
    m = np.searchsorted(t, x, side='right') - 1
    m = np.clip(m, k, nb - 1)
    # Cox-de Boor triangle (degree up to k - deriv), vectorised over points
    kk = k - deriv
    N = np.zeros((len(x), kk + 1))
    N[:, 0] = 1.
    for d in range(1, kk + 1):
        new = np.zeros_like(N)
        for r in range(d + 1):
            # basis index i = m - d + r
            i = m - d + r
            if r > 0:          # left term: N_{i, d-1} (stored at r-1)
                den = t[i + d] - t[i]
                with np.errstate(divide='ignore', invalid='ignore'):
                    new[:, r] += np.where(den > 0, (x - t[i]) / den, 0.) * N[:, r - 1]
            if r < d:          # right term: N_{i+1, d-1} (stored at r)
                den = t[i + d + 1] - t[i + 1]
                with np.errstate(divide='ignore', invalid='ignore'):
                    new[:, r] += np.where(den > 0, (t[i + d + 1] - x) / den, 0.) * N[:, r]
        N = new
    if deriv == 0:
        return m - k, N
    if deriv != 1:
        raise ValueError('only deriv = 0 or 1 is implemented')
    # derivative of degree-k basis from degree-(k-1) values N (indices m-k+1 ... m)
    out = np.zeros((len(x), k + 1))
    for r in range(k + 1):
        i = m - k + r          # degree-k function i
        # B'_{i,k} = k [B_{i,k-1}/(t_{i+k}-t_i) - B_{i+1,k-1}/(t_{i+k+1}-t_{i+1})]
        if r - 1 >= 0:
            den = t[i + k] - t[i]
            with np.errstate(divide='ignore', invalid='ignore'):
                out[:, r] += np.where(den > 0, k / den, 0.) * N[:, r - 1]
        if r <= k - 1:
            den = t[i + k + 1] - t[i + 1]
            with np.errstate(divide='ignore', invalid='ignore'):
                out[:, r] -= np.where(den > 0, k / den, 0.) * N[:, r]
    return m - k, out


def integrals(t, k=3):
    """Integral of every basis function over the whole knot range."""
    nb = n_basis(t, k)
    i = np.arange(nb)
    return (t[i + k + 1] - t[i]) / (k + 1)


def second_difference(n):
    """``(D^T D)`` for the second-difference operator on ``n`` coefficients."""
    if n < 3:
        return np.zeros((n, n))
    D = np.diff(np.eye(n), n=2, axis=0)
    return D.T @ D


class NormalSums:
    """Per-column normal equations of a 1D spline in the spatial coordinate.

    For every spectral column ``x`` we keep ``A_x = sum w B B^T``,
    ``b_x = sum w B y`` and ``s_x = sum w y^2``, separately for even and odd
    integrations (for cross-validation). A tensor-product (2D) spline in
    (spatial offset, column) is assembled from these afterwards
    (:meth:`assemble`), so the column knots can be changed without reading the
    data again, and summing over columns gives the 1D (column-independent) fit.

    Memory: ``2 * ncol * nb**2`` doubles (e.g. 2 x 1500 x 35^2 = 30 MB).
    """

    def __init__(self, ncol, t, k=3):
        self.t = np.asarray(t, float)
        self.k = k
        self.nb = n_basis(self.t, k)
        self.ncol = int(ncol)
        self.A = np.zeros((2, self.ncol, self.nb, self.nb))
        self.b = np.zeros((2, self.ncol, self.nb))
        self.s = np.zeros((2, self.ncol))
        self.n = np.zeros((2, self.ncol), dtype=np.int64)

    def add(self, u, col, y, w, parity):
        """Add pixels.

        Parameters
        ----------
        u : ndarray
            Spatial coordinate (offset from the trace) of each pixel.
        col : ndarray of int
            Column index (0 ... ncol-1) of each pixel.
        y, w : ndarray
            Value and weight (1/variance) of each pixel.
        parity : ndarray of int
            0 for even, 1 for odd integrations.
        """
        if len(u) == 0:
            return
        j0, B = basis(u, self.t, self.k)
        nb, nc = self.nb, self.ncol
        col = np.asarray(col, np.int64)
        par = np.asarray(parity, np.int64)
        base = (par * nc + col)
        size2 = 2 * nc * nb
        for p in range(self.k + 1):
            wp = w * B[:, p]
            ip = base * nb + j0 + p
            self.b += np.bincount(ip, weights=wp * y, minlength=size2).reshape(2, nc, nb)
            for q in range(self.k + 1):
                iq = ip * nb + j0 + q
                self.A += np.bincount(iq, weights=wp * B[:, q],
                                      minlength=size2 * nb).reshape(2, nc, nb, nb)
        self.s += np.bincount(base, weights=w * y * y, minlength=2 * nc).reshape(2, nc)
        self.n += np.bincount(base, minlength=2 * nc).reshape(2, nc)

    def assemble(self, bx_j0, bx, nbx, cols=None):
        """Block-banded normal equations of the tensor spline, for each parity.

        Coefficients are ordered column-basis-major (``index = j * nb + i``),
        so the matrix is block-banded: block ``(j, j + d)`` (``d`` = 0 ... 3)
        couples column basis functions ``j`` and ``j + d``.

        Parameters
        ----------
        bx_j0, bx : ndarray
            Column-direction basis (from :func:`basis`) for every column.
        nbx : int
            Number of column-direction basis functions.
        cols : ndarray of bool or None
            Columns to include (default: all).

        Returns
        -------
        Mb : ndarray ``(2, nbx, nd, nb, nb)``
            Upper blocks ``(j, j + d)`` (``nd = max(kx, 3)``).
        r : ndarray ``(2, nbx, nb)``
        s : ndarray ``(2,)``
        """
        nb = self.nb
        kx = bx.shape[1]
        nd = max(kx, 3)
        Mb = np.zeros((2, nbx + kx, nd, nb, nb))
        R2 = np.zeros((2, nbx + kx, nb))
        use = np.ones(self.ncol, bool) if cols is None else np.asarray(cols, bool)
        for j in np.unique(bx_j0[use]):
            sel = (bx_j0 == j) & use
            A = self.A[:, sel]
            b = self.b[:, sel]
            for p in range(kx):
                R2[:, j + p] += np.einsum('pcn,c->pn', b, bx[sel, p])
                for q in range(p, kx):
                    Mb[:, j + p, q - p] += np.einsum('pcnm,c->pnm', A, bx[sel, p] * bx[sel, q])
        return Mb[:, :nbx], R2[:, :nbx], self.s[:, use].sum(axis=1)


class BandedSystem:
    """Symmetric block-banded matrix (see :meth:`NormalSums.assemble`)."""

    def __init__(self, Mb):
        self.Mb = np.array(Mb, dtype=float)
        self.nbx, self.nd, self.nb, _ = self.Mb.shape

    @property
    def n(self):
        return self.nbx * self.nb

    def trace(self):
        return float(np.einsum('jii->', self.Mb[:, 0]))

    def copy(self):
        return BandedSystem(self.Mb.copy())

    def add_u_penalty(self, lam, DtD):
        """``+ lam * (I_x kron DtD)`` (smoothness along the spatial offset)."""
        self.Mb[:, 0] += lam * DtD

    def add_x_penalty(self, lam, DtD):
        """``+ lam * (DtD kron I_u)`` (smoothness along the columns)."""
        eye = np.eye(self.nb)
        for d in range(min(3, self.nd)):
            dg = np.diagonal(DtD, offset=d)
            self.Mb[:len(dg), d] += lam * dg[:, None, None] * eye

    def add_ridge(self, eps):
        self.Mb[:, 0] += eps * np.eye(self.nb)

    def matvec(self, c):
        c = np.asarray(c).reshape(self.nbx, self.nb)
        y = np.einsum('jab,jb->ja', self.Mb[:, 0], c)
        for d in range(1, self.nd):
            y[:-d] += np.einsum('jab,jb->ja', self.Mb[:-d, d], c[d:])
            y[d:] += np.einsum('jab,ja->jb', self.Mb[:-d, d], c[:-d])
        return y.ravel()

    def to_lower_banded(self):
        nb, nbx, nd = self.nb, self.nbx, self.nd
        kd = nd * nb - 1
        ab = np.zeros((kd + 1, self.n))
        a, b = np.meshgrid(np.arange(nb), np.arange(nb), indexing='ij')
        for d in range(nd):
            for j in range(nbx - d):
                blk = self.Mb[j, d]
                r = (j + d) * nb + b          # lower triangle: transpose of the upper block
                c = j * nb + a
                m = r >= c
                ab[(r - c)[m], c[m]] = blk[m]
        return ab

    def to_dense(self):
        nb, n = self.nb, self.n
        M = np.zeros((n, n))
        for d in range(self.nd):
            for j in range(self.nbx - d):
                M[j * nb:(j + 1) * nb, (j + d) * nb:(j + d + 1) * nb] = self.Mb[j, d]
                if d:
                    M[(j + d) * nb:(j + d + 1) * nb, j * nb:(j + 1) * nb] = self.Mb[j, d].T
        return M

    def solve(self, rhs, dense_max=3000):
        """Solve ``M c = rhs`` (dense Cholesky for moderate sizes, banded otherwise)."""
        from scipy.linalg import (cholesky_banded, cho_solve_banded, cho_factor, cho_solve,
                                  LinAlgError)
        if self.n <= dense_max:
            M = self.to_dense()
            try:
                return cho_solve(cho_factor(M, check_finite=False), np.ravel(rhs), check_finite=False)
            except LinAlgError:
                return np.linalg.lstsq(M, np.ravel(rhs), rcond=None)[0]
        ab = self.to_lower_banded()
        try:
            cb = cholesky_banded(ab, lower=True, check_finite=False)
            return cho_solve_banded((cb, True), np.ravel(rhs), check_finite=False)
        except LinAlgError:
            from scipy.linalg import solveh_banded
            return solveh_banded(ab + 1e-9 * np.abs(ab[0]).max() * (np.arange(len(ab))[:, None] == 0),
                                 np.ravel(rhs), lower=True, check_finite=False)
