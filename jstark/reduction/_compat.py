"""Work-arounds for known issues in upstream packages.

Each patch first checks that the problem is actually present, so it becomes
a no-op once the upstream package is fixed.
"""
import warnings

import numpy as np

_PATCHED = {}


def patch_rotation3d_single_angle():
    """Fix ``Rotation3DToGWA`` with a single rotation angle (stdatamodels 6.0.0).

    With recent astropy versions a one-element angle list becomes a 0-d model
    parameter, and ``Rotation3DToGWA.evaluate`` fails with
    ``TypeError: 'numpy.float64' object is not iterable``. The NIRSpec PRISM
    refraction model (``Snell``) builds exactly such a rotation, so
    ``assign_wcs`` fails for all NIRSpec PRISM data. The patch reshapes the
    angles to the ``(1, n)`` layout that ``evaluate`` expects.
    """
    if _PATCHED.get('rotation3d'):
        return True
    try:
        from stdatamodels.jwst.transforms.models import Rotation3DToGWA
    except ImportError:
        return False
    try:
        Rotation3DToGWA([10.], 'y')(np.array([0.1]), np.array([0.]), np.array([0.99]))
        return False  # works: nothing to do
    except TypeError:
        pass
    original = Rotation3DToGWA.evaluate

    def evaluate(self, x, y, z, angles):
        a = np.asarray(angles)
        if a.ndim < 2:
            a = a.reshape(1, -1)
        return original(self, x, y, z, a)

    Rotation3DToGWA.evaluate = evaluate
    _PATCHED['rotation3d'] = True
    warnings.warn('jstark: patched stdatamodels Rotation3DToGWA for single-angle rotations '
                  '(needed for NIRSpec PRISM assign_wcs with this stdatamodels/astropy).')
    return True


def apply_all():
    """Apply all compatibility patches that are needed."""
    patch_rotation3d_single_angle()
