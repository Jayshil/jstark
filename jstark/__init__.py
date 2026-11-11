try:
    from .version import version as __version__
except ImportError:  # package not installed (e.g. running from a source checkout)
    __version__ = 'unknown'

__all__ = []
