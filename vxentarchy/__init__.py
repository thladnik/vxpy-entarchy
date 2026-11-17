from .schema import *

try:
    from importlib.metadata import version as _dist_version
    __version__ = _dist_version('vxentarchy')
except Exception:  # a checkout that was never installed
    __version__ = '0.0.0'
