"""
optimalgiv
==========

Thin Python wrapper around the registered Julia package **OptimalGIV**.

`giv` and `GIVModel` are re-exported from `. _bridge`.
"""
from ._bridge import giv, GIVModel

__all__ = ["giv", "GIVModel"]
__version__ = "0.1.0"
