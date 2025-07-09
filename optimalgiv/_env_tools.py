from pathlib import Path
from juliacall import Main as jl

_PKG_DIR   = Path(__file__).parent.resolve()
_SENTINEL  = _PKG_DIR / ".optimalgiv_setup_complete"

def update_packages():
    """
    Update Julia dependencies used by OptimalGIV
    and re-precompile them. Safe to call from Python.
    """
    # remove sentinel so boot-strap will know to rebuild next time
    if _SENTINEL.exists():
        _SENTINEL.unlink()

    # kick Julia once *now* so the user doesn’t have to re-import
    jl.seval("import Pkg; Pkg.update(); Pkg.precompile()")

    # recreate sentinel
    _SENTINEL.touch()

    print("Julia packages updated and re-precompiled.")
