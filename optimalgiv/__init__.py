import os
import sys
import pathlib
import warnings

#------------------------------------------------------------------
# Default to 1 Julia thread unless the caller set it explicitly
#------------------------------------------------------------------
os.environ.setdefault("JULIA_NUM_THREADS", "1")

if int(os.getenv("JULIA_NUM_THREADS")) > 1:
    warnings.warn(
        "OptimalGIV is more stable on a single Julia thread. "
        "You have JULIA_NUM_THREADS="
        + os.environ["JULIA_NUM_THREADS"]
        + ".  If you see segfaults, rerun with 1."
    )

from juliacall import Main as jl

################################################################################
#  SPEED-OPTIMISED JULIA ENV BOOTSTRAP
#
#  – First run  : installs + precompiles all deps, then drops a “sentinel” file
#  – Later runs : skips installs/update/precompile and just `instantiate`s
#  – Optional   : set env OPTIMALGIV_FORCE_SETUP=1 to force a fresh rebuild
################################################################################

if not hasattr(sys, "_julia_env_initialized"):
    _pkg_dir = pathlib.Path(__file__).parent.resolve()         # project folder
    os.environ["PYTHON_JULIAPKG_PROJECT"] = str(_pkg_dir)
    sentinel = _pkg_dir / ".optimalgiv_setup_complete"         # touch() on success
    force    = os.getenv("OPTIMALGIV_FORCE_SETUP") == "1"

    jl.seval("import Pkg")
    jl.seval(f'Pkg.activate("{_pkg_dir}")')

    # -------------------------------------------------------------------------
    # Decide whether we need a *full* setup or the quick path
    # -------------------------------------------------------------------------
    need_full_setup = force or not sentinel.exists()

    if need_full_setup:
        # Make sure the General registry exists (cheap if it already does)
        jl.seval('Pkg.Registry.add("General")')

        jl.seval(
            '''
            import Pkg
            Pkg.add(Pkg.PackageSpec(
                url = "https://github.com/FuZhiyu/OptimalGIV.jl",
                rev = "main"
            ))
            '''
        )

        jl.seval(
            '''
            import Pkg
            Pkg.add(Pkg.PackageSpec(
                url = "https://github.com/FuZhiyu/HeteroPCA.jl"
            ))
            '''
        )

        # Add ONLY missing packages (skips the network if already present)
        pkgs = [
            ("PythonCall",  "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"),
            # ("OptimalGIV",  "bf339e5b-51e6-4b7b-82b3-758165633231"),
            ("DataFrames",  "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"),
            ("StatsModels", "3eaba693-59b7-5ba5-a881-562e759f1c8d"),
            ("CategoricalArrays","324d7699-5711-5eae-9e2f-1d82baa6b597"),
            ("StatsFuns",   "4c63d2b9-4356-54db-8cca-17b64c39e42c"),
            ("LineSearches","d3d80556-e9d4-5f37-9878-2ab0fcc64255"),
        ]
        jl.packages_to_add = pkgs   # hand it over once
        jl.seval("""
        pkgs_to_add = packages_to_add   # pulled from Python
        installed  = keys(Pkg.dependencies())
        for (name, uuid) in pkgs_to_add
            if !(name in installed)
                Pkg.add(name=name, uuid=uuid)
            end
        end
        """)

        # Resolve/instantiate & precompile once
        jl.seval("Pkg.instantiate()")
        jl.seval("Pkg.precompile()")

        # Mark success
        sentinel.touch()

    else:
        # Fast path: trust Manifest, just ensure deps exist
        jl.seval("Pkg.instantiate()")

    # -------------------------------------------------------------------------
    # Finally, load the packages we need in this session
    # -------------------------------------------------------------------------
    jl.seval("using PythonCall, OptimalGIV, DataFrames, StatsModels, "
             "CategoricalArrays, StatsFuns, LineSearches, HeteroPCA")

    # Cache flag for rest of Python process
    sys._julia_env_initialized = True


from ._bridge import giv, GIVModel
from ._simulation import simulate_data, SimParam
from ._env_tools import update_packages
from ._pca import HeteroPCAModel

__all__ = ["simulate_data", "SimParam", "giv", "GIVModel", "update_packages", "HeteroPCAModel"]
__version__ = "0.2.0"