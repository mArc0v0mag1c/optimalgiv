import os
import sys
import pathlib
from juliacall import Main as jl

# Guard to prevent re-initialization
if not hasattr(sys, "_julia_env_initialized"):
    # 1) Set project directory
    _pkg_dir = pathlib.Path(__file__).parent.resolve()
    os.environ["PYTHON_JULIAPKG_PROJECT"] = str(_pkg_dir)

    # 2) Activate and set up environment
    jl.seval("import Pkg")
    jl.seval(f'Pkg.activate("{_pkg_dir}")')

    # 3) Add required registries
    jl.seval('Pkg.Registry.add("General")')

    # 4) Install and load packages directly
    packages = [
        ("PythonCall", "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"),
        ("OptimalGIV", "bf339e5b-51e6-4b7b-82b3-758165633231"),
        ("DataFrames", "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"),
        ("StatsModels", "3eaba693-59b7-5ba5-a881-562e759f1c8d"),
        ("CategoricalArrays", "324d7699-5711-5eae-9e2f-1d82baa6b597")
        #("Tables", "bd369af6-aec1-5ad0-b16a-f7cc5008161c")
    ]

    # Install packages using UUIDs
    for name, uuid in packages:
        jl.seval(f'Pkg.add(name="{name}", uuid="{uuid}")')

    # 5) Update, instantiate, and precompile packages
    jl.seval("Pkg.Registry.update()")
    jl.seval("Pkg.update()")
    jl.seval("Pkg.instantiate()")
    jl.seval("Pkg.precompile()")

    # 6) Load packages
    jl.seval("using PythonCall, OptimalGIV, DataFrames, StatsModels, CategoricalArrays")

    sys._julia_env_initialized = True

from ._bridge import giv, GIVModel
from ._simulate import simulate_data, SimParam

__all__ = ["simulate_data", "SimParam", "giv", "GIVModel"]
__version__ = "0.1.9"