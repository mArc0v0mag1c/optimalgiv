import os
import warnings
import juliapkg

#------------------------------------------------------------------
# Default to 1 Julia thread unless the caller set it explicitly
#------------------------------------------------------------------
os.environ.setdefault("JULIA_NUM_THREADS", "1")

if int(os.getenv("JULIA_NUM_THREADS")) > 1:
    warnings.warn(
        "The support of multithreading with JuliaCall is experimental. "
        "You have JULIA_NUM_THREADS="
        + os.environ["JULIA_NUM_THREADS"]
        + ".  If you see segfaults, rerun with 1."
    )

from juliacall import Main as jl


# Load the required Julia packages
jl.seval("using PythonCall, OptimalGIV, DataFrames, CategoricalArrays")


from ._bridge import giv, GIVModel
from ._simulation import simulate_data, SimParam
from ._pca import HeteroPCAModel

__all__ = ["simulate_data", "SimParam", "giv", "GIVModel", "HeteroPCAModel"]
__version__ = "0.2.0"