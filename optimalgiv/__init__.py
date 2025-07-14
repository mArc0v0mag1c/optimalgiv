import os
import warnings
import juliapkg
import json
from pathlib import Path

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

#------------------------------------------------------------------
# Manage Julia dependencies with juliapkg
#------------------------------------------------------------------
# Read expected packages from juliapkg.json
juliapkg_path = Path(__file__).parent / "juliapkg.json"
expected_packages = {}
if juliapkg_path.exists():
    with open(juliapkg_path, "r") as f:
        juliapkg_config = json.load(f)
        expected_packages = juliapkg_config.get("packages", {})

# Resolve dependencies explicitly
juliapkg.resolve()

# Check if all expected packages are present and add them if needed
for pkg_name, pkg_info in expected_packages.items():
    pkg_uuid = pkg_info.get("uuid")
    if pkg_uuid:
        # This will be no-op if already present
        juliapkg.add(pkg=pkg_name, uuid=pkg_uuid)

# Import juliacall after ensuring dependencies are resolved
from juliacall import Main as jl


# Load the required Julia packages
jl.seval("using PythonCall, OptimalGIV, DataFrames, CategoricalArrays")


from ._bridge import giv, GIVModel
from ._simulation import simulate_data, SimParam
from ._pca import HeteroPCAModel

__all__ = ["simulate_data", "SimParam", "giv", "GIVModel", "HeteroPCAModel"]
__version__ = "0.2.0"