"""
Integration tests comparing Julia and Python API results.

This test module runs identical models in both Julia (via pre-generated outputs)
and Python, then compares the results to ensure the Python wrapper correctly
interfaces with the Julia package.
"""

import pytest
import json
import numpy as np
import pandas as pd
import pathlib
import subprocess
import os
from optimalgiv import giv

# Paths
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
JULIA_OUTPUTS_DIR = pathlib.Path(__file__).resolve().parent / "julia_comparison" / "outputs"
JULIA_SCRIPT = pathlib.Path(__file__).resolve().parent / "julia_comparison" / "run_julia_tests.jl"
DATA_PATH = ROOT_DIR / "examples" / "simdata1.csv"

# Tolerances for comparison
COEF_TOL = 1e-6
VCOV_TOL = 1e-5
GENERAL_TOL = 1e-8


def ensure_julia_outputs_exist():
    """Run Julia tests if outputs don't exist."""
    if not JULIA_OUTPUTS_DIR.exists() or not any(JULIA_OUTPUTS_DIR.glob("*.json")):
        print("Julia outputs not found. Running Julia tests...")
        os.makedirs(JULIA_OUTPUTS_DIR, exist_ok=True)
        
        # Set Julia threads to 1 for consistency
        env = os.environ.copy()
        env["JULIA_NUM_THREADS"] = "1"
        
        # Use the Julia from juliacall if available (ensures same Julia version)
        try:
            from juliacall import Main as jl
            julia_cmd = jl.Sys.BINDIR + "/julia"
        except:
            julia_cmd = "julia"
            
        result = subprocess.run(
            [julia_cmd, str(JULIA_SCRIPT)],
            capture_output=True,
            text=True,
            env=env
        )
        
        if result.returncode != 0:
            print("Julia stdout:", result.stdout)
            print("Julia stderr:", result.stderr)
            raise RuntimeError(f"Julia tests failed with code {result.returncode}")
        
        print("Julia tests completed successfully")


def load_julia_output(test_name):
    """Load Julia test output from JSON file."""
    file_path = JULIA_OUTPUTS_DIR / f"{test_name}.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_arrays(py_array, jl_array, name, tolerance=COEF_TOL):
    """Compare numpy arrays with appropriate tolerance."""
    # Handle None/null values
    if jl_array is None and (py_array is None or np.asarray(py_array).size == 0):
        return
    if py_array is None and (jl_array is None or np.asarray(jl_array).size == 0):
        return
        
    py_array = np.asarray(py_array)
    jl_array = np.asarray(jl_array)
    
    # Handle empty arrays - Julia may return 1D empty while Python returns 2D empty
    if py_array.size == 0 and jl_array.size == 0:
        return  # Both empty, consider equal
    
    assert py_array.shape == jl_array.shape, f"{name} shape mismatch: {py_array.shape} vs {jl_array.shape}"
    
    if py_array.size > 0:
        max_diff = np.max(np.abs(py_array - jl_array))
        assert max_diff < tolerance, f"{name} max difference {max_diff} exceeds tolerance {tolerance}"


def compare_model_results(py_model, jl_results):
    """Compare Python model results with Julia results."""
    # Basic coefficients
    compare_arrays(py_model.endog_coef, jl_results["endog_coef"], "endog_coef")
    compare_arrays(py_model.exog_coef, jl_results["exog_coef"], "exog_coef")
    
    # Variance-covariance matrices
    # Special handling for vcov which might contain null/NaN when PC extraction is used
    jl_endog_vcov = jl_results["endog_vcov"]
    jl_exog_vcov = jl_results["exog_vcov"]
    
    # Check if Julia vcov contains null values (PC extraction case)
    if isinstance(jl_endog_vcov, list) and len(jl_endog_vcov) > 0:
        if isinstance(jl_endog_vcov[0], list) and jl_endog_vcov[0][0] is None:
            # PC extraction case - vcov is disabled
            assert np.all(np.isnan(py_model.endog_vcov)), "Expected NaN vcov for PC extraction"
        else:
            compare_arrays(py_model.endog_vcov, jl_endog_vcov, "endog_vcov", VCOV_TOL)
    else:
        compare_arrays(py_model.endog_vcov, jl_endog_vcov, "endog_vcov", VCOV_TOL)
        
    if isinstance(jl_exog_vcov, list) and len(jl_exog_vcov) > 0:
        if isinstance(jl_exog_vcov[0], list) and jl_exog_vcov[0][0] is None:
            # PC extraction case - vcov is disabled
            assert np.all(np.isnan(py_model.exog_vcov)), "Expected NaN vcov for PC extraction"
        else:
            compare_arrays(py_model.exog_vcov, jl_exog_vcov, "exog_vcov", VCOV_TOL)
    else:
        compare_arrays(py_model.exog_vcov, jl_exog_vcov, "exog_vcov", VCOV_TOL)
    
    # Aggregate coefficient
    if isinstance(jl_results["agg_coef"], list):
        compare_arrays(py_model.agg_coef, jl_results["agg_coef"], "agg_coef")
    else:
        assert abs(py_model.agg_coef - jl_results["agg_coef"]) < COEF_TOL
    
    # Model metadata
    assert py_model.complete_coverage == jl_results["complete_coverage"]
    assert py_model.converged == jl_results["converged"]
    assert py_model.N == jl_results["N"]
    assert py_model.T == jl_results["T"]
    assert py_model.nobs == jl_results["nobs"]
    assert py_model.dof == jl_results["dof"]
    assert py_model.dof_residual == jl_results["dof_residual"]
    
    # String comparisons
    assert py_model.responsename == jl_results["responsename"]
    assert py_model.endogname == jl_results["endogname"]
    assert py_model.idvar == jl_results["idvar"]
    assert py_model.tvar == jl_results["tvar"]
    
    # Coefficient names
    assert py_model.endog_coefnames == jl_results["endog_coefnames"]
    assert py_model.exog_coefnames == jl_results["exog_coefnames"]
    
    # PC results if applicable
    assert py_model.n_pcs == jl_results["n_pcs"]
    if py_model.n_pcs > 0:
        # Julia returns factors as T×k, Python as k×T, so transpose for comparison
        jl_pc_factors = np.asarray(jl_results["pc_factors"]).T
        compare_arrays(py_model.pc_factors, jl_pc_factors, "pc_factors")
        # Julia returns loadings as k×N, Python as N×k, so transpose for comparison
        jl_pc_loadings = np.asarray(jl_results["pc_loadings"]).T
        compare_arrays(py_model.pc_loadings, jl_pc_loadings, "pc_loadings")


# Load data once
df = pd.read_csv(DATA_PATH)
df['id'] = df['id'].astype('category')


@pytest.fixture(scope="module", autouse=True)
def setup_julia_outputs():
    """Ensure Julia outputs exist before running tests."""
    # In CI, always run fresh to test against latest Julia package
    if os.getenv("CI") == "true":
        import shutil
        if JULIA_OUTPUTS_DIR.exists():
            shutil.rmtree(JULIA_OUTPUTS_DIR)
    ensure_julia_outputs_exist()


def test_homogeneous_scalar_search():
    """Test homogeneous elasticity with scalar search algorithm."""
    jl_data = load_julia_output("homogeneous_scalar_search")
    
    model = giv(
        df,
        "q + endog(p) ~ 0 + fe(id) & (η1 + η2)",
        id="id", t="t", weight="absS",
        algorithm="scalar_search",
        guess={"Aggregate": 2.0},
        save="all"
    )
    
    compare_model_results(model, jl_data["results"])


def test_homogeneous_iv():
    """Test homogeneous elasticity with IV algorithm."""
    jl_data = load_julia_output("homogeneous_iv")
    
    model = giv(
        df,
        "q + endog(p) ~ 0 + fe(id) & (η1 + η2)",
        id="id", t="t", weight="absS",
        algorithm="iv",
        guess=1.0
    )
    
    compare_model_results(model, jl_data["results"])


def test_heterogeneous_scalar_search():
    """Test heterogeneous elasticity with scalar search."""
    jl_data = load_julia_output("heterogeneous_scalar_search")
    
    model = giv(
        df,
        "q + id & endog(p) ~ 0 + id & (η1 + η2)",
        id="id", t="t", weight="absS",
        algorithm="scalar_search",
        guess={"Aggregate": 2.5}
    )
    
    compare_model_results(model, jl_data["results"])


def test_heterogeneous_iv():
    """Test heterogeneous elasticity with IV algorithm."""
    jl_data = load_julia_output("heterogeneous_iv")
    
    model = giv(
        df,
        "q + id & endog(p) ~ 0 + id & (η1 + η2)",
        id="id", t="t", weight="absS",
        algorithm="iv",
        guess=np.ones(5)
    )
    
    compare_model_results(model, jl_data["results"])


def test_homogeneous_debiased_ols():
    """Test debiased OLS algorithm."""
    jl_data = load_julia_output("homogeneous_debiased_ols")
    
    model = giv(
        df,
        "q + endog(p) ~ 0 + fe(id) & (η1 + η2)",
        id="id", t="t", weight="absS",
        algorithm="debiased_ols",
        guess=1.0
    )
    
    compare_model_results(model, jl_data["results"])


def test_homogeneous_iv_twopass():
    """Test two-pass IV algorithm."""
    jl_data = load_julia_output("homogeneous_iv_twopass")
    
    model = giv(
        df,
        "q + endog(p) ~ 0 + fe(id) & (η1 + η2)",
        id="id", t="t", weight="absS",
        algorithm="iv_twopass",
        guess=1.0
    )
    
    compare_model_results(model, jl_data["results"])


def test_partial_absorption():
    """Test partial absorption of fixed effects."""
    jl_data = load_julia_output("partial_absorption")
    
    model = giv(
        df,
        "q + endog(p) ~ 0 + id & η1 + fe(id) & η2",
        id="id", t="t", weight="absS",
        algorithm="scalar_search",
        guess={"Aggregate": 2.0}
    )
    
    compare_model_results(model, jl_data["results"])


def test_with_exclude_pairs():
    """Test exclude_pairs functionality."""
    jl_data = load_julia_output("with_exclude_pairs")
    
    model = giv(
        df,
        "q + endog(p) ~ 0 + fe(id) & (η1 + η2)",
        id="id", t="t", weight="absS",
        algorithm="iv",
        guess=1.0,
        exclude_pairs={1: [2, 3], 4: [5]}
    )
    
    compare_model_results(model, jl_data["results"])


def test_with_pc_extraction():
    """Test principal component extraction."""
    jl_data = load_julia_output("with_pc_extraction")
    
    model = giv(
        df,
        "q + endog(p) ~ fe(id) + pc(2)",
        id="id", t="t", weight="absS",
        algorithm="iv",
        guess=1.0
    )
    
    compare_model_results(model, jl_data["results"])



def test_with_solver_options():
    """Test custom solver options."""
    jl_data = load_julia_output("with_solver_options")
    
    model = giv(
        df,
        "q + endog(p) ~ 0 + fe(id) & (η1 + η2)",
        id="id", t="t", weight="absS",
        algorithm="iv",
        guess=1.0,
        solver_options={"iterations": 1}
    )
    
    compare_model_results(model, jl_data["results"])


def compare_pca_model_results(py_model, jl_results):
    """Helper to compare PCA-specific results."""
    # First compare regular model results
    compare_model_results(py_model, jl_results)
    
    # Then compare PCA-specific results
    pc = py_model.pc_model
    
    # Mean
    compare_arrays(pc.mean, jl_results["pc_mean"], "pc_mean")
    
    # Projection
    compare_arrays(pc.projection, jl_results["pc_projection"], "pc_projection")
    
    # Principal variances (use slightly higher tolerance for iterative algorithms)
    compare_arrays(pc.prinvars, jl_results["pc_prinvars"], "pc_prinvars", tolerance=2e-6)
    
    # Noise variances (use slightly higher tolerance for iterative algorithms)
    compare_arrays(pc.noisevars, jl_results["pc_noisevars"], "pc_noisevars", tolerance=3e-6)
    
    # R-squared
    assert abs(pc.r2 - jl_results["pc_r2"]) < 1e-6, f"pc_r2 mismatch: {pc.r2} vs {jl_results['pc_r2']}"
    
    # Convergence and iterations
    assert pc.converged == jl_results["pc_converged"], f"pc_converged mismatch"
    assert pc.iterations == jl_results["pc_iterations"], f"pc_iterations mismatch"


def test_pca_standard_pairwise():
    """Test PCA with standard algorithm and pairwise imputation."""
    jl_data = load_julia_output("pca_standard_pairwise")
    
    model = giv(
        df,
        "q + endog(p) ~ 0 + pc(2)",
        id="id", t="t", weight="absS",
        guess={"p": 2.0},
        pca_option=dict(
            impute_method="pairwise",
            algorithm="StandardHeteroPCA",
            demean=True,
            maxiter=100,
            abstol=1e-7,
        ),
        save="all", save_df=True, quiet=True
    )
    
    compare_pca_model_results(model, jl_data["results"])


def test_pca_diagonal_zero():
    """Test PCA with diagonal deletion and zero imputation."""
    jl_data = load_julia_output("pca_diagonal_zero")
    
    model = giv(
        df,
        "q + endog(p) ~ 0 + pc(2)",
        id="id", t="t", weight="absS",
        guess={"p": 2.0},
        pca_option=dict(
            impute_method="zero",
            algorithm="DiagonalDeletion",
            demean=False
        ),
        save="all", save_df=True, quiet=True
    )
    
    compare_pca_model_results(model, jl_data["results"])


def test_pca_deflated_pairwise():
    """Test PCA with deflated algorithm and pairwise imputation."""
    jl_data = load_julia_output("pca_deflated_pairwise")
    
    model = giv(
        df,
        "q + endog(p) ~ 0 + pc(2)",
        id="id", t="t", weight="absS",
        guess={"p": 2.0},
        pca_option=dict(
            impute_method="pairwise",
            algorithm="DeflatedHeteroPCA",
            algorithm_options={
                "t_block": 5,
                "condition_number_threshold": 3.5,
            },
            demean=False,
            α=1.0,
            suppress_warnings=False,
            abstol=1e-6,
        ),
        save="all", save_df=True, quiet=True
    )
    
    compare_pca_model_results(model, jl_data["results"])


if __name__ == "__main__":
    pytest.main(["-v", __file__])