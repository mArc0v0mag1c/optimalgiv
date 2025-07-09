# python3 -m pytest -p no:faulthandler tests/test_simulation_interface.py -v

import pytest
import numpy as np
import pandas as pd
from optimalgiv._simulation import SimParam, simulate_data
from juliacall import Main as jl

@pytest.fixture
def default_params():
    return SimParam(T=5, N=3, K=1)

def test_simulate_data_basic(default_params):
    dfs = simulate_data(default_params, nsims=2, seed=123)
    assert len(dfs) == 2
    for df in dfs:
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {'id', 't', 'q', 'p', 'S', 'ζ'}


def test_missing_values():
    """Test missing value generation"""
    params = SimParam(T=5, N=3, missingperc=0.5)
    dfs = simulate_data(params, nsims=1, seed=123)
    df = dfs[0]

    total_rows = params.T * params.N
    expected_rows = total_rows * (1 - params.missingperc)

    # Allow 20% tolerance for randomness
    lower_bound = int(expected_rows * 0.8)
    upper_bound = min(total_rows, int(expected_rows * 1.2))

    assert lower_bound <= len(df) <= upper_bound

def test_julia_object_return():
    params = SimParam(T=3, N=2)
    julia_dfs = simulate_data(params, nsims=1, as_pandas=False)
    jdf = julia_dfs[0]
    assert int(jl.nrow(jdf)) == 6  # 3*2=6
    assert "q" in [str(c) for c in jl.names(jdf)]

def test_zero_missing_values():
    params = SimParam(T=5, N=3, missingperc=0)
    dfs = simulate_data(params, nsims=1)
    assert len(dfs[0]) == 15  # 5*3=15 rows

# def test_no_common_factors():
#     params = SimParam(T=5, N=3, K=0, ushare=1.0)
#     dfs = simulate_data(params, nsims=1)
#     df = dfs[0]
#     # Should have no η columns
#     assert not any(col.startswith('η') for col in df.columns)

if __name__ == "__main__":
    pytest.main(["-v", __file__])