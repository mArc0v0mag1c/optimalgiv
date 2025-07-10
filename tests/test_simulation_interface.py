import numpy as np
from optimalgiv import SimParam, simulate_data


def test_simulation_dimensions():
    """Verify generated DataFrame has correct dimensions"""
    params = SimParam(T=5, N=3)
    dfs = simulate_data(params, nsims=1, seed=123)
    df = dfs[0]

    # Verify entity and time period counts
    assert df['id'].nunique() == params.N
    assert df['t'].nunique() == params.T
    assert len(df) == params.T * params.N  # Full panel with no missing values


def test_common_factors_presence():
    """Verify presence/absence of factor columns based on K"""
    # Case with factors
    params = SimParam(T=3, N=2, K=1)
    df = simulate_data(params, nsims=1, seed=123)[0]
    assert any('η' in col for col in df.columns)
    assert any('λ' in col for col in df.columns)

    # # Case without factors
    # params = SimParam(T=3, N=2, K=0, ushare=1.0)
    # df = simulate_data(params, nsims=1, seed=123)[0]
    # assert not any('η' in col for col in df.columns)
    # assert not any('λ' in col for col in df.columns)


def test_missing_values_handling():
    """Verify missing values are properly generated and handled"""
    params = SimParam(T=5, N=3, missingperc=0.5)
    dfs = simulate_data(params, nsims=1, seed=123)
    df = dfs[0]

    # Calculate expected rows after missing values
    total_rows = params.T * params.N
    expected_rows = total_rows * (1 - params.missingperc)

    # Allow 20% tolerance for randomness
    lower_bound = int(expected_rows * 0.8)
    upper_bound = min(total_rows, int(expected_rows * 1.2))

    assert lower_bound <= len(df) <= upper_bound


def test_elasticity_distribution():
    """Verify elasticity distribution matches parameters"""
    params = SimParam(T=10, N=5, M=0.8, sigma_zeta=0.5)
    df = simulate_data(params, nsims=1, seed=123)[0]

    # Get unique entities and their ζ values
    entities = df[['id', 'ζ', 'S']].drop_duplicates()

    # Verify weighted mean of ζ ≈ 1/M
    weighted_mean = (entities['ζ'] * entities['S']).sum()
    assert np.isclose(weighted_mean, 1 / params.M, atol=0.1)

    # Verify standard deviation ≈ sigma_zeta
    assert np.isclose(entities['ζ'].std(), params.sigma_zeta, atol=0.1)


def test_price_variance():
    """Verify price variance matches parameter"""
    params = SimParam(T=200, N=50, sigma_p=2.0)
    df = simulate_data(params, nsims=1, seed=123)[0]

    # Get unique price values per time period
    prices = df.drop_duplicates('t')['p']

    # Verify variance ≈ sigma_p²
    assert np.isclose(prices.var(), params.sigma_p ** 2, atol=0.1)

# def test_no_common_factors():
#     """Verify simulation with no common factors"""
#     params = SimParam(T=5, N=3, K=0, ushare=1.0)
#     df = simulate_data(params, nsims=1, seed=123)[0]
#
#     # Should have no factor-related columns
#     assert not any(col.startswith('η') for col in df.columns)
#     assert not any(col.startswith('λ') for col in df.columns)
#
#     # All variation should be idiosyncratic
#     assert 'commonshocks' not in df.columns


def test_full_coverage():
    """Verify no missing values when missingperc=0"""
    params = SimParam(T=5, N=3, missingperc=0)
    df = simulate_data(params, nsims=1, seed=123)[0]
    assert df['q'].isnull().sum() == 0
    assert len(df) == params.T * params.N


def test_parameter_integration():
    """Verify all parameters are correctly reflected in simulation"""
    params = SimParam(
        T=100,
        N=50,
        K=2,
        M=0.8,
        sigma_zeta=0.5,
        h=0.3,
        nu=5.0,
        missingperc=0.2,
        ushare=0.7,
        sigma_u_curv=0.2,
        sigma_p=1.5,
    )

    df = simulate_data(params, nsims=1, seed=123)[0]

    # Verify dimensions
    assert df['id'].nunique() == params.N
    assert df['t'].nunique() == params.T

    # Verify factor columns
    assert sum('η' in col for col in df.columns) == params.K
    assert sum('λ' in col for col in df.columns) == params.K

    # Verify elasticity distribution
    entities = df[['id', 'ζ', 'S']].drop_duplicates()
    weighted_mean_zeta = (entities['ζ'] * entities['S']).sum()
    assert np.isclose(weighted_mean_zeta, 1 / params.M, atol=0.1)

    # Verify size distribution
    hhi = (entities['S'] ** 2).sum()
    excess_hhi = np.sqrt(hhi - 1 / params.N)
    assert np.isclose(excess_hhi, params.h, atol=0.05)

    # Verify missing values
    total_rows = params.T * params.N
    assert len(df) < total_rows  # Some rows removed
    assert params.missingperc * 0.8 < (total_rows - len(df)) / total_rows < params.missingperc * 1.2

    # Verify price variance
    prices = df.drop_duplicates('t')['p']
    scaled_variance = prices.var() * (params.T / (params.T - 1))
    assert np.isclose(scaled_variance, params.sigma_p ** 2, rtol=0.2)