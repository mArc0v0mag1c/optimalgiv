# python3 -m pytest -p no:faulthandler tests/test_pca_accuracy.py -v

import pytest
import pathlib
import numpy as np
import pandas as pd
from optimalgiv import giv

root = pathlib.Path(__file__).resolve().parent.parent
df = pd.read_csv(root / "examples" / "simdata1.csv")
df['id'] = df['id'].astype('category')

# --------------------------------------------------------------------
# PCA Interface Tests
# --------------------------------------------------------------------

def validate_pca_model(model, expected):
    # Number of components
    assert model.n_pcs == expected['n_pcs']

    # PC factors and loadings shapes
    assert model.pc_factors.shape == expected['factors_shape']
    assert model.pc_loadings.shape == expected['loadings_shape']

    # Underlying Julia PC model
    pc = model.pc_model
    assert np.allclose(pc.mean, expected['mean'], atol=1e-6)
    assert pc.projection.shape == expected['projection_shape']
    assert np.allclose(pc.projection, expected['projection'], atol=1e-6)
    assert np.allclose(pc.prinvars, expected['prinvars'], atol=1e-6)
    assert np.allclose(pc.noisevars, expected['noisevars'], atol=1e-6)
    assert np.isclose(pc.r2, expected['r2'], atol=1e-6)
    assert pc.converged == expected['converged']
    assert pc.iterations == expected['iterations']

    # Compare saved DataFrame values
    df5 = model.df.iloc[:5]
    # Loadings: DataFrame columns vs pc_loadings
    assert np.allclose(
        df5[["pc_loading_1", "pc_loading_2"]].values,
        expected['loadings_first5'],
        atol=1e-5
    )
    # Factors: DataFrame columns vs pc_factors
    assert np.allclose(
        df5[["pc_factor_1", "pc_factor_2"]].values,
        expected['factors_first5'],
        atol=1e-5
    )

def test_pca_standard_pairwise():
    expected = {
        'n_pcs': 2,
        'factors_shape': (2, 60),
        'loadings_shape': (5, 2),
        'mean': np.array([0.837174, -0.165458, -1.198694, 0.213525, -0.069848]),
        'projection_shape': (5, 2),
        'projection': np.array([
            [-0.556913, 0.239819],
            [-0.124171, -0.190476],
            [-0.157929, 0.259165],
            [0.383208, -0.699816],
            [-0.708971, -0.591014]
        ]),
        'prinvars': np.array([51.881196, 15.880571]),
        'noisevars': np.array([40.954913, 26.372427, 23.967339, 14.073547, -16.729385]),
        'r2': 0.4332577,
        'converged': False,
        'iterations': 100,
        'factors_first5': np.array([
            [-4.430986, -6.226477],
            [-4.430986, -6.226477],
            [-4.430986, -6.226477],
            [-4.430986, -6.226477],
            [-4.430986, -6.226477]
        ]),
        'loadings_first5': np.array([
            [-0.556913, 0.239819],
            [-0.124171, -0.190476],
            [-0.157929, 0.259165],
            [0.383208, -0.699816],
            [-0.708971, -0.591014]
        ])
    }
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
        ),
        save="all", save_df=True, quiet=True
    )
    validate_pca_model(model, expected)


def test_pca_diagonal_zero():
    expected = {
        'n_pcs': 2,
        'factors_shape': (2, 60),
        'loadings_shape': (5, 2),
        'mean': np.zeros(5),
        'projection_shape': (5, 2),
        'projection': np.array([
            [-0.623450, -0.700603],
            [0.038359, -0.192172],
            [-0.115507, -0.241725],
            [0.607257, -0.620861],
            [-0.477217, 0.168305]
        ]),
        'prinvars': np.array([24.751660, 18.890378]),
        'noisevars': np.array([30.120289, 24.561801, 24.937723, 16.349208, 3.852342]),
        'r2': 0.3042033,
        'converged': True,
        'iterations': 1,
        'factors_first5': np.array([
            [-2.296272, -8.677979],
            [-2.296272, -8.677979],
            [-2.296272, -8.677979],
            [-2.296272, -8.677979],
            [-2.296272, -8.677979]
        ]),
        'loadings_first5': np.array([
            [-0.623450, -0.700603],
            [0.038359, -0.192172],
            [-0.115507, -0.241725],
            [0.607257, -0.620861],
            [-0.477217, 0.168305]
        ])
    }
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
    validate_pca_model(model, expected)


def test_pca_deflated_pairwise():
    expected = {
        'n_pcs': 2,
        'factors_shape': (2, 60),
        'loadings_shape': (5, 2),
        'mean': np.zeros(5),
        'projection_shape': (5, 2),
        'projection': np.array([
            [-0.669470, 0.129415],
            [-0.151545, -0.404209],
            [-0.184027, 0.340131],
            [0.362090, -0.647092],
            [-0.603216, -0.534275]
        ]),
        'prinvars': np.array([49.580153, 9.944093]),
        'noisevars': np.array([37.999216, 25.541981, 25.313024, 18.435119, -5.054325]),
        'r2': 0.3679804,
        'converged': True,
        'iterations': 5,
        'factors_first5': np.array([
            [-5.880742, -9.885945],
            [-5.880742, -9.885945],
            [-5.880742, -9.885945],
            [-5.880742, -9.885945],
            [-5.880742, -9.885945]
        ]),
        'loadings_first5': np.array([
            [-0.669470, 0.129415],
            [-0.151545, -0.404209],
            [-0.184027, 0.340131],
            [0.362090, -0.647092],
            [-0.603216, -0.534275]
        ])
    }
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
                "condition_number_threshold": 3.5
            },
            demean=False
        ),
        save="all", save_df=True, quiet=True
    )
    validate_pca_model(model, expected)


if __name__ == "__main__":
    pytest.main(["-v", __file__])