# python3 -m pytest -p no:faulthandler tests/test_estimates.py -v

import pytest
import pathlib
import numpy as np
import pandas as pd
from optimalgiv import giv

root = pathlib.Path(__file__).resolve().parent.parent
df = pd.read_csv(root / "examples" / "simdata1.csv")
df['id'] = df['id'].astype('category')

# --------------------------------------------------------------------
#  1. HOMOGENEOUS ELASTICITY
# --------------------------------------------------------------------
def test_homogeneous_scalar_search():
    f = "q + endog(p) ~ 0 + fe(id) & (η1 + η2)"

    model = giv(
        df,
        f,
        id = "id",
        t = "t",
        weight = "absS",
        guess={"Aggregate": 2.0},
        algorithm="scalar_search",
        save="all",
    )

    # point-estimate & standard error
    assert np.allclose(model.endog_coef[0] * 2, 2.5341730, atol=1e-4)
    se = np.sqrt(model.endog_vcov[0,0]) * 2
    assert np.allclose(se, 0.2407, atol=1e-4)

    factor_expected = np.array(
        [[0.2419, 1.1729],
         [0.1842, 0.3722],
         [-1.3213, -0.6487],
         [0.6288, -0.4422],
         [0.7269, 1.6341]]
    )
    observed = model.fe.iloc[:, 1:3].to_numpy()
    assert np.max(np.abs(observed - factor_expected)) < 1e-4


def test_partial_absorption():
    f_partial = "q + endog(p) ~ 0 + id & η1 + fe(id) & η2"
    model_partial = giv(
        df,
        f_partial,
        id = "id",
        t = "t",
        weight = "absS",
        guess={"Aggregate": 2.0},
        algorithm="scalar_search",
    )

    expect = np.array([0.2419, 0.1842, -1.3213, 0.6288, 0.7269])
    assert np.allclose(model_partial.exog_coef[:5], expect, atol=1e-4)


def test_homogeneous_algorithms_agree():
    f = "q + endog(p) ~ 0 + fe(id) & (η1 + η2)"

    m_scalar = giv(df, f, id = "id", t = "t", weight = "absS",
                   guess={"Aggregate": 2.0},
                   algorithm="scalar_search")

    m_iv = giv(df, f, id = "id", t = "t", weight = "absS",
               guess=[1.0], algorithm="iv")

    m_ols = giv(df, f, id = "id", t = "t", weight = "absS",
                guess=[1.0], algorithm="debiased_ols")

    assert np.allclose(m_scalar.coef(), m_iv.coef(), atol=1e-6)
    assert np.allclose(m_scalar.coef(), m_ols.coef(), atol=1e-6)


# --------------------------------------------------------------------
#  2. HETEROGENEOUS ELASTICITY
# --------------------------------------------------------------------
def test_heterogeneous_scalar_search():
    f_het = "q + id & endog(p) ~ 0 + id & (η1 + η2)"

    model = giv(df,
                f_het,
                id = "id",
                t = "t",
                weight = "absS",
                guess={"Aggregate": 2.5},
                algorithm="scalar_search")

    expect_elast = np.array([1.59636, 1.657, 1.29643, 3.33497, 0.58443])
    assert np.allclose(model.endog_coef, expect_elast, atol=1e-4)

    se = np.sqrt(np.diag(model.endog_vcov))
    expect_se = np.array([1.7824, 0.4825, 0.3911, 0.3846, 0.1732])
    assert np.allclose(se, expect_se, atol=1e-4)

    factor_expected = np.array(
        [0.3406, 0.301, -1.3125, 1.2485, 0.5224,
         1.4531, 0.7041, -0.6237, 1.3181, 1.053]
    )
    assert np.allclose(model.exog_coef, factor_expected, atol=1e-4)


def test_heterogeneous_algorithms_agree():
    f_het = "q + id & endog(p) ~ 0 + id & (η1 + η2)"

    m_scalar = giv(df,
                   f_het,
                   id = "id",
                   t = "t",
                   weight = "absS",
                   guess={"Aggregate": 2.5},
                   algorithm="scalar_search")

    m_iv = giv(df,
               f_het,
               id = "id",
               t = "t",
               weight = "absS",
               guess=np.ones(5),
               algorithm="iv",
               tol=1e-8)

    m_ols = giv(df,
                f_het,
                id = "id",
                t = "t",
                weight = "absS",
                guess=np.ones(5),
                algorithm="debiased_ols",
                tol=1e-8)

    assert np.allclose(m_scalar.coef(), m_iv.coef(), atol=1e-6)
    assert np.allclose(m_scalar.coef(), m_ols.coef(), atol=1e-6)


if __name__ == "__main__":
    pytest.main(["-v", __file__])