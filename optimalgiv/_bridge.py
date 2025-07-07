"""
_bridge.py
----------

• Boots Julia through **JuliaCall**
• Imports the registered package **OptimalGIV**

"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Optional
from juliacall import Main as jl, AnyValue
from pandas.api.types import CategoricalDtype
import math


# ---------------------------------------------------------------------
# One-time Julia initialisation
# ---------------------------------------------------------------------


def _py_to_julia_guess(guess: dict) -> Any:
    """Handle nested guesses for categorical terms"""
    jl_dict = jl.Dict()
    for term, value in guess.items():
        if isinstance(value, dict):
            jl_subdict = jl.Dict()
            for k, v in value.items():
                jl_subdict[str(k)] = float(v)
            jl_dict[term] = jl_subdict
        elif isinstance(value, (list, np.ndarray)):
            jl_dict[term] = jl.convert(jl.Vector[jl.Float64],
                                        [float(x) for x in value])
        else:
            jl_dict[term] = float(value)
    return jl_dict

# ------------------------------------------------------------------
# Julia DataFrame ↔ pandas DataFrame helpers
# ------------------------------------------------------------------
def _jf_to_pd(jdf):
    """
    Convert a Julia DataFrame to a pandas DataFrame column-by-column.
    """
    get_col = jl.seval("(df, col) -> df[!, Symbol(col)]")
    j_names = jl.seval("names")(jdf)

    cols = {
        str(nm): np.asarray(get_col(jdf, nm))
        for nm in j_names
    }

    return pd.DataFrame(cols)

def _pd_to_jf(df: pd.DataFrame):
    """
    Convert a pandas DataFrame to a Julia DataFrame, translating
    np.nan / None → Julia `missing` and preserving categorical levels.
    """
    cols = {}
    _jmissing = jl.missing

    # Pre-compile the Julia constructors we need once for speed
    _V_M_F64 = jl.seval("Vector{Union{Missing, Float64}}")
    _V_M_I64 = jl.seval("Vector{Union{Missing, Int64}}")
    _V_M_BOOL = jl.seval("Vector{Union{Missing, Bool}}")
    _V_M_STR = jl.seval("Vector{Union{Missing, String}}")

    for name in df.columns:
        jname = jl.Symbol(name)
        col = df[name]

        # ---------- CATEGORICAL ----------
        if isinstance(col.dtype, CategoricalDtype):
            if col.cat.ordered:
                raise ValueError(
                    f"Column '{name}' is an *ordered* categorical, "
                    "which OptimalGIV does not handle. "
                    "Cast it to unordered first."
                )

            levels = list(col.dtype.categories)
            data_vec = [
                _jmissing if pd.isna(x) else x
                for x in col.tolist()
            ]
            jcol = jl.categorical(data_vec, levels=levels, ordered=False)

        # ---------- NUMERIC: FLOAT ----------
        elif col.dtype.kind == "f":
            data_vec = [
                _jmissing if (isinstance(x, float) and math.isnan(x)) else float(x)
                for x in col.to_numpy()
            ]
            jcol = _V_M_F64(data_vec)

        # ---------- NUMERIC: (SIGNED) INT ----------
        elif col.dtype.kind in ("i", "u"):
            # pandas nullable Int64Dtype already uses <NA>; others use NaN
            data_vec = [
                _jmissing if pd.isna(x) else int(x)
                for x in col.to_numpy(dtype="object")
            ]
            jcol = _V_M_I64(data_vec)

        # ---------- BOOLEAN ----------
        elif col.dtype.kind == "b":
            data_vec = [
                _jmissing if pd.isna(x) else bool(x)
                for x in col.to_numpy(dtype="object")
            ]
            jcol = _V_M_BOOL(data_vec)

        # ---------- EVERYTHING ELSE (objects, strings, datetimes) ----------
        else:
            data_vec = [
                _jmissing if pd.isna(x) else str(x)
                for x in col.to_numpy(dtype="object")
            ]
            jcol = _V_M_STR(data_vec)

        cols[jname] = jcol

    return jl.DataFrame(cols)

# ---------------------------------------------------------------------------
# Model Wrapper
# ---------------------------------------------------------------------------
class GIVModel:
    """Python-native wrapper for Julia GIV results"""

    def __init__(self, jl_model: Any):
        self._jl_model = jl_model

        self.endog_coef = np.asarray(jl_model.endog_coef)
        self.exog_coef = np.asarray(jl_model.exog_coef)
        self.endog_vcov = np.asarray(jl_model.endog_vcov)
        self.exog_vcov = np.asarray(jl_model.exog_vcov)

        agg = jl_model.agg_coef
        if isinstance(agg, (int, float)):
            self.agg_coef = float(agg)
        elif hasattr(agg, '__len__') and len(agg) == 1:
            self.agg_coef = float(agg[0])
        else:
            self.agg_coef = np.asarray(agg)

        self.complete_coverage = bool(jl_model.complete_coverage)
        self.formula           = str(jl_model.formula)
        self.formula_schema = str(jl_model.formula_schema)
        self.residual_variance = np.asarray(jl_model.residual_variance)
        self.responsename      = str(jl_model.responsename)
        self.endogname         = str(jl_model.endogname)
        self.endog_coefnames = [str(n) for n in jl_model.endog_coefnames]
        self.exog_coefnames = [str(n) for n in jl_model.exog_coefnames]
        self.idvar             = str(jl_model.idvar)
        self.tvar              = str(jl_model.tvar)
        wv = jl_model.weightvar
        self.weightvar         = str(wv) if wv is not jl.nothing else None

        jl_dict = jl_model.exclude_pairs
        ep = {}
        for k in jl.Base.keys(jl_dict):
            # try int, else str
            try:
                kk = int(k)
            except Exception:
                kk = str(k)
            # same for the values list
            raw = jl_dict[k]
            vals = []
            for x in raw:
                try:
                    vals.append(int(x))
                except Exception:
                    vals.append(str(x))
            ep[kk] = vals
        self.exclude_pairs = ep

        self.converged         = bool(jl_model.converged)
        self.N                 = int(jl_model.N)
        self.T                 = int(jl_model.T)
        self.nobs              = int(jl_model.nobs)
        self.dof               = int(jl_model.dof)
        self.dof_residual      = int(jl_model.dof_residual)

        self.coefdf = _jf_to_pd(jl_model.coefdf)
        self.df = (_jf_to_pd(jl_model.df)
                   if jl_model.df is not jl.Base.nothing else None)
        self.fe = (_jf_to_pd(jl_model.fe)
                   if jl_model.fe is not jl.Base.nothing else None)
        self.residual_df = (_jf_to_pd(jl_model.residual_df)
                            if jl_model.residual_df is not jl.Base.nothing else None)

        self.coef = np.concatenate([self.endog_coef, self.exog_coef])
        self.coefnames = self.endog_coefnames + self.exog_coefnames

        n_endog = len(self.endog_coef)
        n_exog = len(self.exog_coef)
        top = np.hstack([self.endog_vcov, np.full((n_endog, n_exog), np.nan)])
        bottom = np.hstack([np.full((n_exog, n_endog), np.nan), self.exog_vcov])
        self.vcov = np.vstack([top, bottom])

        se_endog = np.sqrt(np.diag(self.endog_vcov))
        se_exog = np.sqrt(np.diag(self.exog_vcov))
        self.stderror = np.concatenate([se_endog, se_exog])

    def confint(self, level=0.95):
        """
        (n×2) NumPy array of confidence intervals at the requested level.
        """
        return np.asarray(
            jl.StatsAPI.confint(self._jl_model, level=level)
        )

    def residuals(self):
        """
        Raw residual vector (NumPy 1-D). Requires the model to have been
        fitted with `save_df=True`; otherwise raises RuntimeError.
        """
        if self.df is None:
            raise RuntimeError(
                "DataFrame not saved. Re-run the model with `save_df=True`"
            )
        col = f"{self.responsename}_residual"
        return self.df[col].to_numpy()

    def coeftable(self, level: float = 0.95) -> pd.DataFrame:
        """
        Return a pandas.DataFrame equivalent to Julia’s `StatsAPI.coeftable`.
        """
        est = self.coef()
        se = self.stderror()
        tstat = est / se
        dof = self.dof_residual

        # p-values: ccdf of F(1, dof) evaluated at t²
        abs_t = np.abs(tstat)
        pvals = np.array([jl.fdistccdf(1, int(dof), float(tt ** 2)) for tt in abs_t])

        ci = self.confint(level=level)  # shape (n, 2)
        lower, upper = ci[:, 0], ci[:, 1]

        colnms = [
            "Estimate",
            "Std. Error",
            "t-stat",
            "Pr(>|t|)",
            f"Lower {int(level * 100)}%",
            f"Upper {int(level * 100)}%",
        ]
        df = pd.DataFrame(
            np.column_stack([est, se, tstat, pvals, lower, upper]),
            columns=colnms,
            index=self.coefnames(),
        )
        return df

    def summary(self):
        jl.Base.show(jl.Base.stdout, self._jl_model)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def giv(
    df: pd.DataFrame,
    formula: str,
    *,
    id: str,
    t: str,
    weight: Optional[str] = None,
    **kwargs: Any, ## allows extra arguments
) -> GIVModel:

    jdf = _pd_to_jf(df)
    jformula = jl.seval(f"@formula({formula})")
    jid      = jl.Symbol(id)
    jt       = jl.Symbol(t)
    jweight  = jl.Symbol(weight) if weight else jl.nothing

    # Handle keyword arguments
    if isinstance(kwargs.get("algorithm"), str):
        kwargs["algorithm"] = jl.Symbol(kwargs["algorithm"])
    if isinstance(kwargs.get("save"), str):
        kwargs["save"] = jl.Symbol(kwargs["save"])

    if isinstance(kwargs.get("contrasts"), dict):
        contrasts_arg = kwargs["contrasts"]
        if contrasts_arg is not None:
            jl_contrasts = jl.seval("Dict{Symbol, Any}()")

            for k, v in contrasts_arg.items():
                # Convert key to Julia Symbol
                jkey = jl.Symbol(k)

                # Handle string specifications
                if isinstance(v, str):
                    jval = jl.seval(f"StatsModels.{v}()")
                # Handle direct Julia objects
                elif hasattr(v, '__call__'):
                    jval = v
                else:
                    raise TypeError(f"Unsupported contrast type: {type(v)}")

                jl_contrasts[jkey] = jval

            kwargs["contrasts"] = jl_contrasts
        else:
            # Create empty Julia Dict{Symbol, Any}
            kwargs["contrasts"] = jl.seval("Dict{Symbol, Any}()")

    if isinstance(kwargs.get("solver_options"), dict):
        py_opts = kwargs.pop("solver_options")
        jl_opts = jl.seval("Dict{Symbol, Any}()")
        for py_key, py_val in py_opts.items():
            jkey = jl.Symbol(py_key)

            if py_key == "linesearch":
                if isinstance(py_val, str):
                    # e.g. "HagerZhang" → LineSearches.HagerZhang()
                    jval = jl.seval(f"LineSearches.{py_val}()")
                else:
                    # assume they already passed jl.LineSearches.HagerZhang()
                    jval = py_val

            elif isinstance(py_val, str):
                jval = jl.Symbol(py_val)

            else:
                jval = py_val

            jl_opts[jkey] = jval

        jl.seval("""
            function create_namedtuple_from_dict(d::Dict{Symbol, Any})
                return (; d...)
            end
        """)
        _create_namedtuple = jl.create_namedtuple_from_dict

        kwargs["solver_options"] = _create_namedtuple(jl_opts)

    g = kwargs.get("guess", None)
    if isinstance(g, dict):
        kwargs["guess"] = _py_to_julia_guess(g)
    elif isinstance(g, (list, tuple, np.ndarray)):
        kwargs["guess"] = jl.seval("Vector{Float64}")([float(x) for x in g])
    elif g is None:
        pass
    else:
        kwargs["guess"] = float(g)

    return GIVModel(jl.giv(jdf, jformula, jid, jt, jweight, **kwargs))