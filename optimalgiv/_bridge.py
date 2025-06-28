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

##  1. def func_name(input: InputType) -> OutputType:
##  2. (guess: dict) hints the parameter named guess is expected to be a dictionary

def _py_to_julia_guess(guess: dict) -> Any:
    """Handle nested guesses for categorical terms"""
    jl_dict = jl.Dict() # an empty Julia dictionary created by jl (juliacall)
    for term, value in guess.items():
        if isinstance(value, dict):
            # if value is a dictionary type i.e. guess is a nested dic, then do following...
            # guess = {
            #     "group": {"A": 1.0, "B": 2.0},       # nested dict
            #     "id": [0.8, 0.9, 1.1],               # numpy array or list
            #     "Constant": 0.5                      # scalar i.e. a single number
            # }
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

# def _pd_to_jf(df: pd.DataFrame):
#     """Convert a pandas DataFrame to a Julia DataFrame, preserving categorical levels."""
#     cols = {}
#     for name in df.columns:
#         jname = jl.Symbol(name)
#         col = df[name]
#
#         if isinstance(col.dtype, CategoricalDtype):
#             if col.cat.ordered:
#                 raise ValueError(f"Column '{name}' is an ordered categorical, which is not supported. "
#                                  "Please cast it to unordered (e.g., `df['{name}'] = df['{name}'].astype('category')`).")
#
#             # levels = [str(cat) for cat in col.dtype.categories]
#             # data_vec = [str(v) for v in col]
#             levels = list(col.dtype.categories)
#             # data_vec = col.to_numpy(copy=False)
#             data_vec = col.tolist()
#
#             jcol = jl.categorical(data_vec, levels=levels, ordered=False)
#
#         else:
#             jcol = col.to_numpy()
#
#         cols[jname] = jcol
#
#     return jl.DataFrame(cols)


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
        # formula::FormulaTerm; convert it to str first then complete conversion in giv()

        self.residual_variance = np.asarray(jl_model.residual_variance)
        # A Symbol is an immutable, interned (i.e. pointer based) identifier while a str is char by char.
        # e.g. Every Symbol('abc') i.e. :abc is identical as there is only one pointer pointing at all Symbol('abc')
        # So, :abc === Symbol('abc') must be true as they share the same pointer => O(1) operation
        # while for str in Python: 'abc' == 'abc' by comparing 'a' == 'a' , 'b' == 'b', 'c' == 'c' => O(n)
        # ====> So, as all names has to be unique, why not store them in the memory with a much more speedy way?
        #       (e.g. variable names, keys in Dict, function names. column names in Dataframe)
        self.responsename      = str(jl_model.responsename)
        self.endogname         = str(jl_model.endogname)
        self.endog_coefnames = [str(n) for n in jl_model.endog_coefnames]
        self.exog_coefnames = [str(n) for n in jl_model.exog_coefnames]
        self.idvar             = str(jl_model.idvar)
        self.tvar              = str(jl_model.tvar)
        wv = jl_model.weightvar
        self.weightvar         = str(wv) if wv is not jl.nothing else None
        # self.exclude_pairs     = [(p.first, p.second)
        #                           for p in jl_model.exclude_pairs] # list of pairs (exclude_pairs::Vector{Pair})
        jl_dict = jl_model.exclude_pairs
        self.exclude_pairs = {
            int(k): [int(x) for x in jl_dict[k]]
            for k in jl.Base.keys(jl_dict)
        }

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

        ## straightforward but always return errors so we now use a munual way instead:

        # self.coefdf = jl.convert(pd.DataFrame, jl_model.coefdf)
        # self.df = jl.convert(pd.DataFrame, jl_model.df)

        # get_col = jl.seval("(df, col) -> df[!, Symbol(col)]") # similar to return df.loc[:, str(col)] in python (not copy)
        #
        # j_coefdf    = jl_model.coefdf
        # j_coef_names = jl.seval("names")(j_coefdf)
        # coefdf_dict = {
        #     str(nm): np.asarray(get_col(j_coefdf, nm))  # Extract each column (as a Julia vector) from the DataFrame
        #     for nm in j_coef_names
        # }
        # self.coefdf = pd.DataFrame(coefdf_dict)
        #
        # j_df = jl_model.df
        # if j_df is not jl.nothing:
        #     j_names = jl.seval("names")(j_df)
        #     df_dict = {
        #         str(nm): np.asarray(get_col(j_df, nm))
        #         for nm in j_names
        #     }
        #     self.df = pd.DataFrame(df_dict)
        # else:
        #     self.df = None

    # def coefficient_table(self) -> pd.DataFrame:
    #     """Return the full coefficient table as DataFrame"""
    #     return coefficient_table(self._jl_model)

    def coef(self):
        return np.concatenate([self.endog_coef, self.exog_coef])

    def coefnames(self):
        return self.endog_coefnames + self.exog_coefnames

    # def endog_coefnames(self):
    #     return self.endog_coefnames
    #
    # def exog_coefnames(self):
    #     return self.exog_coefnames
    #
    # def endog_vcov(self):
    #     return self.endog_vcov
    #
    # def exog_vcov(self):
    #     return self.exog_vcov

    def vcov(self):
        n_endog = len(self.endog_coef)
        n_exog = len(self.exog_coef)
        top = np.hstack([self.endog_vcov, np.full((n_endog, n_exog), np.nan)])
        bottom = np.hstack([np.full((n_exog, n_endog), np.nan), self.exog_vcov])
        return np.vstack([top, bottom])

    def confint(self, level=0.95):
        """
        (n×2) NumPy array of confidence intervals at the requested level.
        """
        return np.asarray(
            jl.StatsAPI.confint(self._jl_model, level=level)
        )

    def stderror(self):
        se_endog = np.sqrt(np.diag(self.endog_vcov))
        se_exog = np.sqrt(np.diag(self.exog_vcov))
        return np.concatenate([se_endog, se_exog])

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
            index=self.coefnames,
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
    """Estimate a GIV model from pandas data."""

    # jdf      = jl.DataFrame(df)
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
        kwargs["contrasts"] = {
            jl.Symbol(k) if isinstance(k, str) else k: v
            for k, v in kwargs["contrasts"].items()
        }
    if isinstance(kwargs.get("solver_options"), dict):
        # Turn {"ftol":1e-8, "xtol":1e-8} -> (; ftol=1e-8, xtol=1e-8)
        kwargs["solver_options"] = jl.NamedTuple(kwargs["solver_options"])

    g = kwargs.get("guess", None)
    if isinstance(g, dict):
        kwargs["guess"] = _py_to_julia_guess(g)
    elif isinstance(g, (list, tuple, np.ndarray)):
        # Julia expects Vector{Float64}
        kwargs["guess"] = jl.seval("Vector{Float64}")([float(x) for x in g])
    elif g is None:
        pass  # let Julia fall back to its default heuristics
    else:  # scalar number
        kwargs["guess"] = float(g)

    return GIVModel(jl.giv(jdf, jformula, jid, jt, jweight, **kwargs))

## e.g. :
## kwargs = {
##    "guess": {"group": [1, 2, 3]},
##    "algorithm": "iv"
## }
## ====> **kwargs means: Unpack the kwargs dictionary and pass each key-value pair as a named argument to the Julia giv(...) function


# ---------------------------------------------------------------------------
# Coefficient Table Generator
# ---------------------------------------------------------------------------

## In givmodels.jl, function coeftable will only return a named tuple with 3 fields, so we have to use PrettyTables.jl to show table output in Julia
## However, output via PrettyTables.jl is printed directly to the terminal so can't be returned as a formatted result through the API.
## ===> So we have to manually extract named tuple output from jl_model.coeftable


# def coefficient_table(jl_model: Any) -> pd.DataFrame:
#     """Get full statistical summary from Julia model"""
#
#     ct = jl.seval("OptimalGIV.coeftable")(jl_model)
#
#     ## cols: list of arrays (data columns)
#     ## colnms: column names (e.g., "Estimate")
#     ## rownms: row labels (e.g., "group: 1")
#
#     cols = jl.seval("""
#     function getcols(ct)
#         cols = [ct.cols[i] for i in 1:length(ct.cols)]
#         (; cols=cols, colnms=ct.colnms, rownms=ct.rownms)
#     end
#     """)(ct)
#
#     df = pd.DataFrame(
#         np.column_stack(cols.cols), ## Combines the list of 1D arrays (e.g., estimates, std errors) into a 2D array
#         columns=list(cols.colnms)   ## Assigns column names like "Estimate", "Std. Error", etc
#     )
#     if cols.rownms:                 ## If row names exist (e.g., "group: 0"), insert them as the first column in the DataFrame, called "Term"
#         df.insert(0, "Term", list(cols.rownms))
#
#     if "Pr(>|t|)" in df.columns:
#         df["Pr(>|t|)"] = df["Pr(>|t|)"].astype(float)
#     ## In Julia, p-values might appear as strings (like "<1e-37") instead of floats
#     ## This line forces them into float type so you can do math, sorting, filtering, etc. in Python
#
#     return df

