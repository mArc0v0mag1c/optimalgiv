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
from juliacall import Main as jl



# ---------------------------------------------------------------------
# One-time Julia initialisation
# ---------------------------------------------------------------------

##  1. def func_name(input: InputType) -> OutputType:
##  2. _ prefix = “private” by convention

##  1. (guess: dict) hints the parameter named guess is expected to be a dictionary

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


# ---------------------------------------------------------------------------
# Model Wrapper
# ---------------------------------------------------------------------------
class GIVModel:
    """Python-native wrapper for Julia GIV results"""

    def __init__(self, jl_model: Any):
        self._jl_model = jl_model

        self.coef              = np.asarray(jl_model.coef)
        self.vcov              = np.asarray(jl_model.vcov)
        self.factor_coef       = np.asarray(jl_model.factor_coef)
        self.factor_vcov       = np.asarray(jl_model.factor_vcov)
        agg = jl_model.agg_coef  # agg_coef::Union{Float64,Vector{Float64}}
        try:
            self.agg_coef = float(agg)
        except (TypeError, ValueError):
            self.agg_coef = np.asarray(agg)
        self.formula           = str(jl_model.formula)
        # formula::FormulaTerm; convert it to str first then complete conversion in giv()
        self.price_factor_coef = np.asarray(jl_model.price_factor_coef)
        self.residual_variance = np.asarray(jl_model.residual_variance)
        # A Symbol is an immutable, interned (i.e. pointer based) identifier while a str is char by char.
        # e.g. Every Symbol('abc') i.e. :abc is identical as there is only one pointer pointing at all Symbol('abc')
        # So, :abc === Symbol('abc') must be true as they share the same pointer => O(1) operation
        # while for str in Python: 'abc' == 'abc' by comparing 'a' == 'a' , 'b' == 'b', 'c' == 'c' => O(n)
        # ====> So, as all names has to be unique, why not store them in the memory with a much more speedy way?
        #       (e.g. variable names, keys in Dict, function names. column names in Dataframe)
        self.responsename      = str(jl_model.responsename)
        self.endogname         = str(jl_model.endogname)
        self.coefnames         = list(jl_model.coefnames)
        self.factor_coefnames  = list(jl_model.factor_coefnames)
        self.idvar             = str(jl_model.idvar)
        self.tvar              = str(jl_model.tvar)
        wv = jl_model.weightvar
        self.weightvar         = str(wv) if wv is not jl.nothing else None
        self.exclude_pairs     = [(p.first, p.second)
                                  for p in jl_model.exclude_pairs] # list of pairs (exclude_pairs::Vector{Pair})
        self.converged         = bool(jl_model.converged)
        self.N                 = int(jl_model.N)
        self.T                 = int(jl_model.T)
        self.nobs              = int(jl_model.nobs)
        self.dof               = int(jl_model.dof)
        self.dof_residual      = int(jl_model.dof_residual)

        # Helper to extract Julia DataFrame columns
        get_col = jl.seval("(df, col) -> df[!, Symbol(col)]") # similar to return df.loc[:, str(col)] in python (not copy)

        j_coefdf    = jl_model.coefdf
        j_coef_names = jl.seval("names")(j_coefdf)
        coefdf_dict = {
            str(nm): np.asarray(get_col(j_coefdf, nm))  # Extract each column (as a Julia vector) from the DataFrame
            for nm in j_coef_names
        }
        self.coefdf = pd.DataFrame(coefdf_dict)

        j_df = jl_model.df
        if j_df is not jl.nothing:
            j_names = jl.seval("names")(j_df)
            df_dict = {
                str(nm): np.asarray(get_col(j_df, nm))
                for nm in j_names
            }
            self.df = pd.DataFrame(df_dict)
        else:
            self.df = None

    def coefficient_table(self) -> pd.DataFrame:
        """Return the full coefficient table as DataFrame"""
        return coefficient_table(self._jl_model)


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

    jdf      = jl.DataFrame(df)
    jformula = jl.seval(f"@formula({formula})")
    jid      = jl.Symbol(id)
    jt       = jl.Symbol(t)
    jweight  = jl.Symbol(weight) if weight else jl.nothing

    # Handle keyword arguments
    if isinstance(kwargs.get("algorithm"), str):
        kwargs["algorithm"] = jl.Symbol(kwargs["algorithm"])
    if isinstance(kwargs.get("guess"), dict):
        kwargs["guess"] = _py_to_julia_guess(kwargs["guess"])

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


def coefficient_table(jl_model: Any) -> pd.DataFrame:
    """Get full statistical summary from Julia model"""

    ct = jl.seval("OptimalGIV.coeftable")(jl_model)

    ## cols: list of arrays (data columns)
    ## colnms: column names (e.g., "Estimate")
    ## rownms: row labels (e.g., "group: 1")

    cols = jl.seval("""
    function getcols(ct)
        cols = [ct.cols[i] for i in 1:length(ct.cols)]
        (; cols=cols, colnms=ct.colnms, rownms=ct.rownms)
    end
    """)(ct)

    df = pd.DataFrame(
        np.column_stack(cols.cols), ## Combines the list of 1D arrays (e.g., estimates, std errors) into a 2D array
        columns=list(cols.colnms)   ## Assigns column names like "Estimate", "Std. Error", etc
    )
    if cols.rownms:                 ## If row names exist (e.g., "group: 0"), insert them as the first column in the DataFrame, called "Term"
        df.insert(0, "Term", list(cols.rownms))

    if "Pr(>|t|)" in df.columns:
        df["Pr(>|t|)"] = df["Pr(>|t|)"].astype(float)
    ## In Julia, p-values might appear as strings (like "<1e-37") instead of floats
    ## This line forces them into float type so you can do math, sorting, filtering, etc. in Python

    return df
