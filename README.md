# optimalgiv

[![CI](https://github.com/mArc0v0mag1c/optimalgiv/actions/workflows/ci.yml/badge.svg)](https://github.com/mArc0v0mag1c/optimalgiv/actions/workflows/ci.yml)

A Python wrapper for the Julia package [OptimalGIV.jl](https://github.com/FuZhiyu/OptimalGIV.jl)

This wrapper uses [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl) to call the Julia package directly from Python. Julia is automatically installed and all dependencies are resolved without manual setup. 

**This python package is under active development** The core algorithms are implemented in Julia, and thoroughly tested under simulations, but documentations are working in progress, and bugs may exists for minor features. Feature requests and bug reports are welcomed. 

> This README focuses on the use for Python API.  For more technical documentation, please see [the Julia package](https://github.com/FuZhiyu/OptimalGIV.jl/blob/main/README.md) and the [companion paper](https://fuzhiyu.me/TreasuryGIVPaper/Treasury_GIV_draft.pdf).

> Support for multithreading by PythonCall.jl is experimental** If you encounter segfault, please set `JULIA_NUM_THREADS = 1` in the environmental variables.
---

## Installation

```python
pip install optimalgiv
```

### First import

The first time you run:

```python
import optimalgiv as og
```

it will:

1. **Install Julia** (if not present; ≈ 1-2 min),
2. **Set up a Julia environment** with `OptimalGIV.jl` and **precompile** (≈ 2–4 min).

Later imports will be much faster (≈ 6–10 s), which is typical for Julia project activation—the environment is compiled once and then reused.


---

## Model Specification

The Granular Instrumental Variables (GIV) model estimated by this package follows the specification:

```math
\begin{aligned}
\left.\begin{array}{c}
\begin{array}{cl}
q_{i,t} & =-p_{t}\times\mathbf{C}_{i,t}'\boldsymbol{\zeta}+\mathbf{X}_{i,t}'\boldsymbol{\beta}+u_{i,t},\\
0 & =\sum_{i}S_{i,t}q_{i,t}
\end{array}\end{array}\right\} \implies & p_{t}=\frac{1}{\mathbf{C}_{S,t}'\boldsymbol{\zeta}}\left[\mathbf{X}_{S,t}'\boldsymbol{\beta}+u_{S,t}\right],
\end{aligned}
```


where:

* $q_{i,t}$ and $p_t$ are endogenous,
* $\mathbf{C}_{i,t}$ is a vector of controls for slopes,
* $\mathbf{X}_{i,t}$ is a vector of controls,
* $\boldsymbol{\zeta}$, $\boldsymbol{\beta}$ are coefficient vectors,
* $u_{i,t}$ is the idiosyncratic shock, and
* $S_{i,t}$ is the weighting variable.

The equilibrium price $p_t$ is derived by imposing the market clearing condition and the model is estimated using the moment condition:

$$
\mathbb{E}[u_{i,t} u_{j,t}] = 0
$$

for all $i \neq j$. This implies orthogonality across sectors' residuals.

---

### Panel Data and Coverage

The GIV model supports unbalanced panel data. However, some estimation algorithms (e.g. "scalar_search" and "debiased_ols") **require complete coverage**, meaning:

$$
\sum_i S_{i,t} q_{i,t} = 0
$$

must hold exactly **within the sample**. This ensures internal consistency of the equilibrium condition. 

If the adding-up constraint is not satisfied, the model will adjust accordingly, but **the interpretation of estimated coefficients should be made with caution**, as residual market imbalances may bias elasticities and standard errors. (See the `complete_coverage` argument below for details.)

---

### Internal PC

Internal PC extractions are supported. With internal PCs, the moment conditions become:

$$
\mathbb E[u_{i,t}u_{j,t}] = \Lambda \Lambda'
$$

where $\Lambda$ is the factor loadings estimated internally using [HeteroPCA.jl](https://github.com/FuZhiyu/HeteroPCA.jl) from $u_{i,t}(z) \equiv q_{i,t} + p_{t}\times\mathbf{C}_{i,t}'\boldsymbol{z}$ at each guess of $z$. 

However, with small samples, the exactly root solving the moment condition may not exist, and users may want to use an minimizer to minimize the error instead. Also, be noted that a model with fully flexible elasticity specification and fully flexible factor loadings is not theoretically identifiable. 


---

## Usage

### Basic Example

```python
import pandas as pd
import numpy as np
from optimalgiv import giv

df = pd.read_csv("./simdata1.csv") # you can find simdata under the git repo examples/
# or simulate using simulate_data below

df['id'] = df['id'].astype('category') # ensure id interactions map to distinct groups

# Define the model formula
formula = "q + id & endog(p) ~ 0 + fe(id) + fe(id) & (η1 + η2)"

# Provide an initial guess (a good guess is critical)
guess = np.ones(5)

# Estimate the model
model = giv(
    df = df,
    formula = "q + id & endog(p) ~ 0 + fe(id) + fe(id) & (η1 + η2)",
    id = "id",
    t = "t",
    weight = "absS",
    algorithm = "iv",
    guess = guess,
    save = 'all', # saves both fixed‐effects (model.fe) and residuals (model.residual_df)
)

# View the result
model.summary()

##                     GIVModel (Aggregate coef: 2.13)                     
## ─────────────────────────────────────────────────────────────────────────
##            Estimate  Std. Error    t-stat  Pr(>|t|)  Lower 95%  Upper 95%
## ─────────────────────────────────────────────────────────────────────────
## id: 1 & p  1.00723     1.30407   0.772377    0.4405  -1.55923    3.57369
## id: 2 & p  1.77335     0.475171  3.73204     0.0002   0.8382     2.70851
## id: 3 & p  1.36863     0.382177  3.58114     0.0004   0.616491   2.12077
## id: 4 & p  3.3846      0.382352  8.85207     <1e-16   2.63212    4.13709
## id: 5 & p  0.619882    0.161687  3.83385     0.0002   0.301676   0.938087


```
---

### Formula Specification

The model formula follows the convention:

```python
q + interactions & endog(p) ~ exog_controls + pc(k)
```

Where:

* `q`: **Response variable** (e.g., quantity).
* `endog(p)`: **Endogenous variable** (e.g., price). Must appear on the **left-hand side**.

  > **Note:** A *positive* estimated coefficient implies a *negative* response of `q` to `p` (i.e., a downward-sloping demand curve).
* `interactions`: Exogenous variables used to parameterize **heterogeneous elasticities**, such as entity identifiers or group characteristics.
* `exog_controls`: Exogenous control variables. Supports **fixed effects** (e.g., `fe(id)`) using the same syntax as `FixedEffectModels.jl`.
* `pc(k)`: Principal component extraction with `k` factors (optional). When specified, `k` common factors are extracted from residuals using HeteroPCA.jl 


#### Examples of formulas:

```
# Homogeneous elasticity with entity-specific loadings (estimated) and fixed effects (absorbed)
formula = "q + endog(p) ~ id & η + fe(id)"

# Heterogeneous elasticity by entity
formula = "q + id & endog(p) ~ id & η + fe(id)"

# Multiple interactions
formula = "q + id & endog(p) + category & endog(p) ~ fe(id) & η1 + η2"

formula = "q + id & endog(p) ~ 0 + id & η"

# With PC extraction (2 factors)
formula = "q + endog(p) ~ 0 + pc(2)"

# exogneous controls with PC extraction
formula = "q + endog(p) ~ fe(id) & η1 + pc(3)"
```
---

### Key Function: `giv()`
```python
giv(df, formula: str, id: str, t: str, weight: str, **kwargs) -> GIVModel
```

#### Required Arguments

* `df`: `pandas.DataFrame` containing panel data. **Must be balanced** for some algorithms (e.g., `scalar_search`).
* `formula`: A **string** representing the model (Julia-style formula syntax). See examples above.
* `id`: Name of the column identifying entities (e.g., `"firm_id"`).
* `t`: Name of the time variable column.
* `weight`: Name of the weight/size column (e.g., market shares `S_i,t`).

#### Keyword Arguments (Optional)

* `algorithm`: One of `"iv"` (default), `"iv_twopass"`, `"debiased_ols"`, or `"scalar_search"`.
* `guess`: Initial guess for ζ coefficients. (See below for usage details)
* `exclude_pairs`: Dictionary excluding pairs from moment conditions.
    * Example: `{1: [2, 3], 4: [5]}` excludes entity pair with code (1,2), (1,3), and (4,5) from the moment conditions entering the estimation. 
* `quiet`: Set `True` to suppress warnings and info messages.
* `save`: `"none"` (default), `"residuals"`, `"fe"`, or `"all"` — controls what is stored on the returned model:

  * `"none"`: neither residuals nor fixed-effects are saved
  * `"residuals"`: saves residuals in `model.residual_df`
  * `"fe"`: saves fixed-effects in `model.fe`
  * `"all"`: saves both `model.residual_df` and `model.fe`

* `save_df`: If `True`, the full estimation dataframe (with residuals, coefficients, fixed effects) is stored in `model.df`.
* `complete_coverage`: Whether the dataset **covers the full market in each time period**, meaning
$\sum_i S_{i,t} q_{i,t} = 0$ holds exactly within the sample.

  * Default is `None`, which triggers auto-detection: the model checks this condition period-by-period and sets the flag to `True` or `False` accordingly.
  * If the condition does not hold (`False`), you can still force estimation by setting `quiet=True`, but results may be biased. Use with caution.
  * Required for `"scalar_search"` and `"debiased_ols"` algorithms.

* `return_vcov`: Whether to compute and return the variance–covariance matrices. (default: `True`)
* `tol`: Convergence tolerance for the solver (: `1e-6`)
* `iterations`: Maximum number of solver iterations (: `100`)
* `pca_option`: Dictionary of options for PC extraction when using `pc(k)` in formula:
  * `'algorithm'`: HeteroPCA algorithm - `DeflatedHeteroPCA`, `'StandardHeteroPCA'`, or `'DiagonalDeletion'`
  * `'impute_method'`: `'zero'` or `'pairwise'` for handling missing values (default: `'zero'`)
  * `'demean'`: Whether to demean data before PCA (default: `False`)
  * `'maxiter'`: Maximum iterations for PCA algorithm (default: `100`)

#### Advanced keyword arguments (Optional; Use with caution)

* **`solver_options`** (`Dict[str, Any]`)
  Extra options passed to the nonlinear system solver from [`NLsolve.jl`](https://github.com/JuliaNLSolvers/NLsolve.jl).
  The Python dict is converted to a Julia `NamedTuple` with keyword-style arguments.
  Common options include:

  * `"method"`: `"newton"` , `"anderson"`, `"trust_region"`, etc.
  * `"ftol"`: absolute residual tolerance
  * `"xtol"`: absolute solution tolerance
  * `"iterations"`: max iterations
  * `"show_trace"`: verbose output

  **Example:**

  ```python
  solver_opts = {
      "method": "newton",
      "ftol": 1e-8,
      "xtol": 1e-8,
      "iterations": 1000,
      "show_trace": True,
  }

  model = giv(df, formula, id="id", t="t", solver_options=solver_opts)
  ```

  For the full list of options, see the [NLsolve.jl documentation](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/).
---

### Algorithms

The package implements four algorithms for GIV estimation:

1. **`"iv"`**  
   - Default, recommended  
   - Uses moment condition $$\(\mathbb{E}[u_i\,u_{S,-i}]=0\)$$  
   - $$O(N)\$$ implementation  
   - Supports `exclude_pairs` (exclude certain pairs $E[u_i u_j] = 0$ from the moment conditions)
   - Supports flexible elasticity specs, unbalanced panels  

2. **`"iv_twopass"`**: Numerically identical to `iv` but uses a more straightforward O(N²) implementation with two passes over entity pairs. This is useful for:
   - Debugging purposes
   - When the O(N) optimization in `iv` might cause numerical issues
   - When there are many pairs to be excluded, which will slow down the algorithm in `iv`
   - Understanding the computational flow of the moment conditions 

5. **`"debiased_ols"`**  
   - Uses $$\mathbb{E}[u_iC_{it}p_{it}] = \sigma_i^2 / \zeta_{St}$$
   - Requires **complete market coverage**  
   - More efficient but restrictive  

6. **`"scalar_search"`**  
   - Finds a single aggregate elasticity  
   - Requires **balanced panel, constant weights, complete coverage** 
   - Useful for diagnostics or initial-guess formation  

---

### Initial Guesses

A good guess is key to stable estimation. If omitted, OLS‐based defaults will typically fail. Examples:

```python
import numpy as np
from optimalgiv import giv
# 1) Scalar guess (for homogeneous elasticity)
guess = 1.0
model1 = giv(
    df,
    "q + endog(p) ~ n1 + fe(id)",
    id="id", t="t", weight="S",
    guess=guess
)

# 2) Dict by group name (heterogeneous by id)
guess = {"id": [1.2, 0.8]}
model2 = giv(
    df,
    "q + id & endog(p) ~ 1",
    id="id", t="t", weight="S",
    guess=guess
)

# 3) Dict for multiple interactions
guess = {
    "id": [1.0, 0.9],
    "n1": [0.5, 0.3]
}
model3 = giv(
    df,
    "q + id & endog(p) + n1 & endog(p) ~ fe(id)",
    id="id", t="t", weight="S",
    guess=guess
)

# 4) Dict keyed by exact coefnames
names = model3.coefnames
guess = {name: 0.1 for name in names}
model4 = giv(
    df,
    "q + id & endog(p) + n1 & endog(p) ~ fe(id)",
    id="id", t="t", weight="S",
    guess=guess
)

# 5) Scalar-search with heterogeneous formula
guess = {"Aggregate": 2.5}
model5 = giv(
    df,
    "q + id & endog(p) ~ 0 + fe(id) + fe(id)&(n1 + n2)",
    id="id", t="t", weight="S",
    algorithm="scalar_search",
    guess=guess
)

# 6) Use estimated ζ from model5 as initial guess
guess = model5.endog_coef
model6 = giv(
    df,
    "q + id & endog(p) ~ 0 + fe(id) + fe(id)&(n1 + n2)",
    id="id", t="t", weight="S",
    guess=guess
)

```
---

### Principal Components (PC) in Formulas

The package supports extracting principal components from residuals to capture unobserved factors:

```python
# Add pc(k) to the formula to extract k principal components
model = giv(
    df,
    "q + endog(p) ~ fe(id) + pc(2)",  # Extract 2 PCs from residuals
    id="id", t="t", weight="S",
    save_df=True  # Needed to access PC factors/loadings in df
)

# Access PC results
model.n_pcs          # Number of PCs extracted
model.pc_factors     # k×T matrix of time factors
model.pc_loadings    # N×k matrix of entity loadings
model.pc_model       # HeteroPCAModel object with details
```

#### Internal PCA

Internal PC extractions are supported. With internal PCs, the moment conditions become $\mathbb E[u_{i,t}u_{j,t}] = \Lambda \Lambda'$, where $\Lambda$ is the factor loadings estimated internally using [HeteroPCA.jl](https://github.com/FuZhiyu/HeteroPCA.jl) from $u_{i,t}(z) \equiv q_{i,t} + p_{t}\times\mathbf{C}_{i,t}'\boldsymbol{z}$ at each guess of $z$. However, following caveats apply:

- With internal PC extraction, the weighting scheme is no longer optimal as it does not consider the covariance in the moment conditions due to common factor estimation. The standard error formula also no longer applies and hence was not returned. One can consider bootstrapping for statistical inference; 

- In small samples, the exactly root solving the moment condition may not exist, and users may want to use an minimizer to minimize the error instead. 

- A model with fully flexible elasticity specification and fully flexible internal factor loadings is not theoretically identifiable. Hence, one needs to assume certain level of homogeneity to estimate factors internally. 


You can customize the PC extraction algorithm using the `pca_option` parameter:

```python
# Example with custom PCA options
model = giv(
    df,
    "q + id & endog(p) ~ X + pc(3)",
    id="id", t="t", weight="S",
    pca_option={
        # Preferred: let the wrapper build the constructor for you
        'algorithm': 'DeflatedHeteroPCA',
        'algorithm_options': dict(
            t_block=20,
            condition_number_threshold=5.0,
        ),

        'impute_method': 'zero',   # auto-converted to :zero
        'demean': False,
        'maxiter': 200,
    }
)

```

Available algorithms:
- `'algorithm': 'DeflatedHeteroPCA', which supports additional 'algorithm_options': {'t_block': 10, 'condition_number_threshold': 4.0}`: Deflated algorithm with adaptive block sizing
- `'algorithm': 'StandardHeteroPCA'`: Standard iterative algorithm
- `'algorithm': 'DiagonalDeletion'`: Single-step diagonal deletion method

When `save_df=True`, PC factors and loadings are added to the saved dataframe with columns like `pc_factor_1`, `pc_factor_2`, `pc_loading_1`, etc.

---


### Working with Results

```python
# Methods
model.summary()            # ▶ print full Julia-style summary
model.residuals()          # ▶ numpy array of the residuals for each observation
model.confint(level=0.95)  # ▶ (n×2) array of confidence intervals
model.coeftable(level=0.95)# ▶ pandas.DataFrame of estimates, SEs, t-stats, p-values

# Fields
model.endog_coef           # ▶ numpy array of ζ coefficients
model.exog_coef            # ▶ numpy array of β coefficients
model.agg_coef             # ▶ float: aggregate elasticity
model.endog_vcov           # ▶ VCOV of ζ coefficients
model.exog_vcov            # ▶ VCOV of β coefficients
model.nobs                 # ▶ int: number of observations
model.dof_residual         # ▶ int: residual degrees of freedom
model.formula              # ▶ str: Julia-style formula
model.formula_schema       # ▶ str: the internal schema of the Julia‐style formula after parsing
model.residual_variance    # ▶ numpy array of the estimated variance of the residuals for each entity (ûᵢ’s variance)
model.N                    # ▶ int: the number of cross‐section entities in the panel
model.T                    # ▶ int: the number of time periods per entity in the panel
model.dof                  # ▶ int: the total number of estimated parameters (length of ζ plus length of β)
model.responsename         # ▶ str: the name of the response variable(s)
model.converged            # ▶ bool: solver convergence status
model.endog_coefnames      # ▶ list[str]: ζ coefficient names
model.exog_coefnames       # ▶ list[str]: β coefficient names
model.idvar                # ▶ str: entity identifier column name
model.tvar                 # ▶ str: time identifier column name
model.weightvar            # ▶ str or None: weight column name
model.exclude_pairs        # ▶ dict: excluded moment-condition pairs
model.n_pcs                # ▶ int: number of principal components extracted
model.pc_factors           # ▶ numpy array (k×T) of PC time factors (if pc(k) used)
model.pc_loadings          # ▶ numpy array (N×k) of PC entity loadings (if pc(k) used)
model.pc_model             # ▶ HeteroPCAModel object with PC details (if pc(k) used)
model.coefdf               # ▶ pandas.DataFrame of entity-specific coefficients
model.fe                   # ▶ pandas.DataFrame of fixed-effects and fixed-effect interaction with exogenous controls (if saved) 
model.residual_df          # ▶ pandas.DataFrame of residuals (if saved)
model.df                   # ▶ pandas.DataFrame of full estimation output (if save_df=True)
model.coef                 # ▶ numpy array of [ζ; β]
model.vcov                 # ▶ full (ζ+β) variance–covariance matrix
model.stderror             # ▶ numpy array of standard errors
model.coefnames            # ▶ list[str]: names of all coefficients (ζ then β)
```
#### Entity-specific Coefficients DataFrame (coefdf)
The `model.coefdf` field provides a convenient way to access and report coefficients organized by categorical variables (e.g., by sector, entity, or other groupings). This DataFrame contains:

* All categorical variable values used in the model (e.g., entity IDs, sectors)
* Estimated coefficients for each term in the formula, stored in columns named `<term>_coef`
* Fixed effect estimates and fixed effect interaction with exogenous controls(if `save = 'fe'` or `save = 'all'` was specified)

Example:
```python

# Using the estimated model above as an example
print(model.coefdf)
# id  id & p_coef     fe_id  fe_id&η1  fe_id&η2
# 1     1.007234  0.770445 -0.075198  0.905689
# 2     1.773353 -0.376699  0.452851  0.825657
# 3     1.368630 -0.827939 -1.033757 -0.512825
# 4     3.384603 -0.275443  1.348865   1.37676
# 5     0.619882 -0.419348  0.663217  1.108182

```
---
## Simulation
The package includes utilities for Monte Carlo simulations using the `simulate_data` function:

```python
from optimalgiv import simulate_data, SimParam

# Generate simulated panel datasets
simulated_dfs = simulate_data(
    params = SimParam(
        N=20,      # Number of entities
        T=50,      # Time periods
        K=3,       # Number of factors
        M=0.7,     # Aggregate elasticity
        sigma_zeta=0.5  # Elasticity dispersion
    ),
    nsims=1,      # Number of simulations
    seed=123      # Random seed
)

# Use the first dataset
df = simulated_dfs[0]
```

### Simulation Parameters
The `SimParam` class accepts the following parameters:

| Parameter     | Description | Default |
|---------------|-------------|---------|
| `N`           | Number of entities | 10 |
| `T`           | Number of time periods | 100 |
| `K`           | Number of common factors | 2 |
| `M`           | Aggregate price elasticity | 0.5 |
| `sigma_zeta`  | Standard deviation of entity elasticities | 1.0 |
| `sigma_p`     | Price volatility to target | 2.0 |
| `h`           | Excess HHI for size distribution | 0.2 |
| `ushare`      | Share of price variation from idiosyncratic shocks | 0.2 (if K>0) |
| `sigma_u_curv`| Curvature for size-dependent volatility | 0.1 |
| `nu`          | Degrees of freedom for t-distribution (Inf = Normal) | np.inf |
| `missingperc` | Percentage of missing values | 0.0 |

### Data Generating Process
The simulated data follows this economic model:

```math
\begin{align}
q_{it} &= u_{it} + \Lambda_i \cdot \eta_t - \zeta_i \cdot p_t \\
p_t &= M \cdot \sum_i S_i \cdot (u_{it} + \Lambda_i \cdot \eta_t)
\end{align}
```

Where:
- `q_it`: Quantity for entity i at time t
- `p_t`: Price (common across entities at time t)
- `u_it`: Idiosyncratic shocks
- `η_t`: Common factors
- `Λ_i`: Factor loadings
- `ζ_i`: Entity-specific elasticities
- `S_i`: Entity size/weights

Entity sizes follow a power law distribution calibrated to match the target excess HHI (`h`).

### Output DataFrame
Each simulation returns a pandas DataFrame with columns:
- `id`: Entity identifier
- `t`: Time period
- `q`: Quantity (response variable)
- `p`: Price (endogenous regressor)
- `S`: Entity size/weight
- `ζ`: True entity-specific elasticity
- `η1, η2, ...`: Common factor realizations
- `λ1, λ2, ...`: Entity-specific factor loadings

---

## Limitations
- **PC extraction limitations**: Only `iv` and `iv_twopass` algorithms support internal PC extraction. The `debiased_ols` and `scalar_search` algorithms do not support PC extraction.
- **Variance-covariance matrix**: When PC extraction is used (pc(k) in formula), the variance-covariance matrix calculation is automatically disabled as it is not correct. One should consider bootstrapping instead.
- **Time fixed effects** are not supported directly, but one can use a single factor pc(1) instead.
- Some algorithms require **balanced panels**.
- The `debiased_ols` and `scalar_search` algorithms require **complete market coverage**

---

## To-do List
- Expose `build_error_function` interface.

---

## References

Please cite:

- Gabaix, Xavier, and Ralph S.J. Koijen. Granular Instrumental Variables. Journal of Political Economy, 132(7), 2024, pp. 2274–2303.
- Chaudhary, Manav, Zhiyu Fu, and Haonan Zhou. Anatomy of the Treasury Market: Who Moves Yields? Available at SSRN: https://ssrn.com/abstract=5021055

