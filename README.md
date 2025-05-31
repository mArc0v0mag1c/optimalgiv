# optimalgiv

A minimal Python wrapper for [OptimalGIV.jl](https://github.com/FuZhiyu/OptimalGIV.jl)

This interface enables Python users to call Granular Instrumental Variables (GIV) estimators directly on pandas DataFrames using JuliaCall.
Julia is automatically installed and all dependencies are resolved without manual setup.

---

## Installation

```bash
pip install optimalgiv
````

On first use, `optimalgiv` will automatically:

* Install Julia (if not already available)
* Install `OptimalGIV.jl` and supporting packages
* Precompile and create a self-contained Julia environment

---

## Quickstart

```python
import pandas as pd
import numpy as np
from optimalgiv import giv

df = pd.DataFrame({
    "id":  np.repeat([1, 2], 5),
    "t":   list(range(1, 6)) * 2,
    "q":   np.random.randn(10),
    "p":   np.random.randn(10),
    "η1":  np.random.randn(10),
    "η2":  np.random.randn(10),
    "absS": np.abs(np.random.randn(10)),
})

model = giv(
    df,
    "q + endog(p) ~ id & (η1 + η2)",
    id="id", t="t", weight="absS",
    algorithm="scalar_search",
    guess={"Aggregate": 2.0}
)

print(model.coef)
print(model.coefficient_table())
```

---

## References

- Gabaix, Xavier, and Ralph S.J. Koijen.  
  *Granular Instrumental Variables.*  
  *Journal of Political Economy*, 132(7), 2024, pp. 2274–2303.

- Chaudhary, Manav, Zhiyu Fu, and Haonan Zhou.  
  *Anatomy of the Treasury Market: Who Moves Yields?*  
  Available at SSRN: [https://ssrn.com/abstract=5021055](https://ssrn.com/abstract=5021055)

