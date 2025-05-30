# optimalgiv

A minimal Python wrapper for [OptimalGIV.jl](https://github.com/FuZhiyu/OptimalGIV.jl),
a Julia package developed by Zhiyu Fu for estimating Generalized Instrumental Variable (GIV) models.

This interface enables Python users to call GIV estimators directly on pandas DataFrames using JuliaCall.
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

## Credits

This package wraps the core functionality of [`OptimalGIV.jl`](https://github.com/FuZhiyu/OptimalGIV.jl), authored by **Zhiyu Fu**.
All modeling logic and algorithms originate from her original Julia implementation.

---

## License

MIT License
