from __future__ import annotations
import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import norm, t as student_t
from ._bridge import jl, _jf_to_pd

class SimParam:
    """Python mirror of Julia's SimParam struct with derived fields"""
    _ASCII2UNICODE = {
        "nu": "ν", "sigma_u_curv": "σᵤcurv",
        "sigma_p": "σp", "sigma_zeta": "σζ"
    }

    def __init__(
        self,
        h: float = 0.2,
        M: float = 0.5,
        T: int = 100,
        K: int = 2,
        N: int = 10,
        nu: float = np.inf,
        ushare: float | None = None,
        sigma_u_curv: float = 0.1,
        sigma_p: float = 2.0,
        sigma_zeta: float = 1.0,
        missingperc: float = 0.0
    ):
        # Store core parameters
        self.h = h
        self.M = M
        self.T = T
        self.K = K
        self.N = N
        self.nu = nu
        self.ushare = ushare or (1.0 if K == 0 else 0.2)
        self.sigma_u_curv = sigma_u_curv
        self.sigma_p = sigma_p
        self.sigma_zeta = sigma_zeta
        self.missingperc = missingperc
        
        # Calculate derived fields
        self.tail_param = self._solve_tail_param()
        self.const_s = self._solve_size_dist()
        self.sigma_u_vec = self._specify_volatility()
        self.DIST = self._get_distribution()

    def _solve_tail_param(self) -> float:
        """Solve power-law tail parameter for target HHI"""
        def h_func(tp):
            k = (1 + np.arange(self.N)) ** (-1/tp)
            S = k / k.sum()
            h_actual = np.sqrt((S**2).sum() - 1/self.N)
            return self.h - h_actual
        return root_scalar(h_func, bracket=[1e-6, 100]).root

    def _solve_size_dist(self) -> np.ndarray:
        """Generate entity size distribution"""
        k = (1 + np.arange(self.N)) ** (-1/self.tail_param)
        return k / k.sum()

    def _specify_volatility(self) -> np.ndarray:
        """Compute entity-specific shock volatilities"""
        log_s = np.log(self.const_s)
        sigma2 = np.exp(-self.sigma_u_curv * log_s)
        b = (self.const_s @ self.const_s) / (self.const_s @ (sigma2 * self.const_s))
        return np.sqrt(sigma2 * b)

    def _get_distribution(self):
        """Return frozen SciPy distribution matching Julia's DIST"""
        if np.isinf(self.nu):
            return norm(loc=0, scale=1)
        scale = np.sqrt((self.nu - 2)/self.nu)
        return student_t(df=self.nu, loc=0, scale=scale)

    def to_julia_dict(self) -> dict:
        """Convert to Julia-compatible kwargs with Unicode keys"""
        return {
            self._ASCII2UNICODE.get(k, k): v
            for k, v in vars(self).items()
            if k not in {"tail_param", "const_s", "sigma_u_vec", "DIST"}
        }


def simulate_data(
        params: Union[SimParam, Dict[str, Union[float, int]], None] = None,
        *,
        nsims: int = 1,
        seed: int | None = None,
        as_pandas: bool = True,
) -> Union[List[pd.DataFrame], List[Any]]:

    """Generate synthetic panels via Julia's OptimalGIV.simulate_data"""

    if params is None:
        params = SimParam()
    elif isinstance(params, dict):
        params = SimParam(**params)

    kw_py = params.to_julia_dict()


    jl_dict = jl.Dict[jl.Symbol, jl.Any]()
    for k, v in kw_py.items():
        jl_dict[jl.Symbol(k)] = v

    if not hasattr(jl, "create_namedtuple_from_dict"):
        jl.seval("""
            function create_namedtuple_from_dict(d::Dict{Symbol, Any})
                return (; d...)
            end
        """)
    create_nt = jl.create_namedtuple_from_dict
    simparams_nt = create_nt(jl_dict)


    jl_kwargs = {"Nsims": int(nsims)}
    if seed is not None:
        jl_kwargs["seed"] = int(seed)

    jl_sims = jl.OptimalGIV.simulate_data(simparams_nt, **jl_kwargs)

    if as_pandas:
        return [_jf_to_pd(df) for df in jl_sims]
    return list(jl_sims)