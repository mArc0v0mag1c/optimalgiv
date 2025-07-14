import numpy as np
from juliacall import Main as jl

class HeteroPCAModel:
    """
    Wrapper around a Julia `HeteroPCA.HeteroPCAModel`.

    Parameters
    ----------
    jl_model : PyJulia object
        The raw Julia model returned by `HeteroPCA.heteropca`.
    """

    # -------- life-cycle -------------------------------------------------
    def __init__(self, jl_model):
        self._jl = jl_model

        self.mean        = np.asarray(jl_model.mean)
        self.projection  = np.asarray(jl_model.proj)
        self.prinvars    = np.asarray(jl_model.prinvars)
        self.noisevars   = np.asarray(jl_model.noisevars)
        self.r2          = float(jl.OptimalGIV.HeteroPCA.r2(jl_model))
        self.converged   = bool(jl_model.converged)
        self.iterations  = int(jl_model.iterations)

    # -------- hybrid helpers – call Julia, return NumPy ---------------
    def loadings(self):
        # un-standardised loadings = √prinvars .* projection
        return np.asarray(jl.OptimalGIV.HeteroPCA.loadings(self._jl))

    def predict(self, X, lam=0.0):
        """
        Project new data (d × n) onto the latent factor space.

        Returns
        -------
        numpy.ndarray
            k × n matrix of factor scores.
        """
        Z = jl.OptimalGIV.HeteroPCA.predict(self._jl, X, λ=float(lam))
        return np.asarray(Z)

    def reconstruct(self, F):
        """
        Low-rank reconstruction given factor scores F (k × n).
        """
        Y = jl.OptimalGIV.HeteroPCA.reconstruct(self._jl, F)
        return np.asarray(Y)

    # -------- niceties --------------------------------------------------
    def __repr__(self):
        k = self.projection.shape[1]
        return (f"HeteroPCAModel(k={k}, r2={self.r2:.3f}, "
                f"converged={self.converged}, iterations={self.iterations})")
