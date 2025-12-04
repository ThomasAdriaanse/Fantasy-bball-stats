import numpy as np
from scipy import signal


class PMF1D:
    """
    Efficient discrete 1D probability mass function stored as a NumPy array.
    Index = outcome value, Value = probability.

    Example:
        p[3] = 0.2  means P(X=3) = 0.2
    """

    def __init__(self, probs: np.ndarray):
        # Ensure float type
        self.p = np.asarray(probs, dtype=float)
        self.normalize()

    @property
    def size(self) -> int:
        return self.p.shape[0]

    def normalize(self):
        total = float(self.p.sum())
        if total > 0:
            self.p /= total


    def shifted(self, k: int) -> "PMF1D":
        """
        Return PMF of X+k (i.e., shift to the right by k).
        Used to add current totals (already accumulated stats).
        """
        k = int(k)
        if k <= 0:
            return PMF1D(self.p.copy())

        padded = np.pad(self.p, (k, 0))
        return PMF1D(padded)


    def convolve(self, other: "PMF1D") -> "PMF1D":
        """
        Return PMF of X+Y for independent variables via FFT convolution.
        Much faster than nested loops for larger distributions.
        """
        arr = signal.fftconvolve(self.p, other.p)
        return PMF1D(arr)


    def mean(self) -> float:
        idx = np.arange(self.size)
        return float((idx * self.p).sum())


    def prob_beats(self, other: "PMF1D") -> float:
        """
        Compute P(X > Y) efficiently using CDF of Y.
        """
        # CDF of Y
        cdf_y = np.cumsum(other.p)

        # P(x > y) = sum_x p(x) * P(Y < x)
        # Use cdf shifted: cdf_y[x-1]
        cdf_shifted = np.concatenate([[0.0], cdf_y[:-1]])

        # Zero-pad to match length
        n = max(len(self.p), len(cdf_shifted))
        a = np.pad(self.p, (0, n - len(self.p)))
        b = np.pad(cdf_shifted, (0, n - len(cdf_shifted)))

        return float((a * b).sum())

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    def copy(self) -> "PMF1D":
        return PMF1D(self.p.copy())
