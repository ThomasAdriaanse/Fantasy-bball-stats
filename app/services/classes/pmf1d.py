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


    def prob_beats(self, other: "PMF1D") -> tuple[float, float, float]:
        """
        Compute P(X > Y), P(X = Y), P(X < Y) for two integer-valued 1D PMFs.

        Returns:
            (p_win, p_tie, p_loss) from this PMF's perspective.
        """
        px = np.asarray(self.p, dtype=float)
        py = np.asarray(other.p, dtype=float)

        # Normalize to valid PMFs (robust to minor drift / scaling)
        sx = px.sum()
        sy = py.sum()
        if sx <= 0 or sy <= 0:
            # Degenerate: treat as pure tie
            return 0.0, 1.0, 0.0
        px /= sx
        py /= sy

        # Zero-pad to common length
        n = max(px.size, py.size)
        if px.size < n:
            px = np.pad(px, (0, n - px.size))
        if py.size < n:
            py = np.pad(py, (0, n - py.size))

        # CDF of Y
        cdf_y = np.cumsum(py)

        # P(Y < x) = CDF_Y[x-1] with P(Y < 0) = 0
        prob_y_less = np.concatenate(([0.0], cdf_y[:-1]))

        # P(Y = x) is just py

        # P(X > Y)
        p_win = float(np.dot(px, prob_y_less))

        # P(X = Y)
        p_tie = float(np.dot(px, py))

        # P(X < Y) from complement (numerically more stable)
        p_loss = 1.0 - p_win - p_tie
        # Clamp tiny FP noise
        if p_loss < 0.0:
            p_loss = 0.0
        elif p_loss > 1.0:
            p_loss = 1.0

        return p_win, p_tie, p_loss


    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    def copy(self) -> "PMF1D":
        return PMF1D(self.p.copy())
