import numpy as np
from scipy import signal


class PMF2D:
    """
    Efficient 2D PMF for ratios like FG% or FT%.

    Underlying representation:
        p[i,j] = P(makes=i, attempts=j)

    Dimensions:
        axis0 = makes
        axis1 = attempts

    Ex:
    cason  wallace FT stats: 
    
    0/0*17
    0/2
    2/2*2
    1/1
    4/4
    1/2
    10/13
    
    2D PMF:
    [[0.75, 0.0, 0.0, 0.0, 0.0], 
    [0.0, 0.05, 0.05, 0.0, 0.0], 
    [0.0, 0.0, 0.1, 0.0, 0.0], 
    [0.0, 0.0, 0.0, 0.0, 0.0], 
    [0.0, 0.0, 0.0, 0.0, 0.05]]
    """

    def __init__(self, matrix: np.ndarray):
        self.p = np.asarray(matrix, dtype=float)
        self.normalize()


    @property
    def shape(self):
        return self.p.shape

    def normalize(self):
        total = float(self.p.sum())
        if total > 0:
            self.p /= total

    def shifted(self, makes: int, attempts: int) -> "PMF2D":
        """
        Return PMF of X + (makes, attempts).
        Equivalent to shifting both axes.
        """
        makes = int(makes)
        attempts = int(attempts)

        padded = np.pad(self.p, ((makes, 0), (attempts, 0)))
        return PMF2D(padded)


    def convolve(self, other: "PMF2D") -> "PMF2D":
        """
        2D convolution via FFT.
        Computes distribution of (M1+M2, A1+A2).
        """
        arr = signal.fftconvolve(self.p, other.p, mode='full')
        return PMF2D(arr)


    def expected_ratio(self) -> float:
        """
        Estimate E[ makes/attempts ] for this 2D PMF.
        Zero-safe.
        """
        m_idx, a_idx = np.indices(self.p.shape)

        makes = (m_idx * self.p).sum()
        attempts = (a_idx * self.p).sum()

        if attempts <= 0:
            return 0.0

        return float(makes / attempts)


    def prob_ratio_beats(self, other: "PMF2D") -> float:
        """
        Approximate P( (M1/A1) > (M2/A2) ).
        Brute force but optimized by only iterating support.
        """

        # Get indices with non-zero mass
        m1, a1 = np.where(self.p > 0)
        m2, a2 = np.where(other.p > 0)

        p = 0.0

        for i, j in zip(m1, a1):
            r1 = i / j if j > 0 else 0.0
            p1 = self.p[i, j]

            # Compare vs nonzero entries in other
            for k, l in zip(m2, a2):
                r2 = k / l if l > 0 else 0.0
                if r1 > r2:
                    p += p1 * other.p[k, l]

        return p


    def copy(self) -> "PMF2D":
        return PMF2D(self.p.copy())
