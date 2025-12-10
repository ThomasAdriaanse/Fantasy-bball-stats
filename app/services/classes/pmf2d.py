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
    sum:
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


    def means(self) -> float:
        """
        This returns E[M] / E[A], which represents the percentage you'd get
        in expectation if you aggregated makes and attempts over many
        repeated realizations of this distribution.

        This is used for comparing player's % stats with eachother because it the 
        volume matters on a per player basis.
        """
        arr = self.p
        m_idx, a_idx = np.indices(arr.shape)

        # Expected makes and attempts
        expected_makes = float((m_idx * arr).sum())
        expected_attempts = float((a_idx * arr).sum())

        if expected_attempts <= 0:
            return 0.0

        return expected_makes / expected_attempts

    def prob_beats(self, other: "PMF2D") -> tuple[float, float, float]:
        """
        Compute P(this % > other %), P(this % = other %), P(this % < other %)
        using ratio distributions derived from the 2D PMFs over (makes, attempts).

        Returns:
            (p_win, p_tie, p_loss) from this PMF's perspective.
        """
        arr1 = np.asarray(self.p, dtype=float)
        arr2 = np.asarray(other.p, dtype=float)

        # ---- Flatten (m, a) -> (ratio, prob) for each PMF ----
        def _ratios_and_probs(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            m_idx, a_idx = np.nonzero(arr)
            if m_idx.size == 0:
                # Degenerate: ratio 0 with prob 1
                return np.array([0.0], dtype=float), np.array([1.0], dtype=float)

            probs = arr[m_idx, a_idx].astype(float)
            ratios = np.zeros_like(probs, dtype=float)

            mask = a_idx > 0
            ratios[mask] = m_idx[mask] / a_idx[mask]  # m/a; 0 if a==0

            s = probs.sum()
            if s > 0:
                probs /= s
            else:
                # all zero? treat as point mass at 0%
                probs = np.array([1.0], dtype=float)
                ratios = np.array([0.0], dtype=float)

            return ratios, probs

        r1, p1 = _ratios_and_probs(arr1)
        r2, p2 = _ratios_and_probs(arr2)

        if r1.size == 0 or r2.size == 0:
            return 0.0, 1.0, 0.0

        r1 = np.asarray(r1, dtype=float)
        r2 = np.asarray(r2, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)

        # Normalize to valid PMFs (extra safety)
        s1 = p1.sum()
        s2 = p2.sum()
        if s1 <= 0 or s2 <= 0:
            return 0.0, 1.0, 0.0
        p1 /= s1
        p2 /= s2

        # ---- CDF of other (Y) over ratio values ----
        order2 = np.argsort(r2)
        r2s = r2[order2]
        p2s = p2[order2]
        cdf2 = np.cumsum(p2s)

        # For each x in r1, get P(Y < x) and P(Y <= x)
        idx_less = np.searchsorted(r2s, r1, side="left")
        idx_leq  = np.searchsorted(r2s, r1, side="right")

        prob_less = np.where(idx_less > 0, cdf2[idx_less - 1], 0.0)  # P(Y < x)
        prob_leq  = np.where(idx_leq  > 0, cdf2[idx_leq  - 1], 0.0)  # P(Y <= x)

        # P(this % > other %)
        p_win = float(np.dot(p1, prob_less))

        # P(this % = other %)
        p_tie = float(np.dot(p1, (prob_leq - prob_less)))

        # P(this % < other %)
        p_loss = 1.0 - p_win - p_tie
        if p_loss < 0.0:
            p_loss = 0.0
        elif p_loss > 1.0:
            p_loss = 1.0

        return p_win, p_tie, p_loss


    def expected_weekly_ratio(self) -> float:
        """
        Compute E[makes/attempts] for this 2D PMF over (m, a).
        This is the expected FG%/FT% *per week* and differs from E[M]/E[A].

        This is used for comparing team % stats with eachother because the ovlume does 
        not matter, it matters if the final % is higher
        """
        arr = self.p
        if arr.size == 0:
            return 0.0

        m_idx, a_idx = np.indices(arr.shape)

        # Only consider states with attempts > 0
        mask = a_idx > 0
        if not mask.any():
            return 0.0

        ratios = (m_idx[mask] / a_idx[mask]).astype(float)
        probs  = arr[mask]

        s = probs.sum()
        if s <= 0:
            return 0.0

        return float((ratios * probs).sum() / s)

    def copy(self) -> "PMF2D":
        return PMF2D(self.p.copy())
