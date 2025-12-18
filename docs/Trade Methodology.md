# DARKO Trade Evaluation: Mathematical Methodology

## Executive Summary
This document details the process used to evaluate trade fairness and impact using Probability Mass Functions (PMFs) adjusted by DARKO projections. The core philosophy is to model player performance not as single averages, but as full probability distributions, allowing us to mathematically calculate the exact probability of one team beating another in any category.

## I. The Foundation: Why PMFs?
Traditional fantasy tools use "Z-scores" or average per-game stats. This fails to capture **variance**:
- **Consistency**: A player scoring 20 PPG every night is different from a player scoring 10 or 30 PPG randomly, even if they have the same average.
- **Matchup Logic**: Fantasy constraints are discrete (e.g., "Win", "Lose", "Tie"). The probability of winning a category depends on the *overlap* of the two teams' potential score distributions.

We represent every player's stat output as a **Probability Mass Function (PMF)**.
- **1D Stats (PTS, REB, etc.)**: A discrete distribution where $P(k)$ is the probability of achieving exactly $k$ stats.
- **2D Stats (FG%, FT%)**: A joint distribution over (Makes, Attempts), preserving the correlation between volume and efficiency.

## II. The Process

### Step 1: Data Ingestion & DARKO Projection
For every player involved, we:
1.  **Load Historical Distributions**: Fetch the player's base PMF derived from their actual game logs. This captures their inherent volatility/consistency.
2.  **Fetch DARKO Projection**: Retrieve the latest Daily-Adjusted Rapid Kalman Optimized (DARKO) projection, which is the "source of truth" for their current true talent level (mean).

### Step 2: Distribution Adjustment (The "Shift")
We assume the *shape* (variance/skew) of a player's historical distribution is correct, but the *location* (mean) needs to match the DARKO projection.

**1. Counting Stats (1D): Probabilistic Shifting**
If a player's historical mean is 15.0 pts and DARKO projects 16.5 pts, we shift the entire distribution right by +1.5.
- Because discrete distributions only exist on integers, we cannot shift by exactly 1.5.
- **Solution**: We split the probability mass. A shift of +1.5 acts like a shift of +1 (50% chance) and +2 (50% chance). This mathematically preserves the exact new mean while maintaining the distribution's integer nature.

**2. Ratio Stats (2D): Bilinear Scaling**
For FG% and FT%, we scale the (Makes, Attempts) grid to align with the projected efficiency and volume.
- **Solution**: We use "bilinear mass distribution". If a grid point at (5 makes, 10 attempts) needs to move to (5.5, 11), we distribute its probability mass to the four surrounding integer coordinates. This prevents "gaps" in the data and aliasing artifacts.

### Step 3: Team Aggregation (Convolution)
How do we sum up the stats for a team for a whole week?
- **Mathematical Property**: The distribution of the sum of independent random variables is the **convolution** of their individual distributions.
- **Process**:
    1.  For each player, we convolve their single-game PMF $N$ times (where $N$ is games played that week). This gives the distribution of *that player's weekly total*.
    2.  We then convolve all the players' weekly PMFs together.
- **Result**: A single, massive PMF representing the probability of the *entire team* scoring exactly $X$ points, $Y$ rebounds, etc., for the week.

### Step 4: Trade Simulation
We perform the aggregation twice:
1.  **Before Trade**: Using current rosters.
2.  **After Trade**: Swapping the traded players and recalculating the Team PMFs for the two involved teams. (Note: Non-trading teams are unaffected).

### Step 5: Win Probability Calculation
To determine "Trade Fairness", we calculate the probability that Team A beats Team B in each category.
For two team distributions $A$ and $B$:
$$ P(A > B) = \sum_{x} P(A=x) \cdot P(B < x) $$
(i.e., sum over all possible scores $x$: "Probability Team A scores $x$" times "Probability Team B scores strictly less than $x$").

We repeat this pairwise comparison for the trading team against **every other team in the league**.
- **Metric**: "Average Win %". If Team A has a 60% chance to win Rebounds against the field, that is a robust measure of their rebounding strength.

## III. Considerations & Sources of Error

1.  **Independence Assumption**: Convolution assumes player performances are independent.
    *   *Reality*: They are correlated (e.g., teammates stealing rebounds from each other, or blowout games affecting all starters).
    *   *Impact*: Our model might slightly overestimate the variance of high-correlation stats, but generally this is negligible for weekly aggregations.
2.  **DARKO Accuracy**: The model assumes DARKO is the ground truth.
    *   *Reality*: Projections have error bars. If DARKO is slow to react to a role change (e.g., an injury to a starter), our evaluation will be too.
3.  **Game Count variance**: We assume a fixed number of games (e.g., 3 games/week) or use the schedule.
    *   *Reality*: Rest days, injuries, and overtime affect raw counts.
4.  **"Streamer" Slots**: The "After" state assumes empty roster spots are just empty.
    *   *Refinement*: High-level analysis should replace traded-away players with a "Replacement Level Streamer" PMF to capture the value of the open roster spot.

## Conclusion
This methodology provides a rigorous, probabilistic view of trade value. By combining the **accuracy** of DARKO projections with the **variance modeling** of historical PMFs, we generate a "Win %" that reflects the true fantasy utility of the trade, far better than simple "Player A averages more points than Player B" heuristics.
