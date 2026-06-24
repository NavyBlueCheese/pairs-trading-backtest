# Pairs Trading Backtest
 
A statistical arbitrage strategy built in Python. Identifies cointegrated stock pairs, constructs a mean-reverting spread, and trades it using a rolling z-score signal on historical daily data from Yahoo Finance.
 
---
 
## Table of Contents
 
- [What Is Pairs Trading?](#what-is-pairs-trading)
- [Why Cointegration, Not Correlation](#why-cointegration-not-correlation)
- [Strategy Design](#strategy-design)
- [Backtest Assumptions](#backtest-assumptions)
- [Performance Metrics](#performance-metrics)
- [File Structure](#file-structure)
- [Getting Started](#getting-started)
- [Possible Extensions](#possible-extensions)
- [Dependencies](#dependencies)
---
 
## What Is Pairs Trading?
 
Find two stocks whose prices move together over the long run due to a shared economic relationship, the same industry, competing products, correlated fundamentals. When the spread between them temporarily diverges beyond its historical range, bet on it reverting: short the overpriced leg, long the underpriced one.
 
The edge is **mean reversion**: the empirical tendency of the spread to come back, backed by a statistical test that tells us whether that tendency is structural or coincidental.
 
---
 
## Why Cointegration, Not Correlation
 
Correlation measures whether two stocks tend to move in the same direction on a given day. Two highly correlated stocks can still diverge permanently, correlation says nothing about whether they share a long-run equilibrium.
 
**Cointegration** is a stronger claim. Two series are cointegrated if there exists a linear combination of them that is *stationary* meaning it has a stable, mean-reverting long-run distribution. This linear combination is the spread. If the pair is cointegrated, there is a statistical gravitational pull drawing the spread back to its mean. That pull is what the strategy profits from.
 
Building a pairs strategy on correlation alone risks holding a position in a spread that wanders off and never returns. Cointegration is the mathematical precondition for mean reversion to be real.
 
**Test used:** Engle-Granger two-step test. Regress `log(S1)` on `log(S2)`, then run an Augmented Dickey-Fuller (ADF) stationarity test on the residuals. If the residuals are stationary (p-value below 5%), the pair is cointegrated and the residuals become the spread.
 
---
 
## Strategy Design
 
**Stock universe:** `KO, PEP, XOM, CVX, JPM, BAC` — classic sector pairs across beverages, energy, and banking. All combinations are tested; the pair with the lowest cointegration p-value is selected.
 
### Step 1 — Train/Test Split
 
Data is split 60/40 by time. Pair selection and hedge ratio estimation happen on the **training set only**. The backtest runs exclusively on the held-out test set.
 
> Selecting the best pair on the same data you backtest on is data snooping, you chose the pair that happened to work historically, guaranteeing overfitting. The out-of-sample test is the only honest performance measure.
 
### Step 2 — Hedge Ratio
 
After selecting the pair, the hedge ratio `β` is estimated via OLS on training data:
 
```
log(S1) = α + β · log(S2) + ε
```
 
`β` is the slope: for every unit of S1 held, short `β` units of S2 so the combined position isolates only the relative spread, insensitive to market-wide moves. Log prices are used for scale invariance.
 
### Step 3 — Spread and Rolling Z-Score
 
```
spread  = log(S1) - β · log(S2)
z-score = (spread - rolling_mean) / rolling_std
```
 
The rolling window is **60 trading days (~3 months)**. Rolling rather than full-sample because the equilibrium spread drifts slowly over time; a 3-month anchor tracks this without excessive noise.
 
**Lookahead bias prevention:** rolling statistics are shifted by 1 bar before computing the z-score. The signal at the close of day *t* uses only data available before day *t*. Without this shift, today's spread value would appear in its own denominator which is a subtle bias that inflates backtest results.
 
### Step 4 — Entry and Exit Rules
 
| Condition | Action |
|---|---|
| z-score < −2.0 | Long spread (buy S1, sell β·S2) |
| z-score > +2.0 | Short spread (sell S1, buy β·S2) |
| z-score crosses ±0.5 | Exit — mean reversion achieved |
| \|z-score\| > 3.5 | Stop-loss — spread blowing out |
| Holding period > 30 bars | Timeout exit |
 
The ±2σ entry threshold captures meaningful deviations without requiring extreme events. The ±0.5σ exit captures the profit without waiting for an exact zero crossing. The stop-loss handles the scenario where cointegration has structurally broken down.
 
---
 
## Backtest Assumptions
 
| Assumption | Detail |
|---|---|
| Transaction costs | 0.1% per leg; 0.4% total per round trip (4 commissions) |
| Execution delay | Signals generated at close of day *t*, executed at open of day *t+1* |
| Short selling | Unconstrained — realistic for liquid large-cap stocks with available borrow |
| Hedge ratio | Fixed; estimated once on training data |
| Leverage | None — strategy is approximately dollar-neutral |
| Starting capital | $100,000 |
 
> **Note on fixed hedge ratio:** re-estimating `β` monthly on a rolling basis would likely improve live performance slightly. Using a fixed ratio is the conservative choice.
 
---
 
## Performance Metrics
 
**Total Return** — raw percentage gain over the test period. Easy to communicate; says nothing about risk.
 
**Annualised Sharpe Ratio** — `(mean_daily_excess_return / daily_std) × √252`. Measures return per unit of volatility. Above 1.0 is acceptable; above 2.0 is strong. The risk-free rate (5% annualised) is subtracted from daily returns before computing this.
 
**Maximum Drawdown** — largest peak-to-trough decline in the equity curve. What a real investor would have experienced at the worst possible entry point. A great Sharpe with a 40% max drawdown is psychologically unrunnable.
 
**Win Rate** — percentage of individual trades that closed with positive gross P&L. A high win rate is typical of mean-reversion strategies. Always read it alongside average win vs. average loss size, which is a 90% win rate, is meaningless if losses are 10× the wins.
 
---
 
## File Structure
 
```
pairs_trading_backtest/
├── backtest.py          # Complete strategy implementation
├── requirements.txt     # Python dependencies
├── .gitignore
└── output/              # Auto-generated on first run
    ├── backtest_chart.png
    ├── trades.csv
    └── equity_curve.csv
```
 
---
 
## Getting Started
 
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/pairs-trading-backtest.git
cd pairs-trading-backtest
 
# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
 
# 3. Install dependencies
pip install -r requirements.txt
 
# 4. Run the backtest
python backtest.py
```
 
The script will print a cointegration table, the selected pair, hedge ratio, performance summary, and first 10 trades, then save a three-panel chart and CSVs to `output/`.
 
To change the universe or parameters, edit the configuration block at the top of `backtest.py`.
 
---
 
## Possible Extensions
 
- **Rolling hedge ratio** — re-estimate `β` monthly using an expanding or rolling window to track structural drift.
- **Johansen test** — for trading a basket of more than two stocks, replace Engle-Granger with the Johansen test, which handles multivariate cointegration.
- **Kalman filter spread** — replace OLS + rolling z-score with a Kalman filter that continuously updates the hedge ratio and spread estimate in a principled Bayesian way.
- **Regime detection** — use a Hidden Markov Model to detect periods when the cointegration relationship is active vs. broken, and only trade during active regimes.
- **Walk-forward validation** — re-run the full pipeline across multiple rolling windows to get a distribution of out-of-sample results rather than a single point estimate.
---
 
## Dependencies
 
| Package | Purpose |
|---|---|
| `yfinance` | Historical price data via Yahoo Finance |
| `pandas` | Data manipulation and time series |
| `numpy` | Numerical computation |
| `matplotlib` | Charting |
| `statsmodels` | Engle-Granger cointegration test, OLS regression |
