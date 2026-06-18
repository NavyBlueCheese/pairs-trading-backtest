# Pairs Trading Backtest

A statistical arbitrage backtest built in Python. The strategy identifies cointegrated stock pairs, constructs a mean-reverting spread, and trades it using a rolling z-score rule. Everything runs on historical daily data fetched from Yahoo Finance.

---

## What Is Pairs Trading?

find two stocks whose prices move together over the long run due to a shared economic relationship (same industry, competing products, correlated fundamentals). When the spread between them temporarily diverges beyond its historical range, bet on it returning to normal, short the overpriced leg, long the underpriced one.

The edge is mean reversion: the empirical observation that the spread tends to come back, and a statistical test that tells us whether that tendency is structural or coincidental.

---

## Why Cointegration, Not Correlation

Correlation measures whether two stocks tend to move in the same direction on any given day. That is a short-term, local relationship. Two highly correlated stocks can still diverge permanently — correlation says nothing about whether they share a long-run equilibrium.

**Cointegration** is a stronger claim. Two series are cointegrated if there exists a linear combination of them that is *stationary* — meaning it has a stable, mean-reverting long-run distribution. This linear combination is the spread. If the pair is cointegrated, there is a statistical "gravitational pull" drawing the spread back to its mean. That pull is what the trade profits from.

If you build a pairs strategy on correlation alone, you risk holding a position in a spread that wanders off and never comes back. Cointegration is the mathematical precondition for mean reversion to be real.

**Test used:** Engle-Granger two-step test. Regress `log(S1)` on `log(S2)`, then run an Augmented Dickey-Fuller (ADF) stationarity test on the residuals. If the residuals are stationary (p-value below our 5% threshold), the pair is cointegrated and the residuals are our spread.

---

## Stock Universe

```
KO, PEP, XOM, CVX, JPM, BAC
```

Classic sector pairs — beverages, energy, banks — where economic relationships create a structural tendency to co-move. All combinations are tested; the pair with the lowest cointegration p-value is selected.

---

## Strategy Design

### Step 1 — Train/Test Split

The data is split 60/40 by time. Pair selection and hedge ratio estimation happen on the training set only. The backtest runs exclusively on the held-out test set.

This matters because if you select the best pair *on the same data you backtest on*, you have committed data snooping — you chose the pair that happened to work historically, which guarantees overfitting. The out-of-sample test is the only honest performance measure.

### Step 2 — Hedge Ratio

After selecting the pair, we estimate the hedge ratio `β` via OLS regression on the training data:

```
log(S1) = α + β · log(S2) + ε
```

`β` is the slope. It tells us: for every unit of S1 we hold, we should short `β` units of S2 so that the combined position is insensitive to market-wide moves and isolates only the relative spread. We use log prices for scale invariance (a 10% move looks the same whether the stock is at $10 or $500).

### Step 3 — Spread and Rolling Z-Score

```
spread  = log(S1) - β · log(S2)
z-score = (spread - rolling_mean) / rolling_std
```

The rolling window is 60 trading days (~3 months). We use a rolling window rather than a full-sample mean because the equilibrium spread drifts slowly over time; a 3-month anchor tracks this without being too noisy.

**Critical detail:** the rolling statistics are shifted by 1 bar before computing the z-score. This means the signal at the close of day *t* uses only data available before day *t*. Without this shift, today's spread value would appear in its own denominator — a subtle lookahead bias that inflates backtest results.

### Step 4 — Entry and Exit Rules

| Condition | Action |
|---|---|
| z-score < −2.0 | Long spread (buy S1, sell β·S2) |
| z-score > +2.0 | Short spread (sell S1, buy β·S2) |
| z-score crosses ±0.5 | Exit (mean reversion achieved) |
| \|z-score\| > 3.5 | Stop-loss (spread blowing out) |
| Holding period > 30 bars | Timeout exit |

The entry threshold of ±2σ represents a meaningful deviation from the mean without requiring an extreme event. The exit at ±0.5σ gives us the profit without waiting for an exact zero crossing. The stop-loss handles the scenario where cointegration has structurally broken down and the spread is not coming back.

---

## Backtest Assumptions

Getting these right separates a realistic backtest from a fantasy.

**Transaction costs:** 0.1% per leg. Each trade involves four commissions (enter S1, enter S2, exit S1, exit S2), costing 0.4% of notional per round trip. Small on any one trade, but it compounds across dozens of trades.

**Execution delay:** Signals are generated at the close of day *t* and executed at the open of day *t+1*. This is modelled by shifting the position series by one bar before computing daily P&L. In real markets you would typically trade at the next open or use limit orders.

**No short-selling constraints:** We assume the ability to short either leg, which is realistic for liquid large-cap stocks with available borrow.

**Fixed hedge ratio:** The ratio is estimated once on the training data. In production, you would re-estimate it on a rolling basis (e.g., every month) to track slow structural drift. Using a fixed ratio is conservative — it is likely to slightly understate performance of a well-maintained live strategy.

**No leverage:** The strategy is approximately dollar-neutral (long one leg, short the other). Portfolio value starts at $100,000.

---

## Performance Metrics

**Total Return:** The raw percentage gain over the test period. Easy to communicate, but says nothing about risk.

**Annualised Sharpe Ratio:** `(mean_daily_excess_return / daily_std) × √252`. Measures return earned per unit of volatility. Above 1.0 is generally considered acceptable; above 2.0 is strong. We subtract the risk-free rate (5% annualised) from daily returns before computing this.

**Maximum Drawdown:** The largest peak-to-trough decline in the equity curve. This is what a real investor would have experienced at the worst possible entry point. A strategy with a great Sharpe but a 40% max drawdown is psychologically unrunnable.

**Win Rate:** Percentage of individual trades that closed with a positive gross P&L. High win rate is characteristic of mean-reversion strategies. Always read it alongside average win size vs. average loss size — a 90% win rate means nothing if the losses are 10x the wins.

---

## File Structure

```
pairs_trading_backtest/
├── backtest.py        # Complete strategy implementation
├── requirements.txt   # Python dependencies
├── .gitignore
└── output/            # Auto-generated when you run the script
    ├── backtest_chart.png
    ├── trades.csv
    └── equity_curve.csv
```

---

## How to Run

```bash
# 1. Clone the repo and enter the directory
git clone https://github.com/YOUR_USERNAME/pairs-trading-backtest.git
cd pairs-trading-backtest

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the backtest
python backtest.py
```

The script will print a cointegration table, the selected pair, hedge ratio, performance summary, and first 10 trades, then save a three-panel chart and CSVs to `output/`.

To change the universe or parameters, edit the configuration block at the top of `backtest.py`.

---

## Possible Extensions

- **Rolling hedge ratio:** Re-estimate `β` every month using an expanding or rolling window to track structural drift in the relationship.
- **Johansen test:** For trading a basket of more than two stocks simultaneously, replace Engle-Granger with the Johansen test, which handles multivariate cointegration.
- **Kalman filter spread:** Replace the OLS + rolling z-score with a Kalman filter that continuously updates the hedge ratio and spread estimate in a principled Bayesian way.
- **Regime detection:** Use a Hidden Markov Model to detect periods when the cointegration relationship is active vs. broken, and only trade in active regimes.
- **Walk-forward validation:** Re-run the entire pipeline (pair selection + hedge estimation + backtest) across multiple rolling windows to get a distribution of out-of-sample results rather than a single point estimate.

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Historical price data via Yahoo Finance |
| `pandas` | Data manipulation and time series |
| `numpy` | Numerical computation |
| `matplotlib` | Charting |
| `statsmodels` | Engle-Granger cointegration test, OLS regression |
                                                                                        