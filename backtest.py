"""
Pairs Trading Backtest
======================
Statistical arbitrage strategy using cointegration.

Workflow
--------
1. Download adjusted close prices for a small universe via yfinance.
2. Identify the most cointegrated pair on the training period (Engle-Granger).
3. Estimate a hedge ratio via OLS on the training period only.
4. Build a rolling z-score of the log-price spread on the test period.
5. Enter mean-reversion trades when |z| > ENTRY_THRESHOLD; exit near 0,
   or on stop-loss / max-holding-period.
6. Compute performance metrics and save charts to output/.
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.patches import Patch
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
# Stock universe: mix of sector pairs so the cointegration scanner has candidates
TICKERS = ["KO", "PEP", "XOM", "CVX", "JPM", "BAC"]

START_DATE  = "2015-01-01"
END_DATE    = "2024-12-31"
TRAIN_RATIO = 0.60           # first 60% of data -> pair selection + hedge ratio
                             # remaining 40% -> out-of-sample backtest

# Rolling z-score parameters
ZSCORE_WINDOW   = 60         # ~3 months of trading days
ENTRY_THRESHOLD = 2.0        # open a position when |z| crosses this
EXIT_THRESHOLD  = 0.5        # close the position when |z| falls inside this band
STOP_LOSS_Z     = 3.5        # close immediately if |z| keeps blowing out
MAX_HOLD_DAYS   = 30         # maximum bars to stay in any single trade

# Cost model
TRANSACTION_COST = 0.001     # 0.1% per leg; a round trip costs 4 x 0.1% = 0.4%
INITIAL_CAPITAL  = 100_000   # starting portfolio value ($)

# Benchmark for Sharpe calculation
RISK_FREE_RATE = 0.05        # annualised; converted to daily inside compute_metrics

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Pull adjusted closing prices for all tickers in one yfinance call.
    Returns a DataFrame with one column per ticker, NaN rows dropped.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna()
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"  WARNING: Tickers not found or empty: {missing}")
    print(f"  Downloaded {len(prices)} trading days for {list(prices.columns)}")
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# 2. COINTEGRATION TESTING
# ─────────────────────────────────────────────────────────────────────────────

def find_best_pair(prices: pd.DataFrame, significance: float = 0.05):
    """
    Run the Engle-Granger cointegration test on every combination of tickers.
    Returns the pair with the lowest p-value that sits below `significance`.
    """
    tickers = list(prices.columns)
    best_pval = 1.0
    best_pair = None

    print("\n  Engle-Granger p-values:")
    print(f"  {'Pair':<12}  {'p-value':>8}   {'Cointegrated?':>14}")
    print("  " + "-" * 42)

    for t1, t2 in combinations(tickers, 2):
        s1 = np.log(prices[t1])
        s2 = np.log(prices[t2])
        _, pval, _ = coint(s1, s2)
        flag = "yes  *" if pval < significance else "no"
        print(f"  {t1}/{t2:<9}   {pval:.4f}   {flag:>14}")
        if pval < best_pval:
            best_pval = pval
            best_pair = (t1, t2)

    print()
    if best_pval >= significance:
        raise ValueError(
            f"No cointegrated pair found at {significance*100:.0f}% significance. "
            "Try expanding the universe or date range."
        )
    print(f"  > Selected pair: {best_pair[0]}/{best_pair[1]}   (p = {best_pval:.4f})")
    return best_pair, best_pval


# ─────────────────────────────────────────────────────────────────────────────
# 3. HEDGE RATIO (OLS)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_hedge_ratio(s1: pd.Series, s2: pd.Series) -> float:
    """
    OLS regression:  log(S1) = alpha + beta * log(S2) + epsilon
    The slope beta is the hedge ratio: how many log-units of S2 offset one of S1.
    Estimated on training data only -- never updated in the backtest.
    """
    log_s1 = np.log(s1)
    log_s2 = np.log(s2)
    result = OLS(log_s1, add_constant(log_s2)).fit()
    beta = float(result.params.iloc[1])
    r2   = result.rsquared
    print(f"  beta = {beta:.4f}   R^2 = {r2:.4f}")
    return beta


# ─────────────────────────────────────────────────────────────────────────────
# 4. SPREAD AND ROLLING Z-SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_spread(s1: pd.Series, s2: pd.Series, beta: float) -> pd.Series:
    """Log-price spread:  spread = log(S1) - beta * log(S2)"""
    return np.log(s1) - beta * np.log(s2)


def compute_zscore(spread: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score with a 1-bar shift on the rolling statistics.
    The shift ensures the signal at close of day t uses only data from
    before day t -- eliminates lookahead bias.
    """
    roll_mean = spread.rolling(window).mean().shift(1)
    roll_std  = spread.rolling(window).std().shift(1)
    # Guard against near-zero std during very flat periods
    roll_std  = roll_std.where(roll_std > 1e-8, other=np.nan)
    return (spread - roll_mean) / roll_std


# ─────────────────────────────────────────────────────────────────────────────
# 5. BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    s1: pd.Series,
    s2: pd.Series,
    beta: float,
    spread: pd.Series,
    zscore: pd.Series,
):
    """
    Iterate over each trading day, applying entry/exit rules.

    Position encoding
    -----------------
    +1  ->  long spread  (buy S1, sell beta units of S2; expect spread to rise)
    -1  ->  short spread (sell S1, buy beta units of S2; expect spread to fall)
     0  ->  flat

    P&L model
    ---------
    Daily spread return = log_ret(S1) - beta * log_ret(S2)
    Portfolio daily return = position[t-1] * daily_spread_return[t]

    Using position[t-1] (via .shift(1)) models a realistic one-day delay:
    signal at close of day t-1 executes at the open of day t.
    """
    n         = len(s1)
    positions = np.zeros(n, dtype=float)
    trades    = []

    position  = 0
    entry_idx = None
    entry_z   = None

    # ── Main loop ─────────────────────────────────────────────────────────────
    for i in range(ZSCORE_WINDOW + 2, n):
        z = zscore.iloc[i]
        if np.isnan(z):
            continue

        # ── EXIT CHECK ───────────────────────────────────────────────────────
        if position != 0:
            holding_bars = i - entry_idx
            should_exit  = False
            exit_reason  = ""

            # Mean reversion: z has crossed back towards 0
            if position == 1 and z >= -EXIT_THRESHOLD:
                should_exit, exit_reason = True, "reversion"
            elif position == -1 and z <= EXIT_THRESHOLD:
                should_exit, exit_reason = True, "reversion"

            # Stop loss: spread blowing out, not reverting
            if abs(z) >= STOP_LOSS_Z:
                should_exit, exit_reason = True, "stop_loss"

            # Max holding period: avoid capital lock-up in stalled trades
            if holding_bars >= MAX_HOLD_DAYS:
                should_exit, exit_reason = True, "timeout"

            if should_exit:
                gross_pnl = float(
                    position * (spread.iloc[i] - spread.iloc[entry_idx])
                )
                trades.append({
                    "entry_date":   s1.index[entry_idx],
                    "exit_date":    s1.index[i],
                    "direction":    "long" if position == 1 else "short",
                    "entry_z":      round(entry_z, 3),
                    "exit_z":       round(z, 3),
                    "holding_days": holding_bars,
                    "exit_reason":  exit_reason,
                    "gross_pnl":    round(gross_pnl, 6),
                })
                position  = 0
                entry_idx = None
                entry_z   = None

        # ── ENTRY CHECK (only enter when flat; no pyramiding) ────────────────
        if position == 0:
            if z < -ENTRY_THRESHOLD:          # spread below mean -> long spread
                position, entry_idx, entry_z = 1, i, z
            elif z > ENTRY_THRESHOLD:         # spread above mean -> short spread
                position, entry_idx, entry_z = -1, i, z

        positions[i] = position

    # ── Force-close any open trade at end of data ─────────────────────────────
    if position != 0 and entry_idx is not None:
        gross_pnl = float(position * (spread.iloc[-1] - spread.iloc[entry_idx]))
        trades.append({
            "entry_date":   s1.index[entry_idx],
            "exit_date":    s1.index[-1],
            "direction":    "long" if position == 1 else "short",
            "entry_z":      round(entry_z, 3),
            "exit_z":       round(float(zscore.iloc[-1]), 3),
            "holding_days": n - 1 - entry_idx,
            "exit_reason":  "end_of_data",
            "gross_pnl":    round(gross_pnl, 6),
        })

    # ── Daily P&L ─────────────────────────────────────────────────────────────
    log_ret1         = np.log(s1).diff()
    log_ret2         = np.log(s2).diff()
    spread_daily_ret = log_ret1 - beta * log_ret2   # return of a long spread unit

    # Shift by 1: yesterday's position drives today's return
    pos_series = pd.Series(positions, index=s1.index).shift(1)
    daily_pnl  = pos_series * spread_daily_ret

    # ── Transaction costs ─────────────────────────────────────────────────────
    # 4 commissions per round trip: enter S1, enter S2, exit S1, exit S2
    cost_series = pd.Series(0.0, index=s1.index)
    for t in trades:
        cost_series[t["entry_date"]] += 2 * TRANSACTION_COST
        cost_series[t["exit_date"]]  += 2 * TRANSACTION_COST

    net_pnl = daily_pnl.fillna(0.0) - cost_series

    # ── Equity curve ──────────────────────────────────────────────────────────
    equity = INITIAL_CAPITAL * (1 + net_pnl).cumprod()

    return equity, pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# 6. PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(equity: pd.Series, trades_df: pd.DataFrame) -> dict:
    """
    Total Return     : (final / initial) - 1, as a percentage.
    Annualised Sharpe: (mean_daily_excess_return / daily_std) * sqrt(252).
                       Return per unit of risk. >1 decent, >2 strong.
    Max Drawdown     : largest peak-to-trough decline. The worst-case risk metric.
    Win Rate         : % of trades that closed with positive gross P&L.
    """
    daily_ret = equity.pct_change().dropna()
    rf_daily  = (1 + RISK_FREE_RATE) ** (1 / 252) - 1

    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    excess = daily_ret - rf_daily
    sharpe = (
        (excess.mean() / excess.std()) * np.sqrt(252)
        if excess.std() > 0 else 0.0
    )

    rolling_max  = equity.cummax()
    drawdown     = (equity - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min()) * 100

    n_trades = len(trades_df)
    if n_trades > 0:
        wins     = (trades_df["gross_pnl"] > 0).sum()
        win_rate = wins / n_trades * 100
        avg_hold = float(trades_df["holding_days"].mean())
        winners  = trades_df.loc[trades_df["gross_pnl"] > 0, "gross_pnl"]
        losers   = trades_df.loc[trades_df["gross_pnl"] < 0, "gross_pnl"]
        avg_win  = float(winners.mean()) if len(winners) else 0.0
        avg_loss = float(losers.mean())  if len(losers)  else 0.0
    else:
        win_rate = avg_hold = avg_win = avg_loss = 0.0

    return {
        "Total Return (%)":         round(total_return, 2),
        "Annualised Sharpe":        round(sharpe, 3),
        "Max Drawdown (%)":         round(max_drawdown, 2),
        "Number of Trades":         n_trades,
        "Win Rate (%)":             round(win_rate, 1),
        "Avg Holding (days)":       round(avg_hold, 1),
        "Avg Win (spread units)":   round(avg_win, 5)  if avg_win  else 0.0,
        "Avg Loss (spread units)":  round(avg_loss, 5) if avg_loss else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    prices: pd.DataFrame,
    t1: str,
    t2: str,
    spread: pd.Series,
    zscore: pd.Series,
    equity: pd.Series,
    trades_df: pd.DataFrame,
) -> None:
    """Three-panel chart: normalised prices | z-score with signals | equity curve."""

    fig = plt.figure(figsize=(14, 13))
    gs  = gridspec.GridSpec(3, 1, hspace=0.45)

    # ── Panel 1: Normalised prices ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    n1  = prices[t1] / prices[t1].iloc[0]
    n2  = prices[t2] / prices[t2].iloc[0]
    ax1.plot(n1.index, n1, label=t1, linewidth=1.3)
    ax1.plot(n2.index, n2, label=t2, linewidth=1.3, linestyle="--", alpha=0.85)
    ax1.set_title(f"Normalised Price Series (test period): {t1} vs {t2}",
                  fontweight="bold")
    ax1.set_ylabel("Normalised Price  (base = 1.0)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    # ── Panel 2: Z-score with trade shading ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(zscore.index, zscore, color="steelblue", linewidth=1.0,
             label="Spread Z-score")

    for level, color, style in [
        ( ENTRY_THRESHOLD, "firebrick", "--"),
        (-ENTRY_THRESHOLD, "firebrick", "--"),
        ( EXIT_THRESHOLD,  "seagreen",  ":" ),
        (-EXIT_THRESHOLD,  "seagreen",  ":" ),
        ( STOP_LOSS_Z,     "darkred",   "-."),
        (-STOP_LOSS_Z,     "darkred",   "-."),
        ( 0,               "black",     "-" ),
    ]:
        lw = 0.9 if abs(level) > 0 else 0.5
        ax2.axhline(level, color=color, linestyle=style, linewidth=lw)

    for _, tr in trades_df.iterrows():
        shade = "lightgreen" if tr["direction"] == "long" else "mistyrose"
        ax2.axvspan(tr["entry_date"], tr["exit_date"], alpha=0.25, color=shade)

    legend_elements = [
        plt.Line2D([0], [0], color="steelblue", label="Z-score"),
        plt.Line2D([0], [0], color="firebrick", linestyle="--",
                   label=f"Entry +-{ENTRY_THRESHOLD}"),
        plt.Line2D([0], [0], color="seagreen",  linestyle=":",
                   label=f"Exit +-{EXIT_THRESHOLD}"),
        plt.Line2D([0], [0], color="darkred",   linestyle="-.",
                   label=f"Stop +-{STOP_LOSS_Z}"),
        Patch(facecolor="lightgreen", alpha=0.4, label="Long spread trade"),
        Patch(facecolor="mistyrose",  alpha=0.4, label="Short spread trade"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=2)
    ax2.set_title("Spread Z-Score with Trade Windows", fontweight="bold")
    ax2.set_ylabel("Z-Score")
    ax2.grid(alpha=0.3)

    # ── Panel 3: Equity curve ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(equity.index, equity, color="navy", linewidth=1.4, label="Strategy")
    ax3.axhline(INITIAL_CAPITAL, color="grey", linestyle="--", linewidth=0.9,
                label=f"Initial capital ${INITIAL_CAPITAL:,.0f}")
    ax3.fill_between(
        equity.index, INITIAL_CAPITAL, equity,
        where=(equity >= INITIAL_CAPITAL), alpha=0.12, color="green",
    )
    ax3.fill_between(
        equity.index, INITIAL_CAPITAL, equity,
        where=(equity < INITIAL_CAPITAL), alpha=0.12, color="red",
    )
    ax3.set_title("Equity Curve (out-of-sample test period)", fontweight="bold")
    ax3.set_ylabel("Portfolio Value ($)")
    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax3.legend(loc="upper left")
    ax3.grid(alpha=0.3)

    plt.suptitle(
        f"Pairs Trading Backtest  |  {t1} / {t2}  "
        f"(entry +-{ENTRY_THRESHOLD}s, exit +-{EXIT_THRESHOLD}s, "
        f"stop +-{STOP_LOSS_Z}s)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    path = OUTPUT_DIR / "backtest_chart.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved -> {path}")
    try:
        plt.show()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    sep = "=" * 62
    print(sep)
    print("  PAIRS TRADING BACKTEST")
    print(sep)

    # 1. Fetch data
    print("\n[1] Fetching price data ...")
    prices = fetch_prices(TICKERS, START_DATE, END_DATE)

    # 2. Train / test split
    split = int(len(prices) * TRAIN_RATIO)
    train = prices.iloc[:split]
    test  = prices.iloc[split:]
    print(
        f"\n  Train : {train.index[0].date()} -> {train.index[-1].date()} "
        f"({len(train)} days)"
    )
    print(
        f"  Test  : {test.index[0].date()} -> {test.index[-1].date()} "
        f"({len(test)} days)"
    )

    # 3. Cointegration testing on training data only
    print("\n[2] Testing cointegration on TRAINING data ...")
    (t1, t2), coint_pval = find_best_pair(train)

    # 4. Estimate hedge ratio on training data only
    print("\n[3] Estimating hedge ratio (OLS on training data) ...")
    beta = estimate_hedge_ratio(train[t1], train[t2])
    print(
        f"  Interpretation: for every 1 share of {t1} held long, "
        f"sell {beta:.4f} shares of {t2} to be spread-neutral."
    )

    # 5. Build spread and z-score on TEST data
    print("\n[4] Computing spread and z-score on TEST data ...")
    spread = compute_spread(test[t1], test[t2], beta)
    zscore = compute_zscore(spread, ZSCORE_WINDOW)
    print(f"  Spread  -- mean: {spread.mean():.5f}   std: {spread.std():.5f}")
    print(f"  Z-score -- range: [{zscore.min():.2f}, {zscore.max():.2f}]")

    # 6. Run backtest
    print("\n[5] Running backtest ...")
    equity, trades_df = run_backtest(test[t1], test[t2], beta, spread, zscore)

    # 7. Performance metrics
    print("\n[6] Performance summary (out-of-sample test period):")
    print(f"    {'Metric':<30} {'Value':>12}")
    print("    " + "-" * 44)
    metrics = compute_metrics(equity, trades_df)
    for k, v in metrics.items():
        print(f"    {k:<30} {str(v):>12}")

    # 8. Trade log
    n_trades = len(trades_df)
    print(f"\n[7] Trade log (showing first 10 of {n_trades}):")
    if n_trades > 0:
        pd.set_option("display.width", 120)
        pd.set_option("display.max_columns", None)
        print(trades_df.head(10).to_string(index=False))
    else:
        print(
            "  No trades generated -- try widening the date range "
            "or lowering ENTRY_THRESHOLD."
        )

    # 9. Save CSVs
    if n_trades > 0:
        trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    equity.to_csv(OUTPUT_DIR / "equity_curve.csv", header=["portfolio_value"])
    print(f"\n  CSV outputs -> ./{OUTPUT_DIR}/")

    # 10. Chart
    print("\n[8] Generating charts ...")
    plot_results(test, t1, t2, spread, zscore, equity, trades_df)

    print(f"\n{sep}")
    print("  DONE")
    print(sep)


if __name__ == "__main__":
    main()
