"""
SNP10 Backtester — Top 10 S&P 500 Companies Strategy
======================================================
Invests only in the top 10 S&P 500 companies by approximate market cap,
rebalanced periodically. Designed as a realistic tool to assist in investing.

Features:
  - Dynamic top-10 selection via market cap (shares_outstanding × price)
  - Equal-weight or market-cap-weight allocations
  - Realistic transaction costs (commission + bid-ask spread)
  - Quarterly / annual / monthly rebalancing options
  - Benchmark comparison vs SPY (S&P 500 ETF)
  - Full performance report: CAGR, Sharpe, Sortino, Max Drawdown, Calmar
  - Trade log export
  - Charts: equity curve, drawdown, monthly returns heatmap, allocation over time

Usage:
  python snp10_backtest.py

Adjust CONFIG below to customise the strategy.
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit here
# ─────────────────────────────────────────────────────────────
CONFIG = {
    # Backtest window
    "start_date": "2015-01-01",
    "end_date": datetime.now().strftime("%Y-%m-%d"),

    # Capital
    "initial_capital": 100_000,

    # Rebalancing: 'QE'=quarterly, 'YE'=annual, 'ME'=monthly
    "rebalance_freq": "QE",

    # Weighting scheme: 'equal' or 'market_cap'
    "weighting": "equal",

    # How many top companies to hold (change to experiment)
    "top_n": 10,

    # Transaction costs
    "commission_pct": 0.001,   # 0.1 % per trade (both sides)
    "spread_pct": 0.0005,      # 0.05 % bid-ask spread (buy-side only)

    # Cache downloaded data locally to speed up reruns
    "use_cache": True,
    "cache_dir": ".data_cache",

    # Output
    "output_dir": "output",
    "save_charts": True,
    "save_trade_log": True,
}

# ─────────────────────────────────────────────────────────────
# CANDIDATE UNIVERSE
# Large-caps that have held top-10 S&P positions at various times.
# A broader universe = more accurate dynamic selection.
# ─────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "NVDA", "TSLA",
    "BRK-B", "JPM", "V", "MA", "JNJ", "XOM", "WMT", "UNH", "HD",
    "BAC", "PG", "CVX", "LLY", "MRK", "PFE", "KO", "PEP", "ABBV",
    "TMO", "COST", "AVGO", "MCD", "ACN", "CRM", "ADBE", "NFLX",
    "CSCO", "INTC", "AMD", "ORCL", "IBM", "GE", "QCOM", "TXN",
    "HON", "CAT", "AMGN", "SBUX", "RTX", "T", "VZ", "DIS",
]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def cache_path(cfg, name):
    return os.path.join(cfg["cache_dir"], f"{name}.parquet")


# ─────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────
def fetch_prices(tickers, start, end, cfg):
    """Download adjusted close prices for all tickers, with local caching."""
    cp = cache_path(cfg, "prices")
    if cfg["use_cache"] and os.path.exists(cp):
        print("  [cache] loading prices from disk …")
        return pd.read_parquet(cp)

    print(f"  Downloading prices for {len(tickers)} tickers …")
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True,
    )
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.ffill().dropna(how="all", axis=1)

    if cfg["use_cache"]:
        prices.to_parquet(cp)
    return prices


def fetch_shares_outstanding(tickers, cfg):
    """
    Fetch current shares outstanding for each ticker via yfinance.
    Used as a fixed denominator to approximate historical market cap.
    (Shares outstanding changes slowly for mega-caps, so this is a good proxy.)
    """
    sp = cache_path(cfg, "shares")
    if cfg["use_cache"] and os.path.exists(sp):
        print("  [cache] loading shares outstanding from disk …")
        with open(sp.replace(".parquet", ".json")) as f:
            return json.load(f)

    shares = {}
    print(f"  Fetching shares outstanding for {len(tickers)} tickers …")
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            so = getattr(info, "shares", None)
            if so and so > 0:
                shares[t] = so
        except Exception:
            pass

    if cfg["use_cache"]:
        with open(sp.replace(".parquet", ".json"), "w") as f:
            json.dump(shares, f)
    return shares


def fetch_benchmark(start, end, cfg):
    """Download SPY as benchmark."""
    cp = cache_path(cfg, "spy")
    if cfg["use_cache"] and os.path.exists(cp):
        df = pd.read_parquet(cp)
        # Return a plain Series regardless of how it was stored
        if isinstance(df, pd.DataFrame):
            return df.iloc[:, 0]
        return df

    print("  Downloading SPY benchmark …")
    raw = yf.download("SPY", start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    # Flatten to a plain Series (yfinance may return DataFrame with ticker sub-column)
    if isinstance(raw, pd.DataFrame):
        spy = raw.iloc[:, 0]
    else:
        spy = raw
    spy.name = "Close"

    if cfg["use_cache"]:
        spy.to_frame("Close").to_parquet(cp)
    return spy


# ─────────────────────────────────────────────────────────────
# MARKET CAP RANKING
# ─────────────────────────────────────────────────────────────
def compute_approx_market_caps(prices, shares):
    """
    Approximate market cap = price × shares_outstanding.
    shares_outstanding is fixed (current), making this an approximation.
    We normalise so that ranking is meaningful even without perfect data.
    """
    available = [t for t in shares if t in prices.columns]
    shares_series = pd.Series({t: shares[t] for t in available})
    mcaps = prices[available].multiply(shares_series, axis=1)
    return mcaps


def get_top_n_at_date(mcaps, date, top_n):
    """Return the top-N tickers by market cap on a given date."""
    row = mcaps.loc[:date].iloc[-1].dropna()
    return row.nlargest(top_n).index.tolist()


# ─────────────────────────────────────────────────────────────
# PORTFOLIO ENGINE
# ─────────────────────────────────────────────────────────────
def get_rebalance_dates(prices, freq):
    """Return the first trading day of each period (Q/A/M)."""
    period_ends = prices.resample(freq).last().index
    dates = []
    for pe in period_ends:
        future = prices.index[prices.index >= pe]
        if len(future):
            dates.append(future[0])
    return sorted(set(dates))


def target_weights(tickers, mcaps, date, weighting):
    """Compute target portfolio weights."""
    if weighting == "equal":
        w = {t: 1.0 / len(tickers) for t in tickers}
    else:  # market_cap
        caps = mcaps.loc[:date].iloc[-1][tickers].dropna()
        total = caps.sum()
        w = (caps / total).to_dict()
    return w


def run_backtest(prices, mcaps, cfg, rebalance_dates):
    """
    Core portfolio simulation.

    Returns
    -------
    portfolio_value : pd.Series   daily portfolio value
    holdings       : pd.DataFrame  daily shares held per ticker
    trade_log      : list of dict  every executed trade
    daily_weights  : pd.DataFrame  daily portfolio weights
    """
    top_n = cfg["top_n"]
    weighting = cfg["weighting"]
    commission = cfg["commission_pct"]
    spread = cfg["spread_pct"]

    cash = cfg["initial_capital"]
    shares_held = {}          # {ticker: float}
    trade_log = []

    # output containers
    all_dates = prices.index
    port_values = pd.Series(index=all_dates, dtype=float)
    daily_weights_list = []

    rebalance_set = set(rebalance_dates)
    current_holdings = []     # tickers currently held

    print(f"\n  Running backtest ({cfg['start_date']} → {cfg['end_date']}) …")
    print(f"  Rebalance: {cfg['rebalance_freq']} | Weighting: {weighting} | Top-N: {top_n}\n")

    for date in all_dates:
        day_prices = prices.loc[date]

        # ── Rebalance ───────────────────────────────────────────
        if date in rebalance_set:
            new_holdings = get_top_n_at_date(mcaps, date, top_n)
            weights = target_weights(new_holdings, mcaps, date, weighting)

            # Mark-to-market current portfolio value before trades
            port_val = cash + sum(
                shares_held.get(t, 0) * day_prices.get(t, 0)
                for t in shares_held
            )

            # 1. Sell positions no longer in top-n
            for ticker in list(shares_held):
                if ticker not in new_holdings and shares_held[ticker] > 0:
                    sell_price = day_prices.get(ticker, 0)
                    if sell_price <= 0:
                        continue
                    proceeds = shares_held[ticker] * sell_price
                    cost = proceeds * commission
                    cash += proceeds - cost
                    trade_log.append({
                        "date": date, "ticker": ticker, "action": "SELL",
                        "shares": shares_held[ticker], "price": sell_price,
                        "commission": cost, "value": proceeds,
                    })
                    shares_held[ticker] = 0.0

            # 2. Re-compute port value after sells
            port_val = cash + sum(
                shares_held.get(t, 0) * day_prices.get(t, float("nan"))
                for t in shares_held
                if not np.isnan(day_prices.get(t, float("nan")))
            )

            # 3. Buy / adjust positions for new top-n
            for ticker in new_holdings:
                target_val = port_val * weights.get(ticker, 0)
                current_val = shares_held.get(ticker, 0) * day_prices.get(ticker, 0)
                diff_val = target_val - current_val
                buy_price = day_prices.get(ticker, 0) * (1 + spread)  # pay spread on buys

                if buy_price <= 0:
                    continue

                if diff_val > 0:
                    # Buy
                    shares_to_buy = diff_val / buy_price
                    cost = shares_to_buy * buy_price * commission
                    total_cost = shares_to_buy * buy_price + cost
                    if total_cost > cash:
                        shares_to_buy = cash / (buy_price * (1 + commission))
                        total_cost = cash
                    cash -= total_cost
                    shares_held[ticker] = shares_held.get(ticker, 0) + shares_to_buy
                    trade_log.append({
                        "date": date, "ticker": ticker, "action": "BUY",
                        "shares": shares_to_buy, "price": buy_price,
                        "commission": cost, "value": shares_to_buy * buy_price,
                    })
                elif diff_val < -1:
                    # Trim
                    shares_to_sell = abs(diff_val) / day_prices[ticker]
                    shares_to_sell = min(shares_to_sell, shares_held.get(ticker, 0))
                    proceeds = shares_to_sell * day_prices[ticker]
                    cost = proceeds * commission
                    cash += proceeds - cost
                    shares_held[ticker] = shares_held.get(ticker, 0) - shares_to_sell
                    trade_log.append({
                        "date": date, "ticker": ticker, "action": "TRIM",
                        "shares": shares_to_sell, "price": day_prices[ticker],
                        "commission": cost, "value": proceeds,
                    })

            current_holdings = new_holdings

        # ── Mark-to-market ──────────────────────────────────────
        port_val = cash
        weight_row = {}
        for ticker, sh in shares_held.items():
            p = day_prices.get(ticker, float("nan"))
            if sh > 0 and not np.isnan(p):
                val = sh * p
                port_val += val
                weight_row[ticker] = val
        if port_val > 0:
            weight_row = {k: v / port_val for k, v in weight_row.items()}
        port_values[date] = port_val
        daily_weights_list.append({"date": date, **weight_row})

    daily_weights = pd.DataFrame(daily_weights_list).set_index("date").fillna(0)
    return port_values, shares_held, trade_log, daily_weights


# ─────────────────────────────────────────────────────────────
# PERFORMANCE ANALYTICS
# ─────────────────────────────────────────────────────────────
def annualised_return(series):
    years = (series.index[-1] - series.index[0]).days / 365.25
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1


def max_drawdown(series):
    peak = series.cummax()
    dd = (series - peak) / peak
    return dd.min()


def sharpe(returns, rf=0.0, periods=252):
    excess = returns - rf / periods
    return excess.mean() / excess.std() * np.sqrt(periods) if excess.std() > 0 else 0


def sortino(returns, rf=0.0, periods=252):
    excess = returns - rf / periods
    downside = excess[excess < 0].std()
    return excess.mean() / downside * np.sqrt(periods) if downside > 0 else 0


def calmar(series):
    cagr = annualised_return(series)
    mdd = abs(max_drawdown(series))
    return cagr / mdd if mdd > 0 else 0


def performance_report(label, series, rf=0.03):
    rets = series.pct_change().dropna()
    total_ret = series.iloc[-1] / series.iloc[0] - 1
    cagr = annualised_return(series)
    mdd = max_drawdown(series)
    sr = sharpe(rets, rf=rf)
    so = sortino(rets, rf=rf)
    cal = calmar(series)
    vol = rets.std() * np.sqrt(252)
    pos_days = (rets > 0).sum() / len(rets)

    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Total Return       : {total_ret:>10.2%}")
    print(f"  CAGR               : {cagr:>10.2%}")
    print(f"  Annual Volatility  : {vol:>10.2%}")
    print(f"  Sharpe Ratio       : {sr:>10.2f}")
    print(f"  Sortino Ratio      : {so:>10.2f}")
    print(f"  Max Drawdown       : {mdd:>10.2%}")
    print(f"  Calmar Ratio       : {cal:>10.2f}")
    print(f"  Positive Days      : {pos_days:>10.2%}")

    return {
        "label": label,
        "total_return": total_ret, "cagr": cagr, "volatility": vol,
        "sharpe": sr, "sortino": so, "max_drawdown": mdd,
        "calmar": cal, "positive_days": pos_days,
    }


# ─────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.8,
    "font.size": 10,
    "legend.framealpha": 0.3,
    "legend.edgecolor": "#30363d",
})

PALETTE = {
    "snp10": "#58a6ff",
    "spy": "#f78166",
    "positive": "#3fb950",
    "negative": "#f78166",
    "neutral": "#8b949e",
}


def plot_equity_curve(snp10_val, spy_val, stats_snp10, stats_spy, cfg):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.suptitle(
        f"SNP{cfg['top_n']} Backtest  •  {cfg['start_date']} → {cfg['end_date']}\n"
        f"Top {cfg['top_n']} by Mkt Cap | {cfg['weighting'].title()} Weight | "
        f"{cfg['rebalance_freq']}-Rebalance | ${cfg['initial_capital']:,.0f} Initial",
        fontsize=12, y=0.98,
    )

    # ── 1. Equity curve (normalised) ───────────────────────────
    ax1 = axes[0]
    snp10_norm = snp10_val / snp10_val.iloc[0] * 100
    spy_norm = spy_val.reindex(snp10_val.index, method="ffill")
    spy_norm = spy_norm / spy_norm.iloc[0] * 100

    ax1.plot(snp10_norm.index, snp10_norm, label=f"SNP{cfg['top_n']}", color=PALETTE["snp10"])
    ax1.plot(spy_norm.index, spy_norm, label="SPY (S&P 500)", color=PALETTE["spy"], alpha=0.8)
    ax1.fill_between(snp10_norm.index, snp10_norm, spy_norm,
                     where=snp10_norm >= spy_norm,
                     alpha=0.08, color=PALETTE["snp10"], label="_nolegend_")
    ax1.fill_between(snp10_norm.index, snp10_norm, spy_norm,
                     where=snp10_norm < spy_norm,
                     alpha=0.08, color=PALETTE["spy"], label="_nolegend_")

    final_snp10 = snp10_norm.iloc[-1]
    final_spy = spy_norm.iloc[-1]
    ax1.set_ylabel("Portfolio Value (base=100)", fontsize=9)
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
    ax1.grid(True, axis="y")

    # annotation box
    info = (
        f"SNP{cfg['top_n']}  CAGR {stats_snp10['cagr']:.1%}  "
        f"Sharpe {stats_snp10['sharpe']:.2f}  MDD {stats_snp10['max_drawdown']:.1%}\n"
        f"SPY      CAGR {stats_spy['cagr']:.1%}  "
        f"Sharpe {stats_spy['sharpe']:.2f}  MDD {stats_spy['max_drawdown']:.1%}"
    )
    ax1.text(0.01, 0.97, info, transform=ax1.transAxes,
             verticalalignment="top", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d", alpha=0.7))

    # ── 2. Drawdown ─────────────────────────────────────────────
    ax2 = axes[1]
    dd_snp10 = (snp10_val / snp10_val.cummax() - 1) * 100
    dd_spy = (spy_val.reindex(snp10_val.index, method="ffill") /
              spy_val.reindex(snp10_val.index, method="ffill").cummax() - 1) * 100

    ax2.fill_between(dd_snp10.index, dd_snp10, 0, alpha=0.4, color=PALETTE["snp10"])
    ax2.plot(dd_snp10.index, dd_snp10, color=PALETTE["snp10"], label=f"SNP{cfg['top_n']}")
    ax2.plot(dd_spy.index, dd_spy, color=PALETTE["spy"], alpha=0.7, label="SPY")
    ax2.set_ylabel("Drawdown (%)", fontsize=9)
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, axis="y")

    # ── 3. Rolling 1Y return spread ─────────────────────────────
    ax3 = axes[2]
    roll_snp10 = snp10_norm.pct_change(252) * 100
    roll_spy = spy_norm.pct_change(252) * 100
    spread = roll_snp10 - roll_spy
    pos = spread.clip(lower=0)
    neg = spread.clip(upper=0)
    ax3.bar(spread.index, pos, color=PALETTE["positive"], alpha=0.7,
            width=1.5, label="Outperformance")
    ax3.bar(spread.index, neg, color=PALETTE["negative"], alpha=0.7,
            width=1.5, label="Underperformance")
    ax3.axhline(0, color="#8b949e", linewidth=0.8)
    ax3.set_ylabel("Rolling 1Y Spread (pp)", fontsize=9)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, axis="y")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(cfg["output_dir"], "equity_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {path}")
    plt.close()


def plot_monthly_heatmap(port_values, cfg):
    rets = port_values.resample("ME").last().pct_change().dropna()
    df = rets.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret") * 100
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.55)))
    fig.suptitle(f"SNP{cfg['top_n']} — Monthly Returns (%)", fontsize=12)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#f78166", "#161b22", "#3fb950"], N=256
    )
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1)
    sns.heatmap(
        pivot, ax=ax, cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
        annot=True, fmt=".1f", linewidths=0.4, linecolor="#0d1117",
        cbar_kws={"shrink": 0.5, "label": "Return (%)"},
        annot_kws={"size": 8},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)

    # Append annual returns on right
    annual = port_values.resample("YE").last().pct_change().dropna() * 100
    for i, (yr, _) in enumerate(pivot.iterrows()):
        if yr in annual.index.year:
            val = annual[annual.index.year == yr].values[0]
            color = PALETTE["positive"] if val >= 0 else PALETTE["negative"]
            ax.text(12.6, i + 0.5, f"{val:+.1f}%", va="center",
                    color=color, fontsize=8, fontweight="bold")
    ax.text(12.6, -0.6, "Annual", va="center", fontsize=8, color="#8b949e")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(cfg["output_dir"], "monthly_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {path}")
    plt.close()


def plot_allocation(daily_weights, cfg):
    """Stacked area chart of portfolio allocation over time."""
    # Only keep columns with non-trivial weights
    top_tickers = daily_weights.mean().nlargest(cfg["top_n"]).index.tolist()
    dw = daily_weights[top_tickers].resample("W").mean()

    colors = plt.cm.tab20.colors

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"SNP{cfg['top_n']} — Portfolio Allocation Over Time", fontsize=12)
    ax.stackplot(dw.index, [dw[t] * 100 for t in top_tickers],
                 labels=top_tickers, colors=colors[:len(top_tickers)], alpha=0.85)
    ax.set_ylabel("Allocation (%)", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", ncol=5, fontsize=8)
    ax.grid(True, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(cfg["output_dir"], "allocation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {path}")
    plt.close()


def plot_summary_bar(stats_snp10, stats_spy, cfg):
    """Side-by-side bar comparison of key metrics."""
    metrics = ["cagr", "volatility", "sharpe", "sortino", "max_drawdown", "calmar"]
    labels = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]

    snp10_vals = [stats_snp10[m] for m in metrics]
    spy_vals = [stats_spy[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"SNP{cfg['top_n']} vs SPY — Performance Comparison", fontsize=12)

    bars1 = ax.bar(x - width / 2, snp10_vals, width,
                   label=f"SNP{cfg['top_n']}", color=PALETTE["snp10"], alpha=0.85)
    bars2 = ax.bar(x + width / 2, spy_vals, width,
                   label="SPY", color=PALETTE["spy"], alpha=0.85)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="#8b949e", linewidth=0.6)
    ax.legend()
    ax.grid(True, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(cfg["output_dir"], "comparison_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# INVESTMENT ASSIST — current snapshot
# ─────────────────────────────────────────────────────────────
def current_portfolio_snapshot(prices, mcaps, shares, cfg):
    """
    Prints what the SNP10 portfolio looks like TODAY:
    which 10 stocks to hold, target weights, and current prices.
    Useful for replication.
    """
    latest_date = prices.index[-1]
    top_tickers = get_top_n_at_date(mcaps, latest_date, cfg["top_n"])
    weights = target_weights(top_tickers, mcaps, latest_date, cfg["weighting"])

    print(f"\n{'═'*55}")
    print(f"  CURRENT SNP{cfg['top_n']} PORTFOLIO SNAPSHOT  ({latest_date.date()})")
    print(f"  Weighting: {cfg['weighting']}   Capital: ${cfg['initial_capital']:,.0f}")
    print(f"{'═'*55}")
    print(f"  {'Ticker':<8} {'Weight':>8} {'Price':>10} {'$ Alloc':>12}  {'Shares':>8}")
    print(f"  {'─'*6:<8} {'─'*6:>8} {'─'*8:>10} {'─'*10:>12}  {'─'*6:>8}")

    total_market_cap = mcaps.loc[:latest_date].iloc[-1][top_tickers].sum()
    for t in top_tickers:
        w = weights.get(t, 0)
        price = prices[t].dropna().iloc[-1] if t in prices.columns else float("nan")
        alloc = cfg["initial_capital"] * w
        shares_needed = alloc / price if price > 0 else 0
        mkt_cap_b = mcaps.loc[:latest_date].iloc[-1].get(t, 0) / 1e9
        print(f"  {t:<8} {w:>8.1%} {price:>10,.2f} {alloc:>12,.0f}  {shares_needed:>8.1f}   (Mkt Cap ~${mkt_cap_b:,.0f}B)")

    print(f"{'─'*55}")
    print(f"  Total allocated: ${cfg['initial_capital']:,.0f}")
    print(f"  Approx. total market cap of holdings: "
          f"${total_market_cap / 1e12:.2f}T")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    cfg = CONFIG
    ensure_dirs(cfg["output_dir"], cfg["cache_dir"])

    print("=" * 55)
    print(f"  SNP{cfg['top_n']} BACKTESTER")
    print("=" * 55)

    # 1. Fetch data
    print("\n[1/5] Fetching market data …")
    prices = fetch_prices(UNIVERSE, cfg["start_date"], cfg["end_date"], cfg)
    shares = fetch_shares_outstanding(UNIVERSE, cfg)
    spy = fetch_benchmark(cfg["start_date"], cfg["end_date"], cfg)

    # 2. Compute approximate market caps
    print("\n[2/5] Computing market caps …")
    mcaps = compute_approx_market_caps(prices, shares)
    coverage = len(shares) / len(UNIVERSE)
    print(f"  Shares data for {len(shares)}/{len(UNIVERSE)} tickers "
          f"({coverage:.0%} universe coverage)")

    # 3. Get rebalance schedule
    print("\n[3/5] Scheduling rebalances …")
    rebalance_dates = get_rebalance_dates(prices, cfg["rebalance_freq"])
    print(f"  {len(rebalance_dates)} rebalance events scheduled")

    # 4. Run backtest
    print("\n[4/5] Simulating portfolio …")
    port_values, final_holdings, trade_log, daily_weights = run_backtest(
        prices, mcaps, cfg, rebalance_dates
    )

    # Align benchmark
    spy_aligned = spy.reindex(port_values.index, method="ffill").dropna()
    spy_norm = spy_aligned / spy_aligned.iloc[0] * cfg["initial_capital"]

    # 5. Performance
    print("\n[5/5] Computing performance metrics …")
    stats_snp10 = performance_report(
        f"SNP{cfg['top_n']} (Top {cfg['top_n']}, {cfg['weighting']} wt, {cfg['rebalance_freq']}-rebal)",
        port_values,
    )
    stats_spy = performance_report("SPY Benchmark", spy_norm)

    # Trade summary
    tl = pd.DataFrame(trade_log)
    if not tl.empty:
        total_comm = tl["commission"].sum()
        print(f"\n  Total trades       : {len(tl)}")
        print(f"  Total commissions  : ${total_comm:,.2f}")
        print(f"  Avg trades / rebal : {len(tl) / len(rebalance_dates):.1f}")

    # Current snapshot (investment assist)
    current_portfolio_snapshot(prices, mcaps, shares, cfg)

    # Charts
    print("\n  Generating charts …")
    plot_equity_curve(port_values, spy_norm, stats_snp10, stats_spy, cfg)
    plot_monthly_heatmap(port_values, cfg)
    plot_allocation(daily_weights, cfg)
    plot_summary_bar(stats_snp10, stats_spy, cfg)

    # Save trade log
    if cfg["save_trade_log"] and not tl.empty:
        tl_path = os.path.join(cfg["output_dir"], "trade_log.csv")
        tl.to_csv(tl_path, index=False)
        print(f"  Trade log saved  → {tl_path}")

    # Save performance summary
    perf_df = pd.DataFrame([stats_snp10, stats_spy])
    perf_path = os.path.join(cfg["output_dir"], "performance_summary.csv")
    perf_df.to_csv(perf_path, index=False)
    print(f"  Performance CSV  → {perf_path}")

    print(f"\n  All outputs in: ./{cfg['output_dir']}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
