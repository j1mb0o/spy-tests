"""
DCA Backtest — €100/month into SNP10 vs SPY over 20 years
==========================================================
Simulates Dollar-Cost Averaging: investing €100 every month into either
the top 10 S&P 500 companies (equal-weight, monthly rebalance) or simply
into SPY (S&P 500 ETF).

Currency note: stocks are USD-denominated. We treat €100 ≈ $100 for
simplicity (ignoring EUR/USD fluctuations). Adjust monthly_contribution
in CONFIG if you want a different amount.

Usage:
  uv run dca_backtest.py
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
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "start_date": "2006-04-01",
    "end_date": datetime.now().strftime("%Y-%m-%d"),

    # Monthly DCA contribution (€ / $ — see note above)
    "monthly_contribution": 100,

    # Top-N selection
    "top_n": 10,
    "weighting": "equal",

    # Transaction costs
    "commission_pct": 0.001,    # 0.1 % per trade
    "spread_pct": 0.0005,       # 0.05 % bid-ask spread (buys only)

    # Cache
    "use_cache": True,
    "cache_dir": ".data_cache",

    # Output
    "output_dir": "output",
}

# Candidate universe — broad enough to capture historical top-10 shifts
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
# DATA FETCHING (same logic as snp10_backtest.py, with yfinance fixes)
# ─────────────────────────────────────────────────────────────
def fetch_prices(tickers, start, end, cfg):
    cp = cache_path(cfg, "dca_prices")
    if cfg["use_cache"] and os.path.exists(cp):
        print("  [cache] loading prices …")
        return pd.read_parquet(cp)

    print(f"  Downloading prices for {len(tickers)} tickers …")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.ffill().dropna(how="all", axis=1)

    if cfg["use_cache"]:
        prices.to_parquet(cp)
    return prices


def fetch_shares_outstanding(tickers, cfg):
    jp = os.path.join(cfg["cache_dir"], "shares.json")
    if cfg["use_cache"] and os.path.exists(jp):
        print("  [cache] loading shares outstanding …")
        with open(jp) as f:
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
        with open(jp, "w") as f:
            json.dump(shares, f)
    return shares


def fetch_spy(start, end, cfg):
    cp = cache_path(cfg, "dca_spy")
    if cfg["use_cache"] and os.path.exists(cp):
        df = pd.read_parquet(cp)
        return df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df

    print("  Downloading SPY …")
    raw = yf.download("SPY", start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.DataFrame):
        spy = raw.iloc[:, 0]
    else:
        spy = raw
    spy.name = "Close"

    if cfg["use_cache"]:
        spy.to_frame("Close").to_parquet(cp)
    return spy


# ─────────────────────────────────────────────────────────────
# MARKET-CAP HELPERS
# ─────────────────────────────────────────────────────────────
def compute_approx_market_caps(prices, shares):
    available = [t for t in shares if t in prices.columns]
    shares_series = pd.Series({t: shares[t] for t in available})
    return prices[available].multiply(shares_series, axis=1)


def get_top_n_at_date(mcaps, date, top_n):
    row = mcaps.loc[:date].iloc[-1].dropna()
    return row.nlargest(top_n).index.tolist()


# ─────────────────────────────────────────────────────────────
# DCA ENGINES
# ─────────────────────────────────────────────────────────────
def get_monthly_investment_dates(prices):
    """Return the first trading day of each calendar month."""
    monthly = prices.resample("MS").first()
    dates = []
    for d in monthly.index:
        future = prices.index[prices.index >= d]
        if len(future):
            dates.append(future[0])
    return sorted(set(dates))


def run_dca_snp10(prices, mcaps, spy_prices, cfg):
    """
    Monthly DCA into top-10 with full rebalance each month.

    Each month:
      1. Mark-to-market existing holdings
      2. Sell everything (commission)
      3. Add monthly contribution
      4. Buy equal-weight top 10 (commission + spread)
    """
    contribution = cfg["monthly_contribution"]
    commission = cfg["commission_pct"]
    spread = cfg["spread_pct"]
    top_n = cfg["top_n"]

    invest_dates = get_monthly_investment_dates(prices)
    invest_set = set(invest_dates)
    all_dates = prices.index

    cash = 0.0
    shares_held = {}          # {ticker: float_shares}
    total_invested = 0.0
    total_commissions = 0.0

    port_values = pd.Series(index=all_dates, dtype=float)
    invested_series = pd.Series(index=all_dates, dtype=float)
    trade_log = []
    composition_log = []      # for tracking what's held over time

    for date in all_dates:
        day_prices = prices.loc[date]

        if date in invest_set:
            # ── 1. Sell everything ────────────────────────────
            for ticker in list(shares_held):
                if shares_held[ticker] > 0:
                    sell_price = day_prices.get(ticker, 0)
                    if sell_price <= 0:
                        continue
                    proceeds = shares_held[ticker] * sell_price
                    cost = proceeds * commission
                    total_commissions += cost
                    cash += proceeds - cost
                    trade_log.append({
                        "date": date, "ticker": ticker, "action": "SELL",
                        "shares": shares_held[ticker], "price": sell_price,
                        "commission": cost,
                    })
                    shares_held[ticker] = 0.0

            # ── 2. Add monthly contribution ───────────────────
            cash += contribution
            total_invested += contribution

            # ── 3. Determine top-N and buy ────────────────────
            top_tickers = get_top_n_at_date(mcaps, date, top_n)
            weight = 1.0 / len(top_tickers)

            for ticker in top_tickers:
                alloc = cash * weight
                buy_price = day_prices.get(ticker, 0) * (1 + spread)
                if buy_price <= 0:
                    continue
                shares_to_buy = alloc / (buy_price * (1 + commission))
                cost = shares_to_buy * buy_price * commission
                total_commissions += cost
                shares_held[ticker] = shares_held.get(ticker, 0) + shares_to_buy
                trade_log.append({
                    "date": date, "ticker": ticker, "action": "BUY",
                    "shares": shares_to_buy, "price": buy_price,
                    "commission": cost,
                })

            # All cash deployed (minus rounding dust)
            spent = sum(
                shares_held.get(t, 0) * day_prices.get(t, 0) * (1 + spread)
                for t in top_tickers if day_prices.get(t, 0) > 0
            )
            cash = max(0, cash - spent)
            # Simpler: assume all cash is deployed
            cash = 0.0

            composition_log.append({
                "date": date,
                "tickers": top_tickers.copy(),
            })

        # ── Mark-to-market ────────────────────────────────────
        port_val = cash
        for ticker, sh in shares_held.items():
            p = day_prices.get(ticker, float("nan"))
            if sh > 0 and not np.isnan(p):
                port_val += sh * p

        port_values[date] = port_val
        invested_series[date] = total_invested

    return port_values, invested_series, trade_log, total_commissions, composition_log


def run_dca_snp10_no_rebal(prices, mcaps, cfg):
    """
    Monthly DCA into top-10 — BUY ONLY, never sell.

    Each month:
      1. Determine current top 10 by market cap
      2. Split €100 equally among them and buy
      3. Old positions ride — never sold
    """
    contribution = cfg["monthly_contribution"]
    commission = cfg["commission_pct"]
    spread = cfg["spread_pct"]
    top_n = cfg["top_n"]

    invest_dates = get_monthly_investment_dates(prices)
    invest_set = set(invest_dates)
    all_dates = prices.index

    cash = 0.0
    shares_held = {}          # {ticker: float_shares}
    total_invested = 0.0
    total_commissions = 0.0

    port_values = pd.Series(index=all_dates, dtype=float)
    invested_series = pd.Series(index=all_dates, dtype=float)

    for date in all_dates:
        day_prices = prices.loc[date]

        if date in invest_set:
            # ── Add monthly contribution ──────────────────────
            cash += contribution
            total_invested += contribution

            # ── Determine top-N and buy ───────────────────────
            top_tickers = get_top_n_at_date(mcaps, date, top_n)
            weight = 1.0 / len(top_tickers)

            for ticker in top_tickers:
                alloc = contribution * weight   # only new money
                buy_price = day_prices.get(ticker, 0) * (1 + spread)
                if buy_price <= 0:
                    continue
                shares_to_buy = alloc / (buy_price * (1 + commission))
                cost = shares_to_buy * buy_price * commission
                total_commissions += cost
                shares_held[ticker] = shares_held.get(ticker, 0) + shares_to_buy

            cash = 0.0  # all deployed

        # ── Mark-to-market ────────────────────────────────────
        port_val = cash
        for ticker, sh in shares_held.items():
            p = day_prices.get(ticker, float("nan"))
            if sh > 0 and not np.isnan(p):
                port_val += sh * p

        port_values[date] = port_val
        invested_series[date] = total_invested

    return port_values, invested_series, total_commissions


def run_dca_spy(spy_prices, cfg):
    """
    Monthly DCA into SPY — simple buy-and-hold.

    Each month: add €100, buy SPY shares. Never sell.
    """
    contribution = cfg["monthly_contribution"]
    commission = cfg["commission_pct"]
    spread = cfg["spread_pct"]

    all_dates = spy_prices.index
    invest_dates = set()
    monthly = spy_prices.resample("MS").first()
    for d in monthly.index:
        future = all_dates[all_dates >= d]
        if len(future):
            invest_dates.add(future[0])

    cash = 0.0
    spy_shares = 0.0
    total_invested = 0.0
    total_commissions = 0.0

    port_values = pd.Series(index=all_dates, dtype=float)
    invested_series = pd.Series(index=all_dates, dtype=float)

    for date in all_dates:
        price = spy_prices.loc[date]

        if date in invest_dates:
            cash += contribution
            total_invested += contribution

            buy_price = price * (1 + spread)
            if buy_price > 0:
                shares_to_buy = cash / (buy_price * (1 + commission))
                cost = shares_to_buy * buy_price * commission
                total_commissions += cost
                spy_shares += shares_to_buy
                cash = 0.0

        port_values[date] = cash + spy_shares * price
        invested_series[date] = total_invested

    return port_values, invested_series, total_commissions


# ─────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────
def annualised_return(series):
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0:
        return 0
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1


def max_drawdown(series):
    peak = series.cummax()
    dd = (series - peak) / peak
    return dd.min()


def sharpe(returns, rf=0.03, periods=252):
    excess = returns - rf / periods
    return excess.mean() / excess.std() * np.sqrt(periods) if excess.std() > 0 else 0


def sortino(returns, rf=0.03, periods=252):
    excess = returns - rf / periods
    downside = excess[excess < 0].std()
    return excess.mean() / downside * np.sqrt(periods) if downside > 0 else 0


def calmar(series):
    cagr = annualised_return(series)
    mdd = abs(max_drawdown(series))
    return cagr / mdd if mdd > 0 else 0


def dca_performance(label, port_values, invested_series):
    """Compute and print DCA-specific performance metrics."""
    final = port_values.iloc[-1]
    invested = invested_series.iloc[-1]
    profit = final - invested
    total_ret = final / invested - 1
    mdd = max_drawdown(port_values)
    rets = port_values.pct_change().dropna()
    vol = rets.std() * np.sqrt(252)
    sr = sharpe(rets)
    so = sortino(rets)
    cal = calmar(port_values)
    years = (port_values.index[-1] - port_values.index[0]).days / 365.25

    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Duration           : {years:>10.1f} years")
    print(f"  Total Invested     : €{invested:>10,.0f}")
    print(f"  Final Value        : €{final:>10,.0f}")
    print(f"  Profit             : €{profit:>10,.0f}")
    print(f"  Total Return       : {total_ret:>10.2%}")
    print(f"  Annual Volatility  : {vol:>10.2%}")
    print(f"  Sharpe Ratio       : {sr:>10.2f}")
    print(f"  Sortino Ratio      : {so:>10.2f}")
    print(f"  Max Drawdown       : {mdd:>10.2%}")
    print(f"  Calmar Ratio       : {cal:>10.2f}")

    return {
        "label": label,
        "years": years,
        "total_invested": invested,
        "final_value": final,
        "profit": profit,
        "total_return": total_ret,
        "volatility": vol,
        "sharpe": sr,
        "sortino": so,
        "max_drawdown": mdd,
        "calmar": cal,
    }


# ─────────────────────────────────────────────────────────────
# VISUALISATIONS (dark theme)
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

C_SNP10_REBAL = "#58a6ff"      # blue  — SNP10 with rebalancing
C_SNP10_BUYONLY = "#d2a8ff"    # purple — SNP10 buy-only
C_SPY = "#f78166"              # orange — SPY
C_INVESTED = "#8b949e"         # grey  — total invested baseline
C_POS = "#3fb950"
C_NEG = "#f78166"


def plot_dca_equity(rebal_val, buyonly_val, spy_val, invested,
                    stats_rebal, stats_buyonly, stats_spy, cfg):
    """Equity curve: 3 strategies + total invested baseline."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1.5]})
    contrib = cfg["monthly_contribution"]
    n = cfg["top_n"]
    fig.suptitle(
        f"DCA Backtest  •  €{contrib}/month  •  "
        f"{cfg['start_date']} → {cfg['end_date']}\n"
        f"SNP{n} Rebalanced vs SNP{n} Buy-Only vs SPY",
        fontsize=12, y=0.98,
    )

    # ── 1. Equity curves ─────────────────────────────────────
    ax1 = axes[0]
    common = rebal_val.dropna().index
    r10 = rebal_val.loc[common]
    b10 = buyonly_val.reindex(common, method="ffill").bfill()
    sp = spy_val.reindex(common, method="ffill").bfill()
    inv = invested.loc[common]

    ax1.plot(r10.index, r10, label=f"SNP{n} Rebalanced", color=C_SNP10_REBAL)
    ax1.plot(b10.index, b10, label=f"SNP{n} Buy-Only", color=C_SNP10_BUYONLY, alpha=0.9)
    ax1.plot(sp.index, sp, label="SPY", color=C_SPY, alpha=0.85)
    ax1.plot(inv.index, inv, label="Total Invested", color=C_INVESTED,
             linestyle="--", linewidth=1.2, alpha=0.7)

    ax1.set_ylabel("Portfolio Value (€)", fontsize=9)
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(
        lambda x, _: f"€{x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, axis="y")

    # Stats annotation — 3 rows
    info = (
        f"SNP{n} Rebal   €{stats_rebal['total_invested']:,.0f} → "
        f"€{stats_rebal['final_value']:,.0f}  "
        f"({stats_rebal['total_return']:+.1%})  "
        f"Sharpe {stats_rebal['sharpe']:.2f}  "
        f"MDD {stats_rebal['max_drawdown']:.1%}\n"
        f"SNP{n} Buy     €{stats_buyonly['total_invested']:,.0f} → "
        f"€{stats_buyonly['final_value']:,.0f}  "
        f"({stats_buyonly['total_return']:+.1%})  "
        f"Sharpe {stats_buyonly['sharpe']:.2f}  "
        f"MDD {stats_buyonly['max_drawdown']:.1%}\n"
        f"SPY            €{stats_spy['total_invested']:,.0f} → "
        f"€{stats_spy['final_value']:,.0f}  "
        f"({stats_spy['total_return']:+.1%})  "
        f"Sharpe {stats_spy['sharpe']:.2f}  "
        f"MDD {stats_spy['max_drawdown']:.1%}"
    )
    ax1.text(0.01, 0.97, info, transform=ax1.transAxes,
             verticalalignment="top", fontsize=8,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d", alpha=0.7))

    # ── 2. Drawdown ──────────────────────────────────────────
    ax2 = axes[1]
    dd_r = (r10 / r10.cummax() - 1) * 100
    dd_b = (b10 / b10.cummax() - 1) * 100
    dd_sp = (sp / sp.cummax() - 1) * 100

    ax2.fill_between(dd_r.index, dd_r, 0, alpha=0.25, color=C_SNP10_REBAL)
    ax2.plot(dd_r.index, dd_r, color=C_SNP10_REBAL, label=f"SNP{n} Rebal", linewidth=1.4)
    ax2.plot(dd_b.index, dd_b, color=C_SNP10_BUYONLY, label=f"SNP{n} Buy", linewidth=1.4, alpha=0.85)
    ax2.plot(dd_sp.index, dd_sp, color=C_SPY, alpha=0.7, label="SPY", linewidth=1.4)
    ax2.set_ylabel("Drawdown (%)", fontsize=9)
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, axis="y")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(cfg["output_dir"], "dca_equity_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {path}")
    plt.close()


def plot_dca_monthly_heatmap(port_values, cfg):
    """Monthly returns heatmap for the DCA portfolio."""
    rets = port_values.resample("ME").last().pct_change().dropna()
    df = rets.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret") * 100

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Only rename columns that exist
    pivot.columns = [month_names[m - 1] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.5)))
    fig.suptitle(f"SNP{cfg['top_n']} DCA — Monthly Returns (%)", fontsize=12)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#f78166", "#161b22", "#3fb950"], N=256
    )
    vals = pivot.values[~np.isnan(pivot.values)]
    vmax = max(abs(vals).max(), 1) if len(vals) else 5
    sns.heatmap(
        pivot, ax=ax, cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
        annot=True, fmt=".1f", linewidths=0.4, linecolor="#0d1117",
        cbar_kws={"shrink": 0.5, "label": "Return (%)"},
        annot_kws={"size": 7},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(cfg["output_dir"], "dca_monthly_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {path}")
    plt.close()


def plot_dca_composition(composition_log, prices, cfg):
    """Pie chart of the latest SNP10 DCA holdings."""
    if not composition_log:
        return
    latest = composition_log[-1]
    tickers = latest["tickers"]
    date = latest["date"]
    day_prices = prices.loc[date]

    vals = {}
    for t in tickers:
        p = day_prices.get(t, 0)
        if p > 0:
            vals[t] = p  # equal-weight so same $ each — just show names

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(
        f"SNP{cfg['top_n']} DCA — Current Holdings ({date.date()})",
        fontsize=12,
    )

    labels = list(vals.keys())
    sizes = [1.0 / len(labels)] * len(labels)  # equal weight
    colors = plt.cm.tab20.colors[:len(labels)]
    explode = [0.02] * len(labels)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_color("#c9d1d9")
    ax.set_facecolor("#0d1117")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(cfg["output_dir"], "dca_composition.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {path}")
    plt.close()


def plot_dca_summary_bars(stats_rebal, stats_buyonly, stats_spy, cfg):
    """Side-by-side bar comparison — 3 strategies."""
    metrics = [
        ("total_invested", "Invested (€)"),
        ("final_value", "Final Value (€)"),
        ("profit", "Profit (€)"),
        ("total_return", "Total Return"),
        ("sharpe", "Sharpe"),
        ("max_drawdown", "Max DD"),
    ]
    keys = [m[0] for m in metrics]
    labels = [m[1] for m in metrics]

    n = cfg["top_n"]
    v_rebal = [stats_rebal[k] for k in keys]
    v_buy = [stats_buyonly[k] for k in keys]
    v_spy = [stats_spy[k] for k in keys]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.suptitle(
        f"DCA Comparison — SNP{n} Rebal vs SNP{n} Buy-Only vs SPY  "
        f"(€{cfg['monthly_contribution']}/mo)",
        fontsize=12,
    )

    bars1 = ax.bar(x - width, v_rebal, width,
                   label=f"SNP{n} Rebal", color=C_SNP10_REBAL, alpha=0.85)
    bars2 = ax.bar(x, v_buy, width,
                   label=f"SNP{n} Buy-Only", color=C_SNP10_BUYONLY, alpha=0.85)
    bars3 = ax.bar(x + width, v_spy, width,
                   label="SPY", color=C_SPY, alpha=0.85)

    for bars, vals in [(bars1, v_rebal), (bars2, v_buy), (bars3, v_spy)]:
        for bar, val in zip(bars, vals):
            h = bar.get_height()
            fmt = f"€{val:,.0f}" if abs(val) > 2 else f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + abs(h) * 0.02 + 0.01,
                    fmt, ha="center", va="bottom", fontsize=7, color="#c9d1d9")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color="#8b949e", linewidth=0.6)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(cfg["output_dir"], "dca_comparison_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    cfg = CONFIG
    ensure_dirs(cfg["output_dir"], cfg["cache_dir"])

    n = cfg["top_n"]
    contrib = cfg["monthly_contribution"]

    print("=" * 60)
    print(f"  DCA BACKTEST — €{contrib}/month  •  SNP{n} vs SPY")
    print("=" * 60)

    # ── 1. Fetch data ─────────────────────────────────────────
    print("\n[1/5] Fetching market data …")
    prices = fetch_prices(UNIVERSE, cfg["start_date"], cfg["end_date"], cfg)
    shares = fetch_shares_outstanding(UNIVERSE, cfg)
    spy_prices = fetch_spy(cfg["start_date"], cfg["end_date"], cfg)

    # ── 2. Market caps ────────────────────────────────────────
    print("\n[2/5] Computing market caps …")
    mcaps = compute_approx_market_caps(prices, shares)
    print(f"  Coverage: {len(shares)}/{len(UNIVERSE)} tickers")

    # ── 3. Run DCA simulations ────────────────────────────────
    print("\n[3/5] Simulating DCA (3 strategies) …")

    # Strategy 1: SNP10 with monthly rebalancing
    rebal_val, rebal_inv, trade_log, rebal_comm, comp_log = run_dca_snp10(
        prices, mcaps, spy_prices, cfg
    )
    # Strategy 2: SNP10 buy-only (no selling)
    buyonly_val, buyonly_inv, buyonly_comm = run_dca_snp10_no_rebal(
        prices, mcaps, cfg
    )
    # Strategy 3: SPY buy-and-hold
    spy_val, spy_inv, spy_comm = run_dca_spy(spy_prices, cfg)

    n_months = int(rebal_inv.iloc[-1] / contrib)
    print(f"  {n_months} monthly contributions of €{contrib}")
    print(f"  SNP{n} Rebal   commissions: €{rebal_comm:,.2f}")
    print(f"  SNP{n} Buy-Only commissions: €{buyonly_comm:,.2f}")
    print(f"  SPY            commissions: €{spy_comm:,.2f}")

    # ── 4. Performance ────────────────────────────────────────
    print("\n[4/5] Computing performance …")
    stats_rebal = dca_performance(
        f"SNP{n} DCA — Rebalanced Monthly", rebal_val, rebal_inv
    )
    stats_buyonly = dca_performance(
        f"SNP{n} DCA — Buy-Only (No Sell)", buyonly_val, buyonly_inv
    )
    stats_spy = dca_performance("SPY DCA — Buy & Hold", spy_val, spy_inv)

    # Ranking
    results = [
        (f"SNP{n} Rebal", stats_rebal["final_value"]),
        (f"SNP{n} Buy-Only", stats_buyonly["final_value"]),
        ("SPY", stats_spy["final_value"]),
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  {'═'*55}")
    print(f"  RANKING:")
    for i, (name, val) in enumerate(results, 1):
        print(f"    {i}. {name:<20} €{val:>10,.0f}")
    print(f"  {'═'*55}")

    # ── 5. Charts ─────────────────────────────────────────────
    print("\n[5/5] Generating charts …")
    plot_dca_equity(rebal_val, buyonly_val, spy_val, rebal_inv,
                    stats_rebal, stats_buyonly, stats_spy, cfg)
    plot_dca_monthly_heatmap(rebal_val, cfg)
    plot_dca_composition(comp_log, prices, cfg)
    plot_dca_summary_bars(stats_rebal, stats_buyonly, stats_spy, cfg)

    # Save summary CSV
    perf_df = pd.DataFrame([stats_rebal, stats_buyonly, stats_spy])
    perf_path = os.path.join(cfg["output_dir"], "dca_performance_summary.csv")
    perf_df.to_csv(perf_path, index=False)
    print(f"  Performance CSV → {perf_path}")

    # Save trade log (rebalanced strategy)
    if trade_log:
        tl_path = os.path.join(cfg["output_dir"], "dca_trade_log.csv")
        pd.DataFrame(trade_log).to_csv(tl_path, index=False)
        print(f"  Trade log       → {tl_path}")

    print(f"\n  All outputs in: ./{cfg['output_dir']}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
