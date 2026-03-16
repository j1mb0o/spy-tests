# SNP10 Backtester

Backtest investing in the **top 10 S&P 500 companies** instead of the whole index. Two scripts compare concentrated top-10 strategies against SPY using real historical data from Yahoo Finance.

## Quick Start

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run the DCA experiment (no setup needed — uv handles everything)
uv run dca_backtest.py

# Run the lump-sum backtest
uv run snp10_backtest.py
```

First run downloads ~20 years of price data and caches it locally in `.data_cache/`.

## Scripts

### `dca_backtest.py` — DCA Experiment (100/month for 20 years)

Compares three dollar-cost averaging strategies side by side:

| Strategy | Description |
|---|---|
| **SNP10 Rebalanced** | Each month: sell everything, buy equal-weight top 10 by market cap |
| **SNP10 Buy-Only** | Each month: buy current top 10 with new money, never sell old positions |
| **SPY** | Each month: buy SPY shares, never sell |

All strategies invest 100/month, include realistic transaction costs (0.1% commission + 0.05% spread), and run from April 2006 to present.

#### Sample Results (2006-2026)

| | SNP10 Rebal | SNP10 Buy-Only | SPY |
|---|---|---|---|
| Total Invested | 24,000 | 24,000 | 24,000 |
| Final Value | 145,365 | 163,836 | 101,562 |
| Total Return | 506% | 583% | 323% |
| Sharpe Ratio | 1.10 | 1.13 | 1.06 |
| Max Drawdown | -45.1% | -29.3% | -35.7% |

The buy-only strategy wins because winners keep compounding without being sold, and commission drag is negligible (24 vs 16,496 for rebalancing).

### `snp10_backtest.py` — Lump-Sum Backtest

Full-featured backtester for a lump-sum top-10 strategy with configurable:
- Rebalancing frequency (quarterly, annually, monthly)
- Weighting scheme (equal-weight or market-cap-weight)
- Transaction costs
- Benchmark comparison vs SPY

## Output

Both scripts generate charts and data in `output/`:

| File | Description |
|---|---|
| `dca_equity_curve.png` | 3-strategy equity curves + drawdown |
| `dca_comparison_bars.png` | Side-by-side metric comparison |
| `dca_monthly_heatmap.png` | Monthly returns heatmap |
| `dca_composition.png` | Current top-10 holdings |
| `dca_performance_summary.csv` | Full metrics for all strategies |
| `dca_trade_log.csv` | Every buy/sell for the rebalanced strategy |

## Configuration

Edit the `CONFIG` dict at the top of either script:

```python
CONFIG = {
    "start_date": "2006-04-01",
    "monthly_contribution": 100,    # change to any amount
    "top_n": 10,                    # top N companies
    "commission_pct": 0.001,        # 0.1% per trade
    "spread_pct": 0.0005,           # 0.05% bid-ask spread
}
```

## How It Works

1. **Universe**: 50 large-cap tickers that have historically been in or near the S&P 500 top 10
2. **Market cap ranking**: `shares_outstanding x price` on each rebalance date (shares outstanding from Yahoo Finance `fast_info`)
3. **Top-N selection**: Pick the N largest by market cap at each investment date
4. **DCA simulation**: Track fractional shares, cash, commissions day-by-day
5. **Mark-to-market**: Portfolio valued daily using closing prices

## Caveats

- **Survivorship bias**: The 50-ticker universe was selected with hindsight. Companies that fell out of the top 50 entirely are not included.
- **Shares outstanding**: Uses current shares outstanding as a proxy for historical values. Stock splits are handled by `yfinance`'s adjusted prices, but share counts don't change historically.
- **Currency**: Prices are in USD. The euro amounts assume 1:1 EUR/USD for simplicity.
- **No dividends reinvested**: Uses price returns only (adjusted close), dividends are not explicitly reinvested.
- **Not financial advice**: This is a backtesting experiment for educational purposes.

## Dependencies

Managed by [uv](https://docs.astral.sh/uv/) via `pyproject.toml`:

- `yfinance` — market data
- `pandas` + `numpy` — data processing
- `matplotlib` + `seaborn` — charts
- `pyarrow` — parquet caching
