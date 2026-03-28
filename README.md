# Borrowed Stability — Philippine Peso Depreciation Analysis

Five Economist-styled charts examining the structural weakness of the Philippine peso (PHP) since the Trump inauguration, isolating dollar-specific movements from genuine peso depreciation.

## Charts produced

| File | Description |
|---|---|
| `chart_1_full_history.png` | Full 2018–present history with ARIMA counterfactual and DXY-normalised rate |
| `chart_2_dxy_dollar_index.png` | US Dollar Index (DXY) with strength/weakness shading since inauguration |
| `chart_3_post_trump_zoom.png` | Post-inauguration zoom: actual vs both counterfactuals |
| `chart_4_php_cushion.png` | The "borrowed shield" — how much USD weakness is masking PHP losses |
| `chart_5_summary_table.png` | Key metrics table in Economist data-table style |

## Methodology

- **ARIMA(1,1,1) with drift** — trained on pre-Trump data (Jul 2020–Jan 2025) to project the peso's pre-inauguration trend forward as a counterfactual.
- **DXY-normalisation** — adjusts the actual USDPHP rate by the change in the dollar index since inauguration day, stripping out broad USD moves to reveal PHP-specific dynamics.
- **User hypothesis** — a reference line at 70 PHP/USD representing the hypothetical rate under a stable dollar environment.

## Prerequisites

[uv](https://docs.astral.sh/uv/) — a fast Python package manager. Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

All Python dependencies are declared inline in the script (PEP 723 format) and are installed automatically by `uv run`.

## Running

```bash
# Default: saves PNGs to ./images/
uv run php_economist_clean.py

# Custom output directory
uv run php_economist_clean.py --out-dir ./charts

# Via environment variable
OUT_DIR=./charts uv run php_economist_clean.py
```

Output PNGs are written to `./images/` by default. The directory is created automatically.

## Data sources

- **Yahoo Finance** — `USDPHP=X` (Philippine peso spot rate), `DX-Y.NYB` (DXY dollar index)
- **Bangko Sentral ng Pilipinas (BSP)** — reference rates for context
- Data is fetched live each run; no local cache is used.

## Design system

Follows The Economist's visual identity:

| Token | Hex | Role |
|---|---|---|
| Canvas | `#F5F4EF` | Background |
| Red | `#E3120B` | Actual data, brand accent |
| Blue | `#2E45B8` | ARIMA counterfactual |
| Teal | `#1DC9A4` | DXY-normalised series |
| London grey | `#595959` | Captions, source lines |

## Python version

Requires Python ≥ 3.11 (pinned in `.python-version`).

```
uv python pin 3.11   # already done
```
