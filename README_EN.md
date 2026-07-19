# StockFish

StockFish is a local-first A-share research and decision-support workspace. It combines market data, financials, news sentiment, Qlib factor models, MiroFish scenario simulation, and a manual paper portfolio.

**It does not connect to a broker, place orders, or provide investment advice.**

[中文 README](README.md)

## Highlights

- Single-stock research across price, technicals, financials, valuation, news, and forum sentiment.
- Optional value, growth, trend, contrarian, and macro decision styles.
- Batch research for custom lists, CSI300, CSI500, and CSI1000 constituents.
- Qlib data updates, model training, walk-forward backtest review, and model inference from the UI.
- Local MiroFish multi-agent scenario simulation and persisted reports.
- Field-level data provenance for quotes, fundamentals, and news.
- Manual SQLite paper portfolio with audit-friendly trade records. No broker integration.
- Persisted smart-simulation history with open, download, and delete actions.

## Quick Start on Windows

Prerequisites: Git, Conda, and Python 3.11.

```powershell
git clone https://github.com/You-lie/a-share-research-workbench.git
Set-Location stock-fish
Copy-Item .env.example .env

conda create -n stock_quant python=3.11 -y
conda run -n stock_quant python -m pip install -r requirements.txt
conda run -n stock_quant python -m pip install -r MiroFish/backend/requirements.txt
conda run -n stock_quant python app.py
```

Fill `LLM_API_KEY` in `.env`. `TUSHARE_TOKEN`, `TAVILY_API_KEY`, and `ZEP_API_KEY` are optional, feature-specific integrations. Visit `http://127.0.0.1:8000`.

When `MIROFISH_AUTO_START=true`, StockFish starts local MiroFish automatically. Keep `HOST=127.0.0.1` for personal local use.

## Optional Qlib Setup

Create a separate environment and point `QLIB_PYTHON` at its interpreter:

```powershell
conda create -n stock_qlib python=3.11 -y
conda run -n stock_qlib python -m pip install pyqlib lightgbm mlflow
```

```env
QLIB_PYTHON=C:\path\to\conda\envs\stock_qlib\python.exe
```

Use the Qlib tools in Batch Analysis to download data, train a CSI300 or CSI500 model, review its backtest, and run inference. Qlib is optional for ordinary single-stock analysis.

## Local-Only Data

The following paths are generated locally and ignored by Git: `.env`, `data/paper_portfolio.db`, `memory/analysis/`, `data/outputs/`, `memory/cache/data/`, `memory/stocks/`, `qlib-zh/runtime/`, `qlib-zh/DATA/`, and `MiroFish/backend/uploads/`.

Copy these local paths separately when moving to another computer. Do not publish them because they may contain API credentials, research records, reports, simulated data, or personal paper-portfolio history.

## Attribution

This is a local-first derivative of [freenowill/stock-fish](https://github.com/freenowill/stock-fish). The upstream StockFish history includes early development by `zhuhai`, while the upstream release license identifies `freenowill` as the copyright holder.

It also integrates [MiroFish](https://github.com/666ghj/MiroFish), [Microsoft Qlib](https://github.com/microsoft/qlib), [AkShare](https://github.com/akfamily/akshare), and [Tushare](https://tushare.pro). See [NOTICE.md](NOTICE.md) for details.

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).

### Third-Party Licenses

The upstream StockFish release carried an MIT license, reproduced in [NOTICE.md](NOTICE.md). The bundled `MiroFish/backend` declares AGPL-3.0 in its own `pyproject.toml`.

Use, modification, and redistribution of each component remain subject to its applicable license. Preserve the original copyright and licensing notices.

## Usage Scope

- This project is for research and decision support only. It does not connect to brokers or place orders.
- Market data may be delayed, incomplete, cached, or sourced from a fallback provider. Consult the displayed provenance and timestamp.
- Qlib backtests, LLM conclusions, and MiroFish simulations do not represent future or live-trading returns.
