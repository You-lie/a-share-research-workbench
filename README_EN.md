# Guanchao (观潮)

**A local-first intelligent research workspace for China A-shares.** Market data, financials, sentiment, multi-agent debate, machine-learning stock selection, and scenario simulation — all wired into a single page, with every piece of data staying on your machine. No broker connection, no automatic trading.

> Not a toy demo: data carries provenance, conclusions pass decision guardrails, backtests come with quality panels, and trades are journaled for review.

[中文 README](README.md) · [Attribution & License](#attribution--license)

![Single-stock analysis](docs/screenshots/01-single-analysis.png)

## Highlights

### Multi-agent debate + decision styles
Each stock is analyzed independently by technical, fundamental, and sentiment agents, then adjudicated by a moderator. Seven investment masters (Buffett, Graham, Lynch, ...) are available as review lenses. Every conclusion passes decision guardrails: contradictory signals are downgraded to "observe", position changes require an entered position, and no buy/sell advice is produced when quotes are unavailable or simulated.

![Multi-agent debate](docs/screenshots/02-agent-debate.png)

### Field-level data provenance
Every quote, financial metric, and news item carries its **source, retrieval time, reporting period, and basis**. Fallback, cached, or degraded data is labeled explicitly — simulated data is never passed off as real quotes.

### Qlib machine-learning stock selection
A complete in-UI Qlib workflow: one-click data update → LightGBM walk-forward training → per-fold out-of-sample quality panel (return, Sharpe, drawdown, IC, win rate, stability) → inference that feeds top candidates back into batch analysis. CSI300 and CSI500 universes. Train once, use for months.

![Batch analysis & Qlib tools](docs/screenshots/03-batch-qlib.png)

### Concurrent batch analysis
Custom lists or index constituents, 5-way concurrency by default (configurable), with a maximum external-call estimate before you start, live per-stock progress, and results ordered by your input.

### MiroFish scenario simulation
Launch multi-agent crowd simulation for key positions: agents role-play market participants and produce scenario reports. Simulation history is stored locally for review.

![Simulation history](docs/screenshots/06-prediction-history.png)

### Position ledger & review loop
A manually-confirmed paper ledger: execution price, fees, linked analysis snapshot, and signal rationale per trade; stale vs. fresh quotes clearly distinguished; positions and P&L recomputed chronologically; void and correct actions keep a full audit trail.

![Position ledger](docs/screenshots/04-position-ledger.png)

### Configuration in the UI
Manage every API key and runtime parameter from the page — changes apply immediately without restart. **LLM balance and today's spend** are shown live; secrets are displayed masked.

![Configuration](docs/screenshots/05-config-settings.png)

### Local-first & private
All data lands on your disk and is excluded by `.gitignore`: analysis snapshots, ledger database, Qlib data/models, simulation artifacts. The server binds to `127.0.0.1` only.

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

Fill `LLM_API_KEY` in `.env` (or in the in-app Configuration page). `TUSHARE_TOKEN`, `TAVILY_API_KEY`, and `ZEP_API_KEY` are optional, feature-specific integrations. Visit `http://127.0.0.1:8000`.

When `MIROFISH_AUTO_START=true`, Guanchao starts local MiroFish automatically.

## Optional Qlib Setup

Create a separate environment and point `QLIB_PYTHON` at its interpreter:

```powershell
conda create -n stock_qlib python=3.11 -y
conda run -n stock_qlib python -m pip install pyqlib lightgbm mlflow
```

```env
QLIB_PYTHON=C:\path\to\conda\envs\stock_qlib\python.exe
```

Then use the Qlib tools in Batch Analysis. Daily flow: update data after market close → run inference with an existing model → review the top candidates in batch analysis. Retrain monthly. Qlib is optional for ordinary single-stock analysis.

## Local-Only Data

The following paths are generated locally and ignored by Git: `.env`, `data/paper_portfolio.db`, `memory/analysis/`, `data/outputs/`, `memory/cache/data/`, `memory/stocks/`, `qlib-zh/runtime/`, `qlib-zh/DATA/`, and `MiroFish/backend/uploads/`.

Copy these paths separately when moving to another computer. Do not publish them — they may contain API credentials, research records, reports, and personal paper-portfolio history.

## Usage Scope

- Research and decision support only. No broker connection, no order placement.
- Market data may be delayed, incomplete, cached, or sourced from a fallback provider. Consult the displayed provenance and timestamp.
- Qlib backtests, LLM conclusions, and MiroFish simulations do not represent future or live-trading returns.

## Attribution & License

This project is a derivative work built on the open-source project [freenowill/stock-fish](https://github.com/freenowill/stock-fish). The local runtime model, data security layers, Qlib local workflow, data provenance, position ledger, configuration management, and simulation history are redesigned and implemented in this project.

- Upstream StockFish copyright: `Copyright (c) 2026 freenowill` (MIT), preserved in [NOTICE.md](NOTICE.md).
- Integrated components: [MiroFish](https://github.com/666ghj/MiroFish) (AGPL-3.0), [Microsoft Qlib](https://github.com/microsoft/qlib), [AkShare](https://github.com/akfamily/akshare), and [Tushare](https://tushare.pro).

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0). Preserve the original copyright and license notices when redistributing.
