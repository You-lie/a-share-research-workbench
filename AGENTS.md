# StockFish Agent Guide / StockFish 开发 Agent 指南

## Scope / 范围

- Work only inside `stock-fish/`. Do not recreate retired scaffolding outside this directory.
- 仅在 `stock-fish/` 内工作；不要在外层重建已经废弃的脚手架。

## Environment / 环境

- StockFish and MiroFish share the `stock_quant` Conda environment (Python 3.11).
- StockFish 与 MiroFish 共用 `stock_quant` Conda 环境（Python 3.11）。
- Qlib uses a separate `stock_qlib` environment. Keep Qlib dependencies out of `stock_quant`.
- Qlib 使用独立的 `stock_qlib` 环境；不要把 Qlib 依赖装入 `stock_quant`。
- On Windows, invoke Python by its resolved full path when Conda activation is unreliable. Resolve the active environment interpreter from Conda instead of hard-coding a user-specific path.
- 在 Windows 上，如 Conda 激活不稳定，请使用已解析的完整 Python 路径；从 Conda 环境中获取解释器，不要硬编码任何用户机器路径。
- Run Qlib through `qlib-zh/local_runtime.py`. Do not reintroduce Docker execution.
- Qlib 必须通过 `qlib-zh/local_runtime.py` 在本地运行；不要重新引入 Docker 执行链路。
- Keep Qlib data and MLflow records under `qlib-zh/runtime/`, and model outputs under `qlib-zh/DATA/analysis_outputs/`.
- Qlib 数据与 MLflow 记录放在 `qlib-zh/runtime/`；模型和训练产物放在 `qlib-zh/DATA/analysis_outputs/`。

## Product Rules / 产品规则

- This is an A-share research and signal system. Never add broker access, order placement, or automatic trading.
- 这是 A 股研究和信号辅助系统。禁止接入券商、下单或自动交易。
- Keep market data, indicators, news sentiment, Qlib signals, LLM judgement, and MiroFish simulation clearly separated in code and UI.
- 行情、指标、新闻舆情、Qlib 信号、LLM 判断和 MiroFish 推演必须在代码与界面上清晰区分。
- Present backtests as historical research only. Never imply guaranteed or live-trading returns.
- 回测只能作为历史研究结果展示，不得暗示保证收益或实盘表现。
- Preserve timestamps, data provenance, configuration snapshots, and cost assumptions in generated outputs.
- 生成结果必须保留时间、数据溯源、配置快照和成本假设。
- Never expose `.env` values, API keys, local reports, paper-portfolio records, or personal paths in code, logs, docs, or responses.
- 不得在代码、日志、文档或回复中泄露 `.env`、API Key、本地报告、纸面组合记录或个人路径。

## Local Runtime / 本地运行

- `app.py` starts StockFish and automatically starts a local MiroFish service when `MIROFISH_AUTO_START=true`.
- `app.py` 启动 StockFish，并在 `MIROFISH_AUTO_START=true` 时自动拉起本地 MiroFish。
- Bind local services to `127.0.0.1` unless the user explicitly requests a network-accessible deployment.
- 除非用户明确要求局域网或公网部署，本地服务必须绑定到 `127.0.0.1`。
- The Qlib UI owns data download, path settings, training, inference, and model deletion. Keep paths project-local by default.
- Qlib 页面负责数据下载、路径设置、训练、推理和模型删除；默认路径必须在项目内部。

## Project Map / 项目结构

- `app.py`: Flask API and application entry point. / Flask API 与主启动入口。
- `static/index.html`: single-page analysis UI. / 单页分析前端。
- `analysis/`: analysis, LLM decisions, and batch workflows. / 分析、LLM 决策和批量任务。
- `market_data/`: quotes, financials, news, and provenance adapters. / 行情、财务、新闻与数据溯源适配器。
- `paper_portfolio.py`: local SQLite paper-portfolio ledger. / 本地 SQLite 纸面组合账本。
- `prediction_report/`: persisted smart-simulation reports. / 持久化的智能推演报告。
- `simulation_bridge/` and `MiroFish/backend/`: optional simulation integration. / 可选的推演服务集成。
- `qlib-zh/`: Qlib data, training, inference, and backtest integration. / Qlib 数据、训练、推理与回测集成。
- `data/outputs/`: local reports and generated artifacts; ignored by Git. / 本地报告和生成物，Git 忽略。

## Verification / 验证

- Run `python -m py_compile app.py` after backend changes.
- 后端修改后运行 `python -m py_compile app.py`。
- Parse the inline JavaScript in `static/index.html` after frontend changes and visually check light, dark, desktop, and mobile layouts.
- 前端修改后检查 `static/index.html` 的内联 JavaScript，并验证浅色、深色、桌面和手机布局。
- Do not delete or reset existing user data, model outputs, or unrelated working-tree changes.
- 不得删除、重置已有用户数据、模型产物或无关的工作区改动。
