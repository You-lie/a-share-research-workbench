# StockFish

本地运行的 A 股研究与投资决策辅助工具。它整合行情、财务、新闻舆情、Qlib 因子模型和 MiroFish 情景推演，帮助你做研究、记录纸面组合和复盘。

**它不连接券商、不执行自动交易、不构成投资建议。**

[English README](README_EN.md)

## 能做什么

- **单股分析**：行情、技术指标、财务数据、估值、新闻和股吧舆情的统一分析。
- **决策风格**：可选价值、成长、趋势、逆向和宏观等投资风格审阅结论。
- **批量分析**：按自选股票池、沪深 300、中证 500 或中证 1000 做批量研究。
- **Qlib 工作流**：在页面中更新 A 股数据、训练模型、查看 walk-forward 样本外回测质量，并进行模型推理。
- **MiroFish 智能推演**：在本机自动启动 MiroFish，对单只股票做多智能体情景推演并生成报告。
- **数据溯源**：行情、财务和新闻会显示来源、抓取时间、报告期、口径和降级状态。
- **纸面组合**：手工确认买入、加仓、减持和清空，记录手续费、风险位和关联的分析快照；不接券商。
- **推演历史**：保存已完成的智能推演报告，重启后仍可查看、下载或删除报告副本。

## 快速开始（Windows + Conda）

前提：已安装 Git、Conda 和 Python 3.11。

```powershell
git clone <你的 GitHub 仓库地址>
Set-Location stock-fish
Copy-Item .env.example .env

conda create -n stock_quant python=3.11 -y
conda run -n stock_quant python -m pip install -r requirements.txt
conda run -n stock_quant python -m pip install -r MiroFish/backend/requirements.txt
```

编辑 `.env`，至少填入可用的 `LLM_API_KEY`；Tushare、Tavily 和 Zep 按需填写。然后启动：

```powershell
conda run -n stock_quant python app.py
```

打开 `http://127.0.0.1:8000`。本地 `MIROFISH_AUTO_START=true` 时，StockFish 会自动启动 MiroFish，无需另开终端。

## 配置要点

| 变量 | 用途 | 是否必需 |
| --- | --- | --- |
| `LLM_API_KEY` | 深度分析、决策风格和 MiroFish 推演 | 使用这些功能时必需 |
| `TUSHARE_TOKEN` | 更完整的 A 股基础与财务数据 | 推荐 |
| `TAVILY_API_KEY` | 新闻搜索与摘要 | 推荐 |
| `ZEP_API_KEY` | MiroFish 图记忆 | 可选 |
| `STOCK_BACKEND` | `advanced`、`tushare`、`akshare` 或 `mock` | 推荐 `advanced` |
| `QLIB_PYTHON` | 独立 Qlib Conda 环境的 Python 路径 | 使用 Qlib 时必需 |

不要提交 `.env`，也不要在 Issue、截图或日志中粘贴密钥。

## Qlib（可选）

Qlib 使用独立环境，避免与主服务和 MiroFish 依赖混在一起：

```powershell
conda create -n stock_qlib python=3.11 -y
conda run -n stock_qlib python -m pip install pyqlib lightgbm mlflow
```

在 `.env` 中填写该环境 Python 的绝对路径，例如：

```env
QLIB_PYTHON=C:\path\to\conda\envs\stock_qlib\python.exe
```

启动 StockFish 后，在“批量分析 → Qlib 批量工具”中依次完成：

1. 更新 Qlib 数据。
2. 选择 CSI300 或 CSI500 训练目标并开始训练。
3. 查看回测质量，再选择模型做推理。

模型推理需要**训练好的模型和最新 Qlib 数据**。普通单股分析不依赖 Qlib。

## 本地数据与隐私

下列目录由程序自动创建并已被 `.gitignore` 排除，不会上传到 GitHub：

| 路径 | 内容 |
| --- | --- |
| `data/paper_portfolio.db` | 纸面组合的交易、设置和行情快照 |
| `memory/analysis/` | 单股深度分析 JSON 快照 |
| `data/outputs/reports/` | 智能推演 HTML / JSON 报告与历史入口 |
| `memory/stocks/`、`memory/cache/data/` | 本地行情、新闻、财务缓存 |
| `qlib-zh/runtime/`、`qlib-zh/DATA/` | Qlib 数据、MLflow 记录、模型与训练产物 |
| `MiroFish/backend/uploads/` | MiroFish 项目、报告、日志和模拟过程文件 |

纸面组合数据库会在首次打开纸面组合时自动创建。更换电脑时，复制这些本地数据目录才能带走历史记录；不要把它们公开上传。

## 目录概览

```text
stock-fish/
├── app.py                    # StockFish 启动入口和 Flask API
├── static/index.html         # 分析前端
├── analysis/                 # 分析、决策风格与批量流程
├── market_data/              # 行情、财务、新闻和数据溯源
├── paper_portfolio.py        # 本地 SQLite 纸面组合
├── prediction_report/        # 智能推演报告生成
├── simulation_bridge/        # StockFish 到 MiroFish 的桥接
├── MiroFish/backend/         # 本地群体智能推演服务
├── qlib-zh/                  # Qlib 数据、训练、推理与回测集成
└── data/                     # 本地运行数据，默认不提交
```

## 上游项目与署名

这是一个面向本地 Windows 使用的**二次开发版本**，在上游基础上重新整理了运行方式、数据安全、Qlib 本地工作流、数据溯源、纸面组合和推演历史。

- StockFish 上游发布仓库：[freenowill/stock-fish](https://github.com/freenowill/stock-fish)。上游 Git 历史中的早期 StockFish 开发提交作者为 `zhuhai`；上游发布版本的版权声明为 `Copyright (c) 2026 freenowill`。
- MiroFish 群体智能推演组件：[666ghj/MiroFish](https://github.com/666ghj/MiroFish)。本仓库包含其后端集成代码。
- Qlib 因子研究框架：[microsoft/qlib](https://github.com/microsoft/qlib)。
- A 股数据主要依赖 [AkShare](https://github.com/akfamily/akshare) 与 [Tushare](https://tushare.pro)。

详见 [NOTICE.md](NOTICE.md)。二次开发不等于原作；请在再分发时保留上游署名和许可信息。

## 许可证说明

上游 StockFish 发布版本带有 MIT 许可证，已在 [NOTICE.md](NOTICE.md) 中保留原始版权和许可文本。当前仓库集成的 `MiroFish/backend` 在其 `pyproject.toml` 中声明为 **AGPL-3.0**。

因此，**不要把整个组合仓库标记为 MIT-only**。公开发布、再分发或接受外部贡献前，请由仓库维护者添加与 AGPL-3.0 兼容的顶层许可证和完整许可证文本，并保留所有第三方通知。

## 风险提示

- 所有行情、财务和新闻数据都可能延迟、缺失或来自备用来源。
- Qlib 回测只是历史样本外研究，不代表未来或实盘收益。
- MiroFish、LLM 和投资风格输出是辅助研究观点，不是交易指令。
- 所有实际买卖、仓位、止损和风险承担都由使用者自行决定。
