"""
StockFish - A 股实时分析 + 股价推演系统 (v2)

API:
  POST /api/analyze     完整多因子分析（技术+基本面+舆情+预测）
  POST /api/predict     启动股价推演（支持 base/bull/bear 场景）
  GET  /api/predict/<id>  推演状态
  GET  /api/predict/<id>/stream  SSE 推演进度流
  GET  /api/predict/<id>/report  推演报告 HTML
  GET  /api/config      系统配置
"""
import atexit
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context, send_file
from flask_cors import CORS
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings
from analysis.agent import StockAnalysisAgent
from analysis.batch_analyzer import BatchAnalyzer
from simulation_bridge.orchestrator import SimulationOrchestrator
from prediction_report.report_generator import PredictionReportGenerator
from paper_portfolio import PaperPortfolioStore
from qlib_backtest import load_backtest

# ---- 解析 qlib-zh 目录（兼容 git worktree，models/DATA 在主仓库中） ----
_qlib_zh_dir = Path(__file__).resolve().parent / "qlib-zh"
if str(_qlib_zh_dir) not in sys.path:
    sys.path.insert(0, str(_qlib_zh_dir))

from local_runtime import (
    ProcessCancelledError as _QlibProcessCancelledError,
    build_runtime_env as _build_qlib_runtime_env,
    configure_runtime_paths as _configure_qlib_runtime_paths,
    get_runtime_paths as _get_qlib_runtime_paths,
    python_command as _qlib_python_command,
    run_streaming as _run_qlib_streaming,
    validate_runtime as _validate_qlib_runtime,
)

def _resolve_qlib_dir() -> Path:
    """返回 qlib-zh 的 DATA 目录所在位置（git worktree 时回退到主仓库）"""
    data_dir = _qlib_zh_dir / "DATA"
    # 如果 DATA 目录存在，直接使用
    if data_dir.exists():
        return _qlib_zh_dir
    # worktree: 解析主仓库路径
    gitfile = _qlib_zh_dir.parent / ".git"
    if gitfile.is_file():
        content = gitfile.read_text().strip()
        if content.startswith("gitdir:"):
            # gitdir: /path/to/main/.git/worktrees/name
            git_dir = Path(content.split(":", 1)[1].strip())
            # .git/worktrees/name → parent 3× 回到主仓库根目录
            main_repo = git_dir.parent.parent.parent if "worktrees" in str(git_dir) else git_dir.parent
            candidate = main_repo / "qlib-zh"
            if candidate.exists():
                return candidate
    return _qlib_zh_dir

_qlib_base_dir = _resolve_qlib_dir()
logger.info(f"Qlib 基础目录: {_qlib_base_dir}")

# ---- Qlib 推理模块 (目录名含连字符，使用 importlib 加载) ----
_spec = importlib.util.spec_from_file_location("infer_runner", _qlib_zh_dir / "infer_runner.py")
_infer_runner_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_infer_runner_mod)
run_qlib_inference = _infer_runner_mod.run_inference

# ---- Qlib 数据更新模块 ----
try:
    _spec_data = importlib.util.spec_from_file_location("data_runner", _qlib_zh_dir / "data_runner.py")
    _data_runner_mod = importlib.util.module_from_spec(_spec_data)
    _spec_data.loader.exec_module(_data_runner_mod)
    run_qlib_data_update = _data_runner_mod.run_data_update
except Exception:
    run_qlib_data_update = None

# ---- Qlib 训练模块 ----
try:
    _spec_train = importlib.util.spec_from_file_location("train_runner", _qlib_zh_dir / "train_runner.py")
    _train_runner_mod = importlib.util.module_from_spec(_spec_train)
    _spec_train.loader.exec_module(_train_runner_mod)
    run_qlib_training = _train_runner_mod.run_training
except Exception:
    run_qlib_training = None


def _qlib_data_path() -> Path:
    return _get_qlib_runtime_paths()["data_dir"]


def _qlib_models_path() -> Path:
    return _get_qlib_runtime_paths()["model_dir"]


def _qlib_model_infer_ready(model_dir: Path) -> bool:
    """A model is inferable only after a walk-forward fold saved its parameters."""
    walk_forward_dir = model_dir / "model_predict" / "walk_forward"
    if not walk_forward_dir.is_dir():
        return False
    return any(
        checkpoint.parent.parent.parent.joinpath("workflow_config_practice.yaml").is_file()
        for checkpoint in walk_forward_dir.glob("*/model_runs/*/mlflow_run/artifacts/params.pkl")
    )


def _sync_qlib_runner_paths(paths: dict[str, Path]) -> None:
    for module_name in ("_data_runner_mod", "_train_runner_mod", "_infer_runner_mod"):
        module = globals().get(module_name)
        if module is None:
            continue
        if hasattr(module, "QLIB_DATA_DIR"):
            module.QLIB_DATA_DIR = paths["data_dir"]
        if hasattr(module, "ANALYSIS_OUTPUTS_DIR"):
            module.ANALYSIS_OUTPUTS_DIR = paths["model_dir"]
        if hasattr(module, "MLRUNS_DIR"):
            module.MLRUNS_DIR = paths["mlruns_dir"]

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# ===== 全局状态 =====
agent = StockAnalysisAgent()
batch_analyzer = BatchAnalyzer()
orchestrator = SimulationOrchestrator()
report_gen = PredictionReportGenerator()
predictions = {}
_predictions_lock = threading.Lock()
analysis_tasks = {}
_analysis_lock = threading.Lock()
batch_tasks = {}
_batch_lock = threading.Lock()
qlib_tasks = {}
_qlib_lock = threading.Lock()
qlib_data_tasks = {}
_qlib_data_lock = threading.Lock()
qlib_train_tasks = {}
_qlib_train_lock = threading.Lock()
qlib_finetune_tasks = {}
_qlib_finetune_lock = threading.Lock()
_paper_portfolio_store = None
_paper_portfolio_lock = threading.Lock()
_mirofish_process = None
_mirofish_process_lock = threading.Lock()
_mirofish_log_handle = None


def _is_local_mirofish_host(host: str) -> bool:
    return str(host or "").strip().lower() in {"localhost", "127.0.0.1", "::1"}


def _mirofish_port_is_open(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if str(host).lower() == "localhost" else host
    try:
        with socket.create_connection((probe_host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _stop_managed_mirofish() -> None:
    """Stop only the MiroFish process started by this StockFish process."""
    global _mirofish_process, _mirofish_log_handle
    with _mirofish_process_lock:
        process = _mirofish_process
        _mirofish_process = None
    if process is not None and process.poll() is None:
        logger.info("停止由 StockFish 启动的 MiroFish 服务...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    if _mirofish_log_handle is not None:
        _mirofish_log_handle.close()
        _mirofish_log_handle = None


def _start_mirofish_if_needed() -> bool:
    """Reuse a local MiroFish service or start one alongside StockFish."""
    global _mirofish_process, _mirofish_log_handle
    if not settings.MIROFISH_AUTO_START:
        logger.info("MiroFish 自动启动已关闭")
        return orchestrator.client.health_check()

    host = str(settings.MIROFISH_HOST or "localhost")
    port = int(settings.MIROFISH_PORT)
    if not _is_local_mirofish_host(host):
        logger.info(f"MiroFish 使用远程地址 {host}:{port}，不由 StockFish 启动")
        return orchestrator.client.health_check()
    if orchestrator.client.health_check():
        logger.info(f"复用已运行的 MiroFish 服务: {orchestrator.client.base_url}")
        return True
    if _mirofish_port_is_open(host, port):
        logger.warning(f"端口 {host}:{port} 已被占用，但不是可用的 MiroFish 服务")
        return False

    backend_dir = Path(__file__).resolve().parent / "MiroFish" / "backend"
    entrypoint = backend_dir / "run.py"
    if not entrypoint.is_file():
        logger.warning(f"MiroFish 启动文件不存在: {entrypoint}")
        return False

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "mirofish-managed.log"
    env = os.environ.copy()
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "FLASK_HOST": "127.0.0.1",
        "FLASK_PORT": str(port),
        "FLASK_DEBUG": "False",
    })
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        _mirofish_log_handle = open(log_path, "a", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, str(entrypoint)],
            cwd=str(backend_dir),
            env=env,
            stdout=_mirofish_log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    except OSError as exc:
        logger.warning(f"MiroFish 启动失败: {exc}")
        if _mirofish_log_handle is not None:
            _mirofish_log_handle.close()
            _mirofish_log_handle = None
        return False

    with _mirofish_process_lock:
        _mirofish_process = process
    deadline = time.monotonic() + max(1.0, float(settings.MIROFISH_START_TIMEOUT_SECONDS))
    while time.monotonic() < deadline:
        if orchestrator.client.health_check():
            logger.info(f"MiroFish 已启动: {orchestrator.client.base_url}")
            return True
        if process.poll() is not None:
            logger.warning(f"MiroFish 启动进程已退出（退出码 {process.returncode}），查看日志: {log_path}")
            return False
        time.sleep(0.5)
    logger.warning(f"MiroFish 启动超时，服务可能仍在初始化。日志: {log_path}")
    return False


atexit.register(_stop_managed_mirofish)


def _get_paper_portfolio_store() -> PaperPortfolioStore:
    """Create the local paper ledger only when the user opens that workflow."""
    global _paper_portfolio_store
    with _paper_portfolio_lock:
        if _paper_portfolio_store is None:
            _paper_portfolio_store = PaperPortfolioStore()
        return _paper_portfolio_store


def _paper_quote(symbol: str):
    quote = agent.provider.get_quote(symbol)
    if quote is not None and not getattr(quote, 'source', ''):
        quote.source = agent.provider.backend_name
        quote.endpoint = '实时行情'
    return quote


def _serialize_qlib_paths() -> dict:
    paths = _get_qlib_runtime_paths()
    data_dir = paths["data_dir"]
    model_dir = paths["model_dir"]
    data_ready = (
        (data_dir / "calendars" / "day.txt").is_file()
        and (data_dir / "features").is_dir()
    )
    model_count = sum(1 for path in model_dir.iterdir() if path.is_dir()) if model_dir.is_dir() else 0
    return {
        "data_dir": str(data_dir),
        "model_dir": str(model_dir),
        "mlruns_dir": str(paths["mlruns_dir"]),
        "default_data_dir": str(paths["default_data_dir"]),
        "default_model_dir": str(paths["default_model_dir"]),
        "data_ready": data_ready,
        "model_count": model_count,
    }


def _qlib_paths_busy_task() -> str:
    active_statuses = {"pending", "running", "cancelling"}
    task_groups = (
        ("数据更新", _qlib_data_lock, qlib_data_tasks),
        ("模型训练", _qlib_train_lock, qlib_train_tasks),
        ("模型推理", _qlib_lock, qlib_tasks),
        ("模型微调", _qlib_finetune_lock, qlib_finetune_tasks),
    )
    for label, lock, tasks in task_groups:
        with lock:
            if any(task.get("status") in active_statuses for task in tasks.values()):
                return label
    return ""


def _resolve_user_qlib_path(raw_path: object, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{label}不能为空")
    candidate = Path(os.path.expandvars(raw_path.strip())).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"{label}必须填写绝对路径")
    resolved = candidate.resolve()
    forbidden_paths = {
        Path(resolved.anchor).resolve(),
        Path.home().resolve(),
        Path(__file__).resolve().parent,
        _qlib_zh_dir.resolve(),
    }
    if resolved in forbidden_paths:
        raise ValueError(f"{label}不能设置为磁盘根目录、用户目录或项目根目录")
    return resolved


@app.route('/api/qlib/paths', methods=['GET', 'PUT'])
def qlib_paths():
    """读取或更新本地 Qlib 数据与模型目录。"""
    if request.method == 'GET':
        return jsonify(_serialize_qlib_paths())

    busy_task = _qlib_paths_busy_task()
    if busy_task:
        return jsonify({'error': f'当前正在执行{busy_task}，结束或停止后才能修改路径'}), 409

    data = request.get_json(silent=True) or {}
    try:
        data_dir = _resolve_user_qlib_path(data.get('data_dir'), '数据目录')
        model_dir = _resolve_user_qlib_path(data.get('model_dir'), '模型目录')
        if data_dir == model_dir or data_dir in model_dir.parents or model_dir in data_dir.parents:
            raise ValueError('数据目录和模型目录不能相同或互相嵌套')
        if data_dir.name.lower() != 'cn_data':
            raise ValueError('数据目录末级文件夹必须命名为 cn_data')
        if data_dir.is_file() or model_dir.is_file():
            raise ValueError('配置路径必须是文件夹，不能是文件')
        if data_dir.is_dir() and any(data_dir.iterdir()):
            recognized_data = (
                (data_dir / 'calendars').is_dir()
                and (data_dir / 'instruments').is_dir()
                and (data_dir / 'features').is_dir()
            )
            if not recognized_data:
                raise ValueError('数据目录不是空目录，也不是有效的 Qlib cn_data，已拒绝保存以防误删文件')
        data_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        paths = _configure_qlib_runtime_paths(data_dir, model_dir)
        _sync_qlib_runner_paths(paths)
        _index_stocks_cache.clear()
        _index_stocks_cache_time.clear()
    except (OSError, ValueError) as exc:
        return jsonify({'error': str(exc)}), 400

    logger.info(f"Qlib 路径已更新: data={data_dir}, models={model_dir}")
    return jsonify(_serialize_qlib_paths())


# ==========================================
#  API: 分析 (Phase 2 - StockEngine Agent)
# ==========================================

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """完整多因子分析"""
    data = request.get_json(silent=True) or {}
    raw_symbol = data.get('symbol', '').strip()
    cost_price = data.get('cost_price', 0)
    master = data.get('master', '').strip().lower()
    shares = data.get('shares', 0) or 0
    total_assets = data.get('total_assets', 0) or 0.0
    available_cash = data.get('available_cash', 0) or 0.0
    if not raw_symbol:
        return jsonify({'error': '请提供股票代码'}), 400

    # 支持股票名称输入解析
    from stock_name_resolver import resolve_symbol as _resolve_symbol
    symbol = _resolve_symbol(raw_symbol) or raw_symbol.upper()

    logger.info(f"开始深度分析 [{symbol}] 成本价={cost_price} 持仓={shares}股 总资产={total_assets} 可用={available_cash} master={master or 'off'}")
    result = agent.analyze(symbol, cost_price=float(cost_price) if cost_price else 0.0,
                           master=master, shares=int(shares),
                           total_assets=float(total_assets), available_cash=float(available_cash))
    logger.info(f"[{symbol}] 分析完成 (状态: {result.get('status')})")
    return jsonify(result)


@app.route('/api/analyze/start', methods=['POST'])
def analyze_start():
    """启动可停止的单股分析任务；同步接口保留给已有调用方。"""
    data = request.get_json(silent=True) or {}
    raw_symbol = data.get('symbol', '').strip()
    if not raw_symbol:
        return jsonify({'error': '请提供股票代码'}), 400

    from stock_name_resolver import resolve_symbol as _resolve_symbol
    symbol = _resolve_symbol(raw_symbol) or raw_symbol.upper()
    task_id = f"analysis_{uuid.uuid4().hex[:12]}"
    task = {
        'task_id': task_id,
        'symbol': symbol,
        'status': 'pending',
        'progress': 0.0,
        'message': '等待开始分析',
        'result': None,
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
        '_cancel_event': threading.Event(),
    }
    with _analysis_lock:
        analysis_tasks[task_id] = task

    cost_price = float(data.get('cost_price') or 0)
    master = str(data.get('master') or '').strip().lower()
    shares = int(data.get('shares') or 0)
    total_assets = float(data.get('total_assets') or 0)
    available_cash = float(data.get('available_cash') or 0)

    def _run_analysis():
        _update_analysis(task_id, 0.1, 'running', '正在采集行情、财报与新闻...')
        try:
            result = agent.analyze(
                symbol,
                cost_price=cost_price,
                master=master,
                shares=shares,
                total_assets=total_assets,
                available_cash=available_cash,
                cancel_event=task['_cancel_event'],
            )
            if task['_cancel_event'].is_set() or result.get('status') == 'cancelled':
                _update_analysis(task_id, task.get('progress', 0), 'cancelled', '分析已停止')
            elif result.get('status') == 'error':
                _update_analysis(task_id, 1.0, 'failed', result.get('error') or '分析失败', result=result)
            else:
                _update_analysis(task_id, 1.0, 'completed', '分析完成', result=result)
        except Exception as exc:
            logger.exception(f"[{symbol}] 可停止分析任务失败: {exc}")
            _update_analysis(task_id, 1.0, 'failed', f'分析异常: {exc}')

    threading.Thread(target=_run_analysis, daemon=True).start()
    return jsonify({'task_id': task_id, 'symbol': symbol, 'status': 'queued'})


@app.route('/api/analyze/<task_id>', methods=['GET'])
def analyze_task_status(task_id: str):
    with _analysis_lock:
        task = analysis_tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify({
        'task_id': task['task_id'],
        'symbol': task['symbol'],
        'status': task['status'],
        'progress': task['progress'],
        'message': task['message'],
        'result': task.get('result'),
        'created_at': task['created_at'],
        'completed_at': task['completed_at'],
    })


@app.route('/api/analyze/<task_id>/cancel', methods=['POST'])
def analyze_task_cancel(task_id: str):
    with _analysis_lock:
        task = analysis_tasks.get(task_id)
        if not task:
            return jsonify({'error': '任务不存在'}), 404
        if task['status'] in ('completed', 'failed', 'cancelled'):
            return jsonify({'error': '任务已结束', 'status': task['status']}), 409
        task['_cancel_event'].set()
        task['status'] = 'cancelling'
        task['message'] = '正在停止，当前请求结束后不会继续执行。'
    return jsonify({'task_id': task_id, 'status': 'cancelling'})


# ==========================================
#  API: 推演 (Phase 3+4 - Bridge + Report)
# ==========================================

@app.route('/api/predict', methods=['POST'])
def predict():
    """启动股价推演"""
    data = request.get_json(silent=True) or {}
    raw_symbol = data.get('symbol', '').strip()
    scenario = data.get('scenario', 'base')
    cost_price = data.get('cost_price', 0)
    master = data.get('master', '').strip().lower()
    shares = data.get('shares', 0) or 0
    total_assets = data.get('total_assets', 0) or 0.0
    available_cash = data.get('available_cash', 0) or 0.0

    if not raw_symbol:
        return jsonify({'error': '请提供股票代码'}), 400

    # 支持股票名称输入解析
    from stock_name_resolver import resolve_symbol as _resolve_symbol
    symbol = _resolve_symbol(raw_symbol) or raw_symbol.upper()

    task_id = f"pred_{uuid.uuid4().hex[:12]}"

    pred_data = {
        'task_id': task_id,
        'symbol': symbol,
        'scenario': scenario,
        'cost_price': float(cost_price) if cost_price else 0.0,
        'master': master,
        'shares': int(shares),
        'total_assets': float(total_assets),
        'available_cash': float(available_cash),
        'status': 'pending',
        'progress': 0.0,
        'message': '',
        'analysis': None,
        'simulation': None,
        'report': None,
        'report_html_path': None,
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
        '_cancel_event': threading.Event(),
    }

    with _predictions_lock:
        predictions[task_id] = pred_data

    logger.info(f"[{symbol}] 启动推演 task_id={task_id}, scenario={scenario}")

    # 后台执行完整流水线
    def _run_pipeline():
        try:
            cancel_event = pred_data['_cancel_event']

            def _cancel_if_requested() -> bool:
                if not cancel_event.is_set():
                    return False
                _update_prediction(task_id, pred_data.get('progress', 0), 'cancelled', '用户已停止智能推演')
                return True

            # Step 1: 分析
            _update_prediction(task_id, 0.1, 'analyzing', '正在进行多因子分析...')
            result = agent.analyze(symbol, cost_price=pred_data.get('cost_price', 0),
                                   master=pred_data.get('master', ''),
                                   shares=pred_data.get('shares', 0),
                                   total_assets=pred_data.get('total_assets', 0),
                                   available_cash=pred_data.get('available_cash', 0),
                                   cancel_event=cancel_event)
            if _cancel_if_requested() or result.get('status') == 'cancelled':
                return
            if result.get('status') == 'error':
                _update_prediction(task_id, 1.0, 'failed', f"分析失败: {result.get('error')}")
                return
            _update_prediction(task_id, 0.4, 'analyzing', '分析完成', analysis=result)

            # Step 2: 模拟推演
            def _sim_progress(p, msg):
                progress = 0.4 + p * 0.4
                _update_prediction(task_id, progress, 'simulating', msg)

            _update_prediction(task_id, 0.4, 'simulating', '启动模拟推演引擎...')
            sim_result = orchestrator.orchestrate(result, scenario=scenario, progress_callback=_sim_progress)

            if _cancel_if_requested():
                return

            # 检查推演是否失败（不再降级，失败即报错）
            if sim_result.get('status') == 'failed':
                err_msg = sim_result.get('error', 'MiroFish 推演失败')
                raise RuntimeError(f"MiroFish 推演失败: {err_msg}")

            _update_prediction(task_id, 0.8, 'simulating', '模拟推演完成', simulation=sim_result)

            # Step 3: 生成报告
            _update_prediction(task_id, 0.9, 'generating_report', '生成预测报告...')
            report = report_gen.generate(result, sim_result)
            html_path = report_gen.save(report)

            if _cancel_if_requested():
                return

            _update_prediction(task_id, 1.0, 'completed', '推演完成', report=report, report_html_path=html_path)

        except Exception as e:
            import traceback
            logger.error(f"[{symbol}] 推演失败: {e}\n{traceback.format_exc()}")
            _update_prediction(task_id, 1.0, 'failed', f"推演异常: {str(e)}")

    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()

    return jsonify({
        'task_id': task_id,
        'symbol': symbol,
        'scenario': scenario,
        'status': 'queued',
    })


@app.route('/api/predict/<task_id>', methods=['GET'])
def predict_status(task_id: str):
    with _predictions_lock:
        pred = predictions.get(task_id)
    if not pred:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify({
        'task_id': pred['task_id'],
        'symbol': pred['symbol'],
        'scenario': pred['scenario'],
        'status': pred['status'],
        'progress': pred['progress'],
        'message': pred['message'],
        'created_at': pred['created_at'],
        'completed_at': pred['completed_at'],
    })


@app.route('/api/predict/<task_id>/stream', methods=['GET'])
def predict_stream(task_id: str):
    """SSE 推演进度流"""
    def generate():
        last_progress = -1
        last_yield_time = time.time()
        while True:
            with _predictions_lock:
                pred = predictions.get(task_id)
            if not pred:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            data = {
                'status': pred['status'],
                'progress': pred['progress'],
                'message': pred['message'],
            }
            current_progress = pred['progress']

            if pred['status'] in ('completed', 'failed', 'cancelled'):
                data['report'] = pred.get('report')
                data['report_html_path'] = pred.get('report_html_path')
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            if current_progress != last_progress:
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                last_progress = current_progress
                last_yield_time = time.time()
            elif time.time() - last_yield_time > 15:
                yield ": heartbeat\n\n"
                last_yield_time = time.time()

            time.sleep(1)

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/predict/<task_id>/cancel', methods=['POST'])
def predict_cancel(task_id: str):
    """请求停止智能推演；当前远程调用结束后不再执行下一阶段。"""
    with _predictions_lock:
        pred = predictions.get(task_id)
        if not pred:
            return jsonify({'error': '任务不存在'}), 404
        if pred['status'] in ('completed', 'failed', 'cancelled'):
            return jsonify({'error': '任务已结束', 'status': pred['status']}), 409
        pred['_cancel_event'].set()
        pred['status'] = 'cancelling'
        pred['message'] = '正在停止，当前步骤结束后不会继续执行。'
    return jsonify({'task_id': task_id, 'status': 'cancelling'})


@app.route('/api/predict/<task_id>/report', methods=['GET'])
def predict_report(task_id: str):
    with _predictions_lock:
        pred = predictions.get(task_id)
    if not pred:
        return jsonify({'error': '任务不存在'}), 404
    if pred['status'] != 'completed':
        return jsonify({'error': '任务尚未完成', 'status': pred['status'], 'progress': pred['progress']}), 200

    # 返回 HTML 报告
    html_path = pred.get('report_html_path')
    if html_path and os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        return Response(html, mimetype='text/html')
    return jsonify(pred.get('report', {}))


_PREDICTION_HISTORY_ID_RE = re.compile(r"^[A-Za-z0-9_-]+_prediction_\d{8}_\d{6}$")


def _prediction_history_root() -> Path:
    """Return the single local directory containing completed StockFish reports."""
    root = Path(report_gen.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _prediction_history_paths(history_id: str) -> tuple[Path, Path]:
    """Resolve one report pair without accepting arbitrary local paths."""
    if not _PREDICTION_HISTORY_ID_RE.fullmatch(history_id or ""):
        raise ValueError("无效的推演历史标识")

    root = _prediction_history_root()
    json_path = (root / f"{history_id}.json").resolve()
    html_path = (root / f"{history_id}.html").resolve()
    if json_path.parent != root or html_path.parent != root:
        raise ValueError("无效的推演历史路径")
    return json_path, html_path


def _prediction_history_entry(json_path: Path) -> dict:
    """Build a compact, UI-safe history row from a completed report snapshot."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("报告格式无效")

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    simulation = payload.get("simulation") if isinstance(payload.get("simulation"), dict) else {}
    mirofish_report = simulation.get("mirofish_report") if isinstance(simulation.get("mirofish_report"), dict) else {}
    generated_at = str(payload.get("generated_at") or datetime.fromtimestamp(json_path.stat().st_mtime).isoformat())
    return {
        "history_id": json_path.stem,
        "symbol": str(payload.get("symbol") or ""),
        "name": str(payload.get("name") or ""),
        "generated_at": generated_at,
        "summary": {
            "outlook": summary.get("outlook"),
            "overall_signal": summary.get("overall_signal"),
            "signal_score": summary.get("signal_score"),
            "confidence": summary.get("confidence"),
            "current_price": summary.get("current_price"),
            "price_target_low": summary.get("price_target_low"),
            "price_target_high": summary.get("price_target_high"),
        },
        "simulation": {
            "status": simulation.get("status"),
            "scenario": simulation.get("scenario"),
            "agent_count": mirofish_report.get("agent_count"),
            "simulation_rounds": mirofish_report.get("simulation_rounds"),
        },
        "has_html": json_path.with_suffix(".html").is_file(),
    }


@app.route('/api/prediction-history', methods=['GET'])
def list_prediction_history():
    """List compact completed-report metadata without exposing MiroFish runtime files."""
    limit = min(max(request.args.get("limit", 100, type=int), 1), 200)
    entries = []
    for json_path in _prediction_history_root().glob("*_prediction_*.json"):
        if not _PREDICTION_HISTORY_ID_RE.fullmatch(json_path.stem):
            continue
        try:
            entries.append(_prediction_history_entry(json_path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(f"跳过无法读取的推演历史 {json_path.name}: {exc}")

    entries.sort(key=lambda item: item["generated_at"], reverse=True)
    return jsonify({"items": entries[:limit], "count": len(entries)})


@app.route('/api/prediction-history/<history_id>/report', methods=['GET'])
def open_prediction_history_report(history_id: str):
    """Open a persisted StockFish HTML report after the in-memory task has expired."""
    try:
        json_path, html_path = _prediction_history_paths(history_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not json_path.is_file() or not html_path.is_file():
        return jsonify({"error": "历史报告不存在或已被删除"}), 404
    return send_file(html_path, mimetype="text/html", as_attachment=False)


@app.route('/api/prediction-history/<history_id>', methods=['DELETE'])
def delete_prediction_history(history_id: str):
    """Delete only the StockFish report pair, never the underlying MiroFish runtime data."""
    try:
        json_path, html_path = _prediction_history_paths(history_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not json_path.is_file():
        return jsonify({"error": "历史记录不存在或已被删除"}), 404

    try:
        json_path.unlink()
        if html_path.is_file():
            html_path.unlink()
    except OSError as exc:
        logger.error(f"删除推演历史失败 {history_id}: {exc}")
        return jsonify({"error": "删除历史报告失败"}), 500
    return jsonify({"success": True, "history_id": history_id})


# ==========================================
#  API: 批量分析 (Batch Analysis)
# ==========================================

@app.route('/api/batch/analyze', methods=['POST'])
def batch_analyze():
    """启动批量股票分析"""
    data = request.get_json(silent=True) or {}
    symbols_raw = data.get('symbols', '').strip()
    cost_prices_raw = data.get('cost_prices', '').strip()
    shares_raw = data.get('shares', '').strip()
    master = data.get('master', '').strip().lower()
    total_assets = data.get('total_assets', 0) or 0.0
    available_cash = data.get('available_cash', 0) or 0.0

    if not symbols_raw:
        return jsonify({'error': '请提供至少一只股票代码（多只以 / 分隔）'}), 400

    # 解析 / 分隔的输入（支持股票名称自动解析为代码）
    from stock_name_resolver import resolve_symbol as _resolve_symbol
    raw_symbols = [s.strip() for s in symbols_raw.split('/') if s.strip()]
    symbols = [(_resolve_symbol(s) or s.upper()) for s in raw_symbols]
    cost_prices = [float(c.strip()) if c.strip() else 0.0 for c in cost_prices_raw.split('/')] if cost_prices_raw else []
    shares_list = [int(s.strip()) if s.strip() else 0 for s in shares_raw.split('/')] if shares_raw else []

    # 校验
    if cost_prices and len(cost_prices) != len(symbols):
        return jsonify({'error': f'成本价数量({len(cost_prices)})与股票数量({len(symbols)})不一致'}), 400
    if shares_list and len(shares_list) != len(symbols):
        return jsonify({'error': f'数量({len(shares_list)})与股票数量({len(symbols)})不一致'}), 400

    # 补齐缺失
    while len(cost_prices) < len(symbols):
        cost_prices.append(0.0)
    while len(shares_list) < len(symbols):
        shares_list.append(0)

    task_id = f"batch_{uuid.uuid4().hex[:12]}"

    task_data = {
        'task_id': task_id,
        'symbols': symbols,
        'cost_prices': cost_prices,
        'shares_list': shares_list,
        'total_assets': float(total_assets),
        'available_cash': float(available_cash),
        'master': master,
        'status': 'pending',
        'message': '',
        'progress': 0.0,
        'results': [],
        'summary': None,
        'quality_pick': None,
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
        '_cancel_event': threading.Event(),
    }

    with _batch_lock:
        batch_tasks[task_id] = task_data

    logger.info(f"批量分析启动 task_id={task_id}, symbols={symbols}, master={master or 'off'}")

    def _run_batch():
        try:
            def _progress(event_type, event_data):
                if event_type == 'progress':
                    current = event_data.get('current', 0)
                    total = event_data.get('total', 1)
                    _update_batch(task_id, current / total,
                                 'running', event_data.get('message', ''),
                                 current_stock=event_data.get('symbol', ''))
                elif event_type == 'stock_result':
                    _update_batch(task_id, None, 'running',
                                 event_data.get('message', ''),
                                 add_result={
                                     'symbol': event_data.get('symbol', ''),
                                     'data': event_data.get('data', {}),
                                 })
                elif event_type == 'batch_summary':
                    _update_batch(task_id, 0.9, 'summarizing', '批量总结完成',
                                 summary=event_data.get('summary'),
                                 quality_pick=event_data.get('quality_pick'))
                elif event_type == 'completed':
                    _update_batch(task_id, 1.0, 'completed', event_data.get('message', ''))

            result = batch_analyzer.run_batch(
                symbols=symbols,
                cost_prices=cost_prices,
                shares_list=shares_list,
                total_assets=float(total_assets),
                available_cash=float(available_cash),
                master=master,
                progress_callback=_progress,
                cancel_event=task_data['_cancel_event'],
            )

            # 如果 completed 事件没发出来（降级路径）
            with _batch_lock:
                bt = batch_tasks.get(task_id)
                if task_data['_cancel_event'].is_set() or result.get('status') == 'cancelled':
                    if bt:
                        bt['status'] = 'cancelled'
                        bt['message'] = '批量分析已停止'
                        bt['completed_at'] = datetime.now().isoformat()
                elif bt and bt['status'] not in ('completed', 'failed', 'cancelled'):
                    bt['status'] = 'completed'
                    bt['progress'] = 1.0
                    bt['message'] = '批量分析完成'
                    bt['summary'] = result.get('summary')
                    bt['quality_pick'] = result.get('quality_pick')
                    bt['completed_at'] = datetime.now().isoformat()

        except Exception as e:
            import traceback
            logger.error(f"批量分析失败: {e}\n{traceback.format_exc()}")
            _update_batch(task_id, 1.0, 'failed', f"批量分析异常: {str(e)}")

    thread = threading.Thread(target=_run_batch, daemon=True)
    thread.start()

    return jsonify({
        'task_id': task_id,
        'symbols': symbols,
        'status': 'queued',
    })


@app.route('/api/batch/analyze/<task_id>/cancel', methods=['POST'])
def batch_cancel(task_id: str):
    """停止当前批量任务；当前正在分析的单只股票结束后不再继续。"""
    with _batch_lock:
        task = batch_tasks.get(task_id)
        if not task:
            return jsonify({'error': '任务不存在'}), 404
        if task['status'] in ('completed', 'failed', 'cancelled'):
            return jsonify({'error': '任务已结束', 'status': task['status']}), 409
        task['_cancel_event'].set()
        task['status'] = 'cancelling'
        task['message'] = '正在停止，当前股票处理完成后不会继续。'
    return jsonify({'task_id': task_id, 'status': 'cancelling'})


@app.route('/api/batch/analyze/<task_id>', methods=['GET'])
def batch_status(task_id: str):
    """查询批量分析任务状态"""
    with _batch_lock:
        bt = batch_tasks.get(task_id)
    if not bt:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify({
        'task_id': bt['task_id'],
        'symbols': bt['symbols'],
        'status': bt['status'],
        'progress': bt['progress'],
        'message': bt['message'],
        'results': bt.get('results', []),
        'results_count': len(bt.get('results', [])),
        'total': len(bt.get('symbols', [])),
        'success_count': sum(1 for r in bt.get('results', []) if r.get('status') == 'complete'),
        'summary': bt.get('summary'),
        'quality_pick': bt.get('quality_pick'),
        'created_at': bt['created_at'],
        'completed_at': bt['completed_at'],
    })


@app.route('/api/batch/analyze/<task_id>/stream', methods=['GET'])
def batch_stream(task_id: str):
    """SSE 批量分析进度流"""
    def generate():
        last_progress = -1
        last_result_count = 0
        last_yield_time = time.time()
        while True:
            with _batch_lock:
                bt = batch_tasks.get(task_id)
            if not bt:
                yield f"data: {json.dumps({'type': 'error', 'message': '任务不存在'})}\n\n"
                break

            current_progress = bt.get('progress', 0)
            results = bt.get('results', [])
            current_result_count = len(results)

            yielded = False

            # 推送新完成的 stock_result
            if current_result_count > last_result_count:
                for r in results[last_result_count:]:
                    yield f"data: {json.dumps({'type': 'stock_result', 'symbol': r['symbol'], 'data': r['data']}, ensure_ascii=False)}\n\n"
                last_result_count = current_result_count
                yielded = True

            # 推送 progress
            if current_progress != last_progress:
                msg = bt.get('message', '')
                current_stock = bt.get('current_stock', '')
                yield f"data: {json.dumps({'type': 'progress', 'progress': current_progress, 'message': msg, 'current_stock': current_stock}, ensure_ascii=False)}\n\n"
                last_progress = current_progress
                yielded = True

            # 终端状态推送 summary + quality_pick
            if bt['status'] in ('completed', 'failed', 'cancelled'):
                if bt['status'] == 'completed':
                    summary = bt.get('summary')
                    quality_pick = bt.get('quality_pick')
                    if summary or quality_pick:
                        yield f"data: {json.dumps({'type': 'batch_summary', 'summary': summary, 'quality_pick': quality_pick}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': bt['status'], 'message': bt.get('message', '')})}\n\n"
                break

            if yielded:
                last_yield_time = time.time()
            elif time.time() - last_yield_time > 15:
                yield ": heartbeat\n\n"
                last_yield_time = time.time()

            time.sleep(1)

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


def _update_batch(task_id, progress, status, message, **kwargs):
    """更新批量分析任务状态（线程安全）"""
    with _batch_lock:
        bt = batch_tasks.get(task_id)
        if not bt:
            return
        if bt.get('status') == 'cancelling' and status not in ('cancelled', 'failed'):
            return
        if progress is not None:
            bt['progress'] = progress
        bt['status'] = status
        bt['message'] = message
        # 追加结果
        add_result = kwargs.pop('add_result', None)
        if add_result:
            bt.setdefault('results', []).append(add_result)
        for k, v in kwargs.items():
            bt[k] = v
        if status in ('completed', 'failed', 'cancelled'):
            bt['completed_at'] = datetime.now().isoformat()


def _update_analysis(task_id, progress, status, message, **kwargs):
    """更新单股分析任务状态（线程安全）。"""
    with _analysis_lock:
        task = analysis_tasks.get(task_id)
        if not task:
            return
        if task.get('status') == 'cancelling' and status not in ('cancelled', 'failed'):
            return
        if progress is not None:
            task['progress'] = progress
        task['status'] = status
        task['message'] = message
        for key, value in kwargs.items():
            task[key] = value
        if status in ('completed', 'failed', 'cancelled'):
            task['completed_at'] = datetime.now().isoformat()


def _update_qlib(task_id, progress, status, message, **kwargs):
    """更新 qlib 推理任务状态（线程安全）"""
    with _qlib_lock:
        qt = qlib_tasks.get(task_id)
        if not qt:
            return
        if progress is not None:
            qt['progress'] = progress
        qt['status'] = status
        qt['message'] = message
        for k, v in kwargs.items():
            qt[k] = v
        if status in ('completed', 'failed'):
            qt['completed_at'] = datetime.now().isoformat()


# ==========================================
#  API: Qlib 推理
# ==========================================

@app.route('/api/qlib/models', methods=['GET'])
def qlib_models():
    """列出所有可用模型（扫描 DATA/analysis_outputs/）"""
    scan_dir = _qlib_models_path()
    models = []

    if scan_dir.exists():
        for d in sorted(scan_dir.iterdir()):
            if not d.is_dir():
                continue
            name = d.name

            market = "unknown"
            if "csi300" in name.lower():
                market = "csi300"
            elif "csi500" in name.lower():
                market = "csi500"
            elif "csi1000" in name.lower():
                market = "csi1000"

            date_part = name[:10] if len(name) >= 10 and name[4] == "-" else ""
            has_scores = (d / "model_predict" / "scores.csv").is_file()
            infer_ready = _qlib_model_infer_ready(d)
            is_finetune = "fintune" in name.lower()

            models.append({
                "name": name,
                "market": market,
                "date": date_part,
                "has_scores": has_scores,
                "infer_ready": infer_ready,
                "is_finetune": is_finetune,
                "in_analysis_outputs": True,  # 始终为 True
            })

    return jsonify(models)


@app.route('/api/qlib/models/<model_name>', methods=['DELETE'])
def qlib_model_delete(model_name):
    """删除一个未被运行中任务使用的本地 Qlib 模型目录。"""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', model_name):
        return jsonify({'error': '模型名称不合法'}), 400

    models_dir = _qlib_models_path().resolve()
    model_dir = (models_dir / model_name).resolve()
    try:
        relative_path = model_dir.relative_to(models_dir)
    except ValueError:
        return jsonify({'error': '模型路径不合法'}), 400
    if len(relative_path.parts) != 1 or relative_path.name != model_name:
        return jsonify({'error': '模型路径不合法'}), 400
    if not model_dir.is_dir():
        return jsonify({'error': f'模型不存在: {model_name}'}), 404

    active_statuses = {'pending', 'running', 'cancelling'}
    with _qlib_lock:
        inference_active = any(
            task.get('model') == model_name and task.get('status') in active_statuses
            for task in qlib_tasks.values()
        )
    if inference_active:
        return jsonify({'error': '该模型正在推理，任务结束后才能删除'}), 409

    with _qlib_train_lock:
        training_active = any(
            task.get('status') in active_statuses
            for task in qlib_train_tasks.values()
        )
    if training_active:
        return jsonify({'error': '当前有模型正在训练，训练结束后才能删除模型'}), 409

    with _qlib_finetune_lock:
        finetune_active = any(
            task.get('status') in active_statuses
            and model_name in {task.get('base_model'), task.get('model_name')}
            for task in qlib_finetune_tasks.values()
        )
    if finetune_active:
        return jsonify({'error': '该模型正在参与微调，任务结束后才能删除'}), 409

    try:
        shutil.rmtree(model_dir)
    except OSError as exc:
        logger.error(f"删除 Qlib 模型失败 model={model_name}: {exc}")
        return jsonify({'error': f'删除失败: {exc}'}), 500

    logger.info(f"Qlib 模型已删除: {model_name}")
    return jsonify({'success': True, 'model': model_name})


@app.route('/api/qlib/models/<model_name>/backtest', methods=['GET'])
def qlib_model_backtest(model_name):
    """Read persisted Qlib walk-forward artifacts without recalculating a model."""
    try:
        return jsonify(load_backtest(_qlib_models_path(), model_name))
    except FileNotFoundError:
        return jsonify({'error': f'模型不存在: {model_name}'}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except OSError as exc:
        logger.error(f"读取 Qlib 回测失败 model={model_name}: {exc}")
        return jsonify({'error': f'读取回测结果失败: {exc}'}), 500


@app.route('/api/qlib/train-targets', methods=['GET'])
def qlib_train_targets():
    """返回可用的训练目标配置（用于训练面板的模型选择器）"""
    targets = [
        {
            "value": "csi300-alpha158",
            "label": "沪深300 Alpha158",
            "market": "csi300",
            "benchmark": "SH000300",
            "description": "沪深300成分股 + Alpha158因子 + LightGBM walk-forward全量训练（约20-40分钟）"
        },
        {
            "value": "csi500-alpha158",
            "label": "中证500 Alpha158",
            "market": "csi500",
            "benchmark": "SH000905",
            "description": "中证500成分股 + Alpha158因子 + LightGBM walk-forward全量训练（约30-60分钟）"
        },
    ]
    return jsonify(targets)


@app.route('/api/qlib/infer', methods=['POST'])
def qlib_infer():
    """启动 qlib 推理任务"""
    data = request.get_json(silent=True) or {}
    model = data.get('model', '').strip()
    holdings = data.get('holdings', '').strip()

    if not model:
        return jsonify({'error': '请选择模型'}), 400
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', model):
        return jsonify({'error': '模型名称不合法'}), 400

    models_dir = _qlib_models_path().resolve()
    model_dir = (models_dir / model).resolve()
    try:
        relative_path = model_dir.relative_to(models_dir)
    except ValueError:
        return jsonify({'error': '模型路径不合法'}), 400
    if len(relative_path.parts) != 1 or relative_path.name != model:
        return jsonify({'error': '模型路径不合法'}), 400
    if not model_dir.is_dir():
        return jsonify({'error': f'模型不存在: {model}'}), 400
    if not _qlib_model_infer_ready(model_dir):
        return jsonify({'error': '模型训练尚未完成，当前只能删除，不能推理'}), 409

    task_id = f"qlib_{uuid.uuid4().hex[:12]}"

    task_data = {
        'task_id': task_id,
        'model': model,
        'status': 'pending',
        'message': '',
        'progress': 0.0,
        'stocks': '',
        'count': 0,
        'scores': [],
        'pred_date': '',
        'error': '',
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
    }

    with _qlib_lock:
        qlib_tasks[task_id] = task_data

    logger.info(f"Qlib 推理启动 task_id={task_id}, model={model}")

    def _run():
        try:
            def _progress(event_data):
                status = event_data.get('status', 'running')
                message = event_data.get('message', '')
                if status == 'completed':
                    _update_qlib(task_id, 1.0, 'completed', message,
                                 stocks=event_data.get('stocks', ''),
                                 count=event_data.get('count', 0),
                                 scores=event_data.get('scores', []),
                                 pred_date=event_data.get('pred_date', ''),
                                 strategy_b=event_data.get('strategy_b', {}))
                else:
                    progress = 0.5 if '推理' in message else 0.1
                    _update_qlib(task_id, progress, 'running', message)

            result = run_qlib_inference(model, top_n=20, progress_callback=_progress, holdings=holdings)

            # 确保完成状态
            with _qlib_lock:
                qt = qlib_tasks.get(task_id)
                if qt and qt['status'] not in ('completed', 'failed'):
                    qt['status'] = 'completed'
                    qt['progress'] = 1.0
                    qt['stocks'] = result.get('stocks', '')
                    qt['count'] = result.get('count', 0)
                    qt['scores'] = result.get('scores', [])
                    qt['pred_date'] = result.get('pred_date', '')
                    qt['message'] = f"完成 — 已选出 {result.get('count', 0)} 只股票"

        except Exception as e:
            logger.error(f"Qlib 推理失败 task_id={task_id}: {e}")
            _update_qlib(task_id, 0.0, 'failed', str(e), error=str(e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({'task_id': task_id, 'status': 'pending'})


@app.route('/api/qlib/infer/<task_id>/stream', methods=['GET'])
def qlib_infer_stream(task_id):
    """SSE 流 — qlib 推理进度"""
    def generate():
        last_progress = -1
        last_message = ''
        last_yield_time = time.time()
        while True:
            with _qlib_lock:
                qt = qlib_tasks.get(task_id)
            if not qt:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            data = {
                'status': qt.get('status'),
                'progress': qt.get('progress', 0),
                'message': qt.get('message', ''),
            }
            current_progress = qt.get('progress', 0)

            if qt.get('status') == 'completed':
                data['stocks'] = qt.get('stocks', '')
                data['count'] = qt.get('count', 0)
                data['pred_date'] = qt.get('pred_date', '')
                data['strategy_b'] = qt.get('strategy_b', {})
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            if qt.get('status') == 'failed':
                data['error'] = qt.get('error', '未知错误')
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            current_message = qt.get('message', '')
            if current_progress != last_progress or current_message != last_message:
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                last_progress = current_progress
                last_message = current_message
                last_yield_time = time.time()
            elif time.time() - last_yield_time > 15:
                yield ": heartbeat\n\n"
                last_yield_time = time.time()

            time.sleep(1)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


# ---- 指数成分股缓存 ----
_index_stocks_cache = {}
_index_stocks_cache_time = {}


@app.route('/api/qlib/index-stocks', methods=['GET'])
def qlib_index_stocks():
    """返回项目内 Qlib 数据中的指数成分股列表。"""
    index_name = request.args.get('index', 'csi300').strip().lower()
    exclude_star = request.args.get('exclude_star', 'false').strip().lower() == 'true'

    if index_name not in ('csi300', 'csi500', 'csi1000'):
        return jsonify({'error': f'不支持的指数: {index_name}，支持 csi300/csi500/csi1000'}), 400

    # 缓存 1 小时
    cache_key = f"{index_name}_{exclude_star}"
    now = time.time()
    if cache_key in _index_stocks_cache and (now - _index_stocks_cache_time.get(cache_key, 0)) < 3600:
        return jsonify(_index_stocks_cache[cache_key])

    # 读取 qlib 数据中的成分股文件
    inst_file = _qlib_data_path() / "instruments" / f"{index_name}.txt"
    if not inst_file.exists():
        return jsonify({'error': f'成分股文件不存在: {inst_file}'}), 404

    # 解析：instrument start_date end_date
    date_groups = {}
    for line in inst_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        inst, start, end = parts[0], parts[1], parts[2]
        date_groups.setdefault(end, []).append(inst)

    # 取最大 end_date 作为当前成分股
    if not date_groups:
        return jsonify({'stocks': '', 'count': 0})

    max_date = max(date_groups.keys())
    stocks = sorted(set(date_groups[max_date]))

    # 转换 instrument 为纯代码: SZ000001 → 000001
    codes = []
    for inst in stocks:
        code = inst[2:] if inst.startswith(('SZ', 'SH', 'BJ')) else inst
        # 剔除科创板 (688xxx)
        if exclude_star and code.startswith('688'):
            continue
        codes.append(code)

    result = {
        'stocks': '/'.join(codes),
        'count': len(codes),
        'index': index_name,
        'date': max_date,
        'exclude_star': exclude_star,
    }

    _index_stocks_cache[cache_key] = result
    _index_stocks_cache_time[cache_key] = now

    return jsonify(result)


# ==========================================
#  API: Qlib 数据更新
# ==========================================

@app.route('/api/qlib/data/update', methods=['POST'])
def qlib_data_update():
    """启动 qlib 数据下载更新任务"""
    with _qlib_data_lock:
        active_task = next(
            (task for task in qlib_data_tasks.values()
             if task.get('status') in {'pending', 'running'}),
            None,
        )
        if active_task:
            return jsonify({
                'error': '已有 Qlib 数据更新任务正在运行，请等待完成',
                'task_id': active_task.get('task_id'),
            }), 409

    task_id = f"qdata_{uuid.uuid4().hex[:12]}"
    task_data = {
        'task_id': task_id,
        'status': 'pending',
        'message': '',
        'progress': 0.0,
        'error': '',
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
    }
    with _qlib_data_lock:
        qlib_data_tasks[task_id] = task_data

    logger.info(f"Qlib 数据更新启动 task_id={task_id}")

    def _run():
        try:
            def _progress(event_data):
                with _qlib_data_lock:
                    qt = qlib_data_tasks.get(task_id)
                    if not qt:
                        return
                    status = event_data.get('status', 'running')
                    msg = event_data.get('message', '')
                    progress = event_data.get('progress')
                    if progress is not None:
                        qt['progress'] = progress
                    qt['status'] = status
                    qt['message'] = msg
                    if status in ('completed', 'failed'):
                        qt['completed_at'] = datetime.now().isoformat()

            if run_qlib_data_update is None:
                with _qlib_data_lock:
                    qt = qlib_data_tasks.get(task_id)
                    if qt:
                        qt['status'] = 'failed'
                        qt['message'] = 'Qlib 数据更新模块加载失败，请检查服务日志'
                return
            result = run_qlib_data_update(progress_callback=_progress)

            with _qlib_data_lock:
                qt = qlib_data_tasks.get(task_id)
                if qt and qt['status'] not in ('completed', 'failed'):
                    qt['status'] = 'completed'
                    qt['progress'] = 1.0
                    qt['message'] = result.get('message', '数据更新完成')

        except Exception as e:
            logger.error(f"Qlib 数据更新失败 task_id={task_id}: {e}")
            with _qlib_data_lock:
                qt = qlib_data_tasks.get(task_id)
                if qt:
                    qt['status'] = 'failed'
                    qt['error'] = str(e)
                    qt['message'] = f'数据更新失败: {e}'

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({'task_id': task_id, 'status': 'pending'})


@app.route('/api/qlib/data/update/<task_id>/stream', methods=['GET'])
def qlib_data_update_stream(task_id):
    """SSE 流 — qlib 数据更新进度"""
    def generate():
        last_progress = -1
        last_message = ''
        last_yield_time = time.time()
        while True:
            with _qlib_data_lock:
                qt = qlib_data_tasks.get(task_id)
            if not qt:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            data = {
                'status': qt.get('status'),
                'progress': qt.get('progress', 0),
                'message': qt.get('message', ''),
            }
            current_progress = qt.get('progress', 0)
            current_message = qt.get('message', '')

            if qt.get('status') == 'completed':
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            if qt.get('status') == 'failed':
                data['error'] = qt.get('error', '未知错误')
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            if current_progress != last_progress or current_message != last_message:
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                last_progress = current_progress
                last_message = current_message
                last_yield_time = time.time()
            elif time.time() - last_yield_time > 15:
                yield ": heartbeat\n\n"
                last_yield_time = time.time()

            time.sleep(1)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


# ==========================================
#  API: Qlib 模型训练
# ==========================================

_QLIB_TRAIN_ACTIVE_STATUSES = {'pending', 'running', 'cancelling'}
_QLIB_TRAIN_TERMINAL_STATUSES = {'completed', 'failed', 'cancelled'}


def _append_qlib_train_log(task: dict, message: str) -> None:
    if not message:
        return
    created_at = datetime.fromisoformat(task['created_at'])
    elapsed = max(0, round((datetime.now() - created_at).total_seconds()))
    logs = task.setdefault('logs', [])
    log_index = task.get('_log_sequence', 0)
    task['_log_sequence'] = log_index + 1
    logs.append({'index': log_index, 'elapsed': elapsed, 'message': message})
    if len(logs) > 2000:
        task['logs'] = logs[-2000:]


def _public_qlib_train_task(task: dict) -> dict:
    return {
        'task_id': task.get('task_id'),
        'status': task.get('status'),
        'message': task.get('message', ''),
        'progress': task.get('progress', 0),
        'phase': task.get('phase', ''),
        'fold_current': task.get('fold_current', 0),
        'fold_total': task.get('fold_total', 0),
        'market': task.get('market', ''),
        'model_name': task.get('model_name', ''),
        'error': task.get('error', ''),
        'backtest_metrics': task.get('backtest_metrics', {}),
        'created_at': task.get('created_at'),
        'completed_at': task.get('completed_at'),
        'logs': list(task.get('logs', [])),
    }


@app.route('/api/qlib/train', methods=['POST'])
def qlib_train():
    """启动 qlib 模型训练任务"""
    data = request.get_json(silent=True) or {}
    market = data.get('market', 'csi300')
    target = data.get('target', '').strip()
    # 如果前端传了 target，从 target 解析 market（如 csi300-alpha158 → csi300）
    if target:
        target_lower = target.lower()
        if 'csi300' in target_lower:
            market = 'csi300'
        elif 'csi500' in target_lower:
            market = 'csi500'
        elif 'csi1000' in target_lower:
            market = 'csi1000'
    model_mode = data.get('model_mode', 'robust')
    try:
        hold_num = int(data.get('hold_num', 20))
        train_years = int(data.get('train_years', 5))
        valid_val = int(data.get('valid_val', 1))
        test_val = int(data.get('test_val', 2))
    except (TypeError, ValueError):
        return jsonify({'error': '训练窗口和持仓数量必须是整数'}), 400
    valid_unit = data.get('valid_unit', 'year')
    test_unit = data.get('test_unit', 'year')

    if market not in ('csi300', 'csi500', 'csi1000'):
        return jsonify({'error': f'不支持的市场: {market}'}), 400
    if model_mode not in ('default', 'robust'):
        return jsonify({'error': f'不支持的模型模式: {model_mode}'}), 400
    if not 1 <= hold_num <= 100:
        return jsonify({'error': '持仓数量必须在 1 到 100 之间'}), 400
    if train_years <= 0 or valid_val <= 0 or test_val <= 0:
        return jsonify({'error': '训练、验证和测试窗口必须大于 0'}), 400
    if valid_unit not in ('year', 'month', 'week') or test_unit not in ('year', 'month', 'week'):
        return jsonify({'error': '窗口单位只支持 year、month 或 week'}), 400

    with _qlib_train_lock:
        active_task = next(
            (task for task in qlib_train_tasks.values()
             if task.get('status') in _QLIB_TRAIN_ACTIVE_STATUSES),
            None,
        )
        if active_task:
            return jsonify({
                'error': '已有 Qlib 训练任务正在运行，请先停止或等待完成',
                'task_id': active_task.get('task_id'),
            }), 409

    task_id = f"qtrain_{uuid.uuid4().hex[:12]}"
    cancel_event = threading.Event()
    task_data = {
        'task_id': task_id,
        'status': 'pending',
        'message': '',
        'progress': 0.0,
        'phase': '准备训练环境',
        'fold_current': 0,
        'fold_total': 0,
        'market': market,
        'model_name': '',
        'error': '',
        'backtest_metrics': {},
        'logs': [],
        '_cancel_event': cancel_event,
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
    }
    with _qlib_train_lock:
        qlib_train_tasks[task_id] = task_data

    logger.info(f"Qlib 训练启动 task_id={task_id}, market={market}")

    def _run():
        try:
            def _progress(event_data):
                with _qlib_train_lock:
                    qt = qlib_train_tasks.get(task_id)
                    if not qt:
                        return
                    status = event_data.get('status', 'running')
                    msg = event_data.get('message', '')
                    progress = event_data.get('progress')
                    if progress is not None:
                        qt['progress'] = progress
                    if event_data.get('phase'):
                        qt['phase'] = event_data['phase']
                    if event_data.get('fold_current') is not None:
                        qt['fold_current'] = event_data['fold_current']
                    if event_data.get('fold_total') is not None:
                        qt['fold_total'] = event_data['fold_total']
                    if qt.get('status') != 'cancelling' or status in _QLIB_TRAIN_TERMINAL_STATUSES:
                        qt['status'] = status
                    qt['message'] = msg
                    _append_qlib_train_log(qt, msg)
                    if status in _QLIB_TRAIN_TERMINAL_STATUSES:
                        qt['completed_at'] = datetime.now().isoformat()

            if run_qlib_training is None:
                with _qlib_train_lock:
                    qt = qlib_train_tasks.get(task_id)
                    if qt:
                        qt['status'] = 'failed'
                        qt['message'] = 'Qlib 本地训练模块加载失败，请检查 stock_qlib 环境'
                        qt['phase'] = '训练模块加载失败'
                        qt['completed_at'] = datetime.now().isoformat()
                        _append_qlib_train_log(qt, qt['message'])
                return
            result = run_qlib_training(
                market=market,
                model_mode=model_mode,
                hold_num=hold_num,
                lightgbm_only=True,
                train_years=train_years,
                valid_val=valid_val,
                valid_unit=valid_unit,
                test_val=test_val,
                test_unit=test_unit,
                progress_callback=_progress,
                cancel_event=cancel_event,
            )

            if cancel_event.is_set():
                raise _QlibProcessCancelledError("Qlib 训练已取消")

            with _qlib_train_lock:
                qt = qlib_train_tasks.get(task_id)
                if qt and qt['status'] not in _QLIB_TRAIN_TERMINAL_STATUSES:
                    qt['status'] = 'completed'
                    qt['progress'] = 1.0
                    qt['phase'] = '训练与回测完成'
                    qt['model_name'] = result.get('model_name', '')
                    qt['message'] = result.get('message', '训练完成')
                    qt['backtest_metrics'] = result.get('backtest_metrics', {})
                    qt['completed_at'] = datetime.now().isoformat()
                    _append_qlib_train_log(qt, qt['message'])

        except _QlibProcessCancelledError:
            logger.info(f"Qlib 训练已取消 task_id={task_id}")
            with _qlib_train_lock:
                qt = qlib_train_tasks.get(task_id)
                if qt:
                    qt['status'] = 'cancelled'
                    qt['message'] = '训练已停止'
                    qt['phase'] = '训练已停止'
                    qt['completed_at'] = datetime.now().isoformat()
                    _append_qlib_train_log(qt, qt['message'])
        except Exception as e:
            logger.error(f"Qlib 训练失败 task_id={task_id}: {e}")
            with _qlib_train_lock:
                qt = qlib_train_tasks.get(task_id)
                if qt:
                    qt['status'] = 'failed'
                    qt['error'] = str(e)
                    qt['message'] = f'训练失败: {e}'
                    qt['phase'] = '训练失败'
                    qt['completed_at'] = datetime.now().isoformat()
                    _append_qlib_train_log(qt, qt['message'])

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({'task_id': task_id, 'status': 'pending'})


@app.route('/api/qlib/train/<task_id>/status', methods=['GET'])
def qlib_train_status(task_id):
    """返回训练任务状态和完整日志，用于页面切换或刷新后恢复。"""
    with _qlib_train_lock:
        task = qlib_train_tasks.get(task_id)
        if not task:
            return jsonify({'error': '训练任务不存在'}), 404
        payload = _public_qlib_train_task(task)
    return jsonify(payload)


@app.route('/api/qlib/train/<task_id>/cancel', methods=['POST'])
def qlib_train_cancel(task_id):
    """请求停止训练并由运行层终止完整子进程树。"""
    with _qlib_train_lock:
        task = qlib_train_tasks.get(task_id)
        if not task:
            return jsonify({'error': '训练任务不存在'}), 404
        status = task.get('status')
        if status in _QLIB_TRAIN_TERMINAL_STATUSES:
            return jsonify({'error': f'训练任务已经结束: {status}'}), 409
        task['status'] = 'cancelling'
        task['message'] = '正在停止训练...'
        task['phase'] = '正在停止训练'
        _append_qlib_train_log(task, task['message'])
        cancel_event = task.get('_cancel_event')
        if cancel_event is not None:
            cancel_event.set()
    return jsonify({'task_id': task_id, 'status': 'cancelling'})


@app.route('/api/qlib/train/<task_id>/stream', methods=['GET'])
def qlib_train_stream(task_id):
    """SSE 流 — qlib 训练进度"""
    def generate():
        last_progress = -1
        last_message = ''
        last_yield_time = time.time()
        while True:
            with _qlib_train_lock:
                qt = qlib_train_tasks.get(task_id)
            if not qt:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            data = {
                'status': qt.get('status'),
                'progress': qt.get('progress', 0),
                'message': qt.get('message', ''),
                'phase': qt.get('phase', ''),
                'fold_current': qt.get('fold_current', 0),
                'fold_total': qt.get('fold_total', 0),
                'log_index': qt.get('logs', [{}])[-1].get('index', -1) if qt.get('logs') else -1,
            }
            current_progress = qt.get('progress', 0)
            current_message = qt.get('message', '')

            if qt.get('status') == 'completed':
                data['model_name'] = qt.get('model_name', '')
                data['backtest_metrics'] = qt.get('backtest_metrics', {})
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            if qt.get('status') == 'failed':
                data['error'] = qt.get('error', '未知错误')
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            if qt.get('status') == 'cancelled':
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            if current_progress != last_progress or current_message != last_message:
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                last_progress = current_progress
                last_message = current_message
                last_yield_time = time.time()
            elif time.time() - last_yield_time > 15:
                yield ": heartbeat\n\n"
                last_yield_time = time.time()

            time.sleep(1)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


# ==========================================
#  API: Qlib 模型微调
# ==========================================

@app.route('/api/qlib/finetune', methods=['POST'])
def qlib_finetune():
    """启动 qlib 模型微调任务"""
    data = request.get_json(silent=True) or {}
    base_model = data.get('base_model', '').strip()

    if not base_model:
        return jsonify({'error': '请选择基础模型'}), 400

    # 验证基础模型存在
    base_model_dir = _qlib_models_path() / base_model
    if not base_model_dir.exists():
        base_model_dir = None

    if not base_model_dir:
        return jsonify({'error': f'基础模型不存在: {base_model}'}), 400

    model_name = f"{base_model}-fintune"
    task_id = f"qft_{uuid.uuid4().hex[:12]}"
    task_data = {
        'task_id': task_id,
        'status': 'pending',
        'message': '',
        'progress': 0.0,
        'base_model': base_model,
        'model_name': model_name,
        'error': '',
        'backtest_metrics': {},
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
    }
    with _qlib_finetune_lock:
        qlib_finetune_tasks[task_id] = task_data

    logger.info(f"Qlib 微调启动 task_id={task_id}, base_model={base_model}")

    def _run():
        try:
            cfg = _resolve_model_config(base_model)
            benchmark = cfg["benchmark"]

            def _log(msg: str, **extra):
                with _qlib_finetune_lock:
                    qt = qlib_finetune_tasks.get(task_id)
                    if not qt:
                        return
                    status = extra.get('status', 'running')
                    progress = extra.get('progress')
                    if progress is not None:
                        qt['progress'] = progress
                    qt['status'] = status
                    qt['message'] = msg

            _log(f"基础模型: {base_model}")
            _log(f"输出名称: {model_name}")
            _log("开始本地 Qlib 微调...")

            _validate_qlib_runtime(require_data=True)
            output_root = _qlib_models_path() / model_name
            runtime_env = _build_qlib_runtime_env({
                "TARGET_MARKET": cfg["market"],
                "TARGET_BENCHMARK": benchmark,
                "CASH_TOTAL": "100000",
                "OMP_NUM_THREADS": "8",
            })
            cmd = _qlib_python_command(
                _qlib_zh_dir / "scripts" / "finetune_alpha158.py",
                "--base-model-dir", base_model_dir,
                "--output-name", model_name,
                "--template", cfg["template"],
                "--output-root", output_root,
                "--experiment-name", model_name,
                "--train-years", "5",
                "--valid-years", "1",
                "--hold-num", "20",
                "--model-mode", "robust",
            )
            _run_qlib_streaming(
                cmd,
                lambda line: _log(f"[Local] {line[:300]}"),
                timeout=3600,
                cwd=_qlib_zh_dir,
                env=runtime_env,
            )

            # 提取回测指标
            summary_path = _qlib_models_path() / model_name / "finetune_summary.json"
            bt_metrics = {}
            if summary_path.exists():
                try:
                    import json as _json
                    summary = _json.loads(summary_path.read_text(encoding="utf-8"))
                    bt_metrics = summary.get("backtest", {})
                except Exception:
                    pass

            msg = f"微调完成: {model_name}"
            if bt_metrics.get("sharpe_ratio") is not None:
                msg += f" | 夏普比: {bt_metrics['sharpe_ratio']}"
            _log(msg, progress=1.0, status="completed")

            with _qlib_finetune_lock:
                qt = qlib_finetune_tasks.get(task_id)
                if qt and qt['status'] not in ('completed', 'failed'):
                    qt['status'] = 'completed'
                    qt['progress'] = 1.0
                    qt['model_name'] = model_name
                    qt['message'] = msg
                    qt['backtest_metrics'] = bt_metrics

        except Exception as e:
            logger.error(f"Qlib 微调失败 task_id={task_id}: {e}")
            with _qlib_finetune_lock:
                qt = qlib_finetune_tasks.get(task_id)
                if qt:
                    qt['status'] = 'failed'
                    qt['error'] = str(e)
                    qt['message'] = f'微调失败: {e}'

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({'task_id': task_id, 'status': 'pending', 'model_name': model_name})


@app.route('/api/qlib/finetune/<task_id>/stream', methods=['GET'])
def qlib_finetune_stream(task_id):
    """SSE 流 — qlib 微调进度"""
    def generate():
        last_progress = -1
        last_message = ''
        last_yield_time = time.time()
        while True:
            with _qlib_finetune_lock:
                qt = qlib_finetune_tasks.get(task_id)
            if not qt:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            data = {
                'status': qt.get('status'),
                'progress': qt.get('progress', 0),
                'message': qt.get('message', ''),
            }
            current_progress = qt.get('progress', 0)
            current_message = qt.get('message', '')

            if qt.get('status') == 'completed':
                data['model_name'] = qt.get('model_name', '')
                data['backtest_metrics'] = qt.get('backtest_metrics', {})
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            if qt.get('status') == 'failed':
                data['error'] = qt.get('error', '未知错误')
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            if current_progress != last_progress or current_message != last_message:
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                last_progress = current_progress
                last_message = current_message
                last_yield_time = time.time()
            elif time.time() - last_yield_time > 15:
                yield ": heartbeat\n\n"
                last_yield_time = time.time()

            time.sleep(1)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


def _resolve_model_config(model_name: str) -> dict:
    """解析模型名称对应的市场配置（用于微调）"""
    name_lower = model_name.lower()
    if "csi1000" in name_lower:
        return {
            "market": "csi1000",
            "benchmark": "SH000852",
            "template": _qlib_zh_dir / "scripts" / "small" / "templates" / "workflow_config_lightgbm_Alpha158_csi1000.yaml",
        }
    if "csi500" in name_lower:
        return {
            "market": "csi500",
            "benchmark": "SH000905",
            "template": _qlib_zh_dir / "scripts" / "small" / "templates" / "workflow_config_lightgbm_Alpha158_csi500.yaml",
        }
    else:
        return {
            "market": "csi300",
            "benchmark": "SH000300",
            "template": _qlib_zh_dir / "examples" / "benchmarks" / "LightGBM" / "workflow_config_lightgbm_Alpha158.yaml",
        }


# ==========================================
#  API: 分析报告下载
# ==========================================

@app.route('/api/report/download', methods=['POST'])
def download_analysis_report():
    """接收分析结果 JSON，生成 HTML 报告并返回下载"""
    data = request.get_json(silent=True) or {}
    if not data or not data.get('symbol'):
        return jsonify({'error': '请提供分析结果数据'}), 400

    symbol = data.get('symbol', 'unknown')
    try:
        report = report_gen.generate(data, simulation_result=None)
        html = report_gen.to_html(report)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        response = Response(html, mimetype='text/html')
        response.headers['Content-Disposition'] = (
            f'attachment; filename="{symbol}_analysis_{timestamp}.html"'
        )
        return response
    except Exception as e:
        logger.error(f"生成分析报告失败 [{symbol}]: {e}")
        return jsonify({'error': f'报告生成失败: {str(e)}'}), 500


# ==========================================
#  API: 系统
# ==========================================

@app.route('/api/config', methods=['GET', 'POST'])
def config():
    if request.method == 'GET':
        return jsonify({
            'backend': agent.provider.backend_name,
            'tushare_token_configured': bool(os.environ.get('TUSHARE_TOKEN')),
            'llm_configured': bool(agent.prediction_node.api_key),
            'llm_model': agent.prediction_node.model,
            'mirofish_available': orchestrator.client.health_check(),
            'mirofish_url': orchestrator.client.base_url,
        })
    data = request.get_json(silent=True) or {}
    if 'backend' in data:
        new_backend = str(data['backend']).strip().lower()
        if new_backend not in {'advanced', 'tushare', 'akshare', 'baostock', 'auto', 'mock'}:
            return jsonify({'error': f'不支持的数据后端: {new_backend}'}), 400
        agent.provider._backend = None
        agent.provider._requested_backend = new_backend
        agent.provider._active_backend_name = new_backend
    return jsonify({'status': 'ok'})


# ==========================================
#  API: 配置健康检查
# ==========================================

@app.route('/api/config/health', methods=['GET'])
def config_health():
    """返回完整的配置健康检查报告"""
    from config_health import ConfigHealthChecker
    from analysis.agent import _get_search_service
    checker = ConfigHealthChecker(
        settings_obj=settings,
        agent_obj=agent,
        orchestrator_obj=orchestrator,
        search_service_obj=_get_search_service(),
    )
    return jsonify(checker.run_all_checks())


# ==========================================
#  API: 记忆系统统计
# ==========================================

@app.route('/api/memory/stats', methods=['GET'])
def memory_stats():
    """返回记忆系统（缓存/分析归档/大师追踪）的统计信息"""
    from memory import get_cache_manager, get_analysis_store, get_master_track_db

    try:
        # 缓存统计
        cache_mgr = get_cache_manager()
        cache_stats = cache_mgr.stats()

        # 分析归档统计
        analysis_store = get_analysis_store()
        symbols_with_analysis = analysis_store.list_all_symbols()

        return jsonify({
            'cache': cache_stats,
            'analysis_store': {
                'symbol_count': len(symbols_with_analysis),
                'symbols': symbols_with_analysis[:100],  # 最多返回 100 个
            },
            'master_track': {
                'available': True,
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==========================================
#  API: 本地纸面组合（不连接券商）
# ==========================================

@app.route('/api/paper-portfolio/overview', methods=['GET'])
def paper_portfolio_overview():
    try:
        return jsonify(_get_paper_portfolio_store().overview(_paper_quote))
    except Exception as exc:
        logger.exception(f"读取纸面组合失败: {exc}")
        return jsonify({'error': f'读取纸面组合失败: {exc}'}), 500


@app.route('/api/paper-portfolio/settings', methods=['GET', 'PUT'])
def paper_portfolio_settings():
    store = _get_paper_portfolio_store()
    try:
        if request.method == 'GET':
            return jsonify(store.get_settings())
        return jsonify(store.update_settings(request.get_json(silent=True) or {}))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/api/paper-portfolio/trades', methods=['GET', 'POST'])
def paper_portfolio_trades():
    store = _get_paper_portfolio_store()
    try:
        if request.method == 'GET':
            return jsonify(store.list_trades())
        trade = store.create_trade(request.get_json(silent=True) or {})
        return jsonify({'success': True, 'trade': trade}), 201
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/api/paper-portfolio/trades/<int:trade_id>/void', methods=['POST'])
def paper_portfolio_void_trade(trade_id: int):
    store = _get_paper_portfolio_store()
    try:
        payload = request.get_json(silent=True) or {}
        trade = store.void_trade(trade_id, str(payload.get('reason') or ''))
        return jsonify({'success': True, 'trade': trade})
    except LookupError as exc:
        return jsonify({'error': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/api/paper-portfolio/trades/<int:trade_id>/correct', methods=['POST'])
def paper_portfolio_correct_trade(trade_id: int):
    try:
        trade = _get_paper_portfolio_store().correct_trade(trade_id, request.get_json(silent=True) or {})
        return jsonify({'success': True, 'trade': trade}), 201
    except LookupError as exc:
        return jsonify({'error': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/api/memory/invalidate', methods=['POST'])
def memory_invalidate():
    """失效指定股票的缓存"""
    from memory import get_cache_manager
    data = request.get_json(silent=True) or {}
    symbol = data.get('symbol', '')
    if not symbol:
        return jsonify({'error': 'symbol is required'}), 400
    cache_mgr = get_cache_manager()
    result = cache_mgr.invalidate_symbol(symbol)
    return jsonify({'invalidated': result})


# ==========================================
#  API: 股票名称搜索（自动补全）
# ==========================================

@app.route('/api/stock/search', methods=['GET'])
def stock_search():
    """搜索股票代码（支持中文名称、拼音、代码片段）"""
    from stock_name_resolver import get_resolver
    q = request.args.get('q', '').strip()
    if not q or len(q) < 1:
        return jsonify([])
    resolver = get_resolver()
    results = resolver.search(q, limit=10)
    return jsonify(results)


@app.route('/api/masters', methods=['GET'])
def list_masters():
    """返回可用的大师决策者列表"""
    from analysis.agents.cio_prompts import list_masters as get_masters
    return jsonify({'masters': get_masters()})


@app.route('/api/predictions', methods=['GET'])
def list_predictions():
    with _predictions_lock:
        items = [
            {
                'task_id': p['task_id'],
                'symbol': p['symbol'],
                'scenario': p['scenario'],
                'status': p['status'],
                'progress': p['progress'],
                'created_at': p['created_at'],
            }
            for p in predictions.values()
        ]
    return jsonify(items)


@app.route('/')
def index():
    with open(os.path.join(app.static_folder, 'index.html'), encoding='utf-8') as f:
        return f.read()


# ==========================================
#  工具
# ==========================================

def _update_prediction(task_id, progress, status, message, **kwargs):
    with _predictions_lock:
        pred = predictions.get(task_id)
        if not pred:
            return
        if pred.get('status') == 'cancelling' and status not in ('cancelled', 'failed'):
            return
        pred['progress'] = progress
        pred['status'] = status
        pred['message'] = message
        for k, v in kwargs.items():
            pred[k] = v
        if status in ('completed', 'failed', 'cancelled'):
            pred['completed_at'] = datetime.now().isoformat()


# ==========================================
#  启动
# ==========================================

def _ensure_single_server(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(
                f"端口 {port} 已有 StockFish 或其他服务运行，请勿重复启动。"
            )

if __name__ == '__main__':
    _ensure_single_server(settings.PORT)
    _start_mirofish_if_needed()
    logger.info(f"StockFish v2 启动: http://{settings.HOST}:{settings.PORT}")
    app.run(
        host=settings.HOST,
        port=settings.PORT,
        debug=settings.DEBUG,
        threaded=True,
        use_reloader=False,
    )
