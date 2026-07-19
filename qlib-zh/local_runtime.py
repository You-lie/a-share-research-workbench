"""Shared local Conda runtime for Qlib jobs on Windows."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
PATH_SETTINGS_FILE = RUNTIME_ROOT / "path_settings.json"

_WINDOWS_DEFAULT_PYTHON = Path(r"D:\anaconda3\envs\stock_qlib\python.exe")
_default_python = _WINDOWS_DEFAULT_PYTHON if os.name == "nt" else Path(sys.executable)

QLIB_PYTHON = Path(os.environ.get("QLIB_PYTHON", str(_default_python))).expanduser()
QLIB_TEMP_DIR = Path(
    os.environ.get("QLIB_TEMP_DIR", str(RUNTIME_ROOT / "tmp"))
).expanduser().resolve()
DEFAULT_QLIB_DATA_DIR = Path(
    os.environ.get(
        "QLIB_DATA_DIR",
        str(RUNTIME_ROOT / "qlib_data" / "cn_data"),
    )
).expanduser().resolve()
MLRUNS_DIR = Path(
    os.environ.get("QLIB_MLRUNS_DIR", str(RUNTIME_ROOT / "mlruns"))
).expanduser().resolve()
DEFAULT_ANALYSIS_OUTPUTS_DIR = (PROJECT_ROOT / "DATA" / "analysis_outputs").resolve()


def _load_saved_paths() -> tuple[Path, Path]:
    data_dir = DEFAULT_QLIB_DATA_DIR
    model_dir = DEFAULT_ANALYSIS_OUTPUTS_DIR
    try:
        payload = json.loads(PATH_SETTINGS_FILE.read_text(encoding="utf-8"))
        if payload.get("data_dir"):
            data_dir = Path(os.path.expandvars(payload["data_dir"])).expanduser().resolve()
        if payload.get("model_dir"):
            model_dir = Path(os.path.expandvars(payload["model_dir"])).expanduser().resolve()
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return data_dir, model_dir


QLIB_DATA_DIR, ANALYSIS_OUTPUTS_DIR = _load_saved_paths()


def get_runtime_paths() -> dict[str, Path]:
    return {
        "data_dir": QLIB_DATA_DIR,
        "model_dir": ANALYSIS_OUTPUTS_DIR,
        "mlruns_dir": MLRUNS_DIR,
        "default_data_dir": DEFAULT_QLIB_DATA_DIR,
        "default_model_dir": DEFAULT_ANALYSIS_OUTPUTS_DIR,
    }


def configure_runtime_paths(data_dir: str | Path, model_dir: str | Path) -> dict[str, Path]:
    global QLIB_DATA_DIR, ANALYSIS_OUTPUTS_DIR

    resolved_data_dir = Path(os.path.expandvars(str(data_dir))).expanduser().resolve()
    resolved_model_dir = Path(os.path.expandvars(str(model_dir))).expanduser().resolve()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_dir": str(resolved_data_dir),
        "model_dir": str(resolved_model_dir),
    }
    temp_file = PATH_SETTINGS_FILE.with_suffix(".json.tmp")
    temp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_file.replace(PATH_SETTINGS_FILE)
    QLIB_DATA_DIR = resolved_data_dir
    ANALYSIS_OUTPUTS_DIR = resolved_model_dir
    return get_runtime_paths()


class ProcessCancelledError(RuntimeError):
    """Raised after a requested local Qlib process cancellation."""


def build_runtime_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build a UTF-8 subprocess environment with local Qlib paths."""
    QLIB_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
            "JOBLIB_MULTIPROCESSING": os.environ.get("JOBLIB_MULTIPROCESSING", "0" if os.name == "nt" else "1"),
            "MLFLOW_ALLOW_FILE_STORE": "true",
            "QLIB_WINDOWS_TEMP_COMPAT": "1",
            "JOBLIB_TEMP_FOLDER": str(QLIB_TEMP_DIR.resolve()),
            "TMPDIR": str(QLIB_TEMP_DIR.resolve()),
            "TEMP": str(QLIB_TEMP_DIR.resolve()),
            "TMP": str(QLIB_TEMP_DIR.resolve()),
            "QLIB_DATA_DIR": str(QLIB_DATA_DIR.resolve()),
            "QLIB_MLRUNS_DIR": str(MLRUNS_DIR.resolve()),
            "QLIB_MODEL_DIR": str(ANALYSIS_OUTPUTS_DIR.resolve()),
            "WORKDIR": str(PROJECT_ROOT.resolve()),
        }
    )
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(PROJECT_ROOT.resolve()), current_pythonpath) if part
    )
    if extra:
        env.update({key: str(value) for key, value in extra.items()})
    return env


@lru_cache(maxsize=1)
def _validate_python() -> None:
    if not QLIB_PYTHON.is_file():
        raise FileNotFoundError(
            f"Qlib Python 不存在: {QLIB_PYTHON}。请在 .env 设置 QLIB_PYTHON。"
        )

    check = subprocess.run(
        [str(QLIB_PYTHON), "-c", "import qlib, lightgbm"],
        cwd=str(PROJECT_ROOT),
        env=build_runtime_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if check.returncode != 0:
        detail = (check.stderr or check.stdout).strip().splitlines()
        message = detail[-1] if detail else f"退出码 {check.returncode}"
        raise RuntimeError(f"Qlib Conda 环境依赖检查失败: {message}")


def validate_runtime(require_data: bool = True) -> None:
    """Validate the configured interpreter and optional local data bundle."""
    _validate_python()
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if not require_data:
        return

    calendar_file = QLIB_DATA_DIR / "calendars" / "day.txt"
    features_dir = QLIB_DATA_DIR / "features"
    if not calendar_file.is_file() or not features_dir.is_dir():
        raise FileNotFoundError(
            f"Qlib 数据不完整: {QLIB_DATA_DIR}。请先在页面更新 Qlib 数据。"
        )


def python_command(script: str | Path, *args: object) -> list[str]:
    return [str(QLIB_PYTHON), str(Path(script).resolve()), *(str(arg) for arg in args)]


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0 and process.poll() is None:
            process.kill()
    else:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def run_streaming(
    command: Iterable[str],
    line_callback: Callable[[str], None] | None = None,
    *,
    timeout: int,
    cwd: str | Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[str]:
    """Run a local process with live merged output and a wall-clock timeout."""
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=env or build_runtime_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None

    output_queue: queue.Queue[str | None] = queue.Queue()

    def _read_output() -> None:
        try:
            for raw_line in process.stdout:
                output_queue.put(raw_line.rstrip())
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=_read_output, daemon=True)
    reader.start()

    try:
        started_at = time.monotonic()
        lines: list[str] = []
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise ProcessCancelledError("Qlib 任务已取消")
            if time.monotonic() - started_at > timeout:
                raise TimeoutError(f"本地 Qlib 任务超时 ({timeout}s)")

            try:
                item = output_queue.get(timeout=0.25)
            except queue.Empty:
                if process.poll() is not None and not reader.is_alive():
                    break
                continue

            if item is None:
                break
            if item:
                lines.append(item)
                if line_callback:
                    line_callback(item)

        return_code = process.wait(timeout=10)
        if return_code != 0:
            detail = "\n".join(lines[-20:])
            raise RuntimeError(
                f"本地 Qlib 进程退出码: {return_code}"
                + (f"\n{detail}" if detail else "")
            )
        return lines
    except BaseException:
        _terminate_process_tree(process)
        raise
    finally:
        reader.join(timeout=2)
        process.stdout.close()
