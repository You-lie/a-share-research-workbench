"""Runtime settings manager for the local web UI.

Lets the user edit API keys and analysis options from the 配置管理 tab instead
of hand-editing .env. Writes are atomic and comment-preserving; secrets are
never returned unmasked to the browser.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import PROJECT_ROOT

ENV_PATH = PROJECT_ROOT / ".env"

_SECRET_KEYS = {
    "LLM_API_KEY", "TUSHARE_TOKEN", "TAVILY_API_KEY", "BOCHA_API_KEY", "ZEP_API_KEY",
    "BRAVE_API_KEY", "SERPAPI_API_KEY", "ANSPIRE_API_KEY", "MINIMAX_API_KEY",
}

CONFIG_GROUPS: List[Dict[str, Any]] = [
    {
        "id": "llm",
        "title": "大语言模型（LLM）",
        "description": "驱动所有 AI 分析和报告；更换 Key 或模型后立即生效。",
        "fields": [
            {"key": "LLM_API_KEY", "label": "API Key", "type": "secret",
             "hint": "DeepSeek 或其他 OpenAI 兼容平台的密钥"},
            {"key": "LLM_BASE_URL", "label": "API 地址", "type": "text",
             "placeholder": "https://api.deepseek.com", "required": True,
             "hint": "接口基础地址，需包含 https://"},
            {"key": "LLM_MODEL_NAME", "label": "模型名称", "type": "text",
             "placeholder": "deepseek-flash", "required": True,
             "hint": "平台支持的模型标识，例如 deepseek-flash"},
        ],
    },
    {
        "id": "datasource",
        "title": "行情数据源",
        "description": "决定行情、财务数据的优先级；修改后立即生效。",
        "fields": [
            {"key": "STOCK_BACKEND", "label": "数据后端", "type": "select",
             "options": [
                 {"value": "advanced", "label": "advanced（多源自动切换，推荐）"},
                 {"value": "tushare", "label": "tushare（需 Token 权限）"},
                 {"value": "akshare", "label": "akshare（免费）"},
                 {"value": "baostock", "label": "baostock（免费，无新闻）"},
                 {"value": "auto", "label": "auto（自动探测）"},
                 {"value": "mock", "label": "mock（模拟数据，仅演示）"},
             ],
             "hint": "默认 advanced；避免使用 mock 做真实判断"},
            {"key": "TUSHARE_TOKEN", "label": "Tushare Token", "type": "secret",
             "hint": "tushare.pro 注册获取；免费账号部分接口受限"},
        ],
    },
    {
        "id": "search",
        "title": "新闻搜索",
        "description": "舆情分析和资讯搜索使用；配置多个来源时按优先级自动切换，修改后立即生效。",
        "fields": [
            {"key": "TAVILY_API_KEY", "label": "Tavily API Key", "type": "secret",
             "hint": "tavily.com 获取；免费额度约 1000 次/月"},
            {"key": "BOCHA_API_KEY", "label": "博查 API Key（可选）", "type": "secret",
             "hint": "bochaai.com 获取；配置后优先使用博查搜索"},
            {"key": "BRAVE_API_KEY", "label": "Brave Search Key（可选）", "type": "secret",
             "hint": "brave.com/search/api 获取；有免费额度"},
            {"key": "SERPAPI_API_KEY", "label": "SerpAPI Key（可选）", "type": "secret",
             "hint": "serpapi.com 获取；付费为主"},
            {"key": "ANSPIRE_API_KEY", "label": "Anspire Key（可选）", "type": "secret",
             "hint": "anspire 搜索服务密钥"},
            {"key": "MINIMAX_API_KEY", "label": "MiniMax Key（可选）", "type": "secret",
             "hint": "MiniMax 搜索服务密钥"},
        ],
    },
    {
        "id": "mirofish",
        "title": "MiroFish 智能推演",
        "description": "仅影响智能推演功能；保存后如 MiroFish 由本程序托管会自动重启。",
        "fields": [
            {"key": "ZEP_API_KEY", "label": "Zep API Key", "type": "secret",
             "hint": "app.getzep.com 获取；缺少则无法启动推演"},
        ],
    },
    {
        "id": "performance",
        "title": "性能",
        "description": "控制批量分析和 LLM 请求的并发；修改后立即生效。",
        "fields": [
            {"key": "BATCH_CONCURRENCY", "label": "批量分析并发数", "type": "number",
             "min": 1, "max": 8,
             "hint": "同时分析的股票数，默认 5；过高可能触发接口限流"},
            {"key": "LLM_MAX_CONCURRENCY", "label": "LLM 并发上限", "type": "number",
             "min": 1, "max": 64,
             "hint": "所有任务合计的最大并发 LLM 请求数，默认 8"},
        ],
    },
]

_ALLOWED_KEYS = {field["key"] for group in CONFIG_GROUPS for field in group["fields"]}

# Saving these requires spawned MiroFish to be restarted to take effect.
_MIROFISH_RESTART_KEYS = {"ZEP_API_KEY"}

_KEY_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_./:@+\-]*$")


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "****"
    return f"****{value[-4:]}"


def _format_env_value(value: str) -> str:
    if _SAFE_VALUE.match(value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _read_env_file() -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not ENV_PATH.is_file():
        return values
    try:
        text = ENV_PATH.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw = stripped.partition("=")
        key = key.strip()
        raw = raw.strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
            raw = raw[1:-1]
            if line.strip().rpartition("=")[2].strip().startswith('"'):
                raw = raw.replace('\\"', '"').replace("\\\\", "\\")
        values[key] = raw
    return values


def read_config() -> Dict[str, Any]:
    """Return grouped field metadata with current (masked) values for the UI."""
    file_values = _read_env_file()
    groups: List[Dict[str, Any]] = []
    for group in CONFIG_GROUPS:
        fields = []
        for field in group["fields"]:
            key = field["key"]
            raw = os.environ.get(key)
            if raw is None:
                raw = file_values.get(key, "")
            raw = (raw or "").strip()
            is_secret = field.get("type") == "secret"
            fields.append({
                **{k: v for k, v in field.items() if k != "key"},
                "key": key,
                "is_set": bool(raw),
                "value": _mask(raw) if is_secret else raw,
            })
        groups.append({
            "id": group["id"],
            "title": group["title"],
            "description": group["description"],
            "fields": fields,
        })
    return {
        "groups": groups,
        "env_file": str(ENV_PATH),
    }


def _validate_updates(updates: Dict[str, Any]) -> Dict[str, str]:
    """Validate incoming values; empty string means "clear this key"."""
    cleaned: Dict[str, str] = {}
    field_defs = {field["key"]: field for group in CONFIG_GROUPS for field in group["fields"]}
    for key, value in updates.items():
        if key not in _ALLOWED_KEYS:
            raise ValueError(f"不支持的配置项: {key}")
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise ValueError(f"{key} 的值格式不正确")
        if "\n" in value or "\r" in value:
            raise ValueError(f"{key} 不能包含换行符")
        value = value.strip()
        if len(value) > 500:
            raise ValueError(f"{key} 长度超限")
        field = field_defs[key]
        ftype = field.get("type")
        if value:
            if ftype == "number":
                try:
                    number = int(value)
                except ValueError:
                    raise ValueError(f"{field['label']} 必须是整数")
                low = int(field.get("min", 1))
                high = int(field.get("max", 1000))
                if not (low <= number <= high):
                    raise ValueError(f"{field['label']} 需在 {low}-{high} 之间")
                value = str(number)
            elif ftype == "select":
                allowed = {opt["value"] for opt in field.get("options", [])}
                if value not in allowed:
                    raise ValueError(f"{field['label']} 不是有效选项")
            elif key == "LLM_BASE_URL":
                if not value.startswith(("http://", "https://")):
                    raise ValueError("API 地址必须以 http:// 或 https:// 开头")
        elif field.get("required"):
            raise ValueError(f"{field['label']} 不能为空")
        cleaned[key] = value
    return cleaned


def apply_updates(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Validate, persist to .env, and push values into the running process.

    Returns {changed: {key: bool_is_set}, restart_hint: str}.
    """
    cleaned = _validate_updates(updates)
    if not cleaned:
        return {"changed": {}, "restart_hint": ""}

    # Preserve file structure: replace in place, append missing keys.
    lines: List[str] = []
    if ENV_PATH.is_file():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    seen = set()
    rewritten: List[str] = []
    for line in lines:
        match = _KEY_LINE.match(line)
        if match and match.group(1) in cleaned:
            key = match.group(1)
            seen.add(key)
            value = cleaned[key]
            # Clearing a value removes the line entirely: leaving "KEY=" breaks
            # pydantic parsing for typed fields and is not a valid "unset".
            if value:
                rewritten.append(f"{key}={_format_env_value(value)}")
        else:
            rewritten.append(line)
    for key, value in cleaned.items():
        if key not in seen and value:
            rewritten.append(f"{key}={_format_env_value(value)}")

    text = "\n".join(rewritten).rstrip("\n") + "\n"
    tmp = ENV_PATH.with_suffix(".env.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, ENV_PATH)

    # Push into the live process so new requests see the values immediately.
    defaults = {
        "BATCH_CONCURRENCY": 5,
        "LLM_MAX_CONCURRENCY": 8,
    }
    numeric_keys = {
        field["key"] for group in CONFIG_GROUPS for field in group["fields"]
        if field.get("type") == "number"
    }
    for key, value in cleaned.items():
        typed_value: Any = value
        if value and key in numeric_keys:
            try:
                typed_value = int(value)
            except ValueError:
                typed_value = value
        if value:
            os.environ[key] = str(value)
        else:
            os.environ.pop(key, None)
        try:
            from config import settings
            if value:
                setattr(settings, key, typed_value)
            elif key in defaults:
                setattr(settings, key, defaults[key])
            else:
                setattr(settings, key, None)
        except Exception:
            pass

    restart_hint = ""
    if any(key in _MIROFISH_RESTART_KEYS for key in cleaned):
        restart_hint = "Zep/ZepKey 已保存；如 MiroFish 由本程序托管，已自动重启以生效。"
    return {
        "changed": {key: bool(value) for key, value in cleaned.items()},
        "restart_hint": restart_hint,
    }
