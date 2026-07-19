"""Normalized, user-facing provenance for market-data analysis results."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable


_SOURCE_LABELS = {
    "advanced": "多源行情后端",
    "akshare": "AkShare / 东方财富",
    "akshare_em": "AkShare / 东方财富",
    "akshare_sina": "AkShare / 新浪财经",
    "akshare_qq": "AkShare / 腾讯财经",
    "efinance": "efinance / 东方财富",
    "tushare": "Tushare Pro",
    "yfinance": "Yahoo Finance",
    "baostock": "Baostock",
    "sina": "新浪财经",
    "tencent": "腾讯财经",
    "fallback": "备用数据源",
    "mock": "模拟数据",
    "模拟数据": "模拟数据",
    "unknown": "来源未记录",
}


def source_label(source: Any) -> str:
    raw = str(source or "unknown").strip()
    return _SOURCE_LABELS.get(raw.lower(), raw or _SOURCE_LABELS["unknown"])


def _unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        # DataFetcherManager may attach diagnostic dictionaries (provider,
        # duration, result) to a broad pipeline trace. They are not sources
        # for the financial value currently displayed to the user.
        if isinstance(value, dict):
            continue
        normalized = str(value or "").strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _status_for(value: Any, source: str, cached: bool, fallback: bool = False) -> str:
    if value is None or value == "":
        return "unavailable"
    if str(source).lower() in {"mock", "模拟数据"}:
        return "mock"
    if cached:
        return "cached"
    if str(source).lower() == "fallback":
        return "fallback"
    if fallback:
        return "fallback"
    return "fresh"


def _field(
    value: Any,
    source: str,
    endpoint: str,
    as_of: str,
    report_period: str,
    basis: str,
    cached: bool,
    source_chain: Iterable[Any],
    fallback_reason: str = "",
) -> dict:
    fallback = bool(fallback_reason)
    status = _status_for(value, source, cached, fallback)
    if value is None or value == "":
        fallback_reason = fallback_reason or "上游未返回该字段"
    return {
        "value": value,
        "source": source_label(source),
        "source_id": str(source or "unknown"),
        "endpoint": endpoint or "未记录接口",
        "retrieved_at": as_of or "",
        "data_date": as_of or "",
        "report_period": report_period or "",
        "basis": basis or "来源未明确口径",
        "status": status,
        "fallback_reason": fallback_reason,
        "source_chain": [source_label(item) for item in _unique(source_chain)],
    }


def _section_status(values: Iterable[Any], source: str, cached: bool, source_chain: list[str]) -> str:
    values = list(values)
    present = [value for value in values if value is not None and value != ""]
    if not present:
        return "unavailable"
    base_status = _status_for(present[0], source, cached)
    if base_status != "fresh":
        return base_status
    if len(source_chain) > 1 or len(present) < len(values):
        return "partial"
    return "fresh"


def build_data_provenance(market: Dict[str, Any], *, cached: bool = False) -> dict:
    """Build a serializable provenance payload without guessing financial bases."""
    generated_at = datetime.now().isoformat(timespec="seconds")
    backend = str(market.get("data_source") or "unknown")
    quote = market.get("quote") or {}
    financial = market.get("financial_summary") or {}
    news = market.get("news") or []

    quote_source = str(quote.get("source") or backend)
    quote_sources = _unique([quote_source, *(quote.get("source_chain") or [])])
    quote_field_sources = quote.get("field_sources") or {}
    quote_bases = quote.get("field_bases") or {}
    quote_endpoint = quote.get("endpoint") or "实时行情"
    quote_time = quote.get("timestamp") or market.get("timestamp") or ""

    financial_source = str(financial.get("source") or backend)
    financial_sources = _unique([financial_source, *(financial.get("source_chain") or [])])
    financial_field_sources = financial.get("field_sources") or {}
    financial_endpoint = financial.get("endpoint") or "财务摘要"
    financial_time = financial.get("retrieved_at") or market.get("timestamp") or ""
    report_date = financial.get("report_date") or ""
    financial_basis = financial.get("period_basis") or "来源未明确口径"

    def quote_field(name: str, default_basis: str = "来源未明确口径") -> dict:
        field_source = str(quote_field_sources.get(name) or quote_source)
        fallback_reason = ""
        if field_source != quote_source:
            fallback_reason = "该字段由备用行情源补齐"
        return _field(
            quote.get(name), field_source, quote_endpoint, quote_time, "",
            quote_bases.get(name) or default_basis, cached,
            [*quote_sources, field_source], fallback_reason,
        )

    def financial_field(name: str) -> dict:
        field_source = str(financial_field_sources.get(name) or financial_source)
        fallback_reason = ""
        if field_source != financial_source:
            fallback_reason = "该字段由备用财务源补齐"
        return _field(
            financial.get(name), field_source, financial_endpoint, financial_time,
            report_date, financial_basis, cached,
            [*financial_sources, field_source], fallback_reason,
        )

    financial_values = [financial.get(key) for key in ("roe", "eps", "revenue", "net_profit")]
    item_sources = _unique(item.get("source") for item in news if isinstance(item, dict))
    news_status = "unavailable" if not news else "mock" if item_sources == ["模拟数据"] else "cached" if cached else "fresh"
    sections = {
        "quote": {
            "label": "行情",
            "status": _section_status([quote.get("price")], quote_source, cached, quote_sources),
            "source_chain": [source_label(item) for item in quote_sources],
            "retrieved_at": quote_time,
            "fallback_reason": "行情字段来自多个来源" if len(quote_sources) > 1 else "",
        },
        "financial": {
            "label": "财务",
            "status": _section_status(financial_values, financial_source, cached, financial_sources),
            "source_chain": [source_label(item) for item in financial_sources],
            "retrieved_at": financial_time,
            "report_period": report_date,
            "basis": financial_basis,
            "fallback_reason": "财务字段来自多个来源" if len(financial_sources) > 1 else "",
        },
        "news": {
            "label": "新闻",
            "status": news_status,
            "source_chain": [source_label(item) for item in item_sources],
            "retrieved_at": market.get("timestamp") or "",
            "count": len(news),
            "fallback_reason": "未获取到新闻" if not news else "",
        },
    }
    return {
        "generated_at": generated_at,
        "cache_status": "cached" if cached else "fresh",
        "sections": sections,
        "fields": {
            "quote.price": quote_field("price", "最新成交价"),
            "quote.pe": quote_field("pe"),
            "quote.pb": quote_field("pb", "市净率（来源未明确报表期）"),
            "financial.roe": financial_field("roe"),
            "financial.eps": financial_field("eps"),
            "financial.revenue": financial_field("revenue"),
            "financial.net_profit": financial_field("net_profit"),
        },
        "news_items": [
            {
                "title": item.get("title", ""),
                "source": source_label(item.get("source")),
                "source_id": item.get("source", ""),
                "publish_time": item.get("publish_time", ""),
                "url": item.get("url", ""),
                "retrieved_at": item.get("retrieved_at") or market.get("timestamp") or "",
                "status": "mock" if str(item.get("source", "")).lower() in {"mock", "模拟数据"} else "cached" if cached else "fresh",
                "summary_generated_by_model": bool(item.get("summary_generated_by_model") or item.get("is_llm_summary")),
            }
            for item in news if isinstance(item, dict)
        ],
    }
