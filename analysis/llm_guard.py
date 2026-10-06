"""Process-wide LLM concurrency cap.

Batch analysis runs several stocks in parallel, and each stock spawns its own
parallel agent calls. Without a global cap, peak request volume can trip the
LLM provider's rate limits and fail whole batches. Acquire a slot around every
outbound chat completion call.
"""
import os
import threading


def _resolve_limit() -> int:
    raw = os.environ.get('LLM_MAX_CONCURRENCY', '').strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    try:
        from config import settings
        return max(1, int(getattr(settings, 'LLM_MAX_CONCURRENCY', 8) or 8))
    except Exception:
        return 8


LLM_SEMAPHORE = threading.BoundedSemaphore(_resolve_limit())


class llm_slot:
    """Context manager that holds one LLM concurrency slot."""

    def __enter__(self):
        LLM_SEMAPHORE.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        LLM_SEMAPHORE.release()
        return False
