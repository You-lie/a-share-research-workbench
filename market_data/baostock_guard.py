"""Process-wide lock for BaoStock access.

The baostock package keeps a single global session (module-level login state),
so concurrent queries from multiple threads corrupt each other's connection.
Serialize every BaoStock call through this shared lock.
"""
import threading

BAOSTOCK_LOCK = threading.RLock()
