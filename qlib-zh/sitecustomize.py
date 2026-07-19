"""Runtime compatibility hooks loaded by local Qlib subprocesses."""

from __future__ import annotations

import atexit
import os
import secrets
import shutil
import tempfile


_created_temp_dirs: set[str] = set()


def _cleanup_windows_temp_dirs() -> None:
    for path in sorted(_created_temp_dirs, key=len, reverse=True):
        shutil.rmtree(path, ignore_errors=True)


def _windows_mkdtemp(suffix=None, prefix=None, dir=None):
    directory = dir if dir is not None else tempfile.gettempdir()
    use_bytes = isinstance(directory, bytes)
    if use_bytes:
        prefix = b"tmp" if prefix is None else os.fsencode(prefix)
        suffix = b"" if suffix is None else os.fsencode(suffix)
    else:
        prefix = "tmp" if prefix is None else os.fsdecode(prefix)
        suffix = "" if suffix is None else os.fsdecode(suffix)

    for _ in range(tempfile.TMP_MAX):
        token = secrets.token_hex(8)
        if use_bytes:
            token = token.encode("ascii")
        path = os.path.join(directory, prefix + token + suffix)
        try:
            os.mkdir(path)
        except FileExistsError:
            continue
        absolute_path = os.path.abspath(path)
        _created_temp_dirs.add(absolute_path)
        return absolute_path
    raise FileExistsError("No usable temporary directory name found")


if os.name == "nt" and os.environ.get("QLIB_WINDOWS_TEMP_COMPAT") == "1":
    tempfile.mkdtemp = _windows_mkdtemp
    atexit.register(_cleanup_windows_temp_dirs)
