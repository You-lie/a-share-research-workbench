"""
Qlib 数据更新执行器 — 下载 qlib_bin.tar.gz 并解压到项目内 runtime/qlib_data/cn_data。

用法:
    from data_runner import run_data_update
    result = run_data_update(progress_callback=cb)
    print(result["message"])
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Callable

import requests

from local_runtime import QLIB_DATA_DIR

# ---- 常量 ----
GITHUB_REPO = "chenditc/investment_data"
RELEASE_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases?per_page=5"
ASSET_NAME = "qlib_bin.tar.gz"

# 下载超时（秒）
DOWNLOAD_TIMEOUT = 600
# 每下载 chunk 后推送间隔（秒）
PROGRESS_INTERVAL = 1.0

# ---- 代理配置 ----
_PROXY_URL = os.environ.get("QLIB_DATA_PROXY", "")
_PROXIES = {"http": _PROXY_URL, "https": _PROXY_URL} if _PROXY_URL else None


def _make_session() -> requests.Session:
    """创建仅使用 QLIB_DATA_PROXY 的 requests Session."""
    session = requests.Session()
    session.trust_env = False
    if _PROXIES:
        session.proxies.update(_PROXIES)
    return session


def _log(progress_callback: Callable | None, message: str, **extra):
    """统一日志推送."""
    if progress_callback:
        progress_callback({"message": message, **extra})


def _get_latest_release_info() -> dict:
    """获取最新包含 qlib_bin.tar.gz 的 release。自动回退到上一个 release。"""
    session = _make_session()
    resp = session.get(
        RELEASE_API_URL,
        headers={"User-Agent": "StockFish/1.0"},
        timeout=30,
    )
    resp.raise_for_status()
    releases = resp.json()

    for rel in releases:
        tag = rel.get("tag_name", "unknown")
        for asset in rel.get("assets", []):
            if asset.get("name") == ASSET_NAME:
                return {
                    "download_url": asset["url"],
                    "tag": tag,
                    "size": asset.get("size", 0),
                }

    raise FileNotFoundError(
        f"在 {GITHUB_REPO} 的最近 5 个 release 中均未找到 {ASSET_NAME}"
    )


def _download_file(
    url: str,
    dest_path: Path,
    total_size: int = 0,
    progress_callback: Callable | None = None,
) -> None:
    """流式下载文件并推送进度."""
    _log(progress_callback, f"开始下载: {url}")
    _log(progress_callback, f"目标文件: {dest_path}")
    if total_size > 0:
        _log(progress_callback, f"文件大小: {total_size / 1024 / 1024:.1f} MB")

    session = _make_session()
    resp = session.get(
        url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": "StockFish/1.0",
        },
        stream=True,
        timeout=DOWNLOAD_TIMEOUT,
    )
    resp.raise_for_status()

    # 实际 content-length
    content_length = int(resp.headers.get("content-length", 0))
    if content_length == 0:
        content_length = total_size

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    last_log_time = 0.0
    import time as _time

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                now = _time.time()
                if content_length > 0 and (now - last_log_time) >= PROGRESS_INTERVAL:
                    pct = downloaded / content_length * 100
                    mb_dl = downloaded / 1024 / 1024
                    mb_total = content_length / 1024 / 1024
                    _log(
                        progress_callback,
                        f"下载中... {mb_dl:.1f}/{mb_total:.1f} MB ({pct:.1f}%)",
                        progress=round(pct / 100, 4),
                    )
                    last_log_time = now

    _log(progress_callback, f"下载完成: {downloaded / 1024 / 1024:.1f} MB")


def _extract_tar_gz(
    archive_path: Path,
    extract_to: Path,
    progress_callback: Callable | None = None,
) -> None:
    """解压 tar.gz 到一个空的暂存目录."""
    _log(progress_callback, f"开始解压: {archive_path}")
    _log(progress_callback, f"解压到: {extract_to}")

    extract_to.mkdir(parents=True, exist_ok=True)
    if any(extract_to.iterdir()):
        raise RuntimeError(f"暂存目录不是空目录: {extract_to}")

    # 解压
    import time as _time

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        total_members = len(members)
        _log(progress_callback, f"共 {total_members} 个文件/目录")

        if not members:
            raise RuntimeError("下载的数据包为空，无法解压")

        path_parts = [PurePosixPath(member.name).parts for member in members if member.name]
        prefixes = [parts[0] for parts in path_parts if len(parts) > 1]
        common_prefix = prefixes[0] if prefixes and len(prefixes) == len(path_parts) and all(
            prefix == prefixes[0] for prefix in prefixes
        ) else None

        # strip-components=1
        last_log_time = 0.0
        for i, member in enumerate(members):
            # 调整路径：去掉统一的顶级目录。
            if common_prefix and member.name.startswith(common_prefix + "/"):
                relative_name = member.name[len(common_prefix) + 1:]
            elif common_prefix and member.name == common_prefix:
                continue
            else:
                relative_name = member.name

            relative_path = PurePosixPath(relative_name)
            if not relative_name or relative_path.is_absolute() or ".." in relative_path.parts:
                raise RuntimeError(f"数据包包含不安全路径: {member.name}")

            target_path = (extract_to / Path(*relative_path.parts)).resolve()
            try:
                target_path.relative_to(extract_to.resolve())
            except ValueError as exc:
                raise RuntimeError(f"数据包路径超出暂存目录: {member.name}") from exc

            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError(f"无法读取数据包文件: {member.name}")
                with source, open(target_path, "wb") as destination:
                    shutil.copyfileobj(source, destination)
            else:
                raise RuntimeError(f"数据包包含不支持的文件类型: {member.name}")

            now = _time.time()
            if (now - last_log_time) >= PROGRESS_INTERVAL:
                pct = (i + 1) / total_members * 100
                _log(
                    progress_callback,
                    f"解压中... {i + 1}/{total_members} ({pct:.0f}%)",
                    progress=round(0.5 + pct / 200, 4),  # 进度映射 0.5~1.0
                )
                last_log_time = now

    _log(progress_callback, "解压完成，正在整理目录结构...")

    # 兼容 qlib_bin/ 或 cn_data/ 这类嵌套根目录。
    for _ in range(3):  # 最多处理 3 层嵌套
        subdirs = [d for d in extract_to.iterdir() if d.is_dir()]
        files = [f for f in extract_to.iterdir() if f.is_file()]
        if len(subdirs) == 1 and len(files) == 0:
            nested = subdirs[0]
            if not any(nested.iterdir()):
                raise RuntimeError(f"数据包只包含空目录: {nested.name}")
            _log(progress_callback, f"检测到嵌套目录 {nested.name}，上移内容...")
            for item in nested.iterdir():
                target = extract_to / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
            nested.rmdir()
        else:
            break
    _log(progress_callback, "目录整理完成")


def _verify_data_dir(data_dir: Path, progress_callback: Callable | None = None) -> list[str]:
    """验证暂存或正式数据目录，缺少关键文件时立即失败."""
    checks = [
        data_dir / "calendars" / "day.txt",
        data_dir / "instruments" / "csi300.txt",
        data_dir / "instruments" / "all.txt",
    ]
    found = []
    missing = []
    for p in checks:
        if p.exists():
            found.append(str(p.relative_to(data_dir)))
        else:
            missing.append(str(p.relative_to(data_dir)))

    _log(progress_callback, f"数据验证: 找到 {len(found)}/{len(checks)} 关键文件")
    if missing:
        raise FileNotFoundError(f"解压结果缺少关键文件: {', '.join(missing)}")

    return found


def _replace_data_dir(staging_dir: Path, target_dir: Path, progress_callback: Callable | None = None) -> None:
    """用已校验的暂存目录替换正式目录，失败时恢复原数据。"""
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = target_dir.parent / f".{target_dir.name}.backup-{uuid.uuid4().hex}"
    had_previous_data = target_dir.exists()

    if had_previous_data:
        _log(progress_callback, "新数据校验通过，正在保留旧数据并切换...")
        target_dir.replace(backup_dir)

    try:
        staging_dir.replace(target_dir)
    except Exception:
        if had_previous_data and backup_dir.exists() and not target_dir.exists():
            backup_dir.replace(target_dir)
        raise

    if backup_dir.exists():
        try:
            shutil.rmtree(backup_dir)
        except OSError as exc:
            _log(progress_callback, f"新数据已生效，旧数据备份暂未清理: {exc}")
    _log(progress_callback, "数据目录切换完成")


def run_data_update(
    progress_callback: Callable | None = None,
) -> dict:
    """
    下载最新 qlib_bin.tar.gz 并解压到项目内 runtime/qlib_data/cn_data。

    Args:
        progress_callback: 可选回调，接收 dict(status, message, progress, ...)

    Returns:
        {"success": True/False, "message": "...", "tag": "...", "verified_files": [...]}
    """
    import time as _time
    start_time = _time.time()
    staging_dir: Path | None = None

    try:
        # Step 1: 获取 release 信息
        _log(progress_callback, "正在获取最新 release 信息...", progress=0.0)
        release_info = _get_latest_release_info()
        tag = release_info["tag"]
        download_url = release_info["download_url"]
        size = release_info["size"]

        _log(
            progress_callback,
            f"最新 release: {tag} | {size / 1024 / 1024:.1f} MB",
            progress=0.05,
        )

        # Step 2: 下载
        _log(progress_callback, f"下载链接: {download_url}", progress=0.05)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            _download_file(download_url, tmp_path, total_size=size, progress_callback=progress_callback)

            staging_dir = QLIB_DATA_DIR.parent / f".{QLIB_DATA_DIR.name}.staging-{uuid.uuid4().hex}"
            _log(progress_callback, "正在解压到暂存目录，不会覆盖现有数据...")
            _extract_tar_gz(tmp_path, staging_dir, progress_callback=progress_callback)
            verified = _verify_data_dir(staging_dir, progress_callback)
            _replace_data_dir(staging_dir, QLIB_DATA_DIR, progress_callback)
        finally:
            # 清理临时文件
            if tmp_path.exists():
                tmp_path.unlink()
                _log(progress_callback, "已清理临时文件")
            if staging_dir is not None and staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

        elapsed = _time.time() - start_time
        msg = (
            f"✅ qlib 数据更新完成！"
            f" 版本: {tag}, 耗时: {elapsed:.0f}s, "
            f"验证通过: {len(verified)} 文件"
        )
        _log(progress_callback, msg, progress=1.0, status="completed")

        return {
            "success": True,
            "message": msg,
            "tag": tag,
            "verified_files": verified,
            "elapsed_seconds": round(elapsed, 1),
        }

    except Exception as e:
        error_msg = f"❌ 数据更新失败: {e}"
        _log(progress_callback, error_msg, status="failed", error=str(e))
        return {
            "success": False,
            "message": error_msg,
            "error": str(e),
        }


def _print_progress(data):
    """CLI 进度回调."""
    msg = data.get("message", "")
    status = data.get("status", "")
    pct = data.get("progress", 0)
    if pct:
        print(f"[{pct*100:.0f}%] {msg}", file=sys.stderr)
    else:
        print(f"[data] {msg}", file=sys.stderr)


# ---- CLI 入口 ----
if __name__ == "__main__":
    result = run_data_update(progress_callback=_print_progress)
    print(json.dumps(result, ensure_ascii=False, indent=2))
