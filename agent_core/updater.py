from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version
from dotenv import dotenv_values

from agent_core.runtime_paths import tg_agent_home
from agent_core.version import get_cli_version

MANIFEST_URL_ENV = "CRAB_UPDATE_MANIFEST_URL"
SKIP_UPDATE_CHECK_ENV = "CRAB_SKIP_UPDATE_CHECK"
OSS_ENDPOINT_URL_ENV = "CRAB_OSS_ENDPOINT_URL"
OSS_BUCKET_ENV = "CRAB_OSS_BUCKET"
OSS_ACCESS_KEY_ID_ENV = "CRAB_OSS_ACCESS_KEY_ID"
OSS_SECRET_ACCESS_KEY_ENV = "CRAB_OSS_SECRET_ACCESS_KEY"
OSS_REGION_ENV = "CRAB_OSS_REGION"
OSS_VERIFY_SSL_ENV = "CRAB_OSS_VERIFY_SSL"
_UPDATE_ENV_NAMES = (
    MANIFEST_URL_ENV,
    SKIP_UPDATE_CHECK_ENV,
    OSS_ENDPOINT_URL_ENV,
    OSS_BUCKET_ENV,
    OSS_ACCESS_KEY_ID_ENV,
    OSS_SECRET_ACCESS_KEY_ENV,
    OSS_REGION_ENV,
    OSS_VERIFY_SSL_ENV,
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_S3",
)


@dataclass(frozen=True)
class UpdateInfo:
    current_version: str
    latest_version: str
    published_at: str | None
    notes: list[str]
    release: dict[str, Any]
    manifest: dict[str, Any]


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def update_state_path() -> Path:
    return tg_agent_home() / "update_state.json"


def update_lock_path() -> Path:
    return tg_agent_home() / "update.lock"


def updates_root() -> Path:
    return tg_agent_home() / "updates"


def _load_update_env() -> None:
    """Load updater-related env vars before the main CLI runtime starts."""
    for env_path in (Path.cwd() / ".env", tg_agent_home() / ".env"):
        if not env_path.exists():
            continue
        try:
            values = dotenv_values(env_path)
        except Exception:
            continue
        for name in _UPDATE_ENV_NAMES:
            value = values.get(name)
            if value is not None and name not in os.environ:
                os.environ[name] = value


def _read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(default or {})
    return payload if isinstance(payload, dict) else dict(default or {})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_update_state() -> dict[str, Any]:
    state = _read_json(update_state_path(), {"auto_check_enabled": True})
    state.setdefault("auto_check_enabled", True)
    return state


def save_update_state(state: dict[str, Any]) -> None:
    _write_json(update_state_path(), state)


def _manifest_url() -> str | None:
    value = os.environ.get(MANIFEST_URL_ENV, "").strip()
    return value or None


def _platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    is_x64 = machine in {"x86_64", "amd64", "x64"}
    if system == "linux" and is_x64:
        return "linux-x64"
    if system == "windows" and is_x64:
        return "windows-x64"
    return f"{system}-{machine}"


def _parse_s3_url(url: str) -> tuple[str, str] | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme:
        return None
    bucket = os.environ.get(OSS_BUCKET_ENV, "").strip()
    key = url.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"{OSS_BUCKET_ENV} and object key url are required")
    return bucket, key


def _s3_client() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for s3:// update URLs") from exc

    endpoint_url = (
        os.environ.get(OSS_ENDPOINT_URL_ENV)
        or os.environ.get("AWS_ENDPOINT_URL_S3")
        or os.environ.get("AWS_ENDPOINT_URL")
    )
    access_key_id = os.environ.get(OSS_ACCESS_KEY_ID_ENV) or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_access_key = os.environ.get(OSS_SECRET_ACCESS_KEY_ENV) or os.environ.get(
        "AWS_SECRET_ACCESS_KEY"
    )
    region_name = os.environ.get(OSS_REGION_ENV) or os.environ.get("AWS_DEFAULT_REGION")
    verify_value = os.environ.get(OSS_VERIFY_SSL_ENV, "").strip().lower()
    verify: bool | str = verify_value not in {"0", "false", "no", "off"}

    kwargs: dict[str, Any] = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    if access_key_id:
        kwargs["aws_access_key_id"] = access_key_id
    if secret_access_key:
        kwargs["aws_secret_access_key"] = secret_access_key
    if region_name:
        kwargs["region_name"] = region_name
    kwargs["verify"] = verify
    return boto3.client("s3", **kwargs)


def _s3_error_message(exc: Exception, *, bucket: str, key: str) -> str:
    response = getattr(exc, "response", None)
    error = response.get("Error", {}) if isinstance(response, dict) else {}
    code = error.get("Code") or exc.__class__.__name__
    message = error.get("Message") or str(exc)
    return (
        "对象存储文件下载失败: "
        f"bucket={bucket}, key={key}, code={code}, message={message}. "
        "请确认 manifest 中的 url 和实际上传到对象存储的 key 完全一致。"
    )


def _download_bytes(url: str) -> bytes:
    s3_location = _parse_s3_url(url)
    if s3_location is not None:
        bucket, key = s3_location
        try:
            response = _s3_client().get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            try:
                return body.read()
            finally:
                body.close()
        except Exception as exc:
            raise RuntimeError(_s3_error_message(exc, bucket=bucket, key=key)) from exc

    request = urllib.request.Request(url, headers={"User-Agent": "CrabCLI-Updater/1"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read()


def _download_json(url: str) -> dict[str, Any]:
    data = _download_bytes(url)
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be a JSON object")
    return payload


def _is_newer(latest: str, current: str) -> bool:
    try:
        return Version(latest) > Version(current)
    except InvalidVersion:
        return latest != current


def _notes_from_manifest(manifest: dict[str, Any]) -> list[str]:
    raw_notes = manifest.get("notes")
    if not isinstance(raw_notes, list):
        return []
    return [item.strip() for item in raw_notes if isinstance(item, str) and item.strip()]


def _rich_console(*, stderr: bool = False) -> Any | None:
    try:
        from rich.console import Console
    except ImportError:
        return None
    return Console(stderr=stderr)


def _print_styled(parts: list[tuple[str, str]], *, stderr: bool = False, end: str = "\n") -> None:
    console = _rich_console(stderr=stderr)
    if console is None:
        stream = sys.stderr if stderr else sys.stdout
        print("".join(text for text, _style in parts), end=end, file=stream)
        return

    from rich.text import Text

    line = Text()
    for text, style in parts:
        line.append(text, style=style)
    console.print(line, end=end)


def _print_white(message: str = "", *, stderr: bool = False) -> None:
    if message:
        _print_styled([(message, "white")], stderr=stderr)
        return
    print(file=sys.stderr if stderr else sys.stdout)


def _prompt_choice() -> str:
    _print_styled([("请输入选项 [1/2]: ", "white")], end="")
    return input().strip()


def check_for_update(*, force_message: bool = False) -> UpdateInfo | None:
    _load_update_env()
    manifest_url = _manifest_url()
    current_version = get_cli_version()
    if manifest_url is None:
        if force_message:
            _print_styled(
                [
                    ("未配置更新地址。请设置 ", "white"),
                    (MANIFEST_URL_ENV, "yellow"),
                    (" 指向 stable.json。", "white"),
                ]
            )
        return None

    manifest = _download_json(manifest_url)
    latest = manifest.get("latest")
    if not isinstance(latest, str) or not latest.strip():
        raise ValueError("manifest.latest is required")
    latest = latest.strip()

    state = load_update_state()
    state["last_check_at"] = _now_iso()
    state["last_seen_version"] = latest
    save_update_state(state)

    if not _is_newer(latest, current_version):
        if force_message:
            _print_styled([("当前已是最新版本：v", "white"), (current_version, "green")])
        return None

    releases = manifest.get("releases")
    if not isinstance(releases, dict):
        raise ValueError("manifest.releases is required")
    platform_key = _platform_key()
    release = releases.get(platform_key)
    if not isinstance(release, dict):
        raise ValueError(f"manifest does not contain release for platform: {platform_key}")

    return UpdateInfo(
        current_version=current_version,
        latest_version=latest,
        published_at=manifest.get("published_at") if isinstance(manifest.get("published_at"), str) else None,
        notes=_notes_from_manifest(manifest),
        release=release,
        manifest=manifest,
    )


def _print_update_info(info: UpdateInfo) -> None:
    _print_styled([("✦ ", "bold #c084fc"), ("发现 Crab CLI 新版本", "bold white")])
    _print_white()
    _print_styled([("当前版本: ", "white"), (info.current_version, "yellow")])
    _print_styled([("最新版本: ", "white"), (info.latest_version, "bold green")])
    if info.published_at:
        _print_styled([("发布时间: ", "white"), (info.published_at, "white")])
    if info.notes:
        _print_white()
        _print_styled([("更新内容:", "bold white")])
        for note in info.notes:
            _print_styled([("  - ", "white"), (note, "white")])


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    s3_location = _parse_s3_url(url)
    if s3_location is not None:
        bucket, key = s3_location
        try:
            response = _s3_client().get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            try:
                with destination.open("wb") as output:
                    shutil.copyfileobj(body, output)
            finally:
                body.close()
        except Exception as exc:
            raise RuntimeError(_s3_error_message(exc, bucket=bucket, key=key)) from exc
        return

    request = urllib.request.Request(url, headers={"User-Agent": "CrabCLI-Updater/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_tar(archive: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    root = target_dir.resolve()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            member_path = (target_dir / member.name).resolve()
            if root not in (member_path, *member_path.parents):
                raise ValueError(f"unsafe archive path: {member.name}")
        if sys.version_info >= (3, 12):
            tar.extractall(target_dir, filter="data")
        else:
            tar.extractall(target_dir)


def _extract_archive(archive: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        root = target_dir.resolve()
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                member_path = (target_dir / member).resolve()
                if root not in (member_path, *member_path.parents):
                    raise ValueError(f"unsafe archive path: {member}")
            zf.extractall(target_dir)
        return
    _safe_extract_tar(archive, target_dir)


def _find_bundle_root(staging_dir: Path) -> Path:
    candidates = [path for path in staging_dir.iterdir() if path.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    return staging_dir


def _deploy_script(bundle_root: Path) -> Path:
    if platform.system().lower() == "windows":
        script = bundle_root / "deploy.bat"
    else:
        script = bundle_root / "deploy.sh"
    if not script.exists():
        raise FileNotFoundError(f"deploy script not found: {script}")
    return script


def _acquire_update_lock() -> bool:
    lock_path = update_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
        json.dump({"pid": os.getpid(), "started_at": _now_iso()}, lock_file)
    return True


def _release_file_name(info: UpdateInfo) -> str:
    raw = info.release.get("file")
    if isinstance(raw, str) and raw.strip():
        return Path(raw.strip()).name
    url = info.release.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("release.url is required")
    return Path(urllib.parse.urlparse(url).path).name


def _installed_python_path() -> Path:
    active_release_path = tg_agent_home() / "active-release.txt"
    try:
        active_release = active_release_path.read_text(encoding="ascii").strip()
    except OSError:
        active_release = ""

    if platform.system().lower() == "windows":
        if active_release:
            return tg_agent_home() / "releases" / active_release / ".venv" / "Scripts" / "python.exe"
        return tg_agent_home() / ".venv" / "Scripts" / "python.exe"
    if active_release:
        return tg_agent_home() / "releases" / active_release / ".venv" / "bin" / "python"
    return tg_agent_home() / ".venv" / "bin" / "python"


def _write_pending_deploy_script(bundle_root: Path, deploy_script: Path, version: str) -> Path:
    root = updates_root()
    if platform.system().lower() == "windows":
        pending_script = root / "pending_update.ps1"
        content = (
            '$ErrorActionPreference = "Stop"\n'
            f'$DeployScript = {json.dumps(str(deploy_script))}\n'
            f'$BundleRoot = {json.dumps(str(bundle_root))}\n'
            f'$LockPath = {json.dumps(str(update_lock_path()))}\n'
            f'$VenvPython = {json.dumps(str(_installed_python_path()))}\n'
            f'$TargetVersion = {json.dumps(version)}\n'
            "try {\n"
            "  Push-Location $BundleRoot\n"
            "  & $DeployScript --update\n"
            "  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n"
            "  & $VenvPython -m agent_core.updater mark-installed --version $TargetVersion\n"
            "}\n"
            "finally {\n"
            "  Pop-Location\n"
            "  Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue\n"
            "}\n"
        )
        pending_script.write_text(content, encoding="utf-8")
        return pending_script

    pending_script = root / "pending_update.sh"
    content = (
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        f"deploy_script={json.dumps(str(deploy_script))}\n"
        f"bundle_root={json.dumps(str(bundle_root))}\n"
        f"lock_path={json.dumps(str(update_lock_path()))}\n"
        f"venv_python={json.dumps(str(_installed_python_path()))}\n"
        f"target_version={json.dumps(version)}\n"
        "cleanup() { rm -f \"${lock_path}\"; }\n"
        "trap cleanup EXIT\n"
        "cd \"${bundle_root}\"\n"
        "\"${deploy_script}\" --update\n"
        "\"${venv_python}\" -m agent_core.updater mark-installed --version \"${target_version}\"\n"
    )
    pending_script.write_text(content, encoding="utf-8")
    pending_script.chmod(0o755)
    return pending_script


def prepare_update_before_launch(info: UpdateInfo) -> Path:
    if not _acquire_update_lock():
        _print_styled([("Crab 正在更新中，请稍后重新启动。", "yellow")])
        raise SystemExit(10)
    try:
        url = info.release.get("url")
        expected_sha = info.release.get("sha256")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("release.url is required")
        if not isinstance(expected_sha, str) or not expected_sha.strip():
            raise ValueError("release.sha256 is required")
        expected_sha = expected_sha.strip().lower()

        root = updates_root()
        downloads_dir = root / "downloads"
        staging_root = root / "staging"
        staging_dir = root / "staging" / info.latest_version
        file_name = _release_file_name(info)
        archive_path = downloads_dir / file_name

        if downloads_dir.exists():
            shutil.rmtree(downloads_dir)
        downloads_dir.mkdir(parents=True, exist_ok=True)
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)

        _print_styled([("正在下载: ", "white"), (url, "white")])
        _download_file(url, archive_path)
        actual_sha = _sha256(archive_path).lower()
        if actual_sha != expected_sha:
            raise ValueError(f"sha256 mismatch: expected {expected_sha}, got {actual_sha}")

        _print_styled([("正在解压: ", "white"), (str(archive_path), "white")])
        _extract_archive(archive_path, staging_dir)
        bundle_root = _find_bundle_root(staging_dir)
        deploy_script = _deploy_script(bundle_root)

        pending_script = _write_pending_deploy_script(bundle_root, deploy_script, info.latest_version)
        state = load_update_state()
        state["pending_update"] = {
            "version": info.latest_version,
            "prepared_at": _now_iso(),
            "script": str(pending_script),
        }
        save_update_state(state)
        _print_styled([("更新已准备好: ", "bold green"), (str(pending_script), "white")])
        return pending_script
    except Exception:
        try:
            update_lock_path().unlink()
        except FileNotFoundError:
            pass
        raise


def check_before_launch() -> int:
    _load_update_env()
    if os.environ.get(SKIP_UPDATE_CHECK_ENV) == "1":
        return 0
    if load_update_state().get("auto_check_enabled") is False:
        return 0
    if update_lock_path().exists():
        _print_styled([("Crab 正在更新中，请稍后重新启动。", "yellow")])
        return 10
    try:
        info = check_for_update(force_message=False)
    except Exception as exc:
        _print_styled(
            [("检查更新失败，继续启动当前版本: ", "bold red"), (str(exc), "white")],
            stderr=True,
        )
        return 0
    if info is None:
        return 0

    _print_update_info(info)
    if not sys.stdin.isatty():
        _print_white()
        _print_white("请关闭所有 Crab 后重新启动以更新。")
        return 0

    _print_white()
    _print_styled([("请选择:", "bold white")])
    _print_styled([("  1.", "bold green"), (" 立即更新", "white")])
    _print_styled([("  2.", "bold yellow"), (" 跳过本次，继续启动", "white")])
    _print_white()
    choice = _prompt_choice()
    if choice != "1":
        return 0

    try:
        prepare_update_before_launch(info)
        return 20
    except Exception as exc:
        _print_styled(
            [("更新失败，继续启动当前版本: ", "bold red"), (str(exc), "white")],
            stderr=True,
        )
        return 0
    return 10


def run_check() -> int:
    try:
        info = check_for_update(force_message=True)
    except Exception as exc:
        _print_styled([("检查更新失败: ", "bold red"), (str(exc), "white")])
        return 1
    if info is None:
        return 0
    _print_update_info(info)
    _print_white()
    _print_white("请关闭所有 Crab 后重新启动以更新。")
    return 0


def run_status() -> int:
    current_version = get_cli_version()
    state = load_update_state()
    last_check_at = state.get("last_check_at") or "-"
    last_install = state.get("last_install")
    last_updated_at = "-"
    if isinstance(last_install, dict):
        last_updated_at = str(last_install.get("finished_at") or "-")
    _print_styled([("当前版本: ", "white"), (current_version, "yellow")])
    _print_styled([("上次检查时间: ", "white"), (str(last_check_at), "white")])
    _print_styled([("上一次成功更新时间: ", "white"), (last_updated_at, "white")])
    return 0


def mark_installed(version: str) -> int:
    state = load_update_state()
    state["last_install"] = {
        "version": version,
        "status": "success",
        "finished_at": _now_iso(),
    }
    state.pop("pending_update", None)
    save_update_state(state)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crab CLI updater")
    parser.add_argument("command", choices=("check-before-launch", "check", "status", "mark-installed"))
    parser.add_argument("--version")
    args = parser.parse_args(argv)
    if args.command == "check-before-launch":
        return check_before_launch()
    if args.command == "check":
        return run_check()
    if args.command == "status":
        return run_status()
    if args.command == "mark-installed":
        if not args.version:
            parser.error("--version is required for mark-installed")
        return mark_installed(args.version)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
