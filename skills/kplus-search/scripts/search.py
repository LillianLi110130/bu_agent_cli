#!/usr/bin/env python3
"""Query the K+ knowledge base."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_SERVICE_ID = "123"


def _load_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON 解析失败: {path}") from exc
    except OSError as exc:
        raise RuntimeError(f"读取文件失败: {path}") from exc


def _candidate_config_paths() -> list[Path]:
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    return [
        cwd / "tg_crab_worker.json",
        home / ".tg_agent" / "tg_crab_worker.json",
    ]


def _resolve_server_host() -> str:
    for path in _candidate_config_paths():
        if not path.exists():
            continue
        payload = _load_json(path)
        value = payload.get("server_host")
        if isinstance(value, str) and value.strip():
            return value.strip().rstrip("/")
    raise RuntimeError("未找到 server_host，请检查 tg_crab_worker.json")


def _candidate_token_paths() -> list[Path]:
    home = Path.home().resolve()
    return [
        home / ".tg_agent" / "token.json",
    ]


def _resolve_authorization() -> str:
    for path in _candidate_token_paths():
        if not path.exists():
            continue
        payload = _load_json(path)
        value = payload.get("authorization")
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise RuntimeError("未找到 authorization，请先完成登录并生成 ~/.tg_agent/token.json")


def _post_json(url: str, authorization: str, payload: dict[str, object]) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": authorization,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json, text/plain;q=0.9, */*;q=0.8",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body


def _search(question: str, serviceId: str) -> tuple[int, str]:
    base_url = _resolve_server_host()
    authorization = _resolve_authorization()
    url = f"{base_url}/cli/kplus/search"
    payload = {"question": question, "serviceId": serviceId}

    return _post_json(url, authorization, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Search K+ knowledge base")
    parser.add_argument("--question", required=True, help="检索问题")
    parser.add_argument(
        "--serviceId",
        default=DEFAULT_SERVICE_ID,
        help=f"知识库 serviceId，默认 {DEFAULT_SERVICE_ID}",
    )
    args = parser.parse_args()

    try:
        status, body = _search(args.question, args.serviceId)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Error: 网络请求失败: {exc}", file=sys.stderr)
        return 1

    if status >= 400:
        print(f"Error: 请求失败，状态码 {status}", file=sys.stderr)
        if body.strip():
            print(body, file=sys.stderr)
        return 1

    text = body.strip()
    if not text:
        print("{}", end="")
        return 0

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        print(text)
        return 0

    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
