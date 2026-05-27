from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def _read_project_version(repo_root: Path) -> str:
    pyproject = repo_root / "pyproject.toml"
    match = re.search(
        r'(?m)^version\s*=\s*"([^"]+)"\s*$',
        pyproject.read_text(encoding="utf-8"),
    )
    if match is None:
        raise SystemExit(f"Could not find project version in {pyproject}")
    return match.group(1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return dict(default or {})
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Crab update manifests")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--platform", required=True, choices=("linux-x64", "windows-x64"))
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--base-url", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    artifact = Path(args.artifact).resolve()
    if not artifact.exists():
        raise SystemExit(f"Artifact not found: {artifact}")

    project_version = _read_project_version(repo_root)
    if project_version != args.version:
        raise SystemExit(
            f"Version mismatch: pyproject.toml has {project_version}, build version is {args.version}"
        )

    release_notes_path = repo_root / "config" / "release_notes.json"
    release_notes = _read_json(release_notes_path)
    if release_notes.get("version") != args.version:
        raise SystemExit(
            f"Version mismatch: {release_notes_path} has {release_notes.get('version')}, "
            f"build version is {args.version}"
        )
    notes = release_notes.get("notes")
    if not isinstance(notes, list) or not all(isinstance(item, str) and item.strip() for item in notes):
        raise SystemExit(f"{release_notes_path} must contain a non-empty notes array")
    published_at = release_notes.get("published_at")
    if not isinstance(published_at, str) or not published_at.strip():
        raise SystemExit(f"{release_notes_path} must contain published_at")

    artifact_sha = _sha256(artifact)
    sha_path = artifact.with_name(f"{artifact.name}.sha256")
    sha_path.write_text(f"{artifact_sha}  {artifact.name}\n", encoding="utf-8")

    base_url = args.base_url.rstrip("/")
    artifact_url = (
        f"{base_url}/releases/{args.version}/{artifact.name}" if base_url else artifact.name
    )

    update_root = output_root / "update"
    release_manifest_path = update_root / "releases" / args.version / "manifest.json"
    manifest = _read_json(
        release_manifest_path,
        {
            "schema": 1,
            "version": args.version,
            "published_at": published_at,
            "releases": {},
            "notes": notes,
        },
    )
    manifest["schema"] = 1
    manifest["version"] = args.version
    manifest["published_at"] = published_at
    manifest["notes"] = notes
    manifest.setdefault("releases", {})
    manifest["releases"][args.platform] = {
        "file": artifact.name,
        "url": artifact_url,
        "sha256": artifact_sha,
        "size": artifact.stat().st_size,
    }
    _write_json(release_manifest_path, manifest)

    stable_manifest = {
        "schema": 1,
        "channel": "stable",
        "latest": args.version,
        "published_at": manifest["published_at"],
        "releases": {
            key: {field: value for field, value in release.items() if field != "file"}
            for key, release in manifest["releases"].items()
        },
        "notes": manifest["notes"],
    }
    stable_path = update_root / "channels" / "stable.json"
    _write_json(stable_path, stable_manifest)

    print(f"[portable] sha256: {sha_path}")
    print(f"[portable] release manifest: {release_manifest_path}")
    print(f"[portable] stable manifest: {stable_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
