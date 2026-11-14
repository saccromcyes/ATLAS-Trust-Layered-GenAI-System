"""Publish this repository to a Hugging Face Space (Docker SDK) and wait until it is live.

Creates the Space if needed, syncs the LLM key into the Space's secrets, uploads the code with a
README that carries the Space metadata, then follows the build until the app answers its health
check. On a failed build it prints the Space's build log. Used by .github/workflows/deploy.yml.

    HF_TOKEN=... [HF_SPACE=username/name] [LLM_API_KEY=...] python scripts/deploy_space.py

Without HF_SPACE the Space is created under the token's own account as SPACE_NAME.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
SPACE_NAME = "atlas"
HEALTH_PATH = "/api/health"
SPACE_HEADER = """---
title: ATLAS
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: RAG that cites sources and refuses to guess
---

"""
# Variables that are safe to expose as Space settings; the API key goes in as a secret.
PUBLIC_VARS = ["LLM_BASE_URL", "LLM_MODEL", "LLM_FAST_MODEL", "MIN_RELEVANCE", "MIN_GROUNDEDNESS"]
FAILED = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}


def log(msg: str) -> None:
    print(msg, flush=True)


def github_output(**values: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            for key, value in values.items():
                f.write(f"{key}={value}\n")


def summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write(text + "\n")


def print_logs(space: str, token: str, kind: str, max_lines: int = 120) -> None:
    """The Space log endpoints stream server-sent events; read what is there and stop."""
    req = urllib.request.Request(
        f"https://huggingface.co/api/spaces/{space}/logs/{kind}", headers={"Authorization": f"Bearer {token}"}
    )
    lines: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            while len(lines) < 2000:
                raw = resp.readline()
                if not raw:
                    break
                text = raw.decode("utf-8", "replace").strip()
                if text.startswith("data:"):
                    try:
                        lines.append(json.loads(text[5:]).get("data", "").rstrip())
                    except json.JSONDecodeError:
                        lines.append(text[5:].strip())
    except Exception as exc:  # the stream stays open; a timeout just means we have everything so far
        if not lines:
            log(f"(could not read {kind} log: {exc})")
    log(f"----- last {max_lines} lines of the {kind} log -----")
    for line in lines[-max_lines:]:
        log(line)


def wait_until_live(api: HfApi, space: str, token: str, url: str, timeout_s: int = 1800) -> bool:
    start, seen_change, last = time.time(), False, None
    time.sleep(20)
    while time.time() - start < timeout_s:
        try:
            stage = str(api.get_space_runtime(space).stage).split(".")[-1]
        except Exception as exc:
            log(f"Could not read Space status ({exc}); retrying")
            time.sleep(20)
            continue
        if stage != last:
            log(f"[{int(time.time() - start):>4}s] Space is {stage}")
            last = stage
        seen_change |= stage != "RUNNING"
        if stage in FAILED:
            print_logs(space, token, "build")
            if stage == "RUNTIME_ERROR":
                print_logs(space, token, "run")
            return False
        # A RUNNING status seen before the rebuild starts belongs to the previous version.
        if stage == "RUNNING" and (seen_change or time.time() - start > 120):
            for _ in range(12):  # the container can need a moment after RUNNING to accept requests
                try:
                    with urllib.request.urlopen(url + HEALTH_PATH, timeout=10) as resp:
                        if resp.status == 200:
                            log(f"Health check passed: {url}{HEALTH_PATH}")
                            return True
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(10)
            log("The Space says RUNNING but the app does not answer its health check.")
            print_logs(space, token, "run")
            return False
        time.sleep(20)
    log("Timed out waiting for the Space to start.")
    print_logs(space, token, "build")
    return False


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is required", file=sys.stderr)
        return 1
    api = HfApi(token=token)
    try:
        user = api.whoami()["name"]
    except Exception as exc:
        print(f"The HF_TOKEN was rejected by Hugging Face ({exc}). Create a new token with Write access.", file=sys.stderr)
        return 1
    space = os.environ.get("HF_SPACE") or f"{user}/{SPACE_NAME}"
    log(f"Deploying to Space {space} as {user}")

    try:
        api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    except Exception as exc:
        print(f"Could not create the Space ({exc}). The HF_TOKEN needs Write access.", file=sys.stderr)
        return 1

    if os.environ.get("LLM_API_KEY"):
        api.add_space_secret(space, "LLM_API_KEY", os.environ["LLM_API_KEY"])
    else:
        log("LLM_API_KEY is not set; the app will run without a language model.")
    for name in PUBLIC_VARS:
        if os.environ.get(name):
            api.add_space_variable(space, name, os.environ[name])

    with tempfile.TemporaryDirectory() as tmp:
        # Only files tracked by git are published (no local caches, .env or databases).
        files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split()
        for rel in files:
            if rel.startswith((".github/", "tests/", "docs/")):
                continue
            dest = Path(tmp) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dest)
        readme = Path(tmp) / "README.md"
        readme.write_text(SPACE_HEADER + readme.read_text())

        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        api.upload_folder(
            folder_path=tmp, repo_id=space, repo_type="space", commit_message=f"Deploy {sha} from GitHub", delete_patterns=["*"]
        )

    page = f"https://huggingface.co/spaces/{space}"
    info = api.space_info(space)
    host = (
        getattr(info, "host", None) or f"https://{space.replace('/', '-').replace('_', '-').replace('.', '-').lower()}.hf.space"
    )
    host = host if host.startswith("http") else f"https://{host}"
    github_output(url=page, app=host)
    log(f"Uploaded. Space page: {page}  App: {host}")

    if not wait_until_live(api, space, token, host):
        summary(f"Deploy of `{sha}` to [{space}]({page}) **failed**. See the build log in this job.")
        return 1
    summary(f"Deployed `{sha}`. Live at [{page}]({page}) (direct app link: {host}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
