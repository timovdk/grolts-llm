"""Submit the exported gpt-5-mini request files to the OpenAI Batch API and collect them.

    # OPENAI_API_KEY comes from the repo-root .env (see .env.example), or the environment
    python submit_openai_batches.py              # dry run: show what would be sent
    python submit_openai_batches.py --submit     # send them

Batches take up to 24 hours. A manifest is written as each batch is created, so the script
can be stopped and restarted without resubmitting anything::

    python submit_openai_batches.py --collect    # resume: poll and download only

Batches are kept **in flight a few at a time** rather than all twenty at once. The Batch API
caps how many tokens an account may have enqueued per model, and these are large requests --
roughly 10M input tokens each, so all twenty at once is ~200M enqueued tokens and would very
likely be rejected. As each batch completes the next is submitted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from pipeline_config import ENV_FILE

EXPORT_DIR = Path("../packages/openai_batches")
OUTPUT_PATH = Path("../eval/batches_out")
MANIFEST = OUTPUT_PATH / "openai_batches_manifest.json"
POLL_SECONDS = 120
TERMINAL = {"completed", "failed", "expired", "cancelled"}


def find_requests(export_dir: Path) -> list[Path]:
    """Every exported request file, sorted so runs are submitted in a predictable order."""
    return sorted(export_dir.glob("*/*.jsonl"))


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def save_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def submit_one(client, path: Path) -> dict:
    """Upload one request file and enqueue it."""
    with open(path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": path.stem},
    )
    return {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "output": f"{path.stem}.jsonl",
        "status": batch.status,
    }


def download(client, entry: dict) -> bool:
    """Write a finished batch's output next to the other model outputs."""
    batch = client.batches.retrieve(entry["batch_id"])
    if not batch.output_file_id:
        return False
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_PATH / entry["output"]
    target.write_text(client.files.content(batch.output_file_id).text)
    print(f"[INFO] downloaded {target.name}")
    return True


def output_for(path: Path) -> Path:
    """Where a request file's result is written: the input's own name."""
    return OUTPUT_PATH / f"{path.stem}.jsonl"


def already_done(entry: dict) -> bool:
    """Whether a manifest entry's output is on disk.

    Guard the empty name: ``OUTPUT_PATH / ""`` is the directory itself, which always
    exists, so an entry without an output would otherwise look complete.
    """
    name = entry.get("output")
    return bool(name) and (OUTPUT_PATH / name).exists()


def run(client, requests: list[Path], manifest: dict, args) -> int:
    """Keep a few batches in flight, submitting more as they finish."""
    queue = [p for p in requests if not output_for(p).exists() and p.stem not in manifest]
    in_flight = {
        name: entry
        for name, entry in manifest.items()
        if not already_done(entry) and entry.get("status") not in {"failed", "expired", "cancelled"}
    }
    failures: list[str] = []

    while queue or in_flight:
        while queue and len(in_flight) < args.max_in_flight:
            path = queue.pop(0)
            entry = submit_one(client, path)
            manifest[path.stem] = entry
            in_flight[path.stem] = entry
            save_manifest(manifest)
            print(f"[INFO] submitted {path.stem}  -> {entry['batch_id']}", flush=True)

        if not in_flight:
            break

        time.sleep(POLL_SECONDS)

        for name, entry in list(in_flight.items()):
            batch = client.batches.retrieve(entry["batch_id"])
            entry["status"] = batch.status
            if batch.status not in TERMINAL:
                counts = getattr(batch, "request_counts", None)
                progress = f" {counts.completed}/{counts.total}" if counts else ""
                print(f"[INFO] {name}: {batch.status}{progress}", flush=True)
                continue

            del in_flight[name]
            if batch.status == "completed" and download(client, entry):
                entry["status"] = "downloaded"
            else:
                print(f"[ERROR] {name}: {batch.status}", file=sys.stderr)
                failures.append(name)
            save_manifest(manifest)

    if failures:
        print(f"\n[ERROR] {len(failures)} batch(es) did not complete: {', '.join(failures)}")
        print("        Re-run with --submit to retry them; completed ones are skipped.")
        return 1

    print(f"\n[OK] all {len(requests)} batches downloaded to {OUTPUT_PATH}")
    print("     next: python ../eval/process_batch_result.py")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--submit", action="store_true", help="actually send the batches")
    parser.add_argument(
        "--collect", action="store_true",
        help="do not submit anything new; poll and download what the manifest already lists",
    )
    parser.add_argument(
        "--max-in-flight", type=int, default=4,
        help="how many batches to keep queued at once (default 4)",
    )
    parser.add_argument("--export-dir", default=str(EXPORT_DIR))
    args = parser.parse_args(argv)

    export_dir = Path(args.export_dir)
    requests = find_requests(export_dir)
    if not requests:
        print(f"[ERROR] no request files under {export_dir}", file=sys.stderr)
        return 1

    manifest = load_manifest()
    total_mb = sum(p.stat().st_size for p in requests) / 1e6
    print(f"{len(requests)} request files in {export_dir} ({total_mb:,.0f} MB)\n")
    for path in requests:
        entry = manifest.get(path.stem)
        if output_for(path).exists():
            state = "done"
        elif entry:
            state = f"in flight ({entry.get('status', '?')})"
        else:
            state = "not submitted"
        n = sum(1 for _ in path.open(encoding="utf-8"))
        print(f"  {path.parent.name:<22} {path.stem[-28:]:<30} {n:>5} requests   {state}")

    outstanding = [p for p in requests if not output_for(p).exists()]
    if not outstanding:
        print("\n[OK] every batch has already been downloaded; nothing to do.")
        return 0

    if not (args.submit or args.collect):
        print(
            f"\n{len(outstanding)} batch(es) outstanding. This is a dry run — nothing was sent."
            "\nRe-run with --submit to send them, or --collect to poll ones already submitted."
        )
        return 0


    if not os.environ.get("OPENAI_API_KEY"):
        print(
            f"[ERROR] OPENAI_API_KEY is not set (looked in the environment and {ENV_FILE})",
            file=sys.stderr,
        )
        return 1

    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        print(
            "[ERROR] the openai package is not installed.\n"
            "        uv sync --only-group eval   (or: pip install openai)",
            file=sys.stderr,
        )
        return 1

    client = OpenAI()
    if args.collect:
        # Nothing new goes out; only what the manifest already knows about is polled.
        requests = [p for p in requests if p.stem in manifest]
        if not requests:
            print("[ERROR] nothing in the manifest to collect", file=sys.stderr)
            return 1

    return run(client, requests, manifest, args)


if __name__ == "__main__":
    raise SystemExit(main())
