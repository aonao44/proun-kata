#!/usr/bin/env python3
"""Batch tester for /transcribe_phonetic API."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
from requests import Response
from tqdm import tqdm

DEFAULT_ROOT = Path("audio_samples")
DEFAULT_PATTERN = "**/*.wav"
DEFAULT_ARTIFACT_DIR = Path("artifacts/json")
DEFAULT_RESULTS_CSV = Path("artifacts/batch_results.csv")


@dataclass
class TestResult:
    path: Path
    status_code: int
    passed: bool
    phones_len: int | None
    kana_len: int | None
    kana_text: str
    latency_ms: float
    error: str | None = None
    raw_json: dict | None = None

    @property
    def pass_flag(self) -> str:
        return "1" if self.passed else "0"


def discover_files(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")
    files = sorted(p for p in root.glob(pattern) if p.is_file())
    return files


def send_request(
    session: requests.Session,
    base_url: str,
    path: Path,
    timeout: float,
) -> tuple[Response, float]:
    url = base_url.rstrip("/") + "/transcribe_phonetic"
    start = time.perf_counter()
    with path.open("rb") as fh:
        response = session.post(
            url,
            files={"audio": (path.name, fh, "audio/wav")},
            timeout=timeout,
        )
    latency_ms = (time.perf_counter() - start) * 1000
    return response, latency_ms


def evaluate_response(response: Response, latency_ms: float, path: Path) -> TestResult:
    try:
        data = response.json()
    except ValueError:
        return TestResult(
            path=path,
            status_code=response.status_code,
            passed=False,
            phones_len=None,
            kana_len=None,
            kana_text="",
            latency_ms=latency_ms,
            error="Non-JSON response",
        )

    phones = data.get("phones")
    kana = data.get("kana")
    kana_text = data.get("kana_text") or ""

    phones_len = len(phones) if isinstance(phones, Iterable) else None
    kana_len = len(kana) if isinstance(kana, Iterable) else None

    passed = (
        response.status_code == 200
        and isinstance(phones_len, int)
        and phones_len >= 1
        and isinstance(kana_text, str)
        and kana_text.strip() != ""
    )

    error = None
    if not passed:
        error = data.get("detail") if isinstance(data.get("detail"), str) else None

    return TestResult(
        path=path,
        status_code=response.status_code,
        passed=passed,
        phones_len=phones_len,
        kana_len=kana_len,
        kana_text=kana_text,
        latency_ms=latency_ms,
        error=error,
        raw_json=data,
    )


def process_file(
    path: Path,
    base_url: str,
    timeout: float,
    save_json: bool,
    json_dir: Path,
) -> TestResult:
    session = requests.Session()
    try:
        response, latency_ms = send_request(session, base_url, path, timeout)
        result = evaluate_response(response, latency_ms, path)
        if save_json and result.raw_json is not None:
            json_dir.mkdir(parents=True, exist_ok=True)
            output_path = json_dir / f"{path.stem}.json"
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(result.raw_json, fh, ensure_ascii=False, indent=2)
        return result
    except Exception as exc:  # noqa: BLE001
        return TestResult(
            path=path,
            status_code=0,
            passed=False,
            phones_len=None,
            kana_len=None,
            kana_text="",
            latency_ms=0.0,
            error=str(exc),
        )
    finally:
        session.close()


def run_batch(
    files: list[Path],
    base_url: str,
    timeout: float,
    max_workers: int,
    save_json: bool,
    json_dir: Path,
) -> list[TestResult]:
    if not files:
        print("No files matched the criteria.")
        return []

    results: list[TestResult] = []

    if max_workers <= 1:
        for path in tqdm(files, desc="Testing", unit="file"):
            results.append(process_file(path, base_url, timeout, save_json, json_dir))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(process_file, path, base_url, timeout, save_json, json_dir): path
                for path in files
            }
            for future in tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc="Testing",
                unit="file",
            ):
                results.append(future.result())

    return results


def write_csv(results: list[TestResult], output_csv: Path, root: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "file",
                "status_code",
                "pass",
                "phones_len",
                "kana_text",
            ]
        )
        for result in sorted(results, key=lambda r: str(r.path)):
            try:
                relative = result.path.relative_to(root)
            except ValueError:
                relative = result.path
            writer.writerow(
                [
                    str(relative),
                    result.status_code,
                    result.pass_flag,
                    result.phones_len if result.phones_len is not None else "",
                    result.kana_text,
                ]
            )


def summarize(results: list[TestResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    success_rate = (passed / len(results) * 100) if results else 0.0

    print("\n=== Summary ===")
    print(f"Total: {len(results)}")
    print(f"PASS : {passed}")
    print(f"FAIL : {failed}")
    print(f"Success Rate: {success_rate:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch tester for /transcribe_phonetic API")
    parser.add_argument(
        "--base",
        dest="base",
        default=os.environ.get("TRANSCRIBE_BASE_URL", "http://localhost:8000"),
        help="Base URL of the API (default: env TRANSCRIBE_BASE_URL or http://localhost:8000)",
    )
    parser.add_argument(
        "--root",
        dest="root",
        default=str(DEFAULT_ROOT),
        help=f"Root directory containing WAV files (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--pattern",
        dest="pattern",
        default=DEFAULT_PATTERN,
        help="Glob pattern relative to root (default: **/*.wav)",
    )
    parser.add_argument(
        "--timeout",
        dest="timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--results",
        dest="results",
        default=str(DEFAULT_RESULTS_CSV),
        help=f"Path to CSV output (default: {DEFAULT_RESULTS_CSV})",
    )
    parser.add_argument(
        "--max-workers",
        dest="max_workers",
        type=int,
        default=1,
        help="Number of concurrent workers",
    )
    parser.add_argument(
        "--save-json",
        dest="save_json",
        action="store_true",
        help=f"Save JSON responses under {DEFAULT_ARTIFACT_DIR}",
    )
    parser.add_argument(
        "--json-dir",
        dest="json_dir",
        default=str(DEFAULT_ARTIFACT_DIR),
        help=(
            "Directory for JSON artifacts when --save-json is set "
            f"(default: {DEFAULT_ARTIFACT_DIR})"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root = Path(args.root).expanduser().resolve()
    pattern = args.pattern

    try:
        files = discover_files(root, pattern)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not files:
        print("No matching WAV files found.")
        return 1

    json_dir = Path(args.json_dir).expanduser().resolve()
    if not args.save_json:
        json_dir = Path(json_dir)

    results = run_batch(
        files=files,
        base_url=args.base,
        timeout=args.timeout,
        max_workers=max(1, args.max_workers),
        save_json=args.save_json,
        json_dir=json_dir,
    )

    results_csv = Path(args.results).expanduser().resolve()
    write_csv(results, results_csv, root)
    summarize(results)

    print(f"CSV written to {results_csv}")
    if args.save_json:
        print(f"JSON artifacts saved under {json_dir}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
