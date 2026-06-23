#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Async benchmark client for the sglang-omni SpeechifyTTS server.

Fires PCM-streaming requests at ``POST /v1/audio/speech`` and reports the same
TTFA / E2E / RTF / throughput summary as the vllm-omni baseline benchmark, so
the two stacks can be compared apples-to-apples.

The reference audio is uploaded once via ``POST /v1/audio/upload_reference``
and the returned server-side path is reused for every request.

Usage::

    # single cell
    python examples/speechify_tts/bench.py \
        --api-base http://127.0.0.1:8030 \
        --length short --concurrency 4 --num-requests 32 \
        --out /tmp/bench_h100/sglang/short_c4.json

    # full matrix {short,medium,long} x {1,2,4,8}
    python examples/speechify_tts/bench.py --matrix \
        --api-base http://127.0.0.1:8030 \
        --num-requests 32 --out-dir /tmp/bench_h100/sglang
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2

DEFAULT_PROMPT_DIR = (
    "/home/kevin/speechify-vllm/packages/vllm-omni/examples/prompt_benchmark"
)
DEFAULT_REF_AUDIO = (
    "/home/kevin/vllm-omni/serve_logs/regress_7lang_cg_20260619/en-us/geffen__0026.wav"
)


# --- helpers ---------------------------------------------------------------

def _load_prompts(length: str, prompt_dir: str) -> list[str]:
    path = os.path.join(prompt_dir, f"{length}_prompts.txt")
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if len(line) < 2 or line.startswith("# "):
                continue
            out.append(line)
    if not out:
        raise ValueError(f"prompt file {path} has no usable lines")
    return out


def _pct(xs: list[float], p: float) -> float:
    """Nearest-rank percentile (same convention as the vllm-omni benchmark)."""
    if not xs:
        return 0.0
    k = max(0, min(len(xs) - 1, int(round((len(xs) - 1) * p))))
    return xs[k]


async def _upload_reference(session, api_base: str, ref_audio: str) -> str:
    """Upload the local reference wav once; return the server-side path."""
    import aiohttp

    url = api_base.rstrip("/") + "/v1/audio/upload_reference"
    with open(ref_audio, "rb") as f:
        data = f.read()
    form = aiohttp.FormData()
    form.add_field(
        "file", data,
        filename=os.path.basename(ref_audio),
        content_type="audio/wav",
    )
    async with session.post(url, data=form) as resp:
        if resp.status != 200:
            err = (await resp.text())[:500]
            raise RuntimeError(f"upload_reference HTTP {resp.status}: {err}")
        payload = await resp.json()
    path = payload.get("path")
    if not path:
        raise RuntimeError(f"upload_reference returned no path: {payload}")
    return path


# --- per-request worker -----------------------------------------------------

async def _generate_one(
    session,
    api_base: str,
    text: str,
    reference_audio: str,
    semaphore: asyncio.Semaphore,
    *,
    temperature: float,
    seed: int,
    timeout: float,
    stream: bool = False,
) -> dict:
    """One streaming request. Returns per-request stats.

    With ``stream=False`` the server ships one (post-generation) PCM body and
    TTFA is effectively the full E2E time. With ``stream=True`` the server uses
    the *framed* transport (``[type:1][len:4 BE][payload]`` with ``A`` audio,
    ``M`` marks, ``E`` end), so TTFA is measured from the first **audio** frame
    and only ``A`` payload bytes count toward the audio duration.
    """
    import aiohttp

    url = api_base.rstrip("/") + "/v1/audio/speech"
    body = {
        "input": text,
        "reference_audio": reference_audio,
        "response_format": "pcm",
        "temperature": temperature,
        "seed": seed,
    }
    if stream:
        body["stream"] = True

    async with semaphore:
        nbytes = 0
        first_audio_t: float | None = None
        t0 = time.perf_counter()
        try:
            async with session.post(
                url, json=body,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    err = (await resp.text())[:500]
                    return {"success": False, "error": f"HTTP {resp.status}: {err}"}
                if stream:
                    buf = b""
                    async for chunk in resp.content.iter_any():
                        if not chunk:
                            continue
                        buf += chunk
                        while len(buf) >= 5:
                            ftype = buf[0:1]
                            length = int.from_bytes(buf[1:5], "big")
                            if len(buf) < 5 + length:
                                break
                            payload = buf[5:5 + length]
                            buf = buf[5 + length:]
                            if ftype == b"A":
                                if first_audio_t is None:
                                    first_audio_t = time.perf_counter()
                                nbytes += len(payload)
                            elif ftype == b"X":
                                return {"success": False,
                                        "error": f"stream error: {payload.decode(errors='replace')[:200]}"}
                else:
                    async for chunk in resp.content.iter_any():
                        if not chunk:
                            continue
                        if first_audio_t is None:
                            first_audio_t = time.perf_counter()
                        nbytes += len(chunk)
        except Exception as e:  # noqa: BLE001
            return {"success": False, "error": f"{type(e).__name__}: {e}"}

    elapsed = time.perf_counter() - t0
    ttfa = (first_audio_t - t0) if first_audio_t is not None else elapsed
    duration = (nbytes // BYTES_PER_SAMPLE) / SAMPLE_RATE if SAMPLE_RATE > 0 else 0.0
    return {
        "success": True,
        "ttfa": ttfa,
        "e2e": elapsed,
        "duration": duration,
        "bytes": nbytes,
    }


# --- driver -----------------------------------------------------------------

async def _run_cell(
    args,
    *,
    length: str,
    concurrency: int,
    session=None,
    reference_audio: str | None = None,
) -> dict:
    """Run one (length, concurrency) cell and return an aggregated result dict."""
    import aiohttp

    prompts = _load_prompts(length, args.prompt_dir)
    texts = [prompts[i % len(prompts)] for i in range(args.num_requests)]

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession()
    try:
        if reference_audio is None:
            reference_audio = await _upload_reference(
                session, args.api_base, args.ref_audio
            )

        sem = asyncio.Semaphore(concurrency)
        print(
            f"\n[{length} c{concurrency}] {args.num_requests} requests "
            f"(corpus {len(prompts)} lines) -> {args.api_base.rstrip('/')}"
            f"/v1/audio/speech",
            flush=True,
        )
        t_start = time.perf_counter()
        coros = [
            _generate_one(
                session, args.api_base, text, reference_audio, sem,
                temperature=args.temperature, seed=args.seed, timeout=args.timeout,
                stream=args.stream,
            )
            for text in texts
        ]
        results = await asyncio.gather(*coros)
        wall_time = time.perf_counter() - t_start
    finally:
        if own_session:
            await session.close()

    succeeded = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if not succeeded:
        first_err = failed[0].get("error", "") if failed else "no results"
        print(f"  ALL FAILED ({len(failed)}); first error: {first_err[:200]}",
              flush=True)
        return {
            "length": length,
            "concurrency": concurrency,
            "num_requests": args.num_requests,
            "succeeded": 0,
            "ttfa": {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0},
            "e2e": {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0},
            "avg_duration": 0.0,
            "avg_rtf": 0.0,
            "wall_time": wall_time,
            "throughput": 0.0,
            "failed": len(failed),
        }

    ttfas = sorted(r["ttfa"] for r in succeeded)
    e2es = sorted(r["e2e"] for r in succeeded)
    durations = [r["duration"] for r in succeeded]

    avg_ttfa = sum(ttfas) / len(ttfas)
    avg_e2e = sum(e2es) / len(e2es)
    avg_dur = sum(durations) / len(durations)
    total_dur = sum(durations)
    avg_rtf = sum(r["e2e"] / max(r["duration"], 0.01) for r in succeeded) / len(succeeded)
    throughput = total_dur / wall_time if wall_time > 0 else 0.0

    result = {
        "length": length,
        "concurrency": concurrency,
        "num_requests": args.num_requests,
        "succeeded": len(succeeded),
        "ttfa": {
            "avg": avg_ttfa,
            "p50": _pct(ttfas, 0.5),
            "p95": _pct(ttfas, 0.95),
            "p99": _pct(ttfas, 0.99),
        },
        "e2e": {
            "avg": avg_e2e,
            "p50": _pct(e2es, 0.5),
            "p95": _pct(e2es, 0.95),
            "p99": _pct(e2es, 0.99),
        },
        "avg_duration": avg_dur,
        "avg_rtf": avg_rtf,
        "wall_time": wall_time,
        "throughput": throughput,
        "failed": len(failed),
    }
    _print_summary(result)
    return result


def _print_summary(r: dict) -> None:
    t = r["ttfa"]
    e = r["e2e"]
    print(
        f"\nSummary ({r['succeeded']} requests, "
        f"concurrency={r['concurrency']}):", flush=True
    )
    print(f"  TTFA avg/p50/p95/p99: {t['avg']:.3f}s / {t['p50']:.3f}s / "
          f"{t['p95']:.3f}s / {t['p99']:.3f}s")
    print(f"  E2E  avg/p50/p95/p99: {e['avg']:.3f}s / {e['p50']:.3f}s / "
          f"{e['p95']:.3f}s / {e['p99']:.3f}s")
    print(f"  Avg audio duration: {r['avg_duration']:.2f}s")
    print(f"  Avg RTF:            {r['avg_rtf']:.3f}x")
    print(f"  Wall time:          {r['wall_time']:.2f}s")
    print(f"  Throughput:         {r['throughput']:.2f}x realtime")
    print(f"  Failed:             {r['failed']}")


def _write_json(result: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  wrote {path}", flush=True)


async def run_single(args) -> None:
    result = await _run_cell(
        args, length=args.length, concurrency=args.concurrency
    )
    if args.out:
        _write_json(result, args.out)


async def run_matrix(args) -> None:
    import aiohttp

    lengths = ["short", "medium", "long"]
    concurrencies = [1, 2, 4, 8]
    os.makedirs(args.out_dir, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        # Upload the reference once and reuse across all cells.
        reference_audio = await _upload_reference(
            session, args.api_base, args.ref_audio
        )
        for length in lengths:
            for concurrency in concurrencies:
                result = await _run_cell(
                    args, length=length, concurrency=concurrency,
                    session=session, reference_audio=reference_audio,
                )
                out_path = os.path.join(args.out_dir, f"{length}_c{concurrency}.json")
                _write_json(result, out_path)


def main():
    p = argparse.ArgumentParser(
        description="Async benchmark client for sglang-omni SpeechifyTTS"
    )
    p.add_argument("--api-base", default="http://127.0.0.1:8030")
    p.add_argument("--length", choices=["short", "medium", "long"], default="short")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--num-requests", type=int, default=32)
    p.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    p.add_argument("--prompt-dir", default=DEFAULT_PROMPT_DIR)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--out", default=None, help="Write a JSON results file (single mode).")
    p.add_argument("--stream", action="store_true",
                   help="Use the framed true-streaming transport; measure TTFA "
                        "from the first audio frame.")
    p.add_argument("--matrix", action="store_true",
                   help="Run {short,medium,long} x {1,2,4,8} sequentially.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for --matrix JSON cells "
                        "(filenames {length}_c{concurrency}.json).")
    args = p.parse_args()

    if args.matrix:
        if not args.out_dir:
            p.error("--matrix requires --out-dir")
        asyncio.run(run_matrix(args))
    else:
        asyncio.run(run_single(args))


if __name__ == "__main__":
    main()
