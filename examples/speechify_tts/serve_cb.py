#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Continuous-batching SpeechifyTTS server (torch-native, batched decode).

Loads :class:`CBEngine` -- a background decode thread that keeps up to
``--max-batch`` requests in flight in one batched forward pass -- and exposes an
OpenAI-style ``POST /v1/audio/speech`` endpoint. Unlike ``serve_offline.py``
(single dedicated worker, one request at a time), this scales throughput with
concurrency by sharing the GPU across requests.

Usage::

    python examples/speechify_tts/serve_cb.py \
        --model-path /path/to/simba3-moe4b-vllm-v5-streaming \
        --host 0.0.0.0 --port 8031 --max-batch 8
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import tempfile

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse


def _build_app(model_path: str, device: str, max_batch: int,
               vocode_workers: int = 2) -> FastAPI:
    os.environ.setdefault("JAXTYPING_DISABLE", "1")
    from sglang_omni.models.speechify_tts.cb_engine import CBEngine

    app = FastAPI(title="SpeechifyTTS Continuous-Batching Server")
    print(f"[serve_cb] loading engine from {model_path} on {device} "
          f"(max_batch={max_batch}, vocode_workers={vocode_workers}) ...")
    engine = CBEngine(model_path, max_batch=max_batch, device=device,
                      vocode_workers=vocode_workers)
    print("[serve_cb] warming up ...", flush=True)
    engine.warmup(3)
    print(f"[serve_cb] ready (sample_rate={engine.sample_rate})", flush=True)

    def _resolve_reference(ref: str) -> str:
        if ref.startswith("data:") or (len(ref) > 256 and "/" not in ref[:64]):
            encoded = ref.split(",", 1)[1] if ref.startswith("data:") else ref
            raw = base64.b64decode(encoded)
            fd, path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            return path
        return ref

    def _frame(ftype: bytes, payload: bytes) -> bytes:
        return ftype + len(payload).to_bytes(4, "big") + payload

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        body = await request.json()
        text = body.get("input") or body.get("text") or ""
        ref = (body.get("reference_audio") or body.get("ref_audio")
               or body.get("voice") or body.get("reference"))
        if not text:
            return JSONResponse(status_code=400, content={"error": "missing 'input' text"})
        if not ref:
            return JSONResponse(status_code=400, content={"error": "missing 'reference_audio'"})
        ref_path = _resolve_reference(ref)
        rate = body.get("speaking_rate")
        if rate is None:
            rate = body.get("speed")

        # ---- true streaming (framed): audio blocks ship as the diffusion
        #      vocoder produces them, overlapping the AR decode (low TTFA) ----
        if body.get("stream") and (body.get("response_format") or "pcm").lower() == "pcm":
            out_q = engine.submit_stream(
                text, ref_path,
                temperature=float(body.get("temperature", 0.8)),
                top_p=float(body.get("top_p", 0.8)),
                top_k=int(body.get("top_k", 10)),
                repetition_penalty=float(body.get("repetition_penalty", 2.0)),
                seed=int(body["seed"]) if body.get("seed") is not None else None,
                speaking_rate=float(rate) if rate is not None else None,
            )

            async def _frames():
                loop = asyncio.get_event_loop()
                while True:
                    evt = await loop.run_in_executor(None, out_q.get)
                    if evt is None:
                        return
                    kind, data = evt
                    if kind == "audio":
                        yield _frame(b"A", data)
                    elif kind == "marks":
                        yield _frame(b"M", json.dumps(data, separators=(",", ":")).encode())
                    elif kind == "final":
                        yield _frame(b"E", json.dumps(data, separators=(",", ":")).encode())
                    elif kind == "error":
                        yield _frame(b"X", str(data).encode())
                        return

            return StreamingResponse(
                _frames(), media_type="application/octet-stream",
                headers={
                    "X-Sample-Rate": str(engine.sample_rate), "X-Channels": "1",
                    "X-Bit-Depth": "16", "X-Stream-Format": "framed",
                    "Access-Control-Expose-Headers": "X-Sample-Rate,X-Channels,X-Bit-Depth,X-Stream-Format",
                },
            )

        fut = engine.submit(
            text, ref_path,
            temperature=float(body.get("temperature", 0.8)),
            top_p=float(body.get("top_p", 0.8)),
            top_k=int(body.get("top_k", 10)),
            repetition_penalty=float(body.get("repetition_penalty", 2.0)),
            seed=int(body["seed"]) if body.get("seed") is not None else None,
            speaking_rate=float(rate) if rate is not None else None,
        )
        try:
            res = await asyncio.wrap_future(fut)
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=400, content={"error": str(e)})

        audio = res["audio"].numpy()
        sr = res["sample_rate"]
        dur = round(len(audio) / sr, 3)

        def _encode_marks(ms: list) -> str:
            return base64.b64encode(json.dumps(ms, separators=(",", ":")).encode()).decode()

        marks = res.get("marks") or []
        marks_hdr = _encode_marks(marks)
        # HTTP header lines are capped (~8 KB in aiohttp/uvicorn); decimate the
        # speechmarks until the base64 payload comfortably fits so long prompts
        # don't trip "Got more than 8190 bytes when reading" on the client.
        while len(marks) > 8 and len(marks_hdr) > 7000:
            marks = marks[::2]
            marks_hdr = _encode_marks(marks)

        fmt = (body.get("response_format") or "wav").lower()
        if fmt == "pcm":
            pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()

            def _gen():
                step = sr
                for i in range(0, len(pcm), step * 2):
                    yield pcm[i:i + step * 2]

            return StreamingResponse(
                _gen(), media_type="audio/pcm",
                headers={
                    "X-Sample-Rate": str(sr), "X-Channels": "1", "X-Bit-Depth": "16",
                    "X-Num-Mel-Codes": str(res["num_mel_codes"]),
                    "X-Duration-S": str(dur),
                    "X-Speechmarks": marks_hdr,
                    "Access-Control-Expose-Headers": "X-Sample-Rate,X-Num-Mel-Codes,X-Duration-S,X-Speechmarks",
                },
            )
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return Response(content=buf.getvalue(), media_type="audio/wav",
                        headers={"X-Duration-S": str(dur),
                                 "X-Num-Mel-Codes": str(res["num_mel_codes"])})

    @app.post("/v1/audio/upload_reference")
    async def upload_reference(file: UploadFile):
        raw = await file.read()
        fd, path = tempfile.mkstemp(suffix=os.path.splitext(file.filename or ".wav")[1] or ".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        return {"path": path}

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "sample_rate": engine.sample_rate, "max_batch": max_batch}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8031)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--vocode-workers", type=int, default=2,
                    help="Number of streaming-vocoder worker threads, each on "
                         "its own CUDA stream. Requests are sharded across them "
                         "by id so the per-block diffusion forwards overlap; "
                         "raises high-concurrency streaming throughput.")
    args = ap.parse_args()
    app = _build_app(args.model_path, args.device, args.max_batch,
                     vocode_workers=args.vocode_workers)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
