#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone offline SpeechifyTTS server (torch-native reference path).

Loads :class:`SpeechifyTTSEngine` once and exposes an OpenAI-compatible
``POST /v1/audio/speech`` endpoint plus a tiny browser demo at ``/``. This runs
the validated eager pipeline (encoder -> MoE AR decode -> DiffiTv3 + FlowVAE)
without the sglang scheduler / CUDA-graph runtime, so it is usable for testing
the API and voice quality while the streaming server integration lands.

Usage::

    python examples/speechify_tts/serve_offline.py \
        --model-path /home/kevin/vllm-omni/vllm_omni/checkpoints/simba3-moe4b-vllm-v5-streaming \
        --host 0.0.0.0 --port 8030

    # synthesize (reference audio = server-local 24 kHz wav path or base64 data URL)
    curl -s http://127.0.0.1:8030/v1/audio/speech \
        -H 'content-type: application/json' \
        -d '{"input":"Hello from sglang omni.","reference_audio":"/path/ref.wav"}' \
        --output out.wav
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import tempfile
import threading

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse


def _build_app(model_path: str, device: str) -> FastAPI:
    os.environ.setdefault("JAXTYPING_DISABLE", "1")
    from sglang_omni.models.speechify_tts.engine import SpeechifyTTSEngine

    app = FastAPI(title="SpeechifyTTS Offline Server")
    print(f"[serve_offline] loading engine from {model_path} on {device} ...")
    engine = SpeechifyTTSEngine(model_path, device=device)
    print("[serve_offline] warming up CUDA kernels ...", flush=True)
    engine.warmup()
    lock = threading.Lock()  # single GPU pipeline; serialize requests
    print(f"[serve_offline] ready (sample_rate={engine.sample_rate})", flush=True)

    def _resolve_reference(ref: str) -> str:
        if ref.startswith("data:") or (len(ref) > 256 and "/" not in ref[:64]):
            encoded = ref.split(",", 1)[1] if ref.startswith("data:") else ref
            raw = base64.b64decode(encoded)
            fd, path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            return path
        return ref

    def _synth(body: dict) -> tuple[np.ndarray, int, dict]:
        text = body.get("input") or body.get("text") or ""
        ref = (
            body.get("reference_audio") or body.get("ref_audio")
            or body.get("voice") or body.get("reference")
        )
        if not text:
            raise ValueError("missing 'input' text")
        if not ref:
            raise ValueError("missing 'reference_audio' (local wav path or base64)")
        ref_path = _resolve_reference(ref)
        temperature = float(body.get("temperature", 0.8))
        top_p = float(body.get("top_p", 0.8))
        top_k = int(body.get("top_k", 10))
        repetition_penalty = float(body.get("repetition_penalty", 2.0))
        seed = body.get("seed")
        # speaking rate override: prefer explicit ``speaking_rate`` (raw AR rate
        # in [1.5, 3.5]); fall back to ``speed`` for OpenAI-style callers.
        rate_override = body.get("speaking_rate")
        if rate_override is None:
            rate_override = body.get("speed")
        with lock:
            res = engine.synthesize(
                text, ref_path,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty,
                seed=int(seed) if seed is not None else None,
                speaking_rate=float(rate_override) if rate_override is not None else None,
            )
        dur = res.audio.shape[-1] / res.sample_rate
        meta = {
            "num_mel_codes": res.num_mel_codes,
            "stop_reason": res.stop_reason,
            "speaking_rate": res.speaking_rate,
            "duration_s": round(dur, 3),
            "marks": res.marks or [],
        }
        tm = res.timings or {}
        rtf = round(tm.get("total_s", 0.0) / max(dur, 1e-6), 3)
        print(
            f"[synth] codes={res.num_mel_codes} audio={dur:.2f}s RTF={rtf} | "
            f"voice={tm.get('voice_s')}s enc={tm.get('encoder_s')}s "
            f"ar={tm.get('ar_s')}s ({tm.get('ar_tok_s')} tok/s) "
            f"voc={tm.get('vocoder_s')}s total={tm.get('total_s')}s",
            flush=True,
        )
        return res.audio.numpy(), res.sample_rate, meta

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        from starlette.concurrency import run_in_threadpool

        body = await request.json()
        try:
            # Run the GPU pipeline in a worker thread. The decode loop is
            # CPU-launch-bound; keeping it off the asyncio event-loop thread
            # avoids GIL/event-loop contention that otherwise inflates per-step
            # latency ~1.5x versus a dedicated process.
            audio, sr, meta = await run_in_threadpool(_synth, body)
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=400, content={"error": str(e)})

        import base64 as _b64
        import json as _json

        # Word speechmarks shipped as a base64(JSON) response header (the offline
        # engine has the full utterance up front). base64 (~1.3x) avoids the ~3x
        # blowup of URL-encoding. Only for pathological lengths do we *decimate*
        # (drop every other mark) so highlighting still spans the whole sentence
        # rather than truncating the tail.
        def _encode_marks(ms: list) -> str:
            return _b64.b64encode(
                _json.dumps(ms, separators=(",", ":")).encode()
            ).decode()

        marks = meta.get("marks") or []
        marks_hdr = _encode_marks(marks)
        while len(marks) > 8 and len(marks_hdr) > 24000:
            marks = marks[::2]
            marks_hdr = _encode_marks(marks)

        fmt = (body.get("response_format") or "wav").lower()
        if fmt == "pcm":
            pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()

            def _gen():
                step = sr  # ~1s chunks
                for i in range(0, len(pcm), step * 2):
                    yield pcm[i:i + step * 2]

            return StreamingResponse(
                _gen(), media_type="audio/pcm",
                headers={
                    "X-Sample-Rate": str(sr), "X-Channels": "1", "X-Bit-Depth": "16",
                    "X-Num-Mel-Codes": str(meta["num_mel_codes"]),
                    "X-Duration-S": str(meta["duration_s"]),
                    "X-Speechmarks": marks_hdr,
                    "Access-Control-Expose-Headers": "X-Sample-Rate,X-Num-Mel-Codes,X-Duration-S,X-Speechmarks",
                },
            )
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return Response(
            content=buf.getvalue(), media_type="audio/wav",
            headers={"X-Duration-S": str(meta["duration_s"]),
                     "X-Num-Mel-Codes": str(meta["num_mel_codes"])},
        )

    @app.post("/v1/audio/upload_reference")
    async def upload_reference(file: UploadFile):
        raw = await file.read()
        fd, path = tempfile.mkstemp(suffix=os.path.splitext(file.filename or ".wav")[1] or ".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        return {"path": path}

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _INDEX_HTML

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "sample_rate": engine.sample_rate}

    return app


_INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>SpeechifyTTS Offline</title>
<style>body{font-family:system-ui;max-width:680px;margin:40px auto;padding:0 16px}
textarea,input{width:100%;margin:6px 0;padding:8px;box-sizing:border-box}
button{padding:10px 18px;font-size:15px}</style></head><body>
<h2>SpeechifyTTS (offline reference)</h2>
<p>Reference voice (wav/mp3, resampled to 24 kHz):</p>
<input type="file" id="ref" accept="audio/*">
<textarea id="text" rows="3">Hello, this is a streaming text to speech test.</textarea>
<label>temperature <input type="number" id="temp" value="0.8" step="0.1" style="width:80px"></label>
<button onclick="go()">Synthesize</button>
<p id="status"></p><audio id="player" controls style="width:100%"></audio>
<script>
async function go(){
  const s=document.getElementById('status'); s.textContent='uploading reference...';
  const f=document.getElementById('ref').files[0];
  if(!f){s.textContent='pick a reference audio first';return;}
  const fd=new FormData(); fd.append('file',f);
  const up=await fetch('/v1/audio/upload_reference',{method:'POST',body:fd});
  const {path}=await up.json();
  s.textContent='synthesizing...';
  const r=await fetch('/v1/audio/speech',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({input:document.getElementById('text').value,reference_audio:path,
      temperature:parseFloat(document.getElementById('temp').value)})});
  if(!r.ok){s.textContent='error: '+(await r.text());return;}
  const blob=await r.blob();
  document.getElementById('player').src=URL.createObjectURL(blob);
  s.textContent='done ('+(r.headers.get('X-Duration-S')||'?')+'s, '+(r.headers.get('X-Num-Mel-Codes')||'?')+' codes)';
  document.getElementById('player').play();
}
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8030)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    app = _build_app(args.model_path, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
