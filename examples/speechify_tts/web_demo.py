#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Web demo for SpeechifyTTS online serving in sglang-omni.

A FastAPI app that proxies to a running TTS server and plays back the **raw PCM
stream** from ``POST /v1/audio/speech`` for low time-to-first-audio. The UI
mirrors the vllm-omni SpeechifyT5 demo (dark theme, gradient, live metric cards,
WAV download, optional speaking-rate override); the transport here is raw PCM16
rather than SSE+base64, matching the sglang-omni streaming contract:

    POST /v1/audio/speech  (stream=true, response_format="pcm")
      -> media_type audio/pcm, body = raw PCM16 mono bytes,
         headers X-Sample-Rate / X-Channels / X-Bit-Depth

Usage::

    # 1) start the model server (separate terminal)
    examples/speechify_tts/run_server_streaming_moe4b.sh   # or serve_offline.py

    # 2) start this demo and open http://127.0.0.1:8001
    python examples/speechify_tts/web_demo.py \
        --api-base http://127.0.0.1:8030 --model speechify-tts --port 8001

Reference-audio uploads are written under ``--ref-audio-dir`` (default
``/tmp/speechify_web_demo``); the model server reads them directly (single-host
dev setup). Launch a real ``sgl-omni serve`` with
``--allowed-local-media-path /tmp`` so it can read them.
"""

from __future__ import annotations

import argparse
import os
import time
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

DEFAULT_SAMPLE_RATE = 24000

app = FastAPI(title="SpeechifyTTS Web Demo")


class Settings:
    api_base = "http://127.0.0.1:8030"
    model = "speechify-tts"
    api_key = "EMPTY"
    ref_audio_dir = "/tmp/speechify_web_demo"


settings = Settings()


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>sglang-omni · SpeechifyTTS MoE 4B MTL</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  background:#0f0f1a;color:#e8e8f0;min-height:100vh;padding:2rem;
}
.container{max-width:760px;margin:0 auto}
h1{
  font-size:1.7rem;margin-bottom:.3rem;
  background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.subtitle{color:#7878a0;font-size:.85rem;margin-bottom:1.5rem}
.subtitle code{color:#9898b0}
.card{
  background:#1a1a2e;border-radius:12px;padding:1.5rem;
  margin-bottom:1rem;border:1px solid #2a2a4a;
}
label{display:block;font-size:.85rem;color:#9898b0;margin-bottom:.4rem;font-weight:500}
input[type="text"],textarea{
  width:100%;padding:.7rem;background:#12121f;
  border:1px solid #3a3a5a;border-radius:8px;
  color:#e8e8f0;font-size:.95rem;margin-bottom:1rem;resize:vertical;
}
input[type="text"]:focus,textarea:focus{outline:none;border-color:#667eea}
textarea{min-height:90px;font-family:inherit;line-height:1.5}
input[type="file"]{width:100%;padding:.5rem;margin-bottom:1rem;color:#9898b0}
.row{display:flex;gap:1rem;flex-wrap:wrap;align-items:flex-end}
.row > div{flex:1;min-width:120px}
input[type="number"]{
  width:100%;padding:.5rem;background:#12121f;border:1px solid #3a3a5a;
  border-radius:6px;color:#e8e8f0;
}
.btn{
  width:100%;padding:.8rem;
  background:linear-gradient(135deg,#667eea,#764ba2);
  border:none;border-radius:8px;color:#fff;
  font-size:1rem;font-weight:600;cursor:pointer;transition:opacity .2s;margin-top:1rem;
}
.btn:hover{opacity:.9}
.btn:disabled{opacity:.5;cursor:not-allowed}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin-top:.75rem}
.metric{background:#12121f;border-radius:8px;padding:.7rem;text-align:center}
.metric .value{font-size:1.2rem;font-weight:700;color:#667eea}
.metric .label{font-size:.7rem;color:#7878a0;margin-top:.2rem}
.status{text-align:center;padding:.5rem;font-size:.9rem;color:#9898b0}
.status.error{color:#e74c3c}
.status.playing{color:#2ecc71}
.text-display{
  background:#12121f;border-radius:8px;padding:1.2rem;
  font-size:1.15rem;line-height:1.9;min-height:40px;margin-bottom:1rem;color:#cfcfe6;
}
.text-display .word{display:inline;padding:2px 4px;border-radius:4px;transition:background .08s,color .08s}
.text-display .word.active{background:#667eea;color:#fff}
.text-display .word.spoken{color:#8a8ab0}
</style>
</head>
<body>
<div class="container">
  <h1>SpeechifyTTS — sglang-omni</h1>
  <div class="subtitle">MoE 4B MTL · model <code id="model">…</code> · raw PCM streaming @ 24 kHz</div>

  <div class="card">
    <label for="text-input">Text</label>
    <textarea id="text-input" maxlength="1000">On our show today, we're diving deep into the phenomenon of biomimicry, which is the practice of looking to nature for solutions to human problems.</textarea>

    <label for="ref-audio">Reference voice (.wav / .mp3 / .flac)</label>
    <input type="file" id="ref-audio" accept="audio/*" onchange="uploadRefAudio(this)">
    <div id="ref-status" style="font-size:.8rem;color:#7878a0;margin-bottom:1rem"></div>

    <div class="row">
      <div>
        <label for="temp">Temperature</label>
        <input type="number" id="temp" value="0.8" min="0" max="1.5" step="0.05">
      </div>
      <div>
        <label style="display:flex;align-items:center;gap:.4rem;cursor:pointer">
          <input type="checkbox" id="use-sr" onchange="toggleSR()" style="width:auto;margin:0"> Override speaking rate
        </label>
        <input type="number" id="sr-value" value="2.0" min="1.5" max="3.5" step="0.1" disabled>
      </div>
    </div>

    <button class="btn" id="generate-btn" onclick="generate()" disabled>Generate</button>
    <button class="btn" id="stop-btn" onclick="stopPlayback()" style="display:none;background:linear-gradient(135deg,#e74c3c,#c0392b)">Stop</button>
  </div>

  <div class="card">
    <div id="status" class="status">Upload a reference voice to begin</div>
    <div class="text-display" id="text-display"></div>
    <button class="btn" id="download-btn" onclick="downloadAudio()" style="display:none;background:linear-gradient(135deg,#2ecc71,#27ae60)">Download WAV</button>
    <div class="metrics">
      <div class="metric"><div class="value" id="m-server-ttfa">--</div><div class="label">Server TTFA</div></div>
      <div class="metric"><div class="value" id="m-client-ttfa">--</div><div class="label">Client TTFA</div></div>
      <div class="metric"><div class="value" id="m-total">--</div><div class="label">Total</div></div>
      <div class="metric"><div class="value" id="m-dur">--</div><div class="label">Audio Duration</div></div>
      <div class="metric"><div class="value" id="m-rtf">--</div><div class="label">RTF</div></div>
      <div class="metric"><div class="value" id="m-marks">--</div><div class="label">Marks</div></div>
    </div>
  </div>
</div>

<script>
let SAMPLE_RATE = 24000;
let audioCtx = null;
let nextStartTime = 0;
let abortCtrl = null;
let cachedRefPath = null;
let isUploading = false;
let allSamples = [];
let speechmarks = [];
let playbackStartTime = 0;
let totalScheduledMs = 0;
let highlightRAF = null;

function setMetric(id, v){ document.getElementById(id).textContent = v; }
function setStatus(msg, cls){
  const el = document.getElementById('status');
  el.textContent = msg; el.className = 'status' + (cls ? ' '+cls : '');
}
function updateBtn(){
  document.getElementById('generate-btn').disabled = !cachedRefPath || isUploading;
}
function toggleSR(){
  document.getElementById('sr-value').disabled = !document.getElementById('use-sr').checked;
}

fetch('/api/model').then(r => r.json()).then(d => {
  document.getElementById('model').textContent = d.model;
});

// --- ref-audio upload ---
async function uploadRefAudio(input){
  const refStatus = document.getElementById('ref-status');
  if(!input.files.length){ cachedRefPath = null; refStatus.textContent=''; updateBtn(); return; }
  const file = input.files[0];
  const sizeMB = (file.size/1024/1024).toFixed(1);
  refStatus.textContent = 'Uploading ' + sizeMB + 'MB...';
  refStatus.style.color = '#9898b0';
  isUploading = true; updateBtn();
  try{
    const t0 = performance.now();
    const form = new FormData(); form.append('file', file);
    const resp = await fetch('/api/ref-audio', { method:'POST', body:form });
    const data = await resp.json();
    if(!resp.ok) throw new Error(data.error || ('HTTP '+resp.status));
    const elapsed = ((performance.now()-t0)/1000).toFixed(1);
    cachedRefPath = data.ref_audio;
    refStatus.textContent = 'Ready (' + elapsed + 's to upload)';
    refStatus.style.color = '#2ecc71';
  }catch(err){
    cachedRefPath = null;
    refStatus.textContent = 'Upload failed: ' + err.message;
    refStatus.style.color = '#e74c3c';
  }finally{
    isUploading = false; updateBtn();
  }
}

// --- WAV download ---
function downloadAudio(){
  if(!allSamples.length) return;
  let total = 0; for(const s of allSamples) total += s.length;
  const merged = new Float32Array(total);
  let off=0; for(const s of allSamples){ merged.set(s, off); off += s.length; }
  const n = merged.length;
  const buffer = new ArrayBuffer(44 + n*2);
  const view = new DataView(buffer);
  const ws = (o,s)=>{ for(let i=0;i<s.length;i++) view.setUint8(o+i, s.charCodeAt(i)); };
  ws(0,'RIFF'); view.setUint32(4, 36+n*2, true); ws(8,'WAVE'); ws(12,'fmt ');
  view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,1,true);
  view.setUint32(24,SAMPLE_RATE,true); view.setUint32(28,SAMPLE_RATE*2,true);
  view.setUint16(32,2,true); view.setUint16(34,16,true); ws(36,'data');
  view.setUint32(40, n*2, true);
  for(let i=0;i<n;i++){ let s=Math.max(-1,Math.min(1,merged[i])); view.setInt16(44+i*2, s<0?s*0x8000:s*0x7FFF, true); }
  const url = URL.createObjectURL(new Blob([buffer], {type:'audio/wav'}));
  const a = document.createElement('a'); a.href=url; a.download = crypto.randomUUID()+'.wav'; a.click();
  URL.revokeObjectURL(url);
}

// --- word highlighting synced to playback time ---
function buildHighlightArea(text){
  const area = document.getElementById('text-display');
  area.innerHTML = '';
  const re = /\S+/g; let m, last = 0;
  while((m = re.exec(text)) !== null){
    if(m.index > last) area.appendChild(document.createTextNode(text.slice(last, m.index)));
    const span = document.createElement('span');
    span.className = 'word'; span.textContent = m[0];
    span.dataset.start = m.index; span.dataset.end = m.index + m[0].length;
    area.appendChild(span);
    last = m.index + m[0].length;
  }
  if(last < text.length) area.appendChild(document.createTextNode(text.slice(last)));
}
function startHighlightLoop(){
  function tick(){
    if(!audioCtx) return;
    const elapsed = Math.min((audioCtx.currentTime - playbackStartTime) * 1000, totalScheduledMs);
    let aS = -1, aE = -1;
    for(const mk of speechmarks){
      if(elapsed >= mk.startTime){ aS = mk.startIndex; aE = mk.endIndex; } else break;
    }
    for(const el of document.querySelectorAll('#text-display .word')){
      const s = +el.dataset.start, e = +el.dataset.end;
      if(aS >= 0 && s < aE && e > aS){ el.classList.add('active'); el.classList.remove('spoken'); }
      else if(aS >= 0 && e <= aS){ el.classList.remove('active'); el.classList.add('spoken'); }
      else { el.classList.remove('active','spoken'); }
    }
    highlightRAF = requestAnimationFrame(tick);
  }
  highlightRAF = requestAnimationFrame(tick);
}
function stopHighlightLoop(){ if(highlightRAF){ cancelAnimationFrame(highlightRAF); highlightRAF = null; } }

function pcm16ToFloat32(bytes){
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const n = bytes.byteLength >> 1;
  const out = new Float32Array(n);
  for(let i=0;i<n;i++) out[i] = view.getInt16(i*2, true)/32768;
  return out;
}
function scheduleChunk(f32){
  if(!f32.length) return;
  const buf = audioCtx.createBuffer(1, f32.length, SAMPLE_RATE);
  buf.copyToChannel(f32, 0);
  const src = audioCtx.createBufferSource();
  src.buffer = buf; src.connect(audioCtx.destination);
  const start = Math.max(nextStartTime, audioCtx.currentTime + 0.02);
  src.start(start);
  nextStartTime = start + buf.duration;
}

function stopPlayback(){
  if(abortCtrl) abortCtrl.abort();
}

// --- generation: raw PCM stream ---
async function generate(){
  const text = document.getElementById('text-input').value.trim();
  if(!text){ setStatus('Enter some text first','error'); return; }
  if(!cachedRefPath){ setStatus('Upload reference audio first','error'); return; }

  if(abortCtrl) abortCtrl.abort();
  abortCtrl = new AbortController();
  if(audioCtx){ audioCtx.close(); audioCtx = null; }
  stopHighlightLoop();

  ['m-server-ttfa','m-client-ttfa','m-total','m-dur','m-rtf','m-marks'].forEach(id => setMetric(id,'--'));
  allSamples = []; speechmarks = []; totalScheduledMs = 0;
  document.getElementById('download-btn').style.display = 'none';
  buildHighlightArea(text);
  document.getElementById('generate-btn').style.display = 'none';
  document.getElementById('stop-btn').style.display = 'block';
  setStatus('Connecting...');

  audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  nextStartTime = 0;

  const t0 = performance.now();
  let firstChunk = true, totalSamples = 0, leftover = new Uint8Array(0);

  const body = { text, ref_audio: cachedRefPath, temperature: parseFloat(document.getElementById('temp').value) };
  if(document.getElementById('use-sr').checked){
    body.speaking_rate = parseFloat(document.getElementById('sr-value').value);
  }

  try{
    const resp = await fetch('/api/tts', {
      method:'POST',
      headers:{ 'Content-Type':'application/json' },
      body: JSON.stringify(body),
      signal: abortCtrl.signal,
    });
    if(!resp.ok){ setStatus('Error: ' + (await resp.text()), 'error'); return; }
    SAMPLE_RATE = parseInt(resp.headers.get('X-Sample-Rate') || '24000', 10);
    const serverTTFA = parseFloat(resp.headers.get('X-Server-TTFA') || '0');
    try {
      const smHdr = resp.headers.get('X-Speechmarks') || '';
      speechmarks = smHdr ? JSON.parse(atob(smHdr)) : [];
    } catch(e){ speechmarks = []; console.warn('speechmarks decode failed', e); }
    setMetric('m-marks', speechmarks.length.toString());

    const reader = resp.body.getReader();
    setStatus('Streaming @ ' + SAMPLE_RATE + ' Hz...', 'playing');
    while(true){
      const { done, value } = await reader.read();
      if(done) break;
      let merged = new Uint8Array(leftover.length + value.length);
      merged.set(leftover,0); merged.set(value, leftover.length);
      const usable = merged.length - (merged.length % 2);
      const f32 = pcm16ToFloat32(merged.subarray(0, usable));
      leftover = merged.subarray(usable);
      allSamples.push(f32);
      scheduleChunk(f32);
      totalSamples += f32.length;
      totalScheduledMs = Math.round((totalSamples / SAMPLE_RATE) * 1000);
      if(firstChunk){
        playbackStartTime = Math.max(nextStartTime - f32.length / SAMPLE_RATE, audioCtx.currentTime);
        const clientTTFA = (performance.now()-t0)/1000;
        setMetric('m-server-ttfa', serverTTFA.toFixed(3)+'s');
        setMetric('m-client-ttfa', clientTTFA.toFixed(3)+'s');
        startHighlightLoop();
        firstChunk = false;
      }
    }

    const totalTime = (performance.now()-t0)/1000;
    const audioDur = totalSamples / SAMPLE_RATE;
    const rtf = audioDur > 0 ? totalTime/audioDur : 0;
    setMetric('m-total', totalTime.toFixed(3)+'s');
    setMetric('m-dur', audioDur.toFixed(2)+'s');
    setMetric('m-rtf', rtf.toFixed(3)+'x');
    setStatus('Done — ' + audioDur.toFixed(2) + 's audio, ' + speechmarks.length + ' marks');
    if(totalSamples > 0) document.getElementById('download-btn').style.display = 'block';
    const remaining = nextStartTime - audioCtx.currentTime;
    setTimeout(stopHighlightLoop, Math.max(0, Math.ceil(remaining*1000)) + 200);
  }catch(err){
    if(err.name !== 'AbortError') setStatus('Error: ' + err.message, 'error');
    else setStatus('Stopped.');
    stopHighlightLoop();
  }finally{
    document.getElementById('generate-btn').style.display = 'block';
    document.getElementById('stop-btn').style.display = 'none';
    updateBtn();
  }
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/api/model")
async def model_info() -> JSONResponse:
    return JSONResponse({"model": settings.model, "api_base": settings.api_base})


@app.post("/api/ref-audio")
async def ref_audio(file: UploadFile) -> JSONResponse:
    os.makedirs(settings.ref_audio_dir, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    path = os.path.join(settings.ref_audio_dir, f"{uuid.uuid4().hex}{ext}")
    try:
        data = await file.read()
        with open(path, "wb") as fh:
            fh.write(data)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"ref_audio": path})


@app.post("/api/tts")
async def tts(request: Request):
    payload = await request.json()
    upstream_body = {
        "model": settings.model,
        "input": payload.get("text", ""),
        "ref_audio": payload.get("ref_audio"),
        "response_format": "pcm",
        "stream": True,
    }
    if payload.get("temperature") is not None:
        upstream_body["temperature"] = float(payload["temperature"])
    if payload.get("speaking_rate") is not None:
        upstream_body["speaking_rate"] = float(payload["speaking_rate"])
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Accept": "audio/pcm",
        "Content-Type": "application/json",
    }
    url = f"{settings.api_base.rstrip('/')}/v1/audio/speech"

    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None))
    t0 = time.perf_counter()
    resp_cm = client.stream("POST", url, json=upstream_body, headers=headers)
    resp = await resp_cm.__aenter__()
    if resp.status_code != 200:
        text = (await resp.aread()).decode(errors="replace")
        await resp_cm.__aexit__(None, None, None)
        await client.aclose()
        return JSONResponse({"error": f"upstream {resp.status_code}: {text[:300]}"}, status_code=502)

    sample_rate = resp.headers.get("X-Sample-Rate", str(DEFAULT_SAMPLE_RATE))
    num_codes = resp.headers.get("X-Num-Mel-Codes", "")
    speechmarks = resp.headers.get("X-Speechmarks", "")
    byte_iter = resp.aiter_bytes().__aiter__()

    # Probe the first chunk to measure server-side TTFA.
    server_ttfa = 0.0
    first = b""
    try:
        first = await byte_iter.__anext__()
        server_ttfa = time.perf_counter() - t0
    except StopAsyncIteration:
        pass

    async def _body():
        try:
            if first:
                yield first
            async for chunk in byte_iter:
                if chunk:
                    yield chunk
        finally:
            await resp_cm.__aexit__(None, None, None)
            await client.aclose()

    return StreamingResponse(
        _body(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Channels": "1",
            "X-Bit-Depth": "16",
            "X-Num-Mel-Codes": str(num_codes),
            "X-Speechmarks": speechmarks,
            "X-Server-TTFA": f"{server_ttfa:.6f}",
            "Access-Control-Expose-Headers": "X-Sample-Rate,X-Server-TTFA,X-Num-Mel-Codes,X-Speechmarks",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SpeechifyTTS web demo")
    parser.add_argument("--api-base", default=settings.api_base, help="TTS server base URL")
    parser.add_argument("--model", default=settings.model, help="served model name")
    parser.add_argument("--api-key", default=settings.api_key)
    parser.add_argument("--ref-audio-dir", default=settings.ref_audio_dir)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    settings.api_base = args.api_base
    settings.model = args.model
    settings.api_key = args.api_key
    settings.ref_audio_dir = args.ref_audio_dir

    print(f"[speechify-web-demo] proxying to {settings.api_base} (model={settings.model!r})")
    print(f"[speechify-web-demo] open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
