# SpeechifyTTS (simba3 MoE 4B MTL) — sglang-omni

Zero-shot voice-cloning TTS ported from vllm-omni. 3-stage pipeline:

```
preprocessing (T5 text encoder + Unified speaker embedding)
  -> tts_engine (Gemma-style MoE AR decoder: mel codes + alignment)
  -> vocoder (DiffiTv3 flow-matching diffusion + FlowVAE -> 24 kHz PCM)
```

## Status

The **serving scaffold is complete and validated** (pipeline config, HF configs,
cross-stage state, request boundary, stage factories, example config, launch
script, and this web demo). The **deep model internals are an in-progress port**
and require a GPU box to validate:

- `tts_engine` MoE decoder forward (SGLang `RadixAttention` self-attn + dual
  cross-attention + conformer-conv state cache + `FusedMoE` + alignment head),
  the alignment-based stop runner, and CUDA-graph capture.
- the text encoder + Unified speaker-embedding extraction in preprocessing.
- the DiffiTv3 + FlowVAE streaming diffusion vocoder + its CUDA-graph runner.

Running the server today will raise a clear `NotImplementedError` at the first
deferred boundary. Track progress in `sglang_omni/models/speechify_tts/`.

## Launch the server

```bash
examples/speechify_tts/run_server_streaming_moe4b.sh
# or explicitly:
sgl-omni serve \
  --model-path /home/kevin/vllm-omni/vllm_omni/checkpoints/simba3-moe4b-vllm-v5-streaming \
  --config examples/configs/speechify_tts_moe4b.yaml \
  --allowed-local-media-path /tmp \
  --host 0.0.0.0 --port 8002
```

The checkpoint uses the vllm-omni layout: AR weights under `ar/`, the diffusion
vocoder under `diffusion/` (the stage factories resolve these subdirs).

## Web demo (streaming playback)

```bash
python examples/speechify_tts/web_demo.py \
  --api-base http://127.0.0.1:8002 --model "Simba MoE 4B MTL" --port 8001
# open http://127.0.0.1:8001, upload a reference clip, type text, Synthesize.
```

The demo proxies `POST /v1/audio/speech` with `stream=true, response_format="pcm"`
and plays the raw PCM16 mono stream (sample rate from the `X-Sample-Rate`
response header) via the Web Audio API, reporting TTFA / RTF.

## curl (raw PCM)

```bash
curl -N -X POST http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Simba MoE 4B MTL",
    "input": "Hello from sglang-omni.",
    "ref_audio": "/tmp/my_reference.wav",
    "response_format": "pcm",
    "stream": true,
    "speed": 1.0
  }' --output out.pcm
# play: ffplay -f s16le -ar 24000 -ac 1 out.pcm
```
