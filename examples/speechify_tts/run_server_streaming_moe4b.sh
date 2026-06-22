#!/usr/bin/env bash
# Launch the SpeechifyTTS MoE 4B MTL (simba3) streaming server in sglang-omni.
#
#   preprocessing -> tts_engine (MoE AR decoder) -> vocoder (DiffiTv3 + FlowVAE)
#
# Mirrors the vllm-omni run_server_streaming_moe4b.sh, but uses the sglang-omni
# CLI (`sgl-omni serve`) and the SpeechifyTTS pipeline config. Override any knob
# via environment variables.
set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/home/kevin/vllm-omni/vllm_omni/checkpoints/simba3-moe4b-vllm-v5-streaming}"
CONFIG="${CONFIG:-${_REPO_ROOT}/examples/configs/speechify_tts_moe4b.yaml}"

GPU="${GPU:-0}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
MODEL_NAME="${MODEL_NAME:-Simba MoE 4B MTL}"
# Directory allowed for file:// / local-path ref_audio (the web demo writes
# uploaded references here). Local media refs are disabled unless this is set.
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/tmp}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"
# SGLang FusedMoE path (analog of vllm-omni VLLM_OMNI_DISABLE_GROUPED_MM=1).
export JAXTYPING_DISABLE="${JAXTYPING_DISABLE:-1}"

# torch 2.11 ships CUDA 13 wheels; the host driver here is 570 (CUDA 12.8).
# Prepend a CUDA-13 forward-compat libcuda so the older driver can run cu130
# (H100 supports forward compatibility). Override CUDA_COMPAT_DIR if needed.
CUDA_COMPAT_DIR="${CUDA_COMPAT_DIR:-/home/kevin/miniconda3/envs/speechify-vllm/cuda-compat}"
if [[ -d "${CUDA_COMPAT_DIR}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}:${LD_LIBRARY_PATH:-}"
fi

# Activate the sglang-omni venv if sgl-omni is not already on PATH.
if ! command -v sgl-omni >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "${_REPO_ROOT}/.venv/bin/activate"
fi

echo "[speechify-tts] model_path = ${MODEL_PATH}"
echo "[speechify-tts] config     = ${CONFIG}"
echo "[speechify-tts] serving on ${BIND_HOST}:${PORT} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "[speechify-tts] first start captures CUDA graphs; this can take minutes."

# shellcheck disable=SC2086
exec sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --config "${CONFIG}" \
  --model-name "${MODEL_NAME}" \
  --host "${BIND_HOST}" \
  --port "${PORT}" \
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}" \
  ${EXTRA_ARGS}
