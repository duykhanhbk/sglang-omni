# SPDX-License-Identifier: Apache-2.0
"""Self-contained autoregressive decode loop for the SpeechifyTTS MoE decoder.

Reproduces the vllm-omni OmniAR scheduler + ``gpu_ar_model_runner`` semantics
without the vLLM continuous-batching machinery (single-request decode):

Prefill / forced-rate protocol (``speechify_t5_tts_async_chunk.py``):
  1. decoder prompt is just ``[decoder_start_token_id]`` (alignment frozen at 0).
  2. the ``speaking_rate_token`` is *forced* as the first model input (not
     sampled); its hidden state does not advance alignment.
  3. from the next step on, audio codes are sampled and alignment advances by
     the multi-class ``alignment_step_head`` expected value.

Recipe-gated EOS + stop policy (MTL / MoE 4B, ``align_stop_offset >= 1``,
``full_vocab_eos_token_id`` samplable) — mirrors GPTTTS-native
``experimental/speechify_t5.py:2429-2443`` + the runner stop criterion:
    round(align) > text_len      -> force EOS    (logit = 1e9)
    round(align) < text_len - 4  -> suppress EOS (logit = -inf)
    text_len-4 <= round(align) <= text_len -> unbiased (sample naturally)
    stop when sampled == EOS OR round(align) >= text_len + align_stop_offset
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def speaking_rate_to_token(
    speaking_rate: float,
    vocab_size: int,
    speaking_rate_vocab_size: int = 5,
    lower_bound: float = 1.5,
    upper_bound: float = 3.5,
) -> int:
    sr = max(lower_bound, min(upper_bound, speaking_rate))
    boundaries = torch.linspace(lower_bound, upper_bound, speaking_rate_vocab_size - 1)
    bucket_idx = int(torch.bucketize(torch.tensor(sr), boundaries))
    return bucket_idx + vocab_size - speaking_rate_vocab_size - 2


def _apply_repetition_penalty(
    logits: torch.Tensor, prev_token_ids, penalty: float,
) -> torch.Tensor:
    """HF/vLLM-style repetition penalty over previously generated tokens:
    positive logits are divided, negative logits multiplied, by ``penalty``."""
    if penalty == 1.0 or not prev_token_ids:
        return logits
    ids = torch.tensor(sorted(set(prev_token_ids)), device=logits.device, dtype=torch.long)
    vals = logits.index_select(0, ids)
    vals = torch.where(vals > 0, vals / penalty, vals * penalty)
    logits = logits.index_copy(0, ids, vals)
    return logits


def _sample(
    logits: torch.Tensor, temperature: float, top_p: float, top_k: int,
    prev_token_ids, repetition_penalty: float,
    generator: torch.Generator | None,
) -> int:
    """Sample one token id from a [V] logit vector, matching the vLLM order:
    repetition penalty -> temperature -> top_k -> top_p -> multinomial."""
    logits = logits.float()
    logits = _apply_repetition_penalty(logits, prev_token_ids, repetition_penalty)
    if temperature <= 0.0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k and top_k > 0 and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumsum - sorted_probs > top_p  # keep tokens whose preceding cumsum <= top_p
        sorted_probs[cutoff] = 0.0
        sorted_probs /= sorted_probs.sum()
        choice = torch.multinomial(sorted_probs, 1, generator=generator)
        return int(sorted_idx[choice])
    return int(torch.multinomial(probs, 1, generator=generator))


@dataclass
class ARResult:
    latents: torch.Tensor          # [T, hidden] decoder hidden states (diffusion input)
    aligned_encoder_latents: torch.Tensor  # [T, hidden] cross-attn weighted text states
    codes: list[int]               # full-vocab audio token ids per step
    mel_codes: list[int]           # audio-vocab code ids (full - audio_token_offset)
    alignments: list[float]        # cumulative alignment after each step
    stop_reason: str


@torch.inference_mode()
def generate(
    decoder,
    config,
    text_hidden: torch.Tensor,        # [Te, hidden]
    text_mask: torch.Tensor,          # [1, Te] (1=valid)
    speaker_emb: torch.Tensor,        # [Ns, spk_hidden]
    speaking_rate: float,
    *,
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 10,
    repetition_penalty: float = 2.0,
    seed: int | None = None,
    plateau_steps: int = 0,
) -> ARResult:
    device = text_hidden.device
    eos_id = config.full_vocab_eos_token_id
    audio_off = config.number_text_tokens + 1
    stop_offset = int(getattr(config, "align_stop_offset", 1) or 0)
    text_len = int(text_mask.sum().item())
    neg_inf = torch.finfo(text_hidden.dtype).min

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    cross = decoder.prepare_cross_kv(text_hidden, speaker_emb)
    kv, conv = decoder.new_caches()
    align = torch.zeros(1, device=device)

    # --- prefill: decoder_start (alignment frozen, output forced to rate) ---
    start = torch.tensor([config.decoder_start_token_id], device=device)
    decoder.forward_step(start, kv, conv, cross, text_mask, align)
    sr_tok = speaking_rate_to_token(
        speaking_rate, config.vocab_size,
        getattr(config, "speaking_rate_vocab_size", 5),
    )

    latents: list[torch.Tensor] = []
    enc_latents: list[torch.Tensor] = []
    codes: list[int] = []
    aligns: list[float] = []
    tok = torch.tensor([sr_tok], device=device)
    stop_reason = "max_tokens"
    prev_round = -1
    plateau = 0

    for _ in range(max_new_tokens):
        logits, latent, step, aligned_enc = decoder.forward_step(
            tok, kv, conv, cross, text_mask, align, text_hidden=text_hidden,
        )
        align = align + step[-1]
        align_r = int(align.round().item())

        # recipe-gated EOS bias
        if align_r > text_len:
            logits[-1, eos_id] = 1e9
        elif align_r < text_len - 4:
            logits[-1, eos_id] = neg_inf

        next_tok = _sample(
            logits[-1], temperature, top_p, top_k,
            codes, repetition_penalty, gen,
        )
        if next_tok == eos_id:
            stop_reason = "eos"
            break

        latents.append(latent[-1])
        if aligned_enc is not None:
            enc_latents.append(aligned_enc[-1])
        codes.append(next_tok)
        aligns.append(float(align.item()))

        # alignment-finished stop (MTL: round(align) >= text_len + offset)
        if align_r >= text_len + stop_offset:
            stop_reason = "alignment"
            break

        # plateau watchdog (optional)
        if plateau_steps > 0:
            if align_r == prev_round:
                plateau += 1
                if plateau >= plateau_steps:
                    stop_reason = "plateau"
                    break
            else:
                plateau = 0
            prev_round = align_r

        tok = torch.tensor([next_tok], device=device)

    if latents:
        latent_seq = torch.stack(latents, dim=0)
    else:
        latent_seq = text_hidden.new_zeros(0, text_hidden.shape[-1])
    if enc_latents:
        enc_seq = torch.stack(enc_latents, dim=0)
    else:
        enc_seq = text_hidden.new_zeros(0, text_hidden.shape[-1])
    mel_codes = [c - audio_off for c in codes]
    return ARResult(
        latents=latent_seq,
        aligned_encoder_latents=enc_seq,
        codes=codes,
        mel_codes=mel_codes,
        alignments=aligns,
        stop_reason=stop_reason,
    )
