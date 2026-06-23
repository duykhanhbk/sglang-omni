# SPDX-License-Identifier: Apache-2.0
"""Continuous-batching TTS engine for the SpeechifyTTS MoE decoder.

Wraps the validated single-request components (voice extractor, text encoder,
diffusion vocoder, tokenizer) and the :class:`BatchedDecoder` in a background
decode thread that keeps up to ``max_batch`` requests in flight at once:

  * a single GPU worker thread runs everything (voice/encode/prefill/decode/
    vocode) so the per-thread cuBLAS handle stays warm and there is no
    event-loop / GIL contention on the launch-bound decode loop;
  * new requests are admitted into free slots and *prefilled* without
    disturbing slots already mid-decode (true continuous batching), so the GPU
    stays busy and throughput scales with concurrency;
  * finished slots are vocoded and their futures resolved immediately, freeing
    the slot for the next queued request.

The decode step is the hot path and has fixed shapes per active-slot count,
which the CUDA-graph layer (``cuda_graph.py``) can capture/replay.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field

import torch

from .ar_loop import speaking_rate_to_token
from .batched import BatchedDecoder
from .engine import SpeechifyTTSEngine, _compute_speechmarks

logger = logging.getLogger(__name__)


def _sample_batch(logits, temp, top_p, top_k, seen, rep_pen, gen):
    """Vectorised version of :func:`ar_loop._sample` over ``[n, V]`` logits.

    Matches the per-row order exactly (repetition penalty -> temperature ->
    top_k -> softmax -> top_p -> multinomial) but processes the whole active
    batch in one shot so the decode loop incurs a single host sync per step
    instead of ~3 per slot. ``seen`` is a ``[n, V]`` bool of previously emitted
    tokens (for the penalty); ``temp/top_p/rep_pen`` are ``[n]`` and ``top_k`` is
    ``[n]`` long.
    """
    logits = logits.float()
    V = logits.shape[1]
    pen = rep_pen.unsqueeze(1)
    penalized = torch.where(logits > 0, logits / pen, logits * pen)
    logits = torch.where(seen, penalized, logits)
    logits = logits / temp.unsqueeze(1).clamp(min=1e-6)
    maxk = int(top_k.clamp(min=0).max().item())
    if 0 < maxk < V:
        vals, _ = torch.topk(logits, maxk, dim=-1)            # [n, maxk]
        kth = vals.gather(1, (top_k.clamp(1, maxk) - 1).unsqueeze(1))  # [n,1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    sp, si = torch.sort(probs, descending=True, dim=-1)
    cum = sp.cumsum(-1)
    cutoff = (cum - sp) > top_p.unsqueeze(1)
    sp = sp.masked_fill(cutoff, 0.0)
    sp = sp / sp.sum(-1, keepdim=True).clamp(min=1e-12)
    choice = torch.multinomial(sp, 1, generator=gen)          # [n,1]
    return si.gather(1, choice).squeeze(1)                     # [n]


@dataclass
class _Req:
    text: str
    ref_audio: object                 # path str or waveform tensor (cpu)
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 10
    repetition_penalty: float = 2.0
    seed: int | None = None
    speaking_rate: float | None = None
    max_new_tokens: int = 2048
    future: Future = field(default_factory=Future)
    t_submit: float = 0.0


@dataclass
class _Slot:
    req: _Req
    vf: object
    text_len: int
    text_n_tokens: int
    rate: float
    gen: torch.Generator | None
    codes: list = field(default_factory=list)
    latents: list = field(default_factory=list)
    enc_latents: list = field(default_factory=list)
    aligns: list = field(default_factory=list)
    steps: int = 0
    t_admit: float = 0.0


class CBEngine:
    """Continuous-batching wrapper around :class:`SpeechifyTTSEngine`."""

    def __init__(self, model_path: str, *, max_batch: int = 8, max_dec_len: int = 1024,
                 max_text_len: int = 512, device: str = "cuda", dtype=torch.bfloat16,
                 use_cuda_graph: bool = True, graph_buckets=(1, 2, 4, 8)):
        self.eng = SpeechifyTTSEngine(model_path, device=device, dtype=dtype)
        self.cfg = self.eng.config
        self.device = device
        self.dtype = dtype
        self.max_batch = max_batch
        self.bd = BatchedDecoder(
            self.eng.decoder, self.cfg, max_batch=max_batch, max_dec_len=max_dec_len,
            max_text_len=max_text_len, device=device, dtype=dtype,
        )
        self.eos_id = self.cfg.full_vocab_eos_token_id
        self.audio_off = self.cfg.number_text_tokens + 1
        self.neg_inf = torch.finfo(dtype).min
        self.frame_dur_ms = (
            self.eng.vocoder.diffusion_upsample_factor * self.eng.vocoder.hop_length
            / self.eng.sample_rate * 1000.0
        )
        self.sample_rate = self.eng.sample_rate

        self.use_cuda_graph = use_cuda_graph and str(device).startswith("cuda")
        self.graph_buckets = tuple(b for b in graph_buckets if b <= max_batch)
        self._captured = False
        self.vocab_size = self.cfg.vocab_size
        # per-slot "previously emitted token" mask for the repetition penalty
        self.seen = torch.zeros(max_batch, self.vocab_size, dtype=torch.bool, device=device)
        self._sample_gen = torch.Generator(device=device)
        self._sample_gen.manual_seed(1234)

        self._queue: "queue.Queue[_Req]" = queue.Queue()
        self._slots: list[_Slot | None] = [None] * max_batch
        self._stop = False
        # vocoder runs on its own thread so the decode loop is never blocked by
        # the (~0.18s) diffusion decode when a slot finishes -- the slot is freed
        # and re-admitted immediately, keeping the GPU decode pipeline full.
        self._vocode_q: "queue.Queue[dict]" = queue.Queue()
        self._vocode_thread = threading.Thread(target=self._vocode_loop, name="cb-vocode", daemon=True)
        self._vocode_thread.start()
        self._thread = threading.Thread(target=self._loop, name="cb-decode", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------- public API
    def submit(self, text: str, ref_audio, **kw) -> Future:
        req = _Req(text=text, ref_audio=ref_audio, t_submit=time.perf_counter(), **kw)
        self._queue.put(req)
        return req.future

    def warmup(self, passes: int = 3) -> None:
        g = torch.Generator().manual_seed(0)
        ref = (torch.randn(3 * 24000, generator=g) * 0.05).clamp(-1, 1)
        txt = ("This is a long warmup utterance that drives the batched decode "
               "loop through several hundred steps so cuBLAS heuristics and every "
               "CUDA kernel reach steady state before the first real request.")
        futs = [self.submit(txt, ref, max_new_tokens=600, seed=0) for _ in range(passes)]
        for f in futs:
            try:
                f.result(timeout=120)
            except Exception as exc:  # noqa: BLE001
                logger.warning("warmup failed (non-fatal): %s", exc)
        torch.cuda.synchronize()
        logger.info("CBEngine warmup complete (%d passes)", passes)

    def shutdown(self) -> None:
        self._stop = True

    # ------------------------------------------------------------- decode loop
    def _free_slots(self) -> list[int]:
        return [i for i in range(self.max_batch) if self._slots[i] is None]

    def _active_slots(self) -> list[int]:
        return [i for i in range(self.max_batch) if self._slots[i] is not None]

    @torch.inference_mode()
    def _admit(self, b: int, req: _Req) -> None:
        eng = self.eng
        vf = eng.voice_extractor.extract(req.ref_audio)
        spk = vf.speaker_embedding.to(self.device).to(self.dtype)
        rate = req.speaking_rate if req.speaking_rate is not None else vf.speaking_rate
        ids = torch.tensor(eng.tokenizer.encode(req.text), device=self.device)
        th = eng.encoder(ids).to(self.dtype)
        tm = torch.ones(1, th.shape[0], dtype=torch.long, device=self.device)
        self.bd.reset_slot(b)
        self.bd.set_conditioning(b, th, spk, tm)
        self.seen[b].zero_()
        self._slots[b] = _Slot(
            req=req, vf=vf, text_len=int(tm.sum().item()),
            text_n_tokens=th.shape[0], rate=float(rate), gen=None,
            t_admit=time.perf_counter(),
        )
        # prefill the decoder-start token (alignment frozen); KV[b,0] populated
        self.bd.tok[b] = self.cfg.decoder_start_token_id
        self.bd.step(torch.tensor([b], device=self.device))
        self.bd.align[b] = 0.0
        sr_tok = speaking_rate_to_token(
            rate, self.cfg.vocab_size, getattr(self.cfg, "speaking_rate_vocab_size", 5),
        )
        self.bd.tok[b] = sr_tok

    @torch.inference_mode()
    def _finalize(self, b: int, stop_reason: str) -> None:
        """Hand the finished slot's latents to the vocoder thread and free the
        slot immediately so the decode loop can admit the next request."""
        st = self._slots[b]
        self._slots[b] = None
        # stack latents now (cheap copy) so the slot's caches can be reused
        latents = torch.stack(st.latents, dim=0) if st.latents else None
        enc = torch.stack(st.enc_latents, dim=0) if st.enc_latents else None
        self._vocode_q.put({
            "req": st.req, "latents": latents, "enc": enc, "vf": st.vf,
            "codes": st.codes, "aligns": st.aligns, "text_n_tokens": st.text_n_tokens,
            "rate": st.rate, "stop_reason": stop_reason,
        })

    @torch.inference_mode()
    def _vocode_loop(self) -> None:
        while not self._stop:
            try:
                job = self._vocode_q.get(timeout=0.5)
            except queue.Empty:
                continue
            req = job["req"]
            try:
                if job["latents"] is not None:
                    audio = self.eng.vocoder.generate(
                        job["latents"], job["enc"],
                        job["vf"].speaker_embedding.to(self.device),
                        job["vf"].speech_prompt_mels, seed=req.seed,
                    )[0].float().cpu()
                else:
                    audio = torch.zeros(0)
                marks = _compute_speechmarks(
                    req.text, job["aligns"], job["text_n_tokens"], self.frame_dur_ms)
                req.future.set_result({
                    "audio": audio,
                    "sample_rate": self.sample_rate,
                    "num_mel_codes": len(job["codes"]),
                    "stop_reason": job["stop_reason"],
                    "speaking_rate": job["rate"],
                    "marks": marks,
                })
            except Exception as exc:  # noqa: BLE001
                if not req.future.done():
                    req.future.set_exception(exc)

    def _loop(self) -> None:
        dev = self.device
        if self.use_cuda_graph and not self._captured:
            try:
                logger.info("capturing CUDA graphs for buckets %s ...", self.graph_buckets)
                self.bd.capture(self.graph_buckets)
                self._captured = True
                logger.info("CUDA graph capture complete")
            except Exception as exc:  # noqa: BLE001
                logger.warning("CUDA graph capture failed (%s); using eager decode", exc)
                self.use_cuda_graph = False
        while not self._stop:
            # 1. admit queued requests into free slots (prefill)
            new = []
            for b in self._free_slots():
                try:
                    req = self._queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._admit(b, req)
                    new.append(b)
                except Exception as exc:  # noqa: BLE001
                    self._slots[b] = None
                    if not req.future.done():
                        req.future.set_exception(exc)

            active = self._active_slots()
            if not active:
                # nothing running; block for the next request then loop
                try:
                    req = self._queue.get(timeout=0.5)
                    self._queue.put(req)
                except queue.Empty:
                    pass
                continue

            # 2. batched decode step over all active slots (CUDA-graph replay
            #    when a bucket fits, else eager)
            with torch.inference_mode():
                out = None
                if self.use_cuda_graph and self._captured:
                    out = self.bd.step_graphed(active)
                if out is None:
                    out = self.bd.step(torch.tensor(active, device=dev))
                logits, latents, step, aligned = out
                self._postprocess(active, logits, latents, step, aligned)

    @torch.inference_mode()
    def _postprocess(self, active, logits, latents, step, aligned) -> None:
        dev = self.device
        n = len(active)
        slots_t = torch.tensor(active, device=dev)
        sts = [self._slots[b] for b in active]

        # advance per-slot alignment (vectorised)
        self.bd.align[slots_t] = self.bd.align[slots_t] + step
        align = self.bd.align[slots_t]                       # [n]
        align_r = align.round()
        text_len = torch.tensor([st.text_len for st in sts], device=dev, dtype=torch.float32)

        # recipe-gated EOS bias (vectorised over the batch)
        lg = logits.float()
        force = align_r > text_len
        suppress = align_r < (text_len - 4)
        lg[force, self.eos_id] = 1e9
        lg[suppress, self.eos_id] = float("-inf")

        # batched sampling
        reqs = [st.req for st in sts]
        temp = torch.tensor([r.temperature for r in reqs], device=dev)
        top_p = torch.tensor([r.top_p for r in reqs], device=dev)
        top_k = torch.tensor([r.top_k for r in reqs], device=dev, dtype=torch.long)
        rep = torch.tensor([r.repetition_penalty for r in reqs], device=dev)
        seen = self.seen[slots_t]                            # [n, V]
        toks = _sample_batch(lg, temp, top_p, top_k, seen, rep, self._sample_gen)

        # feed sampled tokens back as next input in one vectorised write
        # (finished slots are overwritten harmlessly -- they get freed below)
        self.bd.tok[slots_t] = toks
        # record emitted tokens for the repetition-penalty mask (vectorised)
        self.seen[slots_t, toks] = True

        # single host sync for the whole batch
        tok_list = toks.tolist()
        align_list = align.tolist()
        align_r_list = align_r.tolist()

        finished: list[tuple[int, str]] = []
        for j, b in enumerate(active):
            st = sts[j]
            nt = tok_list[j]
            st.steps += 1
            if nt == self.eos_id:
                finished.append((b, "eos"))
                continue
            st.codes.append(nt)
            st.latents.append(latents[j])
            if aligned is not None:
                st.enc_latents.append(aligned[j])
            st.aligns.append(align_list[j])
            if align_r_list[j] >= st.text_len + self.bd.stop_offset:
                finished.append((b, "alignment"))
                continue
            if st.steps >= st.req.max_new_tokens:
                finished.append((b, "max_tokens"))
                continue
        for b, reason in finished:
            self._finalize(b, reason)
