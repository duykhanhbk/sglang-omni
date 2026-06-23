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
import re
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


_RID_COUNTER = 0


def _next_rid() -> int:
    global _RID_COUNTER
    _RID_COUNTER += 1
    return _RID_COUNTER


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
    stream: bool = False
    out_q: object = None              # queue.Queue of streaming events when stream=True
    rid: int = field(default_factory=_next_rid)


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
    # incremental speechmark state (streaming only)
    words: list = field(default_factory=list)   # [{value,start,end}]
    word_ptr: int = 0
    char_total: int = 1


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
        # Per-request streaming vocoder sessions, owned by the vocode thread.
        self._sessions: dict = {}
        self._vocode_thread = threading.Thread(target=self._vocode_loop, name="cb-vocode", daemon=True)
        self._vocode_thread.start()
        self._thread = threading.Thread(target=self._loop, name="cb-decode", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------- public API
    def submit(self, text: str, ref_audio, **kw) -> Future:
        req = _Req(text=text, ref_audio=ref_audio, t_submit=time.perf_counter(), **kw)
        self._queue.put(req)
        return req.future

    def submit_stream(self, text: str, ref_audio, **kw) -> "queue.Queue":
        """Submit a request for *true* streaming synthesis.

        Returns a ``queue.Queue`` of events; the audio starts flowing as soon
        as the first diffusion block is vocoded (overlapping the AR decode),
        giving low time-to-first-audio:

          * ``("audio", pcm16_bytes)``  — one decoded block (mono PCM16 LE)
          * ``("marks", marks_list)``   — word speechmarks (once AR finishes)
          * ``("final", meta_dict)``    — num_mel_codes / stop_reason / rate / duration
          * ``("error", message)``      — synthesis failed
          * ``None``                    — end sentinel
        """
        q: "queue.Queue" = queue.Queue(maxsize=512)
        req = _Req(text=text, ref_audio=ref_audio, t_submit=time.perf_counter(),
                   stream=True, out_q=q, **kw)
        self._queue.put(req)
        return q

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

    def _emit_marks(self, st: _Slot, align_val: float) -> None:
        """Emit any word marks whose alignment threshold is now crossed.

        Mirrors :func:`engine._compute_speechmarks` but incrementally: because
        the per-frame alignment is monotonic, a word's start frame is the first
        frame whose alignment reaches the word's char-start fraction. Emitting
        marks as soon as AR commits that frame (well ahead of when the audio
        plays, since AR runs faster than realtime) lets the UI highlight live
        from the very first word.
        """
        if st.req.out_q is None or not st.words:
            return
        frame_idx = len(st.aligns) - 1
        tl = max(st.text_n_tokens, 1)
        new = []
        while st.word_ptr < len(st.words):
            w = st.words[st.word_ptr]
            if (align_val / tl) >= (w["start"] / st.char_total):
                new.append({
                    "value": w["value"], "startIndex": w["start"], "endIndex": w["end"],
                    "startTime": int(round(frame_idx * self.frame_dur_ms)),
                })
                st.word_ptr += 1
            else:
                break
        if new:
            st.req.out_q.put(("marks", new))

    def _flush_remaining_marks(self, st: _Slot) -> None:
        """Flush trailing words whose threshold was never reached (clamped to
        the final frame, matching the offline speechmark behavior)."""
        if st.req.out_q is None or st.word_ptr >= len(st.words):
            return
        frame_idx = max(len(st.aligns) - 1, 0)
        t = int(round(frame_idx * self.frame_dur_ms))
        new = [{
            "value": w["value"], "startIndex": w["start"], "endIndex": w["end"],
            "startTime": t,
        } for w in st.words[st.word_ptr:]]
        st.word_ptr = len(st.words)
        st.req.out_q.put(("marks", new))

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
        if req.stream:
            # Spin up the streaming vocoder session on the vocode thread so
            # its caches + cuBLAS handle live entirely on that worker.
            self._vocode_q.put({
                "type": "start", "rid": req.rid, "req": req, "spk": spk,
                "prompt_mels": vf.speech_prompt_mels,
            })
            slot = self._slots[b]
            slot.words = [
                {"value": m.group().strip(), "start": m.start(), "end": m.end()}
                for m in re.finditer(r"\S+", req.text)
            ]
            slot.char_total = max(len(req.text), 1)
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
        if st.req.stream:
            # Incremental path: latents were already pushed + marks streamed
            # during decode. Flush any trailing words, then queue the tail.
            self._flush_remaining_marks(st)
            self._vocode_q.put({
                "type": "finish", "rid": st.req.rid, "req": st.req,
                "codes_len": len(st.codes), "rate": st.rate,
                "stop_reason": stop_reason,
            })
            return
        # stack latents now (cheap copy) so the slot's caches can be reused
        latents = torch.stack(st.latents, dim=0) if st.latents else None
        enc = torch.stack(st.enc_latents, dim=0) if st.enc_latents else None
        self._vocode_q.put({
            "type": "batch",
            "req": st.req, "latents": latents, "enc": enc, "vf": st.vf,
            "codes": st.codes, "aligns": st.aligns, "text_n_tokens": st.text_n_tokens,
            "rate": st.rate, "stop_reason": stop_reason,
        })

    def _to_pcm(self, audio: torch.Tensor) -> bytes:
        """``[1, n]`` or ``[n]`` float tensor -> mono PCM16 LE bytes."""
        import numpy as np
        a = audio.reshape(-1).clamp(-1, 1).cpu().numpy()
        return (a * 32767.0).astype("<i2").tobytes()

    @torch.inference_mode()
    def _vocode_loop(self) -> None:
        while not self._stop:
            try:
                job = self._vocode_q.get(timeout=0.5)
            except queue.Empty:
                continue
            jtype = job.get("type", "batch")
            try:
                if jtype == "start":
                    sess = self.eng.vocoder.new_streaming_session(
                        job["spk"], job["prompt_mels"],
                    )
                    sess._req = job["req"]
                    sess._out_samples = 0
                    self._sessions[job["rid"]] = sess
                elif jtype == "push":
                    self._stream_push(job)
                elif jtype == "finish":
                    self._stream_finish(job)
                else:
                    self._vocode_batch(job)
            except Exception as exc:  # noqa: BLE001
                self._fail_job(job, exc)

    def _fail_job(self, job: dict, exc: Exception) -> None:
        rid = job.get("rid")
        if rid is not None and rid in self._sessions:
            self._sessions.pop(rid, None)
        req = job.get("req")
        if req is not None and getattr(req, "stream", False) and req.out_q is not None:
            req.out_q.put(("error", str(exc)))
            req.out_q.put(None)
        elif req is not None and not req.future.done():
            req.future.set_exception(exc)

    @torch.inference_mode()
    def _stream_push(self, job: dict) -> None:
        sess = self._sessions.get(job["rid"])
        if sess is None:
            return
        lat = job["latent"].reshape(1, -1)
        enc = job["enc"].reshape(1, -1) if job["enc"] is not None else lat
        sess.push(lat, enc)
        for chunk in sess.drain():
            sess._out_samples += int(chunk.shape[-1])
            sess._req.out_q.put(("audio", self._to_pcm(chunk)))

    @torch.inference_mode()
    def _stream_finish(self, job: dict) -> None:
        sess = self._sessions.pop(job["rid"], None)
        req = job["req"]
        dur = 0.0
        if sess is not None:
            tail = sess.finish()
            if tail is not None and tail.numel() > 0:
                sess._out_samples += int(tail.shape[-1])
                req.out_q.put(("audio", self._to_pcm(tail)))
            dur = round(sess._out_samples / self.sample_rate, 3)
        req.out_q.put(("final", {
            "num_mel_codes": job["codes_len"],
            "stop_reason": job["stop_reason"],
            "speaking_rate": job["rate"],
            "duration_s": dur,
        }))
        req.out_q.put(None)

    @torch.inference_mode()
    def _vocode_batch(self, job: dict) -> None:
        req = job["req"]
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
            st.aligns.append(align_list[j])
            if st.req.stream:
                # Overlap vocoding: ship this token's latent to the streaming
                # session immediately (a block is vocoded once enough arrive).
                self._vocode_q.put({
                    "type": "push", "rid": st.req.rid,
                    "latent": latents[j], "enc": aligned[j] if aligned is not None else None,
                })
                self._emit_marks(st, align_list[j])
            else:
                st.latents.append(latents[j])
                if aligned is not None:
                    st.enc_latents.append(aligned[j])
            if align_r_list[j] >= st.text_len + self.bd.stop_offset:
                finished.append((b, "alignment"))
                continue
            if st.steps >= st.req.max_new_tokens:
                finished.append((b, "max_tokens"))
                continue
        for b, reason in finished:
            self._finalize(b, reason)
