# SPDX-License-Identifier: Apache-2.0
"""Batched, static-cache decode runner for the SpeechifyTTS MoE AR decoder.

This is the foundation for continuous batching + CUDA-graph capture. Unlike the
single-sequence loop in :mod:`ar_loop` (growing ``torch.cat`` caches, one
request at a time), this runner:

  * holds ``B`` independent decode *slots* in a fixed set of pre-allocated
    buffers (self-attn KV, conformer-conv state, per-layer cross-attn KV, text
    states, per-slot alignment + position), so concurrent requests share one
    batched forward pass and the GPU stays busy;
  * uses *fixed-shape* attention over the full KV buffer masked by each slot's
    position, so a single decode step has constant shapes and is therefore
    CUDA-graph capturable (the graph path lives in ``cuda_graph.py``);
  * supports per-slot positions, so slots at different ages can decode together
    (true continuous batching) and finished slots can be replaced by freshly
    prefilled ones without disturbing the others.

Numerics match :mod:`ar_loop` / :mod:`decoder` exactly for a single slot; the
only difference is full-buffer masked attention instead of a sliced cache.

Dims for the MoE 4B MTL recipe (validated from the checkpoint): hidden 1536,
12 heads x 128 (MHA, no GQA), 12 layers, self-attn/cross/bias on [2,5,8,11],
conformer conv (kernel 4) on all layers, speaker tokens = 10.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class BatchedDecoder:
    """Owns the static KV/conv/cross buffers for ``B`` decode slots and runs one
    batched decode step over all of them.

    The module weights are *shared* with the eager :class:`SpeechifyDecoder`
    passed in; this class only adds batched buffers + batched math.
    """

    def __init__(self, decoder, config, *, max_batch: int, max_dec_len: int,
                 max_text_len: int, device, dtype, speaker_tokens: int = 10):
        self.decoder = decoder
        self.config = config
        dc = config.decoder_config
        self.B = max_batch
        # one extra *physical* slot used as a no-op padding lane so a captured
        # CUDA graph can always run a fixed bucket size G >= active count.
        self.P = max_batch + 1
        self.dummy = max_batch
        self.L = max_dec_len
        self.Tmax = max_text_len
        self.device = device
        self.dtype = dtype

        self.h = dc.hidden_size
        self.nh = dc.num_attention_heads
        self.nkv = dc.num_key_value_heads
        self.hd = dc.head_dim
        self.scale = dc.query_pre_attn_scalar ** -0.5
        self.num_layers = dc.num_hidden_layers
        self.conv_kernel = getattr(dc, "conv_kernel_size", 4)
        self.conv_inner = self.h * 2  # expansion_factor=2

        self.self_layers = [i for i, l in enumerate(decoder.layers) if l.has_self_attn]
        self.conv_layers = [i for i, l in enumerate(decoder.layers) if l.has_conv]
        self.cross_layers = [i for i, l in enumerate(decoder.layers) if l.has_cross]

        self.audio_off = config.number_text_tokens + 1
        self.eos_id = config.full_vocab_eos_token_id
        self.mask_end_idx = config.number_text_tokens + 1
        self.decoder_start_token_id = config.decoder_start_token_id
        self.stop_offset = int(getattr(config, "align_stop_offset", 1) or 0)
        self.predict_align = getattr(config, "predict_alignment", False)
        self.align_steps = getattr(config, "predict_alignment_max_steps", 1)

        P, L, Tmax = self.P, self.L, self.Tmax
        nkv, nh, hd = self.nkv, self.nh, self.hd
        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)  # noqa: E731

        # self-attention KV cache: [P, L, nkv, hd] per self-attn layer
        self.k_self = {i: z(P, L, nkv, hd) for i in self.self_layers}
        self.v_self = {i: z(P, L, nkv, hd) for i in self.self_layers}
        # conformer conv state: [P, inner, kernel-1] per conv layer
        ck = max(self.conv_kernel - 1, 0)
        self.conv_state = {i: z(P, self.conv_inner, ck) for i in self.conv_layers}
        # cross-attn (text) KV: [P, nh, Tmax, hd]; speaker KV: [P, nh, Ns, hd]
        self.k_text = {i: z(P, nh, Tmax, hd) for i in self.cross_layers}
        self.v_text = {i: z(P, nh, Tmax, hd) for i in self.cross_layers}
        self.speaker_tokens = speaker_tokens
        self.k_spk = {i: z(P, nh, speaker_tokens, hd) for i in self.cross_layers}
        self.v_spk = {i: z(P, nh, speaker_tokens, hd) for i in self.cross_layers}
        # padded text states for aligned-encoder-latent (diffusion input)
        self.text_hidden = z(P, Tmax, self.h)
        self.text_mask = torch.zeros(P, Tmax, device=device, dtype=torch.long)
        self.text_len = torch.zeros(P, device=device, dtype=torch.long)

        # per-slot scalars
        self.tok = torch.zeros(P, device=device, dtype=torch.long)
        self.pos = torch.zeros(P, device=device, dtype=torch.long)  # # cached entries
        self.align = torch.zeros(P, device=device, dtype=torch.float32)
        self.arangeL = torch.arange(L, device=device)

        # --- CUDA-graph state ---
        self._graphs: dict[int, dict] = {}     # bucket size -> capture record
        self._graph_slots: dict[int, torch.Tensor] = {}
        self._buckets: list[int] = []

    # ----------------------------------------------------------------- admit
    @torch.inference_mode()
    def reset_slot(self, b: int) -> None:
        for i in self.self_layers:
            self.k_self[i][b].zero_()
            self.v_self[i][b].zero_()
        for i in self.conv_layers:
            self.conv_state[i][b].zero_()
        self.text_mask[b].zero_()
        self.pos[b] = 0
        self.align[b] = 0.0

    @torch.inference_mode()
    def set_conditioning(self, b: int, text_hidden: torch.Tensor,
                         speaker_emb: torch.Tensor, text_mask: torch.Tensor) -> None:
        """Project + store this slot's cross-attn K/V and text states."""
        te = text_hidden.shape[0]
        assert te <= self.Tmax, f"text_len {te} > Tmax {self.Tmax}"
        th = text_hidden.to(self.dtype)
        for i in self.cross_layers:
            layer = self.decoder.layers[i]
            tk, tv = layer.text_cross_attn.cross_attn.project_kv(th)  # [nh, te, hd]
            self.k_text[i][b, :, :te].copy_(tk)
            self.v_text[i][b, :, :te].copy_(tv)
            spk = layer.cond_proj(speaker_emb.to(self.dtype))
            sk, sv = layer.speaker_cross_attn.cross_attn.project_kv(spk)  # [nh, Ns, hd]
            assert sk.shape[1] == self.speaker_tokens, (
                f"speaker tokens {sk.shape[1]} != preallocated {self.speaker_tokens}")
            self.k_spk[i][b].copy_(sk)
            self.v_spk[i][b].copy_(sv)
        self.text_hidden[b].zero_()
        self.text_hidden[b, :te].copy_(th)
        self.text_mask[b].zero_()
        self.text_mask[b, :te] = text_mask.view(-1)[:te].to(self.text_mask.dtype)
        self.text_len[b] = int(text_mask.sum().item())

    # ------------------------------------------------------------- batched fwd
    def _relative_bias(self, slots: torch.Tensor) -> torch.Tensor | None:
        """Alignment cross bias for the given slots -> [n, heads, 1, Tmax]."""
        if not self.predict_align:
            return None
        align = self.align[slots].view(-1, 1)  # [n, 1] FLOOR handled in bucket
        mask = self.text_mask[slots]            # [n, Tmax]
        return self.decoder.relative_bias.compute_bias_for_cross_alignment(
            alignment=align.to(self.dtype), encoder_attention_mask=mask,
        )  # [n, heads, 1, Tmax]

    @torch.inference_mode()
    def step(self, slots: torch.Tensor):
        """Run one batched decode step for ``slots`` (a 1-D long tensor of slot
        indices). Reads ``self.tok[slots]``, ``self.pos[slots]``,
        ``self.align[slots]`` and the per-slot caches; writes the new KV at each
        slot's position. Returns ``(logits[n,V], latent[n,h], step_diff[n],
        aligned_enc[n,h])`` aligned to ``slots`` order.
        """
        dec = self.decoder
        n = slots.shape[0]
        idx = slots
        pos = self.pos[idx]                       # [n]
        h = dec.embed_tokens(self.tok[idx])       # [n, hidden]
        residual = None
        bias = self._relative_bias(idx)           # [n, heads, 1, Tmax] or None
        cross_w_acc: list[torch.Tensor] = []

        for li, layer in enumerate(dec.layers):
            if layer.has_self_attn:
                h, residual = layer._add_norm(h, residual, layer.pre_self_attn_layernorm)
                h = self._self_attn(layer, li, h, idx, pos)
            if layer.has_conv:
                h, residual = layer._add_norm(h, residual, layer.pre_conv_layernorm)
                h = self._conv(layer, li, h, idx)
            if layer.has_cross:
                h, residual = layer._add_norm(h, residual, layer.pre_cross_attn_layernorm)
                layer_bias = bias if layer.receives_bias else None
                h, cw = self._cross_text(layer, li, h, idx, layer_bias)
                cross_w_acc.append(cw)
                h, residual = layer._add_norm(h, residual, layer.pre_cond_cross_attn_layernorm)
                h = self._cross_spk(layer, li, h, idx)
            h, residual = layer._add_norm(h, residual, layer.pre_mlp_layernorm)
            h = layer.mlp(h)
        h = h + residual                          # [n, hidden]

        aligned_enc = None
        if cross_w_acc:
            avg = torch.stack(cross_w_acc, dim=0).mean(dim=0)   # [n, 1, Tmax]
            th = self.text_hidden[idx]                          # [n, Tmax, hidden]
            aligned_enc = torch.bmm(avg.to(th.dtype), th).squeeze(1)  # [n, hidden]

        logits = dec.mel_head(h)                   # [n, V]
        logits[:, : self.mask_end_idx] = torch.finfo(logits.dtype).min
        if self.decoder_start_token_id is not None:
            logits[:, self.decoder_start_token_id] = torch.finfo(logits.dtype).min

        step_diff = None
        if self.predict_align:
            la = dec.alignment_step_head(h)
            if self.align_steps > 1:
                probs = torch.softmax(la.float(), dim=-1)
                rng = torch.arange(self.align_steps + 1, device=probs.device).float()
                step_diff = torch.clamp((rng * probs).sum(-1), 0.02, float(self.align_steps))
            else:
                p = torch.nan_to_num(torch.sigmoid(la[..., 0].float()), nan=0.99)
                step_diff = torch.clamp(p, 0.02, 1.0)

        # advance positions (write happened at `pos`, so next free slot is pos+1)
        self.pos[idx] = pos + 1
        return logits, h, step_diff, aligned_enc

    # ----------------------------------------------------------- CUDA graphs
    @torch.inference_mode()
    def capture(self, buckets) -> None:
        """Capture a CUDA graph of :meth:`step` for each bucket size in
        ``buckets`` (must be <= ``max_batch``). At replay the active slots are
        written into the graph's fixed ``slots`` buffer and padding lanes point
        at the reserved dummy slot, so one replay serves any active count up to
        the bucket."""
        self._buckets = sorted(b for b in buckets if b <= self.B)
        for G in self._buckets:
            gs = (torch.arange(G, device=self.device) % self.B).long().contiguous()
            self._graph_slots[G] = gs
            for j in range(G):
                b = int(gs[j].item())
                self.pos[b] = 1
                self.tok[b] = 0
                self.align[b] = 0.0
                self.text_mask[b, :4] = 1  # avoid all-masked NaN during warmup
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.step(gs)
            torch.cuda.current_stream().wait_stream(s)
            for j in range(G):
                self.pos[int(gs[j].item())] = 1
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self.step(gs)
            self._graphs[G] = {"graph": g, "out": out, "slots": gs}
        torch.cuda.synchronize()

    @torch.inference_mode()
    def step_graphed(self, active: list[int]):
        """Replay the smallest captured graph that fits ``len(active)`` slots.
        Returns ``(logits[n,V], latent[n,h], step_diff[n], aligned[n,h])`` cloned
        out of the graph's reused buffers (caller may retain them)."""
        n = len(active)
        G = next((b for b in self._buckets if b >= n), None)
        if G is None:
            return None  # falls back to eager in the caller
        rec = self._graphs[G]
        gs = rec["slots"]
        gs[:n].copy_(torch.tensor(active, device=self.device, dtype=torch.long))
        if n < G:
            gs[n:].fill_(self.dummy)
        self.pos[self.dummy] = 0
        self.tok[self.dummy] = 0
        self.align[self.dummy] = 0.0
        rec["graph"].replay()
        logits, latent, step, aligned = rec["out"]
        return (logits[:n], latent[:n].clone(), step[:n],
                aligned[:n].clone() if aligned is not None else None)

    def _self_attn(self, layer, li, x, idx, pos):
        sa = layer.self_attn
        n = x.shape[0]
        q = sa.self_attn_q_proj(x).view(n, self.nh, self.hd)
        k = sa.self_attn_k_proj(x).view(n, self.nkv, self.hd)
        v = sa.self_attn_v_proj(x).view(n, self.nkv, self.hd)
        # write new k,v at each slot's position
        self.k_self[li][idx, pos] = k
        self.v_self[li][idx, pos] = v
        kk = self.k_self[li][idx].transpose(1, 2)  # [n, nkv, L, hd]
        vv = self.v_self[li][idx].transpose(1, 2)
        if self.nkv != self.nh:
            rep = self.nh // self.nkv
            kk = kk.repeat_interleave(rep, dim=1)
            vv = vv.repeat_interleave(rep, dim=1)
        qh = q.unsqueeze(2)  # [n, nh, 1, hd]
        valid = self.arangeL.unsqueeze(0) <= pos.unsqueeze(1)          # [n, L] True=attend
        out = F.scaled_dot_product_attention(
            qh, kk, vv, attn_mask=valid[:, None, None, :], scale=self.scale,
        ).squeeze(2)                                                  # [n, nh, hd]
        out = out.reshape(n, self.nh * self.hd)
        return sa.self_attn_o_proj(out)

    def _conv(self, layer, li, x, idx):
        conv = layer.conv
        n = x.shape[0]
        g = conv.pointwise_conv1(x)               # [n, 2*inner]
        a, b = g.chunk(2, dim=-1)
        g = a * b.sigmoid()                       # [n, inner]
        k = self.conv_kernel
        state = self.conv_state[li][idx]          # [n, inner, k-1]
        win = torch.cat([state, g.unsqueeze(-1)], dim=2)  # [n, inner, k]
        if k > 1:
            self.conv_state[li][idx] = win[:, :, 1:].detach()
        y = conv.depthwise_conv(win)              # [n, inner, 1]
        y = F.silu(y).squeeze(-1)                 # [n, inner]
        return conv.pointwise_conv2(y)

    def _cross_text(self, layer, li, x, idx, bias):
        ca = layer.text_cross_attn.cross_attn
        n = x.shape[0]
        q = ca.q_proj(x).view(n, self.nh, self.hd)        # [n, nh, hd]
        kk = self.k_text[li][idx]                          # [n, nh, Tmax, hd]
        vv = self.v_text[li][idx]
        scores = torch.matmul(q.unsqueeze(2), kk.transpose(-1, -2)) * ca.scale  # [n,nh,1,Tmax]
        if bias is not None:
            scores = scores + bias.to(scores.dtype)
        else:
            # still must mask padding positions beyond the text
            valid = self.text_mask[idx].bool()             # [n, Tmax]
            scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)  # [n, nh, 1, Tmax]
        out = torch.matmul(weights, vv).squeeze(2).reshape(n, self.nh * self.hd)
        wmean = weights.mean(dim=1)                         # [n, 1, Tmax]
        return ca.o_proj(out), wmean

    def _cross_spk(self, layer, li, x, idx):
        ca = layer.speaker_cross_attn.cross_attn
        n = x.shape[0]
        q = ca.q_proj(x).view(n, self.nh, self.hd)
        kk = self.k_spk[li][idx]
        vv = self.v_spk[li][idx]
        out = F.scaled_dot_product_attention(
            q.unsqueeze(2), kk, vv, attn_mask=None, scale=ca.scale,
        ).squeeze(2).reshape(n, self.nh * self.hd)
        return ca.o_proj(out)
