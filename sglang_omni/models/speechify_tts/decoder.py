# SPDX-License-Identifier: Apache-2.0
"""Torch-native SpeechifyTTS MoE AR decoder (self-contained decode loop).

Port of vllm-omni ``speechify_t5_tts/decoder.py`` without the vLLM coupling
(RadixAttention KV cache, mamba conv-state cache, ``fused_experts``,
``@support_torch_compile``). Parameter names match the converted checkpoint so
weights load directly (see :meth:`SpeechifyDecoder.load_weights`).

Decoder layer (MoE 4B MTL recipe), per the trained config:
  - self-attention on layers [2, 5, 8, 11] (NoPE, causal, MHA 12x128, scale 256^-.5)
  - conformer causal conv on ALL layers (Linear pointwise, kernel 4, 3-frame state)
  - text cross-attention on [2, 5, 8, 11] with alignment relative-position bias
  - speaker cross-attention on [2, 5, 8, 11] (cond_proj 1024->1536)
  - dense SwiGLU MLP on layers 0,1; Sparse-MoE (48 experts, top-4) on 2..11

The residual chain mirrors vLLM's fused add+RMSNorm exactly: each sub-layer's
norm folds the previous sub-layer's residual add, and the final hidden state is
``h + residual`` with no final norm (GPTTTS has none).

Alignment: a per-request float position advances by the multi-class
``alignment_step_head`` expected value each decode step; the text cross-attention
bias is centred on ``floor(alignment)`` (MTL recipe, ``align_stop_offset>=1``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang_omni.models.speechify_tts.layers import RMSNorm, SwiGLUMLP
from sglang_omni.models.speechify_tts.moe import SparseMoeBlock, is_moe_layer


def _relative_position_bucket(
    relative_position: torch.Tensor,
    bidirectional: bool = True,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )
    relative_buckets += torch.where(is_small, relative_position.to(torch.long), relative_position_if_large)
    return relative_buckets


class T5RelativeAttentionBias(nn.Module):
    """Alignment cross-attention bias (T5-style relative-position embedding)."""

    def __init__(self, relative_attention_num_buckets: int, relative_attention_max_distance: int, n_heads: int):
        super().__init__()
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, n_heads)

    def compute_bias_for_cross_alignment(
        self, alignment: torch.Tensor, encoder_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """alignment [B, Q], encoder_attention_mask [B, K] (1=valid). -> [B, n_heads, Q, K]."""
        device = alignment.device
        key_length = encoder_attention_mask.shape[1]
        bsz, query_len = alignment.shape
        memory_position = torch.arange(key_length, dtype=torch.long, device=device).view(1, 1, key_length)
        relative_position = memory_position - alignment[:, :, None].to(torch.long)
        bucket = _relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        ).clamp(0, self.relative_attention_num_buckets - 1)
        bias = self.relative_attention_bias.weight.index_select(0, bucket.view(-1)).view(
            bsz, query_len, key_length, self.n_heads
        )
        bias = bias.permute(0, 3, 1, 2)  # [B, h, Q, K]
        add_mask = (1.0 - encoder_attention_mask[:, None, None, :].to(bias.dtype)) * torch.finfo(
            bias.dtype
        ).min
        return bias + add_mask


class _SelfAttention(nn.Module):
    """Causal MHA with an incremental KV cache (NoPE)."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, scale):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.self_attn_q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.self_attn_k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.self_attn_v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.self_attn_o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: dict) -> torch.Tensor:
        # x [T, hidden]; kv_cache holds growing K,V [Tcache, kvh, hd]
        t = x.shape[0]
        q = self.self_attn_q_proj(x).view(t, self.num_heads, self.head_dim)
        k = self.self_attn_k_proj(x).view(t, self.num_kv_heads, self.head_dim)
        v = self.self_attn_v_proj(x).view(t, self.num_kv_heads, self.head_dim)
        if kv_cache.get("k") is None:
            k_all, v_all = k, v
        else:
            k_all = torch.cat([kv_cache["k"], k], dim=0)
            v_all = torch.cat([kv_cache["v"], v], dim=0)
        kv_cache["k"] = k_all
        kv_cache["v"] = v_all
        # [h, Tq, d] x [h, Tk, d]
        qh = q.transpose(0, 1)
        kh = k_all.transpose(0, 1)
        vh = v_all.transpose(0, 1)
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            kh = kh.repeat_interleave(rep, dim=0)
            vh = vh.repeat_interleave(rep, dim=0)
        tk = k_all.shape[0]
        if t == 1:
            # Single-token decode: the lone query attends to every cached key
            # (all causal), so no mask is needed -- avoids a per-step host-side
            # mask build that serialises the decode loop.
            attn_mask = None
        else:
            # Prefill: query position i (offset by tk-t) attends to keys <= abs pos.
            offset = tk - t
            attn_mask = torch.full((1, 1, t, tk), float("-inf"), device=x.device, dtype=qh.dtype)
            for i in range(t):
                attn_mask[0, 0, i, : offset + i + 1] = 0.0
        out = F.scaled_dot_product_attention(
            qh.unsqueeze(0), kh.unsqueeze(0), vh.unsqueeze(0),
            attn_mask=attn_mask, scale=self.scale,
        ).squeeze(0)
        out = out.transpose(0, 1).reshape(t, self.num_heads * self.head_dim)
        return self.self_attn_o_proj(out)


class _CrossAttention(nn.Module):
    """Cross-attention with pre-projected K,V cache and optional additive bias."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, kv_dim, scale):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def project_kv(self, enc: torch.Tensor):
        # enc [Te, kv_dim] -> k,v [kvh, Te, hd]
        te = enc.shape[0]
        k = self.k_proj(enc).view(te, self.num_kv_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(enc).view(te, self.num_kv_heads, self.head_dim).transpose(0, 1)
        return k.contiguous(), v.contiguous()

    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                bias: torch.Tensor | None = None,
                return_weights: bool = False):
        t = x.shape[0]
        q = self.q_proj(x).view(t, self.num_heads, self.head_dim).transpose(0, 1)
        kh, vh = k, v
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            kh = kh.repeat_interleave(rep, dim=0)
            vh = vh.repeat_interleave(rep, dim=0)
        attn_mask = None
        if bias is not None:
            # bias [h, Q, K]; broadcast Q to t
            attn_mask = bias.to(q.dtype)
            if attn_mask.shape[1] == 1 and t > 1:
                attn_mask = attn_mask.expand(-1, t, -1)
        if return_weights:
            # Explicit softmax so we can return per-head attention weights
            # (used to build aligned_encoder_latent for the diffusion stage).
            scores = torch.matmul(q, kh.transpose(-2, -1)) * self.scale  # [h, Q, K]
            if attn_mask is not None:
                scores = scores + attn_mask
            weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)  # [h, Q, K]
            out = torch.matmul(weights, vh)  # [h, Q, hd]
            out = out.transpose(0, 1).reshape(t, self.num_heads * self.head_dim)
            return self.o_proj(out), weights.mean(dim=0)  # weights_mean [Q, K]
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0), kh.unsqueeze(0), vh.unsqueeze(0),
            attn_mask=None if attn_mask is None else attn_mask.unsqueeze(0), scale=self.scale,
        ).squeeze(0)
        out = out.transpose(0, 1).reshape(t, self.num_heads * self.head_dim)
        return self.o_proj(out)


class _DecoderConv(nn.Module):
    """Conformer causal conv: Linear pw1 -> GLU -> depthwise(kernel) -> SiLU -> Linear pw2."""

    def __init__(self, dim, expansion_factor=2, kernel_size=4):
        super().__init__()
        self.inner_dim = dim * expansion_factor
        self.kernel_size = kernel_size
        self.pointwise_conv1 = nn.Linear(dim, self.inner_dim * 2, bias=True)
        self.depthwise_conv = nn.Conv1d(
            self.inner_dim, self.inner_dim, kernel_size, groups=self.inner_dim, bias=True
        )
        self.pointwise_conv2 = nn.Linear(self.inner_dim, dim, bias=True)

    def forward(self, x: torch.Tensor, conv_cache: dict) -> torch.Tensor:
        # x [T, dim]
        t = x.shape[0]
        g = self.pointwise_conv1(x)  # [T, 2*inner]
        a, b = g.chunk(2, dim=-1)
        g = a * b.sigmoid()  # GLU -> [T, inner]
        gt = g.t().unsqueeze(0)  # [1, inner, T]
        k = self.kernel_size
        prev = conv_cache.get("state")
        if prev is None:
            left = F.pad(gt, (k - 1, 0))
        else:
            left = torch.cat([prev, gt], dim=2)
        conv_cache["state"] = left[:, :, -(k - 1):].detach() if k > 1 else None
        y = self.depthwise_conv(left)  # [1, inner, T]
        y = F.silu(y)
        y = y.squeeze(0).t()  # [T, inner]
        return self.pointwise_conv2(y)


class _DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        dc = config.decoder_config
        self.layer_idx = layer_idx
        h = dc.hidden_size
        nh, nkv, hd = dc.num_attention_heads, dc.num_key_value_heads, dc.head_dim
        scale = dc.query_pre_attn_scalar ** -0.5

        self_idx = getattr(dc, "self_attention_layer_indices", list(range(dc.num_hidden_layers)))
        self.has_self_attn = layer_idx in self_idx
        if self.has_self_attn:
            self.pre_self_attn_layernorm = RMSNorm(h, eps=1e-5)
            self.self_attn = _SelfAttention(h, nh, nkv, hd, scale)

        conv_idx = dc.conv_layer_indices if dc.conv_layer_indices is not None else list(range(dc.num_hidden_layers))
        self.has_conv = layer_idx in conv_idx
        if self.has_conv:
            self.pre_conv_layernorm = RMSNorm(h, eps=1e-5)
            self.conv = _DecoderConv(h, 2, getattr(dc, "conv_kernel_size", 4))

        cross_idx = getattr(dc, "cross_attention_layer_indices", list(range(dc.num_hidden_layers)))
        self.has_cross = layer_idx in cross_idx
        bias_idx = getattr(dc, "cross_attention_bias_layer_indices", list(range(dc.num_hidden_layers)))
        self.receives_bias = layer_idx in bias_idx
        if self.has_cross:
            self.pre_cross_attn_layernorm = RMSNorm(h, eps=1e-5)
            self.text_cross_attn = _CrossAttentionWrapper(
                _CrossAttention(h, nh, nh, hd, dc.text_cross_attention_hidden_size, scale)
            )
            self.pre_cond_cross_attn_layernorm = RMSNorm(h, eps=1e-5)
            self.cond_proj = nn.Linear(dc.speaker_cross_attention_hidden_size, h, bias=False)
            self.speaker_cross_attn = _CrossAttentionWrapper(
                _CrossAttention(h, nh, nh, hd, h, scale)
            )

        self.pre_mlp_layernorm = RMSNorm(h, eps=1e-5)
        self.is_moe = is_moe_layer(
            layer_idx, getattr(dc, "num_experts", 1),
            getattr(dc, "num_dense_layers", 0), getattr(dc, "moe_layer_stride", 1),
        )
        if self.is_moe:
            self.mlp = SparseMoeBlock.from_decoder_config(dc)
        else:
            self.mlp = SwiGLUMLP(dim=h, mult=dc.intermediate_size // h, activation_fn=dc.mlp_activation_fn)

    @staticmethod
    def _add_norm(h, residual, norm):
        if residual is None:
            return norm(h), h
        residual = residual + h
        return norm(residual), residual

    def forward(self, h, residual, kv_cache, conv_cache, cross_kv, bias,
                return_cross_weights: bool = False):
        cross_weights = None
        if self.has_self_attn:
            h, residual = self._add_norm(h, residual, self.pre_self_attn_layernorm)
            h = self.self_attn(h, kv_cache)
        if self.has_conv:
            h, residual = self._add_norm(h, residual, self.pre_conv_layernorm)
            h = self.conv(h, conv_cache)
        if self.has_cross:
            h, residual = self._add_norm(h, residual, self.pre_cross_attn_layernorm)
            tk, tv = cross_kv["text"]
            layer_bias = bias if self.receives_bias else None
            if return_cross_weights:
                h, cross_weights = self.text_cross_attn.attn(
                    h, tk, tv, layer_bias, return_weights=True,
                )
            else:
                h = self.text_cross_attn.attn(h, tk, tv, layer_bias)
            h, residual = self._add_norm(h, residual, self.pre_cond_cross_attn_layernorm)
            sk, sv = cross_kv["speaker"]
            h = self.speaker_cross_attn.attn(h, sk, sv, None)
        h, residual = self._add_norm(h, residual, self.pre_mlp_layernorm)
        h = self.mlp(h)
        return h, residual, cross_weights


class _CrossAttentionWrapper(nn.Module):
    """Holds a ``cross_attn`` submodule so param names match the checkpoint."""

    def __init__(self, attn: _CrossAttention):
        super().__init__()
        self.cross_attn = attn

    def attn(self, *args, **kwargs):
        return self.cross_attn(*args, **kwargs)


class SpeechifyDecoder(nn.Module):
    """Full AR decoder: embedding + layers + mel/alignment heads + relative bias."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        dc = config.decoder_config
        self.embed_tokens = nn.Embedding(config.vocab_size, dc.hidden_size)
        self.layers = nn.ModuleList(
            [_DecoderLayer(config, i) for i in range(dc.num_hidden_layers)]
        )
        self.mel_head = nn.Linear(dc.hidden_size, config.vocab_size, bias=False)
        self.predict_alignment = getattr(config, "predict_alignment", False)
        if self.predict_alignment:
            steps = getattr(config, "predict_alignment_max_steps", 1)
            self.predict_alignment_max_steps = steps
            num_outputs = steps + 1 if steps > 1 else 1
            self.alignment_step_head = nn.Linear(dc.hidden_size, num_outputs)
            self.relative_bias = T5RelativeAttentionBias(
                getattr(dc, "relative_attention_num_buckets", 32),
                getattr(dc, "relative_attention_max_distance", 128),
                dc.num_key_value_heads,
            )
        # token-masking constants
        self.mask_end_idx = config.number_text_tokens + 1
        self.decoder_start_token_id = config.decoder_start_token_id

    # --- conditioning ---------------------------------------------------- #
    @torch.inference_mode()
    def prepare_cross_kv(self, text_hidden: torch.Tensor, speaker_emb: torch.Tensor) -> list[dict]:
        """Project per-layer text + speaker cross-attention K,V once per request.

        ``text_hidden`` [Te, hidden], ``speaker_emb`` [Ns, spk_hidden].
        Returns a per-layer list of ``{"text": (k,v), "speaker": (k,v)}``.
        """
        cross = []
        for layer in self.layers:
            if not layer.has_cross:
                cross.append(None)
                continue
            tk, tv = layer.text_cross_attn.cross_attn.project_kv(text_hidden)
            spk = layer.cond_proj(speaker_emb)
            sk, sv = layer.speaker_cross_attn.cross_attn.project_kv(spk)
            cross.append({"text": (tk, tv), "speaker": (sk, sv)})
        return cross

    def new_caches(self) -> tuple[list[dict], list[dict]]:
        kv = [{} for _ in self.layers]
        conv = [{} for _ in self.layers]
        return kv, conv

    @torch.inference_mode()
    def forward_step(
        self, input_ids: torch.Tensor, kv_caches, conv_caches, cross_kv,
        text_mask: torch.Tensor | None, alignment: torch.Tensor | None,
        text_hidden: torch.Tensor | None = None,
    ):
        """Run one (or a few prefill) tokens.

        Returns ``(logits[T,V], latent[T,h], step_diff[T], aligned_enc[T,h])``.
        ``aligned_enc`` is the cross-attention-weighted average of the text
        encoder states (per the diffusion stage's ``aligned_encoder_latent``);
        ``None`` when ``text_hidden`` is not supplied.
        """
        h = self.embed_tokens(input_ids)
        residual = None
        bias = None
        if self.predict_alignment and alignment is not None and text_mask is not None:
            align_floor = alignment.view(1, -1).to(h.dtype)  # FLOOR (MTL); [1, Q]
            bias_full = self.relative_bias.compute_bias_for_cross_alignment(
                alignment=align_floor, encoder_attention_mask=text_mask,
            )  # [1, heads, Q, K]
            bias = bias_full[0]  # [heads, Q, K]
        want_weights = text_hidden is not None
        cross_weights_acc: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            h, residual, cw = layer(
                h, residual, kv_caches[i], conv_caches[i], cross_kv[i], bias,
                return_cross_weights=want_weights,
            )
            if cw is not None:
                cross_weights_acc.append(cw)
        h = h + residual
        aligned_enc = None
        if want_weights and cross_weights_acc:
            # mean over cross-attn layers -> [Q, K]; weighted sum with text states
            avg = torch.stack(cross_weights_acc, dim=0).mean(dim=0)  # [Q, K]
            aligned_enc = torch.matmul(avg.to(text_hidden.dtype), text_hidden)  # [Q, h]
        logits = self.mel_head(h)
        logits[:, : self.mask_end_idx] = torch.finfo(logits.dtype).min
        if self.decoder_start_token_id is not None:
            logits[:, self.decoder_start_token_id] = torch.finfo(logits.dtype).min
        step_diff = None
        if self.predict_alignment:
            la = self.alignment_step_head(h)
            if self.predict_alignment_max_steps > 1:
                probs = torch.softmax(la.float(), dim=-1)
                rng = torch.arange(self.predict_alignment_max_steps + 1, device=probs.device).float()
                step_diff = torch.clamp((rng * probs).sum(-1), 0.02, float(self.predict_alignment_max_steps))
            else:
                p = torch.nan_to_num(torch.sigmoid(la[..., 0].float()), nan=0.99)
                step_diff = torch.clamp(p, 0.02, 1.0)
        return logits, h, step_diff, aligned_enc

    # --- weight loading -------------------------------------------------- #
    def load_weights(self, weights) -> set[str]:
        transformed: dict[str, torch.Tensor] = {}
        for name, param in weights:
            new = None
            if name == "model.embed_tokens.weight":
                new = "embed_tokens.weight"
            elif name == "mel_head.weight":
                new = "mel_head.weight"
            elif name.startswith("model.alignment_step_head."):
                new = name[len("model.") :]
            elif name.startswith("model.relative_bias."):
                new = name[len("model.") :]
            elif name.startswith("model.decoder."):
                dn = name[len("model.") :]  # decoder.*
                if ".self_attn.q_proj.weight" in dn:
                    new = dn.replace(".self_attn.q_proj.", ".self_attn.self_attn_q_proj.")
                elif ".self_attn.k_proj.weight" in dn:
                    new = dn.replace(".self_attn.k_proj.", ".self_attn.self_attn_k_proj.")
                elif ".self_attn.v_proj.weight" in dn:
                    new = dn.replace(".self_attn.v_proj.", ".self_attn.self_attn_v_proj.")
                elif ".self_attn.o_proj.weight" in dn:
                    new = dn.replace(".self_attn.o_proj.", ".self_attn.self_attn_o_proj.")
                elif ".conv.depthwise_conv.conv." in dn:
                    new = dn.replace(".conv.depthwise_conv.conv.", ".conv.depthwise_conv.")
                else:
                    new = dn  # cross_attn / cond_proj / pre_*_layernorm / conv.pointwise* / mlp
                # strip leading "decoder." -> our module root is the layer stack
                new = new[len("decoder.") :] if new.startswith("decoder.") else new
            if new is not None:
                transformed[new] = param

        loaded: set[str] = set()
        param_dict = dict(self.named_parameters())
        param_dict.update(dict(self.named_buffers()))
        for name, weight in transformed.items():
            if name not in param_dict:
                continue
            param = param_dict[name]
            if weight.dim() == 3 and weight.shape[2] == 1 and param.dim() == 2:
                weight = weight.squeeze(2)
            if name == "mel_head.weight" and param.shape != weight.shape:
                offset = self.config.number_text_tokens + 1
                param.data.zero_()
                end = min(offset + weight.shape[0], param.shape[0])
                param.data[offset:end, :].copy_(weight[: end - offset].to(param.dtype))
                loaded.add(name)
                continue
            if param.shape != weight.shape:
                continue
            param.data.copy_(weight.to(param.dtype))
            loaded.add(name)
        return loaded
