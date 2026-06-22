# SPDX-License-Identifier: Apache-2.0
"""HF configuration classes for SpeechifyTTS (speechify_t5_tts).

Ported verbatim from vllm-omni
(``vllm_omni/model_executor/models/speechify_t5_tts/configuration_speechify_t5_tts.py``).
These are pure ``transformers.PretrainedConfig`` subclasses with no framework
coupling, so they are reused unchanged. Registration with ``AutoConfig`` happens
in :func:`register_speechify_hf_configs` (called from the stage factories).

Architecture:
- Text Encoder: T5-style encoder for text-to-synthesize.
- Speaker tower: Unified/QKV speaker embedding from reference audio (external).
- Decoder: Gemma-style decoder with dual cross-attention (text + speaker) and
  optional MoE; alignment-prediction head drives streaming stop.
- Output: discrete audio (mel) codes; diffusion vocoder turns them into audio.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class SpeechifyT5ExtraCondsConfig(PretrainedConfig):
    """Configuration for the extra_conds_model (speaker embedding tower)."""

    model_type = "speechify_t5_extra_conds"

    def __init__(
        self,
        class_name: str = "QKVEmoSpkEmbeddingWithDec",
        in_channels: int = 100,
        dim: int = 512,
        features_dim: int = 1024,
        dec_dim: int = 1024,
        encoder_num_layers: int = 6,
        decoder_num_layers: int = 6,
        conv1_dim: int = 64,
        conv2_dim: int = 128,
        use_flash_att: bool = False,
        use_post_norm: bool = False,
        use_post_norm_encoder: bool = False,
        use_out_norm: bool = False,
        # QKVEmoSpkEmbeddingWithDec-specific (ignored for Unified)
        m: int = 1,
        output_num_layers: int = 6,
        n_emo_bins: int = 32,
        is_separate: bool = True,
        # UnifiedSpkEmbeddingWithDec-specific (ignored for QKV)
        nb_speaker_features: int = 6,
        feature_predictor_num_layers: int = 6,
        features_config: list[dict] | None = None,
        # Input layer config (TargetMelSpectrogram)
        n_mel_channels: int = 100,
        sampling_rate: int = 24000,
        mel_fmax: int = 12000,
        do_normalization: bool = True,
        # Legacy compatibility (hidden_size maps to features_dim for cross-attn)
        hidden_size: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.class_name = class_name
        self.in_channels = in_channels
        self.dim = dim
        self.features_dim = features_dim
        self.dec_dim = dec_dim
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.conv1_dim = conv1_dim
        self.conv2_dim = conv2_dim
        self.use_flash_att = use_flash_att
        self.use_post_norm = use_post_norm
        self.use_post_norm_encoder = use_post_norm_encoder
        self.use_out_norm = use_out_norm
        self.m = m
        self.output_num_layers = output_num_layers
        self.n_emo_bins = n_emo_bins
        self.is_separate = is_separate
        self.nb_speaker_features = nb_speaker_features
        self.feature_predictor_num_layers = feature_predictor_num_layers
        self.features_config = features_config or []
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.mel_fmax = mel_fmax
        self.do_normalization = do_normalization
        self.hidden_size = hidden_size if hidden_size is not None else features_dim

    @property
    def is_unified(self) -> bool:
        """True iff the active speaker-embedding family is Unified."""
        return "Unified" in (self.class_name or "")

    @property
    def num_speaker_tokens(self) -> int:
        """Number of speaker-token positions consumed by cross-attention."""
        if self.is_unified:
            return self.nb_speaker_features + len(self.features_config or [])
        return self.m * self.output_num_layers

    def to_extra_conds_kwargs(self) -> dict:
        """Convert config to kwargs for the speaker-embedding tower."""
        return {
            "in_channels": self.in_channels,
            "dim": self.dim,
            "m": self.m,
            "features_dim": self.features_dim,
            "dec_dim": self.dec_dim,
            "encoder_num_layers": self.encoder_num_layers,
            "decoder_num_layers": self.decoder_num_layers,
            "output_num_layers": self.output_num_layers,
            "conv1_dim": self.conv1_dim,
            "conv2_dim": self.conv2_dim,
            "n_emo_bins": self.n_emo_bins,
            "use_flash_att": self.use_flash_att,
            "use_post_norm": self.use_post_norm,
            "use_post_norm_encoder": self.use_post_norm_encoder,
            "use_out_norm": self.use_out_norm,
            "is_separate": self.is_separate,
            "input_layer_config": {
                "n_mel_channels": self.n_mel_channels,
                "sampling_rate": self.sampling_rate,
                "mel_fmax": self.mel_fmax,
                "do_normalization": self.do_normalization,
            },
        }


class SpeechifyT5TTSEncoderConfig(PretrainedConfig):
    """Configuration for the T5-style text encoder."""

    model_type = "speechify_t5_tts_encoder"

    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 768,
        mlp_multiplier: int = 4,
        mlp_dropout: float = 0.1,
        num_attention_heads: int = 8,
        num_kv_heads: int | None = None,
        head_dim: int = 64,
        kv_head_dim: int | None = None,
        query_pre_attn_scalar: float = 256.0,
        attention_dropout: float = 0.1,
        attn_logit_softcapping: float = 0.0,
        is_causal: bool = False,
        is_conv_causal: bool = True,
        qk_norm: bool = False,
        conv_kernel_size: int = 9,
        conv_dropout: float = 0.1,
        conv_layer_indices: list[int] | None = None,
        self_attention_layer_indices: list[int] | None = None,
        mlp_activation_fn: str = "geglu",
        add_cond_cross_attn: bool = False,
        spk_emb_dim: int = 1024,
        use_aux_context: bool = False,
        aux_context_model_dim: int = 1024,
        aux_context_cross_attention_indices: list[int] | None = None,
        aux_context_cross_attention_bias_layer_indices: list[int] | None = None,
        self_attention_pos_embedding_type: str = "nope",
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        rope_type: str = "default",
        rope_max_position_embeddings: int = 8192,
        rope_config: dict | None = None,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.mlp_multiplier = mlp_multiplier
        self.mlp_dropout = mlp_dropout
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = (
            num_kv_heads if num_kv_heads is not None else num_attention_heads
        )
        self.head_dim = head_dim
        self.kv_head_dim = kv_head_dim if kv_head_dim is not None else head_dim
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.attention_dropout = attention_dropout
        self.attn_logit_softcapping = attn_logit_softcapping
        self.is_causal = is_causal
        self.is_conv_causal = is_conv_causal
        self.qk_norm = qk_norm
        self.conv_kernel_size = conv_kernel_size
        self.conv_dropout = conv_dropout
        self.conv_layer_indices = (
            conv_layer_indices
            if conv_layer_indices is not None
            else list(range(num_layers))
        )
        self.self_attention_layer_indices = (
            self_attention_layer_indices
            if self_attention_layer_indices is not None
            else list(range(num_layers))
        )
        self.mlp_activation_fn = mlp_activation_fn
        self.add_cond_cross_attn = add_cond_cross_attn
        self.spk_emb_dim = spk_emb_dim
        self.use_aux_context = use_aux_context
        self.aux_context_model_dim = aux_context_model_dim
        self.aux_context_cross_attention_indices = (
            aux_context_cross_attention_indices
            if aux_context_cross_attention_indices is not None
            else (list(range(num_layers)) if use_aux_context else [])
        )
        self.aux_context_cross_attention_bias_layer_indices = (
            aux_context_cross_attention_bias_layer_indices
            if aux_context_cross_attention_bias_layer_indices is not None
            else (list(range(num_layers)) if use_aux_context else [])
        )
        self.self_attention_pos_embedding_type = self_attention_pos_embedding_type
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.rope_type = rope_type
        self.rope_max_position_embeddings = rope_max_position_embeddings
        self.rope_config = rope_config
        self.pad_token_id = pad_token_id


class SpeechifyT5TTSDecoderConfig(PretrainedConfig):
    """Configuration for the Gemma-style decoder with dual cross-attention."""

    model_type = "speechify_t5_tts_decoder"

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int | None = None,
        text_cross_attention_hidden_size: int | None = None,
        speaker_cross_attention_hidden_size: int | None = None,
        hidden_act: str = "gelu",
        layer_norm_epsilon: float = 1e-6,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        head_dim: int | None = None,
        conv_kernel_size: int = 4,
        conv_dropout: float = 0.1,
        conv_layer_indices: list[int] | None = None,
        is_conv_causal: bool = True,
        self_attention_layer_indices: list[int] | None = None,
        cross_attention_layer_indices: list[int] | None = None,
        cross_attention_bias_layer_indices: list[int] | None = None,
        mlp_activation_fn: str = "relu",
        mlp_multiplier: int | None = None,
        mlp_dropout: float = 0.1,
        self_attention_pos_embedding_type: str = "rope",
        query_pre_attn_scalar: float = 256.0,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        # MoE knobs (decoder-only). Defaults yield a dense decoder.
        num_experts: int = 1,
        num_experts_per_tok: int = 1,
        moe_intermediate_size: int | None = None,
        num_dense_layers: int = 0,
        moe_layer_stride: int = 1,
        num_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
        use_expert_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads if num_key_value_heads else num_attention_heads
        )
        self.text_cross_attention_hidden_size = (
            text_cross_attention_hidden_size
            if text_cross_attention_hidden_size
            else hidden_size
        )
        self.speaker_cross_attention_hidden_size = (
            speaker_cross_attention_hidden_size
            if speaker_cross_attention_hidden_size
            else hidden_size
        )
        self.hidden_act = hidden_act
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.head_dim = head_dim if head_dim else hidden_size // num_attention_heads
        self.conv_kernel_size = conv_kernel_size
        self.conv_dropout = conv_dropout
        self.conv_layer_indices = (
            conv_layer_indices
            if conv_layer_indices is not None
            else list(range(num_hidden_layers))
        )
        self.is_conv_causal = is_conv_causal
        self.self_attention_layer_indices = (
            self_attention_layer_indices
            if self_attention_layer_indices is not None
            else list(range(num_hidden_layers))
        )
        self.cross_attention_layer_indices = (
            cross_attention_layer_indices
            if cross_attention_layer_indices is not None
            else list(range(num_hidden_layers))
        )
        self.cross_attention_bias_layer_indices = (
            cross_attention_bias_layer_indices
            if cross_attention_bias_layer_indices is not None
            else list(range(num_hidden_layers))
        )
        self.mlp_activation_fn = mlp_activation_fn
        self.mlp_multiplier = (
            mlp_multiplier
            if mlp_multiplier is not None
            else intermediate_size // hidden_size
        )
        self.mlp_dropout = mlp_dropout
        self.self_attention_pos_embedding_type = self_attention_pos_embedding_type
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        if moe_intermediate_size is None:
            self.moe_intermediate_size = self.hidden_size * self.mlp_multiplier
        else:
            self.moe_intermediate_size = moe_intermediate_size
        self.num_dense_layers = num_dense_layers
        self.moe_layer_stride = moe_layer_stride
        self.num_shared_experts = num_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.use_expert_bias = use_expert_bias


class SpeakingStatsConfig(PretrainedConfig):
    """Configuration for the SpeakingStatsPredictor (speaking-rate from mel)."""

    model_type = "speaking_stats_predictor"

    def __init__(
        self,
        conv1_dim: int = 64,
        conv2_dim: int = 128,
        in_channels: int = 100,
        dim: int = 512,
        num_layers: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv1_dim = conv1_dim
        self.conv2_dim = conv2_dim
        self.in_channels = in_channels
        self.dim = dim
        self.num_layers = num_layers


class SpeechifyT5TTSConfig(PretrainedConfig):
    """Top-level configuration for the SpeechifyT5TTS model."""

    model_type = "speechify_t5_tts"

    def __init__(
        self,
        vocab_size: int = 32000,
        audio_vocab_size: int = 2048,
        number_text_tokens: int = 1101,
        number_mel_codes: int = 2048,
        decoder_start_token_id: int | None = None,
        pad_token_id: int = 0,
        eos_token_id: int | None = None,
        bos_token_id: int = 2,
        enable_speaking_rate_ift: bool = False,
        speaking_rate_vocab_size: int = 5,
        speaking_rate_config: dict | SpeakingStatsConfig | None = None,
        extra_conds_config: dict | SpeechifyT5ExtraCondsConfig | None = None,
        encoder_config: dict | SpeechifyT5TTSEncoderConfig | None = None,
        decoder_config: dict | SpeechifyT5TTSDecoderConfig | None = None,
        tie_word_embeddings: bool = True,
        predict_alignment: bool = False,
        predict_alignment_max_steps: int = 1,
        predict_alignment_mode: str = "cumsum",
        body_end_anchor_offset: int = 2,
        align_stop_offset: int = -1,
        alignment_plateau_max_steps: int = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.audio_vocab_size = audio_vocab_size
        self.number_text_tokens = number_text_tokens
        self.number_mel_codes = number_mel_codes
        if decoder_start_token_id is None:
            decoder_start_token_id = vocab_size - 1
        self.decoder_start_token_id = decoder_start_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        self.enable_speaking_rate_ift = enable_speaking_rate_ift
        self.speaking_rate_vocab_size = speaking_rate_vocab_size

        if speaking_rate_config is None:
            self.speaking_rate_config = SpeakingStatsConfig()
        elif isinstance(speaking_rate_config, dict):
            self.speaking_rate_config = SpeakingStatsConfig(**speaking_rate_config)
        else:
            self.speaking_rate_config = speaking_rate_config

        self.predict_alignment = predict_alignment
        self.predict_alignment_max_steps = predict_alignment_max_steps
        self.predict_alignment_mode = predict_alignment_mode
        self.body_end_anchor_offset = int(body_end_anchor_offset)
        self.align_stop_offset = int(align_stop_offset)
        self.alignment_plateau_max_steps = int(alignment_plateau_max_steps)

        if extra_conds_config is None:
            self.extra_conds_config = SpeechifyT5ExtraCondsConfig()
        elif isinstance(extra_conds_config, dict):
            self.extra_conds_config = SpeechifyT5ExtraCondsConfig(**extra_conds_config)
        else:
            self.extra_conds_config = extra_conds_config

        if encoder_config is None:
            self.encoder_config = SpeechifyT5TTSEncoderConfig()
        elif isinstance(encoder_config, dict):
            self.encoder_config = SpeechifyT5TTSEncoderConfig(**encoder_config)
        else:
            self.encoder_config = encoder_config

        if decoder_config is None:
            self.decoder_config = SpeechifyT5TTSDecoderConfig()
        elif isinstance(decoder_config, dict):
            self.decoder_config = SpeechifyT5TTSDecoderConfig(**decoder_config)
        else:
            self.decoder_config = decoder_config

        if self.decoder_config.text_cross_attention_hidden_size is None:
            self.decoder_config.text_cross_attention_hidden_size = (
                self.encoder_config.hidden_size
            )
        if self.decoder_config.speaker_cross_attention_hidden_size is None:
            self.decoder_config.speaker_cross_attention_hidden_size = (
                self.extra_conds_config.hidden_size
            )

    # Top-level properties mirroring the active (decoder) backbone so generic
    # SGLang config probes resolve to the AR decoder dimensions.
    @property
    def num_attention_heads(self) -> int:
        return self.decoder_config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.decoder_config.num_key_value_heads

    @property
    def num_hidden_layers(self) -> int:
        return self.decoder_config.num_hidden_layers

    @property
    def hidden_size(self) -> int:
        return self.decoder_config.hidden_size

    @property
    def head_dim(self) -> int:
        return self.decoder_config.head_dim or (
            self.decoder_config.hidden_size // self.decoder_config.num_attention_heads
        )

    @property
    def effective_number_mel_codes(self) -> int:
        if self.enable_speaking_rate_ift:
            return self.number_mel_codes + self.speaking_rate_vocab_size
        return self.number_mel_codes

    @property
    def decoder_eos_token_id(self) -> int:
        return self.effective_number_mel_codes

    @property
    def full_vocab_eos_token_id(self) -> int:
        return self.number_text_tokens + self.effective_number_mel_codes + 1

    @property
    def audio_token_offset(self) -> int:
        return self.number_text_tokens + 1


def register_speechify_hf_configs() -> None:
    """Register SpeechifyTTS HF configs with ``transformers.AutoConfig``.

    The checkpoint's top-level ``config.json`` only carries
    ``{"model_type": "speechify_t5_tts"}`` (no stock transformers type), so
    ``AutoConfig.from_pretrained`` needs the mapping registered before load.
    Idempotent: a duplicate registration raises ``ValueError`` which we ignore.
    """
    from transformers import AutoConfig

    for cfg_cls in (
        SpeechifyT5ExtraCondsConfig,
        SpeechifyT5TTSEncoderConfig,
        SpeechifyT5TTSDecoderConfig,
        SpeakingStatsConfig,
        SpeechifyT5TTSConfig,
    ):
        try:
            AutoConfig.register(cfg_cls.model_type, cfg_cls)
        except ValueError:
            pass
