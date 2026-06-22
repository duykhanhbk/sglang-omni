from diffusers.models.autoencoders.autoencoder_oobleck import OobleckEncoder, OobleckDecoder, OobleckDecoderOutput
import torch
import numpy as np
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm
from typing import Optional, Tuple, Union, List


class CausalConv1d(nn.Conv1d):
    """
    Causal 1D convolution with left padding only and optional streaming cache.
    Ensures the output at time t only depends on inputs at times <= t.

    Inherits from nn.Conv1d to be compatible with weight_norm.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        # Causal padding: pad only on the left side
        self.causal_padding = (kernel_size - 1) * dilation

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,  # No padding in conv, we handle it manually
            dilation=dilation,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache_trim: int = 0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional streaming cache.

        Args:
            x: [B, C, T] input tensor
            cache: [B, C, causal_padding] cached values from previous chunk
            use_cache: whether to return updated cache
            cache_trim: number of trailing frames to exclude from cache.
                When streaming with lookahead, the input contains extra future
                frames that should not pollute the cache for the next chunk.
                The padded input is [old_cache, current, lookahead]; with
                cache_trim=L we save x[..., -(cp+L):-L] instead of x[..., -cp:].

        Returns:
            output, or (output, new_cache) if use_cache=True
        """
        if cache is not None:
            # Use cached values instead of zero padding
            x = torch.cat([cache, x], dim=-1)
        else:
            # First chunk or non-streaming: zero pad
            x = nn.functional.pad(x, (self.causal_padding, 0))

        out = super().forward(x)

        if use_cache:
            if self.causal_padding > 0:
                if cache_trim > 0:
                    new_cache = x[..., -(self.causal_padding + cache_trim):-cache_trim].clone()
                else:
                    new_cache = x[..., -self.causal_padding:].clone()
            else:
                new_cache = None
            return out, new_cache
        return out


class Snake1d(nn.Module):
    """
    A 1-dimensional Snake activation function module.
    """

    def __init__(self, hidden_dim, logscale=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, hidden_dim, 1))

        self.alpha.requires_grad = True
        self.beta.requires_grad = True
        self.logscale = logscale

    def forward(self, hidden_states):
        shape = hidden_states.shape

        alpha = self.alpha if not self.logscale else torch.exp(self.alpha)
        beta = self.beta if not self.logscale else torch.exp(self.beta)

        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)
        hidden_states = hidden_states.reshape(shape)
        return hidden_states


class CausalOobleckResidualUnit(nn.Module):
    """
    A causal residual unit composed of Snake1d and weight-normalized CausalConv1d layers with dilations.
    """

    def __init__(self, dimension: int = 16, dilation: int = 1):
        super().__init__()

        self.snake1 = Snake1d(dimension)
        self.conv1 = weight_norm(CausalConv1d(dimension, dimension, kernel_size=7, dilation=dilation))
        self.snake2 = Snake1d(dimension)
        self.conv2 = weight_norm(CausalConv1d(dimension, dimension, kernel_size=1))

    def forward(
        self,
        hidden_state: torch.Tensor,
        cache: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
        use_cache: bool = False,
        cache_trim: int = 0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]]:
        """
        Forward pass through the causal residual unit.

        Args:
            hidden_state: Input tensor [B, C, T]
            cache: Tuple of (conv1_cache, conv2_cache) from previous chunk
            use_cache: Whether to return updated cache
            cache_trim: Number of trailing frames (lookahead) to exclude from cache

        Returns:
            output_tensor, or (output_tensor, (conv1_cache, conv2_cache)) if use_cache=True
        """
        conv1_cache = cache[0] if cache is not None else None
        conv2_cache = cache[1] if cache is not None else None

        output_tensor = self.snake1(hidden_state)
        if use_cache:
            output_tensor, new_conv1_cache = self.conv1(output_tensor, cache=conv1_cache, use_cache=True, cache_trim=cache_trim)
        else:
            output_tensor = self.conv1(output_tensor, cache=conv1_cache)
            new_conv1_cache = None

        output_tensor = self.snake2(output_tensor)
        if use_cache:
            output_tensor, new_conv2_cache = self.conv2(output_tensor, cache=conv2_cache, use_cache=True, cache_trim=cache_trim)
        else:
            output_tensor = self.conv2(output_tensor, cache=conv2_cache)
            new_conv2_cache = None

        output_tensor = hidden_state + output_tensor

        if use_cache:
            return output_tensor, (new_conv1_cache, new_conv2_cache)
        return output_tensor


class CausalOobleckDecoderBlock(nn.Module):
    """Causal decoder block used in CausalOobleckDecoder."""

    # Type alias for block cache: (conv_t1_lookback, res_unit1, res_unit2, res_unit3)
    # conv_t1_lookback: [B, C, 1] - last "current" feature frame before ConvTranspose1d
    # Each res_unit cache is (conv1_cache, conv2_cache)
    BlockCache = Tuple[
        Optional[torch.Tensor],  # conv_t1_lookback
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],  # res_unit1
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],  # res_unit2
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],  # res_unit3
    ]

    def __init__(self, input_dim, output_dim, stride: int = 1):
        super().__init__()
        import math

        self.snake1 = Snake1d(input_dim)
        # ConvTranspose1d is kept as is (not replaced with causal version)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )
        )
        self.res_unit1 = CausalOobleckResidualUnit(output_dim, dilation=1)
        self.res_unit2 = CausalOobleckResidualUnit(output_dim, dilation=3)
        self.res_unit3 = CausalOobleckResidualUnit(output_dim, dilation=9)

    def forward(
        self,
        hidden_state: torch.Tensor,
        cache: Optional[BlockCache] = None,
        use_cache: bool = False,
        cache_trim: int = 0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, BlockCache]]:
        """
        Forward pass through the decoder block.

        Args:
            hidden_state: Input tensor [B, C, T]
            cache: Tuple of (conv_t1_lookback, res1_cache, res2_cache, res3_cache)
            use_cache: Whether to return updated cache
            cache_trim: Number of trailing frames (lookahead) at this block's
                input scale to exclude from caches

        Returns:
            output, or (output, new_cache) if use_cache=True
        """
        conv_t1_lookback = cache[0] if cache is not None else None
        res1_cache = cache[1] if cache is not None else None
        res2_cache = cache[2] if cache is not None else None
        res3_cache = cache[3] if cache is not None else None

        hidden_state = self.snake1(hidden_state)

        # Save lookback: last "current" frame (before lookahead) for next call's conv_t1
        new_lookback = None
        if use_cache:
            if cache_trim > 0:
                new_lookback = hidden_state[..., -(1 + cache_trim):-cache_trim].clone()
            else:
                new_lookback = hidden_state[..., -1:].clone()

        # ConvTranspose1d with lookback for past context
        if conv_t1_lookback is not None:
            conv_t1_input = torch.cat([conv_t1_lookback, hidden_state], dim=-1)
        else:
            conv_t1_input = hidden_state

        hidden_state = self.conv_t1(conv_t1_input)

        # Skip lookback's output samples (1 frame * stride)
        if conv_t1_lookback is not None:
            hidden_state = hidden_state[..., self.conv_t1.stride[0]:]

        # res_units at upsampled scale: cache_trim scales by stride
        upsampled_trim = cache_trim * self.conv_t1.stride[0]

        if use_cache:
            hidden_state, new_res1_cache = self.res_unit1(hidden_state, cache=res1_cache, use_cache=True, cache_trim=upsampled_trim)
            hidden_state, new_res2_cache = self.res_unit2(hidden_state, cache=res2_cache, use_cache=True, cache_trim=upsampled_trim)
            hidden_state, new_res3_cache = self.res_unit3(hidden_state, cache=res3_cache, use_cache=True, cache_trim=upsampled_trim)
            return hidden_state, (new_lookback, new_res1_cache, new_res2_cache, new_res3_cache)
        else:
            hidden_state = self.res_unit1(hidden_state, cache=res1_cache)
            hidden_state = self.res_unit2(hidden_state, cache=res2_cache)
            hidden_state = self.res_unit3(hidden_state, cache=res3_cache)
            return hidden_state


class CausalOobleckDecoder(nn.Module):
    """Causal Oobleck Decoder - all regular Conv1d replaced with CausalConv1d, ConvTranspose1d unchanged."""

    # Cache structure: dict with keys 'conv1', 'blocks' (list), 'conv2'
    DecoderCache = dict

    def __init__(self, channels, input_channels, audio_channels, upsampling_ratios, channel_multiples):
        super().__init__()

        strides = upsampling_ratios
        channel_multiples = [1] + channel_multiples

        # Add first conv layer (causal)
        self.conv1 = weight_norm(CausalConv1d(input_channels, channels * channel_multiples[-1], kernel_size=7))

        # Add upsampling + MRF blocks (causal)
        block = []
        for stride_index, stride in enumerate(strides):
            block += [
                CausalOobleckDecoderBlock(
                    input_dim=channels * channel_multiples[len(strides) - stride_index],
                    output_dim=channels * channel_multiples[len(strides) - stride_index - 1],
                    stride=stride,
                )
            ]

        self.block = nn.ModuleList(block)
        self.num_blocks = len(block)
        output_dim = channels
        self.snake1 = Snake1d(output_dim)
        self.conv2 = weight_norm(CausalConv1d(channels, audio_channels, kernel_size=7, bias=False))

    def forward(
        self,
        hidden_state: torch.Tensor,
        cache: Optional[DecoderCache] = None,
        use_cache: bool = False,
        lookahead_frames: int = 0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, DecoderCache]]:
        """
        Forward pass through the causal decoder.

        Args:
            hidden_state: Input tensor [B, C, T]
            cache: Dict with keys 'conv1', 'blocks', 'conv2'
            use_cache: Whether to return updated cache
            lookahead_frames: Number of trailing latent frames that are lookahead.
                Caches are saved as if these frames were not processed:
                - CausalConv1d caches exclude lookahead via cache_trim
                - ConvTranspose1d lookback saves the last "current" frame
                The cache_trim value scales with resolution through the decoder.
        """
        # Extract caches
        conv1_cache = cache.get('conv1') if cache else None
        block_caches = cache.get('blocks', [None] * self.num_blocks) if cache else [None] * self.num_blocks
        conv2_cache = cache.get('conv2') if cache else None

        # Running cache_trim: starts at latent scale, scales up after each upsample
        trim = lookahead_frames

        # Input conv (latent scale)
        if use_cache:
            hidden_state, new_conv1_cache = self.conv1(hidden_state, cache=conv1_cache, use_cache=True, cache_trim=trim)
        else:
            hidden_state = self.conv1(hidden_state, cache=conv1_cache)
            new_conv1_cache = None

        # Decoder blocks
        new_block_caches = []
        for i, layer in enumerate(self.block):
            block_cache = block_caches[i] if i < len(block_caches) else None
            if use_cache:
                hidden_state, new_block_cache = layer(hidden_state, cache=block_cache, use_cache=True, cache_trim=trim)
                new_block_caches.append(new_block_cache)
            else:
                hidden_state = layer(hidden_state, cache=block_cache)
            # Scale trim after this block's upsample
            trim *= layer.conv_t1.stride[0]

        # Output conv (full audio scale)
        hidden_state = self.snake1(hidden_state)
        if use_cache:
            hidden_state, new_conv2_cache = self.conv2(hidden_state, cache=conv2_cache, use_cache=True, cache_trim=trim)
        else:
            hidden_state = self.conv2(hidden_state, cache=conv2_cache)
            new_conv2_cache = None

        if use_cache:
            new_cache = {
                'conv1': new_conv1_cache,
                'blocks': new_block_caches,
                'conv2': new_conv2_cache,
            }
            return hidden_state, new_cache
        return hidden_state

    @property
    def required_lookahead(self) -> int:
        """Minimum lookahead latent frames needed for correct streaming.

        Each ConvTranspose1d with kernel > stride introduces ~1 frame of future
        dependency at its input resolution. Summing across all blocks and
        converting to the original latent scale gives the total.
        """
        import math
        future_dep = 0.0
        cumulative_stride = 1
        for block in self.block:
            future_dep += 1.0 / cumulative_stride
            cumulative_stride *= block.conv_t1.stride[0]
        return math.ceil(future_dep)


class WN(nn.Module):
    """
    WaveNet-style residual block used as the conditioner in coupling layers.
    
    This is the core building block that makes the flow expressive.
    Uses dilated convolutions with gated activations (tanh * sigmoid).
    
    Reference: VITS (Kim et al., 2021)
    """
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 4,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = weight_norm(
                nn.Conv1d(
                    hidden_channels,
                    2 * hidden_channels,  # For gated activation (tanh * sigmoid)
                    kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.in_layers.append(in_layer)

            # Last layer outputs only skip (no residual needed)
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            res_skip_layer = weight_norm(
                nn.Conv1d(hidden_channels, res_skip_channels, 1)
            )
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, hidden_channels, T]
            
        Returns:
            Output tensor of shape [B, hidden_channels, T]
        """
        output = torch.zeros_like(x)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            
            # Gated activation: tanh(x1) * sigmoid(x2)
            acts = torch.tanh(x_in[:, :self.hidden_channels]) * torch.sigmoid(x_in[:, self.hidden_channels:])
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                # Split into residual and skip connections
                res_acts = res_skip_acts[:, :self.hidden_channels]
                x = x + res_acts
                output = output + res_skip_acts[:, self.hidden_channels:]
            else:
                output = output + res_skip_acts

        return output

    def remove_weight_norm(self):
        for layer in self.in_layers:
            remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            remove_weight_norm(layer)


class Flip(nn.Module):
    """
    Simple channel flip layer for ensuring all channels get transformed.
    
    This is crucial: without flipping (or 1x1 conv), the first half of channels
    in affine coupling would never be transformed.
    """
    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.flip(x, [1])
        logdet = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        return x, logdet


class ResidualCouplingLayer(nn.Module):
    """
    VITS-style Residual Coupling Layer.
    
    Unlike simple affine coupling, this uses:
    1. WaveNet residual blocks for the conditioner network
    2. Only translation (mean_only=True) or full affine (mean_only=False)
    3. Proper initialization for stable training
    
    The transformation is:
        x0, x1 = split(x)
        stats = WN(x0)
        if not mean_only:
            m, logs = split(stats)
            x1 = m + x1 * exp(logs)
        else:
            m = stats
            x1 = m + x1
        y = concat(x0, x1)
    
    Reference: VITS (Kim et al., 2021)
    """
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 4,
        p_dropout: float = 0.0,
        mean_only: bool = False,
        init_scale: float = 5.0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.half_channels = channels // 2
        self.mean_only = mean_only

        # Input projection
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        
        # WaveNet conditioner
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout,
        )
        
        # Output projection: outputs mean (and optionally log_scale)
        self.post = nn.Conv1d(
            hidden_channels,
            self.half_channels * (1 if mean_only else 2),
            1,
        )
        # Initialize to identity transform
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

        # Learnable scaling factor for soft log_scale bounds
        # Using softplus(raw) + eps to ensure s_fac > 0 and smooth gradients
        # Initialize so that initial s_fac ≈ init_scale (default 5.0)
        # softplus inverse: raw = log(exp(init_scale - eps) - 1) ≈ init_scale for large values
        init_raw = np.log(np.exp(init_scale - 0.5) - 1)  # softplus^(-1)(init_scale - 0.5)
        self.scale_factor_raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    def _get_scale_factor(self):
        """
        Get the scaling factor for log_scale bounds.
        Uses softplus to ensure positivity with smooth gradients.
        Output range: (0.5, inf), but practically limited by training dynamics.
        """
        # softplus gives smooth, always-positive output
        # Add 0.5 to ensure minimum scale factor (prevents division issues)
        return nn.functional.softplus(self.scale_factor_raw) + 0.5

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [B, channels, T]
            reverse: If True, apply inverse transformation
            
        Returns:
            y: Transformed tensor of shape [B, channels, T]
            logdet: Log determinant of Jacobian, shape [B]
        """
        x0, x1 = torch.split(x, [self.half_channels, self.half_channels], dim=1)
        
        # Condition on x0
        h = self.pre(x0)
        h = self.enc(h)
        stats = self.post(h)
        
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels, self.half_channels], dim=1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        # Apply soft bounds using tanh with learnable scaling factor
        # log_scale = tanh(log_scale_raw / s_fac) * s_fac
        # This gives smooth gradients everywhere and range [-s_fac, s_fac]
        s_fac = self._get_scale_factor()
        logs = torch.tanh(logs / s_fac) * s_fac

        if not reverse:
            # Forward: x1 -> x1 * exp(logs) + m
            x1 = m + x1 * torch.exp(logs)
            y = torch.cat([x0, x1], dim=1)
            logdet = logs.sum(dim=(1, 2))
        else:
            # Inverse: x1 -> (x1 - m) * exp(-logs)
            x1 = (x1 - m) * torch.exp(-logs)
            y = torch.cat([x0, x1], dim=1)
            logdet = -logs.sum(dim=(1, 2))

        return y, logdet

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class NormalizingFlow(nn.Module):
    """
    VITS-style Normalizing Flow.
    
    Stack of residual coupling layers with flip layers for channel mixing.
    Each flow step consists of:
        1. ResidualCouplingLayer (transforms second half conditioned on first half)
        2. Flip (swaps channel halves so all channels get transformed)
    
    This architecture ensures:
    - All channels are transformed (via flipping)
    - Long-range temporal dependencies (via WaveNet dilated convolutions)
    - Stable training (via proper initialization)
    
    Reference: VITS (Kim et al., 2021)
    """
    def __init__(
        self,
        dim: int,
        n_flows: int = 4,
        hidden_size: int = 192,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 4,
        p_dropout: float = 0.0,
        mean_only: bool = False,
        flow_type: str = 'vits',  # Keep for backward compatibility
    ):
        super().__init__()
        self.dim = dim
        self.n_flows = n_flows
        
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels=dim,
                    hidden_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    p_dropout=p_dropout,
                    mean_only=mean_only,
                )
            )
            self.flows.append(Flip())

    def forward(self, z: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sequence of flow transformations.
        
        Args:
            z: Input tensor from encoder, shape [B, dim, T]
            reverse: If True, apply inverse transformation (for generation)
            
        Returns:
            z_tilde: Transformed tensor, shape [B, dim, T]
            total_logdet: Total log determinant, shape [B]
        """
        total_logdet = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        
        if not reverse:
            for flow in self.flows:
                z, logdet = flow(z, reverse=False)
                total_logdet = total_logdet + logdet
        else:
            for flow in reversed(self.flows):
                z, logdet = flow(z, reverse=True)
                total_logdet = total_logdet + logdet
        
        return z, total_logdet

    def remove_weight_norm(self):
        for flow in self.flows:
            if hasattr(flow, 'remove_weight_norm'):
                flow.remove_weight_norm()


class FlowVAE(nn.Module):
    r"""
    An autoencoder for encoding waveforms into latents and decoding latent representations into waveforms.
    Supports both standard VAE and Flow-VAE modes.
    
    In Flow-VAE mode, a VITS-style normalizing flow transforms the encoder's latent distribution,
    allowing the encoder to learn a more flexible distribution while still enabling
    KL divergence computation against a standard normal prior.

    Parameters:
        encoder_hidden_size (`int`, *optional*, defaults to 128):
            Intermediate representation dimension for the encoder.
        downsampling_ratios (`List[int]`, *optional*, defaults to `[2, 4, 4, 4, 4]`):
            Ratios for downsampling in the encoder. These are used in reverse order for upsampling in the decoder.
        channel_multiples (`List[int]`, *optional*, defaults to `[1, 2, 4, 8, 16]`):
            Multiples used to determine the hidden sizes of the hidden layers.
        decoder_channels (`int`, *optional*, defaults to 128):
            Intermediate representation dimension for the decoder.
        decoder_input_channels (`int`, *optional*, defaults to 64):
            Input dimension for the decoder. Corresponds to the latent dimension.
        audio_channels (`int`, *optional*, defaults to 1):
            Number of channels in the audio data. Either 1 for mono or 2 for stereo.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        n_flows (`int`, *optional*, defaults to 4):
            Number of flow steps (each step = coupling layer + flip).
        flow_hidden_size (`int`, *optional*, defaults to 192):
            Hidden size for WaveNet blocks in coupling layers.
        flow_kernel_size (`int`, *optional*, defaults to 5):
            Kernel size for dilated convolutions in WaveNet blocks.
        flow_dilation_rate (`int`, *optional*, defaults to 1):
            Base dilation rate for WaveNet blocks (dilation = rate^layer).
        flow_n_layers (`int`, *optional*, defaults to 4):
            Number of WaveNet layers per coupling block.
        flow_p_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability in WaveNet blocks.
        flow_mean_only (`bool`, *optional*, defaults to False):
            If True, coupling layers only predict mean (no scale). More stable but less expressive.
        is_flow_vae (`bool`, *optional*, defaults to False):
            Whether to use Flow-VAE mode with normalizing flows.
        flow_type (`str`, *optional*, defaults to 'vits'):
            Type of flow architecture. Currently only 'vits' is supported.
        is_decoder_conv_causal (`bool`, *optional*, defaults to False):
            If True, use CausalOobleckDecoder where all regular Conv1d are replaced with
            CausalConv1d (left padding only), while ConvTranspose1d remain unchanged.
    """
    def __init__(
        self,
        encoder_hidden_size=128,
        downsampling_ratios=[2, 4, 4, 4, 4],
        channel_multiples=[1, 2, 4, 8, 16],
        decoder_channels=128,
        decoder_input_channels=64,
        audio_channels=1,
        sampling_rate=24000,
        n_flows=4,
        flow_hidden_size=64,
        flow_kernel_size=5,
        flow_dilation_rate=1,
        flow_n_layers=4,
        flow_p_dropout=0.0,
        flow_mean_only=False,
        is_flow_vae=False,
        flow_type='vits',
        is_decoder_conv_causal=False,
    ):
        super().__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.downsampling_ratios = downsampling_ratios
        self.decoder_channels = decoder_channels
        self.decoder_input_channels = decoder_input_channels
        self.upsampling_ratios = downsampling_ratios[::-1]
        self.hop_length = int(np.prod(downsampling_ratios))
        self.sampling_rate = sampling_rate
        self.is_flow_vae = is_flow_vae
        self.is_decoder_conv_causal = is_decoder_conv_causal

        self.encoder = OobleckEncoder(
            encoder_hidden_size=encoder_hidden_size,
            audio_channels=audio_channels,
            downsampling_ratios=downsampling_ratios,
            channel_multiples=channel_multiples,
        )

        decoder_cls = CausalOobleckDecoder if is_decoder_conv_causal else OobleckDecoder
        self.decoder = decoder_cls(
            channels=decoder_channels,
            input_channels=decoder_input_channels,
            audio_channels=audio_channels,
            upsampling_ratios=self.upsampling_ratios,
            channel_multiples=channel_multiples,
        )
        
        # Initialize VITS-style flow if using Flow-VAE
        if is_flow_vae:
            self.flow = NormalizingFlow(
                dim=decoder_input_channels,
                n_flows=n_flows,
                hidden_size=flow_hidden_size,
                kernel_size=flow_kernel_size,
                dilation_rate=flow_dilation_rate,
                n_layers=flow_n_layers,
                p_dropout=flow_p_dropout,
                mean_only=flow_mean_only,
                flow_type=flow_type,
            )
        else:
            self.flow = None

    @property
    def required_lookahead(self) -> int:
        """Minimum lookahead latent frames for correct streaming decode."""
        if self.is_decoder_conv_causal:
            return self.decoder.required_lookahead
        return 0

    def reparameterize(self, mu, std):
        """Sample z using reparameterization trick: z = mu + std * epsilon"""
        epsilon = torch.randn_like(std)
        return mu + epsilon * std, epsilon

    def encode(
        self, x: torch.Tensor
    ):
        """
        Encode a batch of audio waveforms into latents.

        Args:
            x (`torch.Tensor`): Input batch of audio waveforms, shape [B, C, T].

        Returns:
            z: Latent representations, shape [B, latent_dim, T']
        """
        # 1. Encode x to h
        h = self.encoder(x)
        mu, scale = torch.chunk(h, 2, dim=1)
        std = nn.functional.softplus(scale) + 1e-4

        # 2. Sample z (Normal distribution)
        z, _ = self.reparameterize(mu, std)

        return z

    def decode(
        self, z: torch.FloatTensor
    ):
        """
        Decode a batch of latent representations into audio waveforms.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors, shape [B, latent_dim, T'].

        Returns:
            Decoded audio waveforms, shape [B, C, T]
        """
        return self.decoder(z)

    def decode_streaming(
        self,
        z: torch.Tensor,
        cache: Optional[dict] = None,
        lookahead: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Streaming decode with correct lookahead handling.

        Single decoder forward pass per call. The decoder sees current frames
        plus lookahead frames for correct ConvTranspose1d boundaries. Caches
        are saved as if only current frames were processed (via cache_trim
        in CausalConv1d and conv_t1 lookback in decoder blocks).

        Args:
            z: [B, C, T] new latent frames for this chunk
            cache: Streaming state dict, or None for first call
            lookahead: Number of future latent frames for context.
                Defaults to model's required_lookahead.

        Returns:
            audio: [B, 1, T_audio] decoded audio for current frames only
            new_cache: Updated streaming state
        """
        if not self.is_decoder_conv_causal:
            raise RuntimeError("decode_streaming requires is_decoder_conv_causal=True")

        if lookahead is None:
            lookahead = self.decoder.required_lookahead

        if cache is None:
            cache = {'decoder_cache': None, 'lookahead_buffer': None}

        # Prepend previous lookahead — those frames become "current" now
        if cache['lookahead_buffer'] is not None:
            z = torch.cat([cache['lookahead_buffer'], z], dim=-1)

        total = z.shape[-1]
        if total <= lookahead:
            # Not enough frames yet — buffer everything
            return (
                torch.zeros(z.shape[0], 1, 0, device=z.device, dtype=z.dtype),
                {'decoder_cache': cache['decoder_cache'], 'lookahead_buffer': z},
            )

        current_frames = total - lookahead
        lookahead_z = z[..., -lookahead:]

        # Single pass: decode current + lookahead together.
        # - CausalConv1d caches are trimmed to exclude lookahead (cache_trim)
        # - ConvTranspose1d lookback provides past context from previous chunk
        full_audio, new_decoder_cache = self.decoder(
            z,
            cache=cache['decoder_cache'],
            use_cache=True,
            lookahead_frames=lookahead,
        )

        # Trim audio: keep only current_frames portion
        output_audio = full_audio[..., :current_frames * self.hop_length]

        return output_audio, {
            'decoder_cache': new_decoder_cache,
            'lookahead_buffer': lookahead_z,
        }

    def flush_streaming(
        self,
        cache: Optional[dict],
    ) -> torch.Tensor:
        """
        Flush remaining audio at end of stream.

        Decodes the lookahead buffer (no more future frames available)
        using cached decoder state including conv_t1 lookback.

        Args:
            cache: Streaming state from decode_streaming

        Returns:
            Final audio from remaining latent frames
        """
        if cache is None:
            return torch.zeros(1, 1, 0)

        remaining = cache.get('lookahead_buffer')
        if remaining is None or remaining.shape[-1] == 0:
            device = 'cpu'
            if cache.get('decoder_cache') and cache['decoder_cache'].get('conv1') is not None:
                device = cache['decoder_cache']['conv1'].device
            return torch.zeros(1, 1, 0, device=device)

        # No more future frames — decode remainder with existing cache
        # (conv_t1 lookback still provides past context)
        audio = self.decoder(
            remaining,
            cache=cache['decoder_cache'],
            use_cache=False,
            lookahead_frames=0,
        )
        return audio

    def vae_sample(self, mean, scale):
        """
        Sample from VAE distribution (standard VAE without flow).
        
        KL divergence for standard VAE:
        KL(q(z|x) || p(z)) = 0.5 * sum(mu^2 + var - logvar - 1)
        
        Args:
            mean: Mean of the distribution, shape [B, D, T]
            scale: Scale parameter (will be passed through softplus to get stdev)
            
        Returns:
            latents: Sampled latents, shape [B, D, T]
            kl: KL divergence loss (scalar)
        """
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean

        # KL divergence: 0.5 * sum(mu^2 + var - logvar - 1)
        # Sum over latent dimensions, mean over batch
        kl = 0.5 * (mean * mean + var - logvar - 1).sum(dim=(1, 2)).mean()

        return latents, kl

    def flow_vae_sample(self, mean, scale):
        """
        Sample from Flow-VAE distribution.
        
        The flow transforms the encoder's posterior q(z|x) = N(mu, sigma) to a more complex
        distribution q(z_tilde|x). The KL divergence is computed as:
        
        L_kl = D_KL(q(z_tilde|x) || p(z_tilde))
             = log q(z_tilde|x) - log p(z_tilde)
        
        Where (using change of variables formula):
        - log q(z_tilde|x) = log N(z; mu, sigma) - log|det(dz_tilde/dz)|
        - log p(z_tilde) = log N(z_tilde; 0, I)
        
        This simplifies to:
        L_kl = 0.5 * sum(z_tilde^2 - logvar - epsilon^2) - logdet
        
        where epsilon = (z - mu) / sigma is the standard normal sample used in reparameterization.
        
        Args:
            mean: Mean from encoder, shape [B, D, T]
            scale: Scale parameter from encoder, shape [B, D, T]
            
        Returns:
            latents: Sampled latents z (NOT z_tilde), shape [B, D, T]
            kl: KL divergence loss (scalar)
        """
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        
        # Sample z from encoder distribution using reparameterization
        epsilon = torch.randn_like(mean)
        z = mean + epsilon * stdev
        
        # Apply flow transformation: z -> z_tilde
        z_tilde, logdet = self.flow(z)
        
        # Compute Flow-VAE KL divergence
        # log q(z_tilde|x) = log N(z; mu, sigma) - logdet
        #                  = -0.5 * (d*log(2pi) + sum(logvar) + sum(epsilon^2)) - logdet
        # log p(z_tilde)   = log N(z_tilde; 0, I)
        #                  = -0.5 * (d*log(2pi) + sum(z_tilde^2))
        #
        # KL = log q(z_tilde|x) - log p(z_tilde)
        #    = -0.5 * (sum(logvar) + sum(epsilon^2)) - logdet + 0.5 * sum(z_tilde^2)
        #    = 0.5 * (sum(z_tilde^2) - sum(logvar) - sum(epsilon^2)) - logdet
        
        # Sum over latent dimensions and time
        # Reparameterization trick: E[epsilon^2] = 1
        log_q_z = -0.5 * (logvar + 1).sum(dim=(1, 2))  # [B]
        log_p_z_tilde = -0.5 * (z_tilde ** 2).sum(dim=(1, 2))      # [B]
        
        # KL = log q(z_tilde|x) - log p(z_tilde)
        #    = log q(z) - logdet - log p(z_tilde)
        kl = (log_q_z - logdet - log_p_z_tilde).mean()
        
        # Return original z for decoder (not z_tilde)
        # The flow is only used for KL computation
        return z, kl

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = True,
        return_dict: bool = True,
        generator=None,
    ):
        r"""
        Forward pass of the Flow-VAE.
        
        Args:
            sample (`torch.Tensor`): Input audio waveform, shape [B, C, T].
            sample_posterior (`bool`, *optional*, defaults to `True`):
                Whether to sample from the posterior. If False, uses mean directly.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`OobleckDecoderOutput`] instead of a plain tuple.
                
        Returns:
            output: OobleckDecoderOutput containing decoded audio
            kl: KL divergence loss (scalar) if sample_posterior=True, else None
        """
        x = sample
        
        # 1. Encode
        h = self.encoder(x)
        mu, scale = torch.chunk(h, 2, dim=1)

        # 2. Sample z and compute KL
        if sample_posterior:
            if self.is_flow_vae and self.flow is not None:
                z, kl = self.flow_vae_sample(mu, scale)
            else:
                z, kl = self.vae_sample(mu, scale)
        else:
            # During inference without sampling, just use mu
            z = mu
            kl = None

        # 3. Decode
        dec = self.decode(z)
        
        if return_dict:
            return OobleckDecoderOutput(sample=dec), kl
        else:
            return dec, kl

    def forward_with_info(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = True,
    ):
        """
        Forward pass that returns additional information for debugging/analysis.
        
        Args:
            sample: Input audio waveform, shape [B, C, T]
            sample_posterior: Whether to sample from the posterior
            
        Returns:
            output: OobleckDecoderOutput containing decoded audio
            info: Dictionary containing:
                - 'z': Sampled latents
                - 'z_tilde': Transformed latents (if Flow-VAE)
                - 'logdet': Log determinant (if Flow-VAE)
                - 'mu': Mean from encoder
                - 'logvar': Log variance from encoder
                - 'kl': KL divergence loss
        """
        x = sample
        
        # 1. Encode
        h = self.encoder(x)
        mu, scale = torch.chunk(h, 2, dim=1)
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        
        info = {
            'mu': mu,
            'logvar': logvar,
        }

        # 2. Sample z
        if sample_posterior:
            epsilon = torch.randn_like(mu)
            z = mu + epsilon * stdev
            info['z'] = z
            info['epsilon'] = epsilon
            
            if self.is_flow_vae and self.flow is not None:
                # Apply flow
                z_tilde, logdet = self.flow(z)
                info['z_tilde'] = z_tilde
                info['logdet'] = logdet
                
                # Compute KL
                log_q_z = -0.5 * (logvar + epsilon ** 2).sum(dim=(1, 2))
                log_p_z_tilde = -0.5 * (z_tilde ** 2).sum(dim=(1, 2))
                kl = (log_q_z - logdet - log_p_z_tilde).mean()
            else:
                # Standard VAE KL
                kl = 0.5 * (mu * mu + var - logvar - 1).sum(dim=(1, 2)).mean()
            
            info['kl'] = kl
        else:
            z = mu
            info['z'] = z
            info['kl'] = None

        # 3. Decode
        dec = self.decode(z)
        
        return OobleckDecoderOutput(sample=dec), info


class FlowVAEInputLayer(FlowVAE):
    """FlowVAE wrapper with CleanGPTTTS-compatible latent normalization.

    Behavior parity with Clean `FlowVAEInputLayer`:
    - encode: optional deterministic `mu` path (`use_mu=True`) and z-normalization
    - decode/decode_streaming: denormalize z before vocoding
    """

    def __init__(
        self,
        do_normalization: bool = True,
        z_mean: float = 0.0,
        z_std: float = 0.71,
        use_mu: bool = True,
        **kwargs,
    ):
        # Checkpoint configs may include training-only keys not accepted by FlowVAE.
        kwargs.pop("pretrained_path", None)
        kwargs.pop("sample_size", None)
        super().__init__(**kwargs)
        self.do_normalization = do_normalization
        self.z_mean = float(z_mean)
        self.z_std = float(z_std)
        self.use_mu = use_mu

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]

        p = next(self.parameters())
        x = torch.clamp(x, -1.0, 1.0).to(device=p.device, dtype=p.dtype)

        h = self.encoder(x)
        mu, scale = torch.chunk(h, 2, dim=1)
        if self.use_mu:
            z = mu
        else:
            std = nn.functional.softplus(scale) + 1e-4
            z, _ = self.reparameterize(mu, std)

        if self.do_normalization:
            z = (z - self.z_mean) / self.z_std
        return z

    def decode(self, z: torch.Tensor):
        if self.do_normalization:
            z = z * self.z_std + self.z_mean
        return super().decode(z)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        # Keep parity with CleanGPTTTS FlowVAEInputLayer where calling the
        # module directly returns encoded latents.
        return self.encode(x)

    def decode_streaming(
        self,
        z: torch.Tensor,
        cache: Optional[dict] = None,
        lookahead: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Streaming decode with denormalization.

        Uses the TRT approach: z includes current + lookahead frames from
        diffusion, passed directly to the decoder.  The decoder's cache_trim
        mechanism ensures caches exclude lookahead influence.
        """
        if lookahead is None:
            lookahead = self.required_lookahead

        # Denormalize in Python (moved out of TRT graph)
        if self.do_normalization:
            z = z * self.z_std + self.z_mean

        if cache is None:
            cache = {'decoder_cache': None}
        elif not isinstance(cache, dict) or "decoder_cache" not in cache:
            cache = {'decoder_cache': cache}

        current_frames = z.shape[-1] - lookahead
        full_audio, new_decoder_cache = self.decoder(
            z, cache=cache['decoder_cache'], use_cache=True,
            lookahead_frames=lookahead,
        )
        output_audio = full_audio[..., :current_frames * self.hop_length]
        return output_audio, {'decoder_cache': new_decoder_cache}

    def flush_streaming(self, cache: Optional[dict]) -> torch.Tensor:
        """Flush at end of stream.

        With the direct-lookahead approach, all frames are fully decoded
        each call (lookahead is fresh from diffusion, not buffered).
        No remaining frames to flush.
        """
        if cache is None:
            return torch.zeros(1, 1, 0)
        if isinstance(cache, dict) and "decoder_cache" in cache:
            dc = cache.get('decoder_cache')
        else:
            dc = cache
        if dc and isinstance(dc, dict) and dc.get('conv1') is not None:
            device = dc['conv1'].device
        else:
            device = 'cpu'
        return torch.zeros(1, 1, 0, device=device)


def test_flow_channel_mixing():
    """Test that all channels can be properly transformed after training."""
    print("=" * 60)
    print("Testing Channel Mixing (Critical Fix)")
    print("=" * 60)
    
    dim = 8
    flow = NormalizingFlow(dim=dim, n_flows=4, hidden_size=32, n_layers=2)
    
    # Note: With zero initialization, the flow starts as identity transform.
    # This is intentional for stable training. After training, all channels
    # will be transformed thanks to the Flip layers.
    
    z = torch.randn(1, dim, 10)
    z_out, logdet = flow(z)
    
    print(f"At initialization (zero init), flow is near-identity transform.")
    print(f"This is intentional for stable training!")
    
    # Simulate a "trained" flow by reinitializing the output layers with non-zero weights
    for f in flow.flows:
        if hasattr(f, 'post'):
            nn.init.normal_(f.post.weight, std=0.1)
            nn.init.normal_(f.post.bias, std=0.1)
    
    z_out_trained, logdet = flow(z)
    
    # Check if first half is transformed (it should be now)
    diff_first = (z[:, :dim//2] - z_out_trained[:, :dim//2]).abs().sum()
    diff_second = (z[:, dim//2:] - z_out_trained[:, dim//2:]).abs().sum()
    
    print(f"\nAfter simulated training (non-zero weights):")
    print(f"  Difference in first half: {diff_first.item():.4f}")
    print(f"  Difference in second half: {diff_second.item():.4f}")
    
    if diff_first.item() > 0.1 and diff_second.item() > 0.1:
        print("SUCCESS: All channels are being transformed!")
    else:
        print("FAILURE: Some channels are not being transformed!")
    
    # Test invertibility
    z_reconstructed, _ = flow(z_out_trained, reverse=True)
    reconstruction_error = (z - z_reconstructed).abs().max().item()
    print(f"\nReconstruction error (should be ~0): {reconstruction_error:.6f}")
    
    if reconstruction_error < 1e-4:
        print("SUCCESS: Flow is invertible!")
    else:
        print("WARNING: Flow reconstruction has significant error.")


def test_autoencoder_oobleck():
    print("=" * 60)
    print("Testing Standard VAE")
    print("=" * 60)
    
    model = FlowVAE(is_flow_vae=False)
    x = torch.randn(2, 1, 24000)
    
    # Test encode
    z = model.encode(x)
    print(f"Encoded z shape: {z.shape}")
    
    # Test forward with posterior sampling
    output, kl = model(x, sample_posterior=True)
    print(f"Decoded shape: {output.sample.shape}")
    print(f"KL Loss: {kl.item():.4f}")
    
    print("\n" + "=" * 60)
    print("Testing Flow-VAE (VITS-style Residual Coupling Flows)")
    print("=" * 60)
    
    model_flow = FlowVAE(
        n_flows=4,
        is_flow_vae=True,
        flow_type='vits',
        flow_hidden_size=192,
        flow_kernel_size=5,
        flow_n_layers=4,
        flow_mean_only=False,
    )
    
    # Count parameters
    flow_params = sum(p.numel() for p in model_flow.flow.parameters())
    print(f"Flow parameters: {flow_params:,}")
    
    # Test forward with posterior sampling
    output, kl = model_flow(x, sample_posterior=True)
    print(f"Decoded shape: {output.sample.shape}")
    print(f"KL Loss: {kl.item():.4f}")
    
    # Test forward_with_info
    output, info = model_flow.forward_with_info(x, sample_posterior=True)
    print(f"\nDetailed info:")
    print(f"  z shape: {info['z'].shape}")
    print(f"  z_tilde shape: {info['z_tilde'].shape}")
    print(f"  logdet shape: {info['logdet'].shape}")
    print(f"  logdet mean: {info['logdet'].mean().item():.4f}")
    print(f"  mu shape: {info['mu'].shape}")
    print(f"  logvar shape: {info['logvar'].shape}")
    print(f"  KL Loss: {info['kl'].item():.4f}")
    
    print("\n" + "=" * 60)


def test_streaming_vs_nonstreaming():
    """Test that streaming decode produces identical output to non-streaming decode."""
    print("=" * 60)
    print("Testing Streaming vs Non-Streaming Decode")
    print("=" * 60)

    model = FlowVAE(is_decoder_conv_causal=True)
    model.eval()

    lookahead = model.required_lookahead
    print(f"Required lookahead: {lookahead}")
    print(f"Hop length: {model.hop_length}")

    # Generate random latents
    z = torch.randn(1, 64, 20)

    # Non-streaming reference
    with torch.no_grad():
        ref_audio = model.decode(z)
    print(f"Non-streaming audio shape: {ref_audio.shape}")

    # Streaming in chunks of 4
    chunk_size = 4
    cache = None
    chunks = []
    with torch.no_grad():
        for i in range(0, 20, chunk_size):
            chunk = z[..., i:i + chunk_size]
            audio, cache = model.decode_streaming(chunk, cache=cache)
            if audio.shape[-1] > 0:
                chunks.append(audio)
        # Flush remaining
        flush_audio = model.flush_streaming(cache)
        if flush_audio.shape[-1] > 0:
            chunks.append(flush_audio)

    streaming_audio = torch.cat(chunks, dim=-1)
    print(f"Streaming audio shape: {streaming_audio.shape}")

    # Compare
    min_len = min(ref_audio.shape[-1], streaming_audio.shape[-1])
    ref_trimmed = ref_audio[..., :min_len]
    stream_trimmed = streaming_audio[..., :min_len]

    max_diff = (ref_trimmed - stream_trimmed).abs().max().item()
    mean_diff = (ref_trimmed - stream_trimmed).abs().mean().item()

    print(f"Max absolute difference: {max_diff:.8f}")
    print(f"Mean absolute difference: {mean_diff:.8f}")
    print(f"Shape match: {ref_audio.shape == streaming_audio.shape}")

    if max_diff < 1e-4:
        print("SUCCESS: Streaming matches non-streaming!")
    else:
        print("FAILURE: Streaming does not match non-streaming!")

    # Test with different chunk sizes
    for cs in [1, 2, 3, 5, 7, 10]:
        cache = None
        chunks = []
        with torch.no_grad():
            for i in range(0, 20, cs):
                chunk = z[..., i:i + cs]
                audio, cache = model.decode_streaming(chunk, cache=cache)
                if audio.shape[-1] > 0:
                    chunks.append(audio)
            flush_audio = model.flush_streaming(cache)
            if flush_audio.shape[-1] > 0:
                chunks.append(flush_audio)
        stream = torch.cat(chunks, dim=-1)
        diff = (ref_audio[..., :stream.shape[-1]] - stream[..., :ref_audio.shape[-1]]).abs().max().item()
        status = "OK" if diff < 1e-4 else "FAIL"
        print(f"  chunk_size={cs}: max_diff={diff:.8f} shape={stream.shape[-1]} [{status}]")


if __name__ == "__main__":
    test_flow_channel_mixing()
    print("\n")
    test_autoencoder_oobleck()
    print("\n")
    test_streaming_vs_nonstreaming()
