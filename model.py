import math

import torch
import torch.nn.functional as F
import torchaudio.transforms as T

from einops import rearrange

from torch import nn


def _check_shape(x: torch.Tensor, size: int) -> torch.Tensor:
    if len(x.shape) != 3:
        raise ValueError("Input should have three axes: batch x seq x dim")
    elif x.shape[2] != size:
        raise ValueError(
            f"The size on the third axis of x is"
            f" {x.shape[2]}, but expected {size}."
        )
    return x.dtype


def rotate_rope(
    pos: torch.Tensor,
    dim: int,
    theta: int = 10000,
) -> torch.Tensor:
    """The flux2 style rotation matrix generator using einops."""
    assert dim % 2 == 0, dim
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    # Compute the angles
    out = torch.einsum("...n, d -> ...nd", pos, omega)

    cos, sin = torch.cos(out), torch.sin(out)
    # Stack elements to form rotation matrix components
    out = torch.stack([cos, -sin, sin, cos], dim=-1)
    # Shape into (..., head_dim//2, 2, 2)
    return rearrange(out, "... d (i j) -> ... d i j", i=2, j=2)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    # 1. Split head_dim into pairs (b, h, s, d, 2)
    xq_ = rearrange(xq, "b h s (d i) -> b h s d i", i=2)
    xk_ = rearrange(xk, "b h s (d i) -> b h s d i", i=2)

    # 2. freqs_cis is (b, s, d, 2, 2)
    # Use einsum to multiply the 2-dim vector by the 2x2 rotation matrix
    # Pattern: ...i (vector), ...ij (matrix) -> ...j (rotated vector)
    xq_out = torch.einsum("bhsd i, bsd i j -> bhsd j", xq_, freqs_cis)
    xk_out = torch.einsum("bhsd i, bsd i j -> bhsd j", xk_, freqs_cis)

    # 3. Flatten back to (b, h, s, head_dim)
    xq_final = rearrange(xq_out, "b h s d j -> b h s (d j)")
    xk_final = rearrange(xk_out, "b h s d j -> b h s (d j)")

    return xq_final.type_as(xq), xk_final.type_as(xk)


class RMSNorm(nn.Module):
    def __init__(
        self,
        size: int,
        eps: float = 1e-6,
        use_scale: bool = True,
    ):
        super().__init__()
        self.size = size
        self.eps = eps
        if use_scale:
            self.scale = torch.nn.Parameter(torch.ones(size))
        else:
            self.scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Assume Batch x Seq x Dim input and returns the same."""
        x_dtype = x.dtype
        # Make sure x is float32 for norm layers.
        x32 = x.to(torch.float32)
        invrms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        x_ = x32 * invrms
        if self.scale is not None:
            return (self.scale * x_).to(x_dtype)
        return x_.to(x_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        input_size: int,
        factor: float | None = None,
        size: int | None = None,
    ):
        """The internal representation either has size 'size'
        or 'input_size' * 'factor'."""
        super().__init__()
        self.input_size = input_size
        self.size = size
        if factor is None and size is None:
            raise ValueError("One of 'factor' or 'size' should be not None.")
        if factor is not None:
            size = int(input_size * factor)
        self.act = nn.SiLU()
        self.in_proj = nn.Linear(input_size, size)
        self.gate_proj = nn.Linear(input_size, size)
        self.out_proj = nn.Linear(size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We don't need fp32 here, fp16, bf16 etc. is enough.
        _ = _check_shape(x, self.input_size)
        return self.out_proj(self.act(self.gate_proj(x)) * self.in_proj(x))


class Attention(nn.Module):
    def __init__(
        self,
        input_size: int,
        head_size: int,
        n_heads: int,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.use_sdpa = use_sdpa
        self.input_size = input_size
        self.head_size = head_size
        self.n_heads = n_heads
        self.qkv = nn.Linear(input_size, head_size * n_heads * 3)
        self.out_proj = nn.Linear(head_size * n_heads, input_size)
        self.q_norm = RMSNorm(head_size)
        self.k_norm = RMSNorm(head_size)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_shape(x, size=self.input_size)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q_ = rearrange(q, "b s (h d) -> b h s d", d=self.head_size)
        k_ = rearrange(k, "b s (h d) -> b h s d", d=self.head_size)
        v_ = rearrange(v, "b s (h d) -> b h s d", d=self.head_size)
        q_n = self.q_norm(q_)
        k_n = self.k_norm(k_)
        if pos is None:
            pos = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
            # Expand to [1, seq_len]
            pos = rearrange(pos, "s -> 1 s")
        freqs = rotate_rope(pos, self.head_size)
        q_rot, k_rot = apply_rope(q_n, k_n, freqs)
        if self.use_sdpa:
            v_mixed = F.scaled_dot_product_attention(
                q_rot,
                k_rot,
                v_,
                is_causal=False,  # Since you are building an encoder
            )
        else:
            scores = torch.matmul(
                q_rot, rearrange(k_rot, "b h s d -> b h d s", d=self.head_size)
            ) / math.sqrt(self.head_size)
            probs = F.softmax(scores, dim=-1)
            v_mixed = torch.matmul(probs, v_)
        v_back = rearrange(v_mixed, "b h s d -> b s (h d)", d=self.head_size)
        return self.out_proj(v_back)


class TrfBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        n_heads: int,
        mlp_factor: int | None = None,
        mlp_size: int | None = None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attention = Attention(hidden_size, head_size, n_heads)
        self.mlp = SwiGLU(hidden_size, mlp_factor, mlp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_pre = self.norm1(x)
        x_attn = self.attention(x_pre)
        x_res1 = x_attn + residual
        residual = x_res1
        x_post = self.norm2(x_res1)
        x_mlp = self.mlp(x_post)
        return x_mlp + residual


# The AudioMNIST paper gets 95% accuracy with only 8k SR.
class AudioTokenizer(nn.Module):
    def __init__(
        self,
        sample_rate: int = 8000,
        augment: bool = False,
    ):
        super().__init__()
        self.augment = augment
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=100,
            f_min=0.0,
            f_max=sample_rate // 2,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
        )

        self.db = T.AmplitudeToDB(
            stype="power",
            top_db=80.0,
        )

        # SpecAugment (only active during training)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=20)

    def forward(self, wav):
        """
        wav: (B, 1, T) or (1, T)
        returns: (B, n_mels, time)
        """
        x = self.mel(wav)
        x = self.db(x)

        if self.training and self.augment:
            x = self.freq_mask(x)
            x = self.time_mask(x)

        x = rearrange(x, "b m t -> b t m")
        return x


class AudioClassifier(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_classes: int,
        n_mels: int,
        n_blocks: int,
        hidden_size: int,
        head_size: int,
        n_heads: int,
        mlp_factor: int | None = None,
        mlp_size: int | None = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.tokenizer = AudioTokenizer(sample_rate=sample_rate)
        self.trf_blocks = nn.Sequential(*[
            TrfBlock(
                hidden_size=hidden_size,
                head_size=head_size,
                n_heads=n_heads,
                mlp_factor=mlp_factor,
                mlp_size=mlp_size,
            )
            for _ in range(n_blocks)
        ])
        if n_classes == 2:
            self.output_layer = nn.Linear(hidden_size, 1)
            self.output_act = nn.Sigmoid()
            self.threshold = 0.5
        elif n_classes > 2:
            self.output_layer = nn.Linear(hidden_size, n_classes)
            self.output_act = nn.Softmax(dim=-1)
            self.threshold = None
        else:
            raise ValueError(
                f"'n_classes' should be > 2, but found: {n_classes}."
            )

    @property
    def binary(self):
        return self.n_classes == 2

    def set_threshold(self, threshold: float):
        if self.threshold is None:
            raise ValueError(
                "Trying to set 'threshold' for softmax classifier."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B x Seq x D.
        returns only logits to be normalized by sigmoid.
        """
        tokenized = self.tokenizer(x)
        trf = self.trf_blocks(tokenized)
        # Trying last pooling, because its the cheapest.
        last = trf[:, -1, :]
        return self.output_layer(last)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B x Seq x D.
        returns 0 - 1 decisions.
        """
        tokenized = self.tokenizer(x)
        trf = self.trf_blocks(tokenized)
        last = trf[:, -1, :]
        probs = self.output_act(self.output_layer(last)).squeeze()
        if self.binary:
            preds = probs > self.threshold
        else:
            preds = probs.argmax(dim=-1)
        return preds.detach().numpy().astype(int)
