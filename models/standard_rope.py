import torch
from timm.models.vision_transformer import Attention
from typing import Tuple

def compute_axial_cis(dim, coords: torch.Tensor, theta = 100.0):
    """
        Args:
            dim: head dimension
            coords: [N, 3]
        Returns:
            freq_cis: [N, head_dim]
    """
    assert dim % 6 == 0, f"dim ({dim}) must be divisible by 6 for 3D RoPE"
    x, y, z = coords.unbind(-1)

    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 6)[: dim // 6].float() / dim)) # dim / 6
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 6)[: dim // 6].float() / dim))
    freqs_z = 1.0 / (theta ** (torch.arange(0, dim, 6)[: dim // 6].float() / dim))

    freq_x = torch.outer(x, freqs_x.to(x.device))
    freq_y = torch.outer(y, freqs_y.to(y.device))
    freq_z = torch.outer(z, freqs_z.to(z.device))

    freq_cis_x = torch.polar(torch.ones_like(freq_x), freq_x) 
    freq_cis_y = torch.polar(torch.ones_like(freq_y), freq_y) 
    freq_cis_z = torch.polar(torch.ones_like(freq_z), freq_z)
    return torch.cat([freq_cis_x, freq_cis_y, freq_cis_z], dim=-1)

def reshape_for_broadcast(
        freqs_cis: torch.Tensor, 
        x: torch.Tensor
        ) -> torch.Tensor:
    """
        Args:
            freqs_cis: [L, head_dim // 2] or [num_heads, L, head_dim // 2]
            x: [B, H, L, head_dim // 2]
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]  # [1, 1, L, head_dim, 2]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]  # [1, num_heads, L, head_dim/2, 2]
    elif freqs_cis.shape[-2:] == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 or i == 0 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Args:
            xq/xk: [B, H, L, head_dim]
            freqs_cis: [L, head_dim // 2] or [num_heads, L, head_dim // 2]
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2).contiguous()) # [B, H, L, head_dim // 2]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2).contiguous())

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_) # [1, H, L, head_dim // 2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # merge the dimesions from the third dimension to the last dimension
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3) # [batch_size, num_heads, length, head_dim/2, 2] -> [batch_size, num_heads, length, head_dim]
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device) # [batch_size, num_heads, length, head_dim]


class RoPEAttention(Attention):
    def __init__(self, freqs_cis, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        self.freqs_cis = freqs_cis

    def forward(self, x: torch.Tensor):
        self.freqs_cis = self.freqs_cis.to(x.device)

        B, N, C = x.shape
        # [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
    
        # Apply rotary position encoding to [1:] tokens, skipping CLS
        q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis)

        # Follow the timm source code
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RoPEAttentionCLS(Attention):
    def __init__(self, freqs_cis, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        # Store RoPE parameters as module attributes, so they can be accessed directly in forward
        self.freqs_cis = freqs_cis

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        self.freqs_cis = self.freqs_cis.to(x.device)

        B, N, C = x.shape
        # [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
    
        # Apply rotary position encoding to [1:] tokens, skipping CLS
        q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=self.freqs_cis)

        # Follow the timm source code
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x