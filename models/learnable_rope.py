import torch
import torch.nn as nn
from .standard_rope import apply_rotary_emb, RoPEAttention as StandardRoPEAttention, RoPEAttentionCLS as StandardRoPEAttentionCLS
from timm.models.vision_transformer import Attention

class CayleyLearnerPerHead(nn.Module):
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_heads, head_dim, head_dim))

    def forward(self):
        A = self.W - self.W.transpose(-1, -2)  # Ensure skew-symmetry per head
        I = torch.eye(A.size(-1), device=A.device).expand_as(A)
        Q = torch.linalg.solve((I + A), (I - A))  # Cayley transform
        return Q  # shape: (num_heads, head_dim, head_dim)


class GivensRotationPerHead(nn.Module):
    def __init__(self, num_heads: int, dim: int, num_rotations: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        # Default to all possible Givens rotations
        if num_rotations is None:
            num_rotations = dim * (dim - 1) // 2
        self.rotation_indices = self._create_rotation_indices(dim, num_rotations)

        # Initialize rotation angles theta (one set per head)
        self.thetas = nn.Parameter(torch.randn(num_heads, len(self.rotation_indices)) * 0.01)

    def _create_rotation_indices(self, n, num_rotations):
        """ Create Givens rotation indices (i, j) where i < j """
        indices = []
        for i in range(n):
            for j in range(i + 1, n):
                indices.append((i, j))
                if len(indices) >= num_rotations:
                    return indices
        return indices

    def apply_batch_givens(self, Q, thetas, indices):
        """ Apply Givens rotations to Q, keeping the computation graph safe """
        Q_new = Q.clone()  # Avoid in-place write operations that break autograd

        for k, (i, j) in enumerate(indices):
            theta = thetas[:, k]  # (H,)
            c = torch.cos(theta).unsqueeze(-1)  # (H, 1)
            s = torch.sin(theta).unsqueeze(-1)  # (H, 1)

            # Clone rows to avoid inplace operations that break the backward pass
            Qi = Q_new[:, i, :].clone()
            Qj = Q_new[:, j, :].clone()

            Q_new[:, i, :] = c * Qi + s * Qj
            Q_new[:, j, :] = -s * Qi + c * Qj

        return Q_new

    def forward(self):
        """ Generate orthogonal rotation matrix Q for each head, shape: (H, D, D) """
        device = self.thetas.device
        Q_init = torch.eye(self.dim, device=device).expand(self.num_heads, self.dim, self.dim).clone()
        return self.apply_batch_givens(Q_init, self.thetas, self.rotation_indices)


class HouseholderPerHead(nn.Module):
    def __init__(self, num_heads: int, dim: int, num_reflections: int = 6):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.k = num_reflections
        self.vs = nn.Parameter(torch.randn(num_heads, self.k, dim))

    def forward(self):
        Q = torch.eye(self.dim, device=self.vs.device).expand(self.num_heads, self.dim, self.dim).clone()  # (H, D, D)

        for i in range(self.k):
            v = self.vs[:, i, :]  # (H, D)
            v = v / v.norm(dim=-1, keepdim=True)  # Normalize each head's vector
            H = torch.eye(self.dim, device=v.device) - 2 * torch.einsum("hi,hj->hij", v, v)  # (H, D, D)
            Q = torch.einsum("hij,hjk->hik", H, Q)

        return Q  # shape: (H, D, D)


class LearnerRoPEAttention(Attention):
    def __init__(self, freqs_cis: torch.Tensor, learner_type: str, dim, num_heads=8, **kwargs):
        super().__init__(dim, num_heads, **kwargs)
        self.head_dim = dim // num_heads
        self.learner_type = learner_type
        
        if learner_type == 'standard':
            # For standard RoPE, we don't need Q_learner
            self.Q_learner = None
        else:
            learner_map = {
                'cayley': CayleyLearnerPerHead,
                'givens': GivensRotationPerHead,
                'householder': HouseholderPerHead
            }
            self.Q_learner = learner_map[learner_type](self.num_heads, self.head_dim)
        self.freqs_cis = freqs_cis

    def forward(self, x: torch.Tensor):
        self.freqs_cis = self.freqs_cis.to(x.device)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.Q_learner is not None:
            Q = self.Q_learner()
            self.last_Q = Q
            q = torch.einsum('bhnc, hcd -> bhnd', q, Q.transpose(-1, -2))
            k = torch.einsum('bhnc, hcd -> bhnd', k, Q.transpose(-1, -2))
        q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))
    
    def get_Q(self):
        return getattr(self, 'last_Q', None)  


class LearnerRoPEAttentionCLS(Attention):
    def __init__(self, freqs_cis: torch.Tensor, learner_type: str, dim, num_heads=8, **kwargs):
        super().__init__(dim, num_heads, **kwargs)
        self.head_dim = dim // num_heads
        self.learner_type = learner_type
        
        if learner_type == 'standard':
            # For standard RoPE, we don't need Q_learner
            self.Q_learner = None
        else:
            learner_map = {
                'cayley': CayleyLearnerPerHead,
                'givens': GivensRotationPerHead,
                'householder': HouseholderPerHead
            }
            self.Q_learner = learner_map[learner_type](self.num_heads, self.head_dim)
        self.freqs_cis = freqs_cis

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        self.freqs_cis = self.freqs_cis.to(x.device)
        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.Q_learner is not None:
            Q = self.Q_learner()
            self.last_Q = Q
            q = torch.einsum('bhnc, hcd -> bhnd', q, Q.transpose(-1, -2))
            k = torch.einsum('bhnc, hcd -> bhnd', k, Q.transpose(-1, -2))
        
        q_cls, q_pos = q[:, :, :1, :], q[:, :, 1:, :]
        k_cls, k_pos = k[:, :, :1, :], k[:, :, 1:, :]
        rot_q_pos, rot_k_pos = apply_rotary_emb(q_pos, k_pos, freqs_cis=self.freqs_cis)
        q = torch.cat([q_cls, rot_q_pos], dim=2)
        k = torch.cat([k_cls, rot_k_pos], dim=2)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))
    
    def get_Q(self):
        if self.learner_type == 'standard':
            return None
        return getattr(self, 'last_Q', None)  