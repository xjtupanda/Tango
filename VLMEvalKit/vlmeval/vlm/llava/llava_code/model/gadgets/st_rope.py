import torch
import torch.nn as nn
import math


# Copied from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class SpatioTemporalRoPE(nn.Module):
    def __init__(self, config) -> None:
        """
        Args:
            base_time: Base theta for temporal dimension. The bigger, the slower of decay (allow for long range merging).
            base_space: Base theta for spatial dimension. The bigger, the slower of decay (allow for long range merging).
            strope_section: Size for each dimension.
        """
        super().__init__()
        base_time = getattr(config, 'base_time', 10000.0)
        base_space = getattr(config, 'base_space', 1000.0)
        self.strope_section = getattr(config, 'strope_section', [1196, 1194, 1194])
        self.d_t, self.d_h, self.d_w = self.strope_section
        
        
        inv_freq_t = 1.0 / (base_time ** (torch.arange(0, self.d_t, 2, dtype=torch.float) / self.d_t))
        inv_freq_h = 1.0 / (base_space ** (torch.arange(0, self.d_h, 2, dtype=torch.float) / self.d_h))
        inv_freq_w = 1.0 / (base_space ** (torch.arange(0, self.d_w, 2, dtype=torch.float) / self.d_w))
        
        self.register_buffer("inv_freq_t", inv_freq_t, persistent=False)
        self.register_buffer("inv_freq_h", inv_freq_h, persistent=False)
        self.register_buffer("inv_freq_w", inv_freq_w, persistent=False)

    def _apply_rotary_emb(self, x, freqs):
        # x: (B, L, D'), 
        # freqs: (B, L, D'/2)
        orig_x_dtype = x.dtype
        x = x.float()
        
        freqs = torch.cat([freqs, freqs], dim=-1) # (B, N, D)
        cos, sin = freqs.cos(), freqs.sin()
        
        x_embed = (x * cos) + (rotate_half(x) * sin)
        x_embed = x_embed.to(orig_x_dtype)
        return x_embed

    def forward(self, x, t_ids=None):
        """
        Args:
            x: (T, N, D)
            t_ids: List or Tensor of shape (T,). 
                   Optional. If provided, represents the physical time (in seconds) for each frame.
                   If None, defaults to uniform interval of 1.
        """
        T, N, D = x.shape
        H = W = math.isqrt(N)
        
        device = x.device
        
        if t_ids is not None:
            if isinstance(t_ids, list):
                t_idx = torch.tensor(t_ids, device=device)
            else:
                t_idx = t_ids.to(device)
            
            t_idx = t_idx.type_as(self.inv_freq_t)
        else:
            t_idx = torch.arange(T, device=device).type_as(self.inv_freq_t)
        h_idx = torch.arange(H, device=device).type_as(self.inv_freq_h)
        w_idx = torch.arange(W, device=device).type_as(self.inv_freq_w)
        
        grid_t, grid_h, grid_w = torch.meshgrid(t_idx, h_idx, w_idx, indexing='ij')
        
        flat_t = grid_t.reshape(-1)
        flat_h = grid_h.reshape(-1)
        flat_w = grid_w.reshape(-1)
        
        angles_t = torch.outer(flat_t, self.inv_freq_t) # (T*N, D//2)
        angles_h = torch.outer(flat_h, self.inv_freq_h)
        angles_w = torch.outer(flat_w, self.inv_freq_w)
        
        x_flat = x.reshape(-1, D)
        
        x_t, x_h, x_w = torch.split(x_flat, self.strope_section, dim=-1)
        out_t = self._apply_rotary_emb(x_t, angles_t)
        out_h = self._apply_rotary_emb(x_h, angles_h)
        out_w = self._apply_rotary_emb(x_w, angles_w)

        out = torch.cat([out_t, out_h, out_w], dim=-1)
        return out.reshape(T, N, D)
