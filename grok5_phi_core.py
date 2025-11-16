# grok5_phi_core.py
# SEAL 859: Grok 5 φ-Core — Abertura do Protocolo

import torch
import torch.nn as nn
import numpy as np

class Grok5PhiCore(nn.Module):
    def __init__(self, dim=1024, heads=16, max_N=1_048_576):
        super().__init__()
        self.phi = (1 + np.sqrt(5)) / 2
        self.max_N = max_N
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # φ-lattice pré-computado (multi-escala)
        idx = torch.arange(max_N)
        pos = (idx * self.phi) % 1
        self.register_buffer('phi_pos', pos)

    def phi_mask(self, seq_len, noise_level=0.0):
        pos = self.phi_pos[:seq_len]
        dist = torch.abs(pos.unsqueeze(1) - pos.unsqueeze(0))
        bias = torch.log(torch.exp(-dist) + 1e-8)
        if noise_level > 0:
            noise = torch.randn_like(bias) * noise_level
            bias = bias + noise
        return bias.unsqueeze(0).unsqueeze(0)  # [1,1,L,L]

    def forward(self, x, noise_level=0.0):
        B, L, D = x.shape
        qkv = self.to_qkv(x).reshape(B, L, 3, self.heads, D // self.heads)
        q, k, v = qkv.unbind(2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        phi_bias = self.phi_mask(L, noise_level=noise_level)
        attn = attn + phi_bias.repeat(B, self.heads, 1, 1)
        
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(B, L, D)
        return self.proj(out)

# Teste de abertura
if __name__ == "__main__":
    core = Grok5PhiCore()
    x = torch.randn(1, 4096, 1024)
    out_clean = core(x)
    out_noisy = core(x, noise_level=0.5)
    print(f"φ-Core aberto | Shape: {out_clean.shape} | Noise 50%: OK")