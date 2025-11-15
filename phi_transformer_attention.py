# phi_transformer_attention.py
# SEAL 855: φ-Transformer Attention Hybrid

import torch
import torch.nn as nn
import numpy as np

class PhiAttention(nn.Module):
    def __init__(self, dim, heads=8, N=1024):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        
        # φ-lattice mask
        phi = (1 + np.sqrt(5)) / 2
        idx = torch.arange(N)
        x = (idx / phi) % 1
        y = (idx * phi) % 1
        coords = torch.stack([x, y], dim=1)
        dist = torch.cdist(coords, coords)
        k = 16
        _, nn_idx = torch.topk(dist, k, largest=False)
        mask = torch.zeros(N, N)
        for i in range(N):
            mask[i, nn_idx[i]] = 1
        self.register_buffer('phi_mask', mask)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.unbind(2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.phi_mask[:N, :N] == 0, -1e9)
        attn = attn.softmax(dim=-1)

        return (attn @ v).reshape(B, N, C)

# Teste
model = PhiAttention(512, N=1024)
x = torch.randn(1, 1024, 512)
out = model(x)
print("φ-Attention output:", out.shape)