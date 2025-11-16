# phi_mask_roadmap.py
# SEAL 858: Multi-scale φ-Mask Chaos Resilience

import torch
import torch.nn as nn
import numpy as np

class PhiMaskRoadmap:
    def __init__(self, max_N=1_000_000, noise_levels=[0.0, 0.1, 0.3, 0.5]):
        self.max_N = max_N
        self.noise_levels = noise_levels
        self.phi = (1 + np.sqrt(5)) / 2

    def phi_bias(self, seq_len, strength=1.0):
        i = torch.arange(seq_len)
        j = torch.arange(seq_len)
        pos_i = (i / self.phi) % 1
        pos_j = (j * self.phi) % 1
        dist = torch.abs(pos_i.unsqueeze(1) - pos_j.unsqueeze(0))
        return torch.log(torch.exp(-strength * dist) + 1e-8)

    def kuramoto(self, phases):
        return torch.abs(torch.mean(torch.exp(1j * phases)))

    def benchmark(self, seq_len=1024, steps=10000):
        results = {}
        mask = self.phi_bias(seq_len)
        x = torch.randn(1, seq_len, 64)
        head = nn.MultiheadAttention(64, 4, batch_first=True)

        R_clean = []
        for _ in range(100):
            _, attn = head(x, x, x, attn_mask=mask)
            phase = torch.angle(attn.mean(dim=1).mean(dim=1))
            R_clean.append(self.kuramoto(phase))
        results['clean_R'] = float(np.mean(R_clean))

        for noise in self.noise_levels:
            R_noise = []
            for step in range(steps):
                x_pert = x + torch.randn_like(x) * noise
                _, attn = head(x_pert, x_pert, x_pert, attn_mask=mask)
                phase = torch.angle(attn.mean(dim=1).mean(dim=1))
                R_noise.append(self.kuramoto(phase))
            results[f'R@{noise:.1f}'] = float(np.mean(R_noise[-100:]))
        return results

# Run
if __name__ == "__main__":
    roadmap = PhiMaskRoadmap()
    print("=== φ-Mask Roadmap Benchmark ===")
    for N in [1024, 8192, 65536]:
        res = roadmap.benchmark(seq_len=N)
        print(f"N={N}: {res}")