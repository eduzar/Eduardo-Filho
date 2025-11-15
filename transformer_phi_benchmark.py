import torch
import torch.nn as nn
from phi_mask.phi_attention import PhiAttentionMask
from utils.metrics import coherence_drop, kuramoto_order_parameter


# ------------------------------
# Basic Transformer Attention Head
# ------------------------------

class SimpleAttentionHead(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

    def forward(self, x, phi_bias=None):
        B, L, D = x.shape

        q = self.Wq(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)

        if phi_bias is not None:
            attn_scores = attn_scores + phi_bias  # add φ-topological bias

        attn_weights = attn_scores.softmax(dim=-1)
        out = attn_weights @ v
        return out.transpose(1, 2).reshape(B, L, -1), attn_weights


# ------------------------------
# Benchmark: Vanilla vs φ-Mask
# ------------------------------

def transformer_phi_benchmark(seq_len=64, d_model=64):

    device = "cpu"
    x = torch.randn(1, seq_len, d_model, device=device)

    head = SimpleAttentionHead(d_model=d_model).to(device)

    phi_mask = PhiAttentionMask(max_len=seq_len, strength=1.0, device=device)
    phi_bias = phi_mask(seq_len)

    # forward (vanilla)
    _, A_vanilla = head(x, phi_bias=None)

    # forward (phi)
    _, A_phi = head(x, phi_bias=phi_bias)

    # perturb state
    x_pert = x.clone()
    idx = torch.randperm(seq_len)[: int(seq_len * 0.20)]
    x_pert[:, idx] += torch.randn_like(x_pert[:, idx]) * 0.25  # chaos perturbation

    # forward perturbed
    _, A_vanilla_pert = head(x_pert, phi_bias=None)
    _, A_phi_pert = head(x_pert, phi_bias=phi_bias)

    # Convert attention to phases for Kuramoto metric
    phase_v = torch.angle(torch.polar(torch.ones_like(A_vanilla[...,0,0]), A_vanilla.mean(dim=2).mean(dim=1)))
    phase_p = torch.angle(torch.polar(torch.ones_like(A_phi[...,0,0]), A_phi.mean(dim=2).mean(dim=1)))

    R_v = kuramoto_order_parameter(phase_v)
    R_p = kuramoto_order_parameter(phase_p)

    R_vp = kuramoto_order_parameter(
        torch.angle(
            torch.polar(
                torch.ones_like(A_vanilla_pert[...,0,0]),
                A_vanilla_pert.mean(dim=2).mean(dim=1)
            )
        )
    )

    R_pp = kuramoto_order_parameter(
        torch.angle(
            torch.polar(
                torch.ones_like(A_phi_pert[...,0,0]),
                A_phi_pert.mean(dim=2).mean(dim=1)
            )
        )
    )

    drop_v = coherence_drop(R_v, R_vp)
    drop_p = coherence_drop(R_p, R_pp)

    return {
        "vanilla_R": float(R_v),
        "phi_R": float(R_p),
        "vanilla_drop%": float(drop_v),
        "phi_drop%": float(drop_p)
    }



if __name__ == "__main__":
    results = transformer_phi_benchmark(seq_len=64, d_model=64)
    print("=== Transformer φ-Mask Stability Test ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")