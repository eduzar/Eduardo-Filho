"""
SEAL 850 — φ-Lattice Extended-N Noise-as-Information Scan

This script explores how φ-inspired aperiodic couplings plus simple
phase/amplitude damping can still sustain coherence pockets as system
size N grows.

It is NOT a full physical model — it's a classical/heuristic
"quantum-walk style" toy model designed to:
  • treat noise as structured information
  • scan coherence vs. N
  • log results for further analysis (e.g., in Jupyter / Colab)

Author: Jose De Oliveira Filho (Kael Zar)
"""

import numpy as np
import csv


PHI = (1.0 + 5.0 ** 0.5) / 2.0  # golden ratio


def build_aperiodic_couplings(N: int, phi: float = PHI) -> np.ndarray:
    """
    Build a 1D aperiodic coupling pattern inspired by φ.

    We use a simple modulation:
      J_i = 1.0 + eps * sign(sin(2π * φ * i))

    This creates an aperiodic sequence of stronger / weaker links.
    """
    i = np.arange(N, dtype=float)
    eps = 0.3
    pattern = np.sin(2.0 * np.pi * phi * i)
    couplings = 1.0 + eps * np.sign(pattern)
    # Ensure no zero couplings
    couplings[couplings == 0.0] = 1.0
    return couplings


def l1_offdiag_coherence(psi: np.ndarray) -> float:
    """
    Approximate L1 off-diagonal coherence:

        C_L1 ≈ Σ_{i != j} |ψ_i ψ*_j|

    For a pure state this is equivalent to:
        (Σ_i |ψ_i|)^2 - Σ_i |ψ_i|^2
    """
    mags = np.abs(psi)
    s1 = np.sum(mags)
    s2 = np.sum(mags ** 2)
    return float(s1 ** 2 - s2)


def step_dynamics(
    psi: np.ndarray,
    couplings: np.ndarray,
    gamma_phase: float,
    gamma_amp: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Single time step:
      1) aperiodic "hopping" with nearest neighbours
      2) phase noise (random phases)
      3) amplitude damping
      4) renormalization
    """
    N = psi.shape[0]

    # Nearest-neighbour hopping with aperiodic couplings
    new_psi = np.zeros_like(psi, dtype=np.complex128)
    for i in range(N):
        left = (i - 1) % N
        right = (i + 1) % N
        J_left = couplings[left]
        J_right = couplings[i]
        # Simple symmetric update
        new_psi[i] = (
            0.4 * psi[i]
            + 0.3 * J_left * psi[left]
            + 0.3 * J_right * psi[right]
        )

    psi = new_psi

    # Phase noise: random phases with variance ~ gamma_phase
    if gamma_phase > 0.0:
        phases = rng.normal(loc=0.0, scale=gamma_phase, size=N)
        psi *= np.exp(1j * phases)

    # Amplitude damping
    if gamma_amp > 0.0:
        psi *= np.exp(-gamma_amp)

    # Renormalize to avoid numerical blow-up
    norm = np.linalg.norm(psi)
    if norm > 0:
        psi /= norm

    return psi


def run_simulation(
    N: int,
    timesteps: int,
    gamma_phase: float,
    gamma_amp: float,
    seed: int,
) -> np.ndarray:
    """
    Run a single φ-lattice toy simulation and return
    the coherence time series.
    """
    rng = np.random.default_rng(seed)

    # initial state: localized + small random perturbation
    psi = np.zeros(N, dtype=np.complex128)
    psi[N // 2] = 1.0
    psi += 0.05 * (rng.normal(size=N) + 1j * rng.normal(size=N))
    psi /= np.linalg.norm(psi)

    couplings = build_aperiodic_couplings(N)

    coherence = np.zeros(timesteps, dtype=float)
    for t in range(timesteps):
        psi = step_dynamics(psi, couplings, gamma_phase, gamma_amp, rng)
        coherence[t] = l1_offdiag_coherence(psi)

    return coherence


def scan_extended_N(
    N_list,
    timesteps: int = 400,
    gamma_phase: float = 0.10,
    gamma_amp: float = 0.02,
    seeds=(0, 1, 2),
    out_csv: str = "seal850_extendedN_coherence_scan.csv",
):
    """
    Scan coherence for several N and random seeds.
    Stores summary stats per run in a CSV:
      N, seed, gamma_phase, gamma_amp,
      t_max, coherence_max, coherence_mean
    """
    rows = []

    for N in N_list:
        for seed in seeds:
            coherence = run_simulation(
                N=N,
                timesteps=timesteps,
                gamma_phase=gamma_phase,
                gamma_amp=gamma_amp,
                seed=seed,
            )
            t_max = int(np.argmax(coherence))
            c_max = float(np.max(coherence))
            c_mean = float(np.mean(coherence))

            rows.append(
                {
                    "N": N,
                    "seed": seed,
                    "gamma_phase": gamma_phase,
                    "gamma_amp": gamma_amp,
                    "timesteps": timesteps,
                    "t_max": t_max,
                    "coherence_max": c_max,
                    "coherence_mean": c_mean,
                }
            )

    fieldnames = [
        "N",
        "seed",
        "gamma_phase",
        "gamma_amp",
        "timesteps",
        "t_max",
        "coherence_max",
        "coherence_mean",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[SEAL 850] Saved results to {out_csv}")


if __name__ == "__main__":
    # Default scan — you can edit these values as you like.
    N_list = [64, 128, 256, 512]
    scan_extended_N(N_list)