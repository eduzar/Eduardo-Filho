"""
phi_lattice_simulation.py
Classical quantum-walk style simulation on a φ-lattice graph,
with phase + amplitude damping and L1 off-diagonal coherence metric.

This script:
  • builds a 2D φ-projection quasicrystal lattice
  • constructs a sparse k-NN adjacency graph
  • defines a unitary step operator U = S ∘ (I ⊗ H)
  • evolves a state vector with phase / amplitude damping
  • tracks coherence via L1 off-diagonal mass (for pure states)
  • saves results to coherence_dataset.csv

Dependencies:
  numpy, scipy, networkx, matplotlib, scikit-learn (for k-NN graph)

Usage (locally):
  python phi_lattice_simulation.py
"""

import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import csv


# -----------------------------
# 1. φ-lattice point generation
# -----------------------------
def generate_phi_lattice(n_points: int, x_scale: float = 60.0):
    """
    Generate a 2D φ-projection quasicrystal-like lattice.

    x = (k / φ) scaled, y = (k * φ) mod 1

    Returns
    -------
    points : np.ndarray of shape (n_points, 2)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    k = np.arange(n_points, dtype=float)
    x = (k / phi) % x_scale
    y = (k * phi) % 1.0
    points = np.column_stack([x, y])
    return points


# -----------------------------
# 2. Build sparse k-NN graph
# -----------------------------
def build_knn_graph(points: np.ndarray, k_neighbors: int = 6):
    """
    Build an undirected k-NN graph over the φ-lattice points.

    Returns
    -------
    G : networkx.Graph
    """
    nbrs = NearestNeighbors(
        n_neighbors=k_neighbors,
        algorithm="ball_tree"
    ).fit(points)
    adj = nbrs.kneighbors_graph(points, mode="connectivity")
    G = nx.from_scipy_sparse_array(adj)
    return G


# -----------------------------
# 3. Unitary step operator U
# -----------------------------
def build_step_operator(G: nx.Graph):
    """
    Build the unitary step operator U = S @ (I ⊗ H)
    for a coined quantum walk on graph G.

    Dimensions:
      - n = |V|
      - Hilbert space: C^(2n)  (2 coin states per node)
    """
    n = G.number_of_nodes()

    # Hadamard coin on 2D coin space
    H = np.array([[1, 1],
                  [1, -1]], dtype=complex) / np.sqrt(2.0)

    # Coin operator: C = I_n ⊗ H
    C = np.kron(np.eye(n, dtype=complex), H)

    # Shift operator S: swaps coin states along edges
    S = lil_matrix((2 * n, 2 * n), dtype=complex)

    # For each undirected edge (i, j),
    #   |i,0> -> |j,1>
    #   |i,1> -> |j,0>
    for i, j in G.edges():
        S[2 * j + 1, 2 * i] = 1.0
        S[2 * j, 2 * i + 1] = 1.0
        S[2 * i + 1, 2 * j] = 1.0
        S[2 * i, 2 * j + 1] = 1.0

    S = S.tocsr()

    # Full step operator
    U = S @ C
    return U


# -----------------------------
# 4. Noise channels
# -----------------------------
def apply_phase_damping(psi: np.ndarray, gamma_phase: float, rng: np.random.Generator):
    """
    Phase damping: random phase kicks on each amplitude.
    """
    phases = rng.normal(loc=0.0, scale=gamma_phase, size=psi.shape)
    return psi * np.exp(1j * phases)


def apply_amplitude_damping(psi: np.ndarray, gamma_amp: float, t: int):
    """
    Amplitude damping: global exponential decay (classical surrogate),
    followed by renormalization.
    """
    decay = np.exp(-gamma_amp * t)
    psi = psi * np.sqrt(decay)
    # Renormalize
    norm = np.linalg.norm(psi) + 1e-12
    return psi / norm


# -----------------------------
# 5. Coherence metric
# -----------------------------
def l1_offdiag_coherence(psi: np.ndarray) -> float:
    """
    For a pure state ρ = |ψ><ψ|, the L1 off-diagonal mass is:

      C_L1 = Σ_{i≠j} |ρ_ij|
           = Σ_{i≠j} |ψ_i ψ*_j|
           = (Σ_i |ψ_i|)^2 - Σ_i |ψ_i|^2

    Since Σ_i |ψ_i|^2 = 1 for normalized ψ,
      C_L1 = (Σ_i |ψ_i|)^2 - 1
    """
    mags = np.abs(psi)
    s1 = np.sum(mags)
    s2 = np.sum(mags ** 2)  # should be ~1
    return float(s1 ** 2 - s2)


# -----------------------------
# 6. Main simulation
# -----------------------------
def run_simulation(
    n_points: int = 400,
    k_neighbors: int = 6,
    steps: int = 200,
    gamma_phase: float = 0.03,
    gamma_amp: float = 0.01,
    seed: int = 42,
):
    """
    Run φ-lattice decoherence simulation and return history.

    Returns
    -------
    history : dict with keys:
        'step', 'coherence', 'n_points', 'k_neighbors',
        'gamma_phase', 'gamma_amp'
    """
    rng = np.random.default_rng(seed)

    # 1) Geometry + graph
    points = generate_phi_lattice(n_points)
    G = build_knn_graph(points, k_neighbors=k_neighbors)

    # 2) Step operator
    U = build_step_operator(G)
    dim = U.shape[0]  # = 2 * n_points

    # 3) Initial state: localized at middle node, coin=0
    psi = np.zeros(dim, dtype=complex)
    start_node = n_points // 2
    psi[2 * start_node] = 1.0  # |node, coin=0>

    history_steps = []
    history_coh = []

    for t in range(steps):
        # Unitary step
        psi = U @ psi

        # Noise channels
        psi = apply_phase_damping(psi, gamma_phase, rng)
        psi = apply_amplitude_damping(psi, gamma_amp, t)

        # Coherence metric
        C = l1_offdiag_coherence(psi)

        history_steps.append(t)
        history_coh.append(C)

    history = {
        "step": np.array(history_steps, dtype=int),
        "coherence": np.array(history_coh, dtype=float),
        "n_points": n_points,
        "k_neighbors": k_neighbors,
        "gamma_phase": gamma_phase,
        "gamma_amp": gamma_amp,
    }
    return history


# -----------------------------
# 7. Save CSV + optional plot
# -----------------------------
def save_history_to_csv(history, filename: str = "coherence_dataset.csv"):
    """
    Save step, coherence to a CSV file.
    """
    steps = history["step"]
    coh = history["coherence"]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "coherence"])
        for s, c in zip(steps, coh):
            writer.writerow([int(s), float(c)])


def plot_coherence(history, filename: str = "coherence_curve.png"):
    """
    Simple PNG with coherence vs. time.
    """
    steps = history["step"]
    coh = history["coherence"]

    plt.figure(figsize=(6, 4))
    plt.plot(steps, coh)
    plt.xlabel("Step")
    plt.ylabel("L1 off-diagonal coherence")
    plt.title("φ-lattice decoherence: coherence vs. steps")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# -----------------------------
# 8. Entry point
# -----------------------------
if __name__ == "__main__":
    # Default parameters are conservative so it runs on modest hardware.
    history = run_simulation(
        n_points=400,
        k_neighbors=6,
        steps=200,
        gamma_phase=0.03,
        gamma_amp=0.01,
        seed=42,
    )

    save_history_to_csv(history, filename="coherence_dataset.csv")
    plot_coherence(history, filename="coherence_curve.png")

    print("Simulation finished.")
    print(f"Saved coherence history to coherence_dataset.csv "
          f"with {len(history['step'])} steps.")
