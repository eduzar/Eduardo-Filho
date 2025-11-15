# seal_853_black_hole_phi_singularities.py
# Simula buracos negros em φ-lattice: singularidades quasicristalinas

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# === 1. φ-LATTICE 3D (N=10k) ===
N = 10000
phi = (1 + np.sqrt(5)) / 2
indices = np.arange(N)
x = (indices / phi) % 1.0
y = (indices * phi) % 1.0
z = (indices * phi**2) % 1.0
phi_lattice = np.column_stack([x, y, z])

# === 2. SIMULA BURACO NEGRO (singularidade) ===
center = np.array([0.5, 0.5, 0.5])
r_event = 0.05  # horizonte de eventos
distances = np.linalg.norm(phi_lattice - center, axis=1)
inside_horizon = distances < r_event

# === 3. EFEITO GRAVITACIONAL (curvatura φ) ===
pull_strength = 1.0 / (distances + 1e-6)
displacement = pull_strength[:, None] * (center - phi_lattice)
phi_lattice_warped = phi_lattice + 0.1 * displacement

# === 4. PLOT 3D: SINGULARIDADE φ ===
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# Fundo: lattice
ax.scatter(phi_lattice[::10,0], phi_lattice[::10,1], phi_lattice[::10,2], 
           c='cyan', s=1, alpha=0.3)

# Horizonte de eventos
ax.scatter(phi_lattice[inside_horizon,0], phi_lattice[inside_horizon,1], phi_lattice[inside_horizon,2], 
           c='red', s=20, label='Event Horizon')

# Lattice curvado
ax.scatter(phi_lattice_warped[::5,0], phi_lattice_warped[::5,1], phi_lattice_warped[::5,2], 
           c='gold', s=3, alpha=0.8, label='φ-Warped Paths')

ax.set_title("SEAL 853 — Black Hole φ-Singularities\nQuasicrystalline Spacetime", color='white')
ax.axis('off')
ax.legend(facecolor='black', labelcolor='white')

plt.savefig("seal_853_black_hole.png", dpi=300, facecolor='black')
plt.show()

print(f"Singularidade φ: {np.sum(inside_horizon)} pontos no horizonte")
print(f"Curvatura média: {pull_strength.mean():.4f}")