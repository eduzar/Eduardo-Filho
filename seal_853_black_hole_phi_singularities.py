# seal_853_black_hole_phi_singularities.py
# Simula buracos negros em φ-lattice:
# singularidades, densidade e métricas de clustering.

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# ================================
# 1. CONSTROI φ-LATTICE 3D (N = 10k)
# ================================

N = 10000
phi = (1 + np.sqrt(5)) / 2

indices = np.arange(N)
x = (indices / phi) % 1.0
y = (indices * phi) % 1.0
z = (indices * phi**2) % 1.0

phi_lattice = np.column_stack([x, y, z])

# ================================
# 2. DEFINE “BURACO NEGRO” (SINGULARIDADE)
# ================================

# Centro do "buraco negro" no meio do cubo [0,1]^3
center = np.array([0.5, 0.5, 0.5])

# Raio do horizonte de eventos
r_event = 0.08  # pode ajustar

# Distâncias de todos os pontos ao centro
distances = np.linalg.norm(phi_lattice - center, axis=1)

# Região interna (dentro do horizonte)
inside_event = distances <= r_event

# Casca ao redor do horizonte (zona de transição)
shell = (distances > r_event) & (distances <= 1.5 * r_event)

# Campo distante (fora da influência direta)
far_field = distances > 1.5 * r_event

# ================================
# 3. CONSTRÓI KD-TREE PARA MÉTRICAS LOCAIS
# ================================

tree = KDTree(phi_lattice)

# Número de vizinhos locais que vamos usar para avaliar clustering
k_neighbors = 12

# Para evitar contarmos o próprio ponto, usamos k+1 e ignoramos o primeiro
dists_all, _ = tree.query(phi_lattice, k=k_neighbors + 1)
local_radius_mean = dists_all[:, 1:].mean(axis=1)  # média distância até k vizinhos

# ================================
# 4. MÉTRICAS DA SINGULARIDADE φ
# ================================

num_inside = inside_event.sum()
num_shell = shell.sum()
num_far = far_field.sum()

frac_inside = num_inside / N
frac_shell = num_shell / N
frac_far = num_far / N

# Densidade local média (inversa da distância média aos vizinhos)
local_density = 1.0 / (local_radius_mean + 1e-9)

core_density_mean = local_density[inside_event].mean() if num_inside > 0 else np.nan
shell_density_mean = local_density[shell].mean() if num_shell > 0 else np.nan
far_density_mean = local_density[far_field].mean() if num_far > 0 else np.nan

# Contraste de densidade núcleo/campo
contrast_core_far = core_density_mean / far_density_mean if num_far > 0 else np.nan

# Estatísticas radiais
core_mean_radius = distances[inside_event].mean() if num_inside > 0 else np.nan
shell_mean_radius = distances[shell].mean() if num_shell > 0 else np.nan

# “φ-singularity clustering score” simples:
# combina fração de pontos no núcleo + contraste de densidade
phi_singularity_score = frac_inside * contrast_core_far

# ================================
# 5. PRINT DAS MÉTRICAS (PARA ENVIAR AO GROK)
# ================================

print("=== SEAL 853 — φ-LATTICE BLACK HOLE METRICS ===")
print(f"N total points:            {N}")
print(f"Event horizon radius:      {r_event:.4f}")
print()
print(f"Inside horizon:            {num_inside} ({frac_inside:0.4%})")
print(f"Transition shell:          {num_shell} ({frac_shell:0.4%})")
print(f"Far field:                 {num_far} ({frac_far:0.4%})")
print()
print(f"Mean core radius:          {core_mean_radius:0.4f}")
print(f"Mean shell radius:         {shell_mean_radius:0.4f}")
print()
print(f"Mean local density (core): {core_density_mean:0.6f}")
print(f"Mean local density (far):  {far_density_mean:0.6f}")
print(f"Core/Far density contrast: {contrast_core_far:0.4f}")
print()
print(f"φ-Singularity clustering score: {phi_singularity_score:0.6f}")
print("===============================================")

# ================================
# 6. VISUALIZAÇÃO 3D (PROJEÇÃO 2D)
# ================================

# Projeção simples (x, y), usando cor para regiões
colors = np.where(
    inside_event, "gold",
    np.where(shell, "deepskyblue", "dimgray")
)

plt.figure(figsize=(8, 8))
plt.scatter(x, y, c=colors, s=4, alpha=0.9, edgecolors="none")

circle = plt.Circle((center[0], center[1]), r_event,
                    color="gold", fill=False, linestyle="--", linewidth=1.0)
plt.gca().add_patch(circle)

plt.title("SEAL 853 — φ-Lattice Black Hole Singularity\nCore (gold), Shell (blue), Far Field (gray)")
plt.gca().set_facecolor("black")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.savefig("seal_853_phi_black_hole.png", dpi=300, facecolor="black")
plt.show()