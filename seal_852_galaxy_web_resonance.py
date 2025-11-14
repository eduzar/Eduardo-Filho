# seal_852_galaxy_web_resonance.py
# Versão ajustada com: pseudo-Z realista, astropy para 3D verdadeiro, plot otimizado

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u

# === 1. BAIXA DADOS DE GALÁXIAS (SDSS DR16 - 100k com RA, DEC, Z) ===
url = "https://www.sdss.org/dr16/data_access/bulk/?table_name=specobj&format=csv&ra_col=ra&dec_col=dec&z_col=z&limit=100000"
data = requests.get(url).text.splitlines()

galaxies = []
for line in data[1:100001]:
    parts = line.split(',')
    if len(parts) < 3:
        continue
    try:
        ra  = float(parts[0])
        dec = float(parts[1])
        z   = float(parts[2])  # Redshift
        if z > 0:  # Apenas galáxias com redshift válido
            galaxies.append([ra, dec, z])
    except ValueError:
        continue

galaxies = np.array(galaxies)
N = len(galaxies)
print(f"Galáxias carregadas com redshift: {N}")

# === 2. CONVERTE RA/DEC/Z → COORDENADAS GALÁCTICAS 3D (X, Y, Z) ===
coords = SkyCoord(ra=galaxies[:, 0]*u.deg, dec=galaxies[:, 1]*u.deg, distance=(galaxies[:, 2]*3.0857e22)*u.m)  # ~Mpc
x_gal = coords.cartesian.x.value
y_gal = coords.cartesian.y.value
z_gal = coords.cartesian.z.value
galaxies_3d = np.column_stack([x_gal, y_gal, z_gal])

# Normaliza para [0,1]^3
galaxies_norm = (galaxies_3d - galaxies_3d.min(axis=0)) / (galaxies_3d.max(axis=0) - galaxies_3d.min(axis=0) + 1e-12)

# === 3. φ-LATTICE 3D (mesmo volume) ===
phi = (1 + np.sqrt(5)) / 2
indices = np.arange(N)
x_phi = (indices / phi)      % 1.0
y_phi = (indices * phi)      % 1.0
z_phi = (indices * phi**2)   % 1.0
phi_lattice = np.column_stack([x_phi, y_phi, z_phi])

# === 4. RESONÂNCIA φ: DISTÂNCIA MÍNIMA AO LATTICE ===
tree = KDTree(phi_lattice)
dist, _ = tree.query(galaxies_norm, k=1)
resonance_map = dist

# === 5. DETECTA FILAMENTOS CÓSMICOS φ-RESONANTES ===
threshold = np.percentile(resonance_map, 10)
filaments = resonance_map < threshold

# === 6. PLOT 3D DO WEB CÓSMICO φ-RESONANTE ===
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# Fundo: galáxias amostradas
sample = slice(0, N, 15)
ax.scatter(
    galaxies_norm[sample, 0],
    galaxies_norm[sample, 1],
    galaxies_norm[sample, 2],
    c='gray', s=0.2, alpha=0.4, depthshade=False
)

# Filamentos φ-resonantes (dourados)
ax.scatter(
    galaxies_norm[filaments, 0],
    galaxies_norm[filaments, 1],
    galaxies_norm[filaments, 2],
    c='gold', s=8, alpha=0.95, edgecolors='orange', linewidth=0.3, label='φ-Resonant Filaments'
)

ax.set_title(
    "SEAL 852 — Galaxy Web φ-Resonance\n100k SDSS Galaxies (3D) Mapped to φ-Lattice",
    color='white', fontsize=16, pad=20
)
ax.set_xlabel('X (normalized)', color='white')
ax.set_ylabel('Y (normalized)', color='white')
ax.set_zlabel('Z (normalized)', color='white')
ax.tick_params(colors='white')
ax.axis('off')
ax.legend(loc='upper left', face30='black', edgecolor='white', labelcolor='gold')

plt.tight_layout()
plt.savefig("seal_852_galaxy_web.png", dpi=300, facecolor='black', bbox_inches='tight')
plt.show()

# === 7. RESULTADOS ===
print(f"Filamentos φ-resonantes detectados: {np.sum(filaments)} / {N}")
print(f"Resonância média: {resonance_map.mean():.6f}")
print(f"Percentil 10% (threshold): {threshold:.6f}")