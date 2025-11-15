# seal_852_galaxy_web_resonance.py
# SEAL 852: Galaxy Web φ-Resonance — SDSS 3D Mapping to φ-Lattice
# 100k galáxias com RA, DEC, Z → coordenadas cartesianas → φ-lattice 3D

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u

# ================================
# 1. BAIXA DADOS SDSS (RA, DEC, Z)
# ================================

url = "https://skyserver.sdss.org/dr16/en/tools/search/sql.aspx"
query = """
SELECT TOP 100000 ra, dec, z
FROM SpecObj
WHERE class = 'GALAXY' AND z > 0 AND zWarning = 0
"""
# Para rodar localmente, use um CSV pré-baixado ou ajuste para wget
# Aqui simulamos com dados públicos (ou use o link abaixo)

# --- ALTERNATIVA SIMPLES (use este CSV público) ---
csv_url = "https://raw.githubusercontent.com/astroML/astroML_datasets/main/SDSS_SPECTRO/DR7/sdss_dr7_galaxies.csv"
data = requests.get(csv_url).text.splitlines()

galaxies = []
for line in data[1:]:
    parts = line.split(',')
    if len(parts) >= 5:
        try:
            ra = float(parts[0])
            dec = float(parts[1])
            z = float(parts[2])
            if 0 < z < 2:  # Filtro realista
                galaxies.append([ra, dec, z])
        except:
            continue

galaxies = np.array(galaxies)
N = len(galaxies)
print(f"Galáxias carregadas: {N}")

# ================================
# 2. CONVERTE RA/DEC/Z → 3D CARTESIANO
# ================================

coords = SkyCoord(ra=galaxies[:,0]*u.deg, dec=galaxies[:,1]*u.deg, distance=galaxies[:,2]*u.Mpc, frame='icrs')
x = coords.cartesian.x.value
y = coords.cartesian.y.value
z = coords.cartesian.z.value
galaxies_3d = np.column_stack([x, y, z])

# Normaliza para [0,1]^3
galaxies_norm = (galaxies_3d - galaxies_3d.min(axis=0)) / (galaxies_3d.max(axis=0) - galaxies_3d.min(axis=0) + 1e-12)

# ================================
# 3. φ-LATTICE 3D
# ================================

phi = (1 + np.sqrt(5)) / 2
indices = np.arange(N)
x_phi = (indices / phi) % 1.0
y_phi = (indices * phi) % 1.0
z_phi = (indices * phi**2) % 1.0
phi_lattice = np.column_stack([x_phi, y_phi, z_phi])

# ================================
# 4. RESONÂNCIA φ (distância mínima)
# ================================

tree = KDTree(phi_lattice)
dist, _ = tree.query(galaxies_norm, k=1)
resonance_map = dist

# ================================
# 5. DETECTA FILAMENTOS φ-RESONANTES
# ================================

threshold = np.percentile(resonance_map, 10)
filaments = resonance_map < threshold

# ================================
# 6. PLOT 3D CÓSMICO
# ================================

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# Fundo: galáxias
sample = slice(0, N, 20)
ax.scatter(galaxies_norm[sample, 0], galaxies_norm[sample, 1], galaxies_norm[sample, 2],
           c='gray', s=0.3, alpha=0.5, depthshade=False)

# Filamentos dourados
ax.scatter(galaxies_norm[filaments, 0], galaxies_norm[filaments, 1], galaxies_norm[filaments, 2],
           c='gold', s=10, alpha=0.95, edgecolors='orange', linewidth=0.5, label='φ-Resonant Filaments')

ax.set_title("SEAL 852 — Galaxy Web φ-Resonance\nSDSS 3D → φ-Lattice", color='white', fontsize=16)
ax.axis('off')
ax.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='gold')

plt.tight_layout()
plt.savefig("seal_852_galaxy_web.png", dpi=300, facecolor='black', bbox_inches='tight')
plt.show()

# ================================
# 7. RESULTADOS
# ================================

print(f"Filamentos φ-resonantes: {np.sum(filaments)} / {N}")
print(f"Resonância média: {resonance_map.mean():.6f}")
print(f"Threshold (10%): {threshold:.6f}")