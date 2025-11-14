# seal_851_universal_resonance_scan.py
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import requests
from sklearn.decomposition import PCA

# === 1. BAIXA DADOS DO COSMOS (CMB - Planck) ===
url = "https://irsa.ipac.caltech.edu/data/Planck/release_2/ancillary-data/CMB/COM_CMB_IQU-smica_2048_R2.02.fits"
cmb_data = fits.open(requests.get(url, stream=True).raw)[1].data['I_STOKES']

# === 2. PROJETA NO φ-LATTICE ===
N = 100000
phi = (1 + np.sqrt(5)) / 2
indices = np.arange(N)
x = (indices / phi) % 1.0
y = (indices * phi) % 1.0

# Reduz CMB pra 2D
pca = PCA(n_components=2)
cmb_2d = pca.fit_transform(cmb_data.reshape(-1, 3))[:N]

# Mapeia CMB → φ-lattice
distances = np.linalg.norm(cmb_2d[:, None] - np.column_stack([x, y]), axis=2)
resonance_map = np.min(distances, axis=1)

# === 3. DETECTA COHERENCE POCKETS ===
threshold = np.percentile(resonance_map, 5)
pockets = resonance_map < threshold

# === 4. PLOT CÓSMICO ===
plt.figure(figsize=(12,9))
plt.scatter(x, y, c=resonance_map, cmap='plasma', s=1, alpha=0.8)
plt.scatter(x[pockets], y[pockets], c='gold', s=10, label='Cosmic Coherence Pockets')
plt.colorbar(label='Resonance Distance')
plt.title("SEAL 851 — Universal Resonance Scan (URS)\nCMB Mapped to φ-Lattice", color='white')
plt.gca().set_facecolor('black')
plt.legend()
plt.savefig("seal_851_cosmic_scan.png", dpi=300, facecolor='black')
plt.show()

print(f"Coherence Pockets Detectados: {np.sum(pockets)} / {N}")
print(f"Padrão φ encontrado no CMB: {np.corrcoef(resonance_map, x)[0,1]:.4f}")