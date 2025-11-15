# phi_lattice_100k_benchmark.py
# SEAL 854: φ-Lattice @ N=100K vs Grids/Random

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

def phi_lattice(N):
    phi = (1 + np.sqrt(5)) / 2
    x = (np.arange(N) / phi) % 1
    y = (np.arange(N) * phi) % 1
    return np.column_stack([x, y])

def build_graph(coords, k=6):
    G = nx.Graph()
    for i in range(len(coords)):
        dists = np.sum((coords - coords[i])**2, axis=1)
        dists[i] = np.inf
        nn = np.argpartition(dists, k)[:k]
        for j in nn: G.add_edge(i, j)
    return G

N = 100000
phi_coords = phi_lattice(N)
G_phi = build_graph(phi_coords)

# Grid 316x316
side = int(np.sqrt(N))
grid_coords = np.array([[i/side, j/side] for i in range(side) for j in range(side)])
G_grid = build_graph(grid_coords, k=4)

# Random
rand_coords = np.random.rand(N, 2)
G_rand = build_graph(rand_coords)

# Métricas
def metrics(G):
    L = nx.laplacian_matrix(G).astype(float)
    vals = eigsh(L, k=3, which='SM', return_eigenvectors=False)
    gap = vals[1] - vals[0]
    clustering = nx.average_clustering(G)
    return gap, clustering

gap_phi, clust_phi = metrics(G_phi)
gap_grid, clust_grid = metrics(G_grid)
gap_rand, clust_rand = metrics(G_rand)

print(f"φ: gap={gap_phi:.4f}, clust={clust_phi:.4f}")
print(f"Grid: gap={gap_grid:.4f}, clust={clust_grid:.4f}")
print(f"Random: gap={gap_rand:.4f}, clust={clust_rand:.4f}")