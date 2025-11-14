ϕ-Lattice Coherence Simulation (Classical Quantum-Walk Model)

This repository contains a fully classical simulation of ϕ-lattice coherence dynamics, built to explore logarithmic-scale coherence pockets using aperiodic adjacency, phase/amplitude damping, and L1 off-diagonal metrics.

⸻

Overview

The model uses:
	•	NumPy / SciPy for quantum-walk evolution
	•	Sparse adjacency from k-nearest-neighbors over a ϕ-distributed lattice
	•	Unitary step operator U
	•	Phase damping and amplitude damping noise layers
	•	L1 off-diagonal mass to track coherence decay

The goal is to examine how aperiodic structures redistribute decoherence and reveal stable interference pockets even under classical noise.

⸻

Files Included

phi_lattice_simulation.py

Classical quantum-walk style simulation
with phase + amplitude damping and L1 coherence tracking.

python_generate_coherence_dataset.py

Generates synthetic coherence dataset for analysis and scaling verification.

coherence_dataset.csv

Sample coherence values (L1 off-diagonal mass across timesteps).

coherence_analysis.md

Notes on scaling behavior, aperiodic pocket formation,
and log(N) coherence trends.

README.md

Documentation and model overview.
