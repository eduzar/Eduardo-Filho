# φ-Lattice Coherence Simulation (Classical Quantum-Walk Model)

This repository contains a fully classical simulation of φ-lattice coherence dynamics, built to explore logarithmic-scale coherence pockets using aperiodic adjacency, phase/amplitude damping, and L1 off-diagonal metrics.

## Overview

The model uses:
- NumPy / SciPy for quantum-walk evolution  
- Sparse adjacency from k-nearest-neighbors over a φ-distributed lattice  
- Unitary step operator **U**  
- Phase damping and amplitude damping noise layers  
- L1 off-diagonal mass to track coherence decay  

The goal is to examine how aperiodic structures redistribute decoherence and reveal stable interference pockets without quantum hardware.

## Files

- **phi_lattice_simulation.py** → main simulation  
- **coherence_dataset.csv** → L1 coherence traces  
- **notebook.ipynb** → exploration and visualization  

## Requirements
## Notes
This repository was created for reproducibility and verification by xAI’s Grok research team for joint testing of long-horizon AGI stability under noise.
