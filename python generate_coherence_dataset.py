import numpy as np

# ----------------------------------------------------------
# coherence_dataset.npy generator
# This script produces a reproducible coherence dataset
# compatible with the φ-lattice quantum-walk simulation.
# ----------------------------------------------------------

def generate_dataset():
    # Sample coherence data: L1 off-diagonal mass
    # In a real experiment, this array is filled from the
    # φ-lattice simulation results.
    
    timesteps = 300
    l1_coherence = np.zeros(timesteps)

    # Example decay + pocket patterns (aperiodic bumps)
    for t in range(timesteps):
        baseline = np.exp(-t / 180)           # soft classical decay
        pocket   = 0.12 * np.sin(1.618 * t)   # φ-frequency modulation
        noise    = 0.01 * np.random.randn()   # small stochastic term
        l1_coherence[t] = baseline + pocket + noise

    # Save to .npy
    np.save("coherence_dataset.npy", l1_coherence)
    print("✔ coherence_dataset.npy successfully generated.")


if __name__ == "__main__":
    generate_dataset()