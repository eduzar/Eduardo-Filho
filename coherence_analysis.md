import numpy as np

data = np.loadtxt("coherence_dataset.csv", delimiter=",")
print("Dataset shape:", data.shape)
import matplotlib.pyplot as plt

plt.plot(data, label="L1 coherence")
plt.xlabel("timesteps")
plt.ylabel("coherence mass")
plt.title("Φ-Lattice L1 Coherence Evolution")
plt.legend()
plt.show()
from scipy.signal import find_peaks
import numpy as np

peaks, _ = find_peaks(data, prominence=0.02)  
print("Pocket indices:", peaks)
print("Approx. log(N) scaling:", np.log(len(data)))
noise_floor = np.mean(np.abs(np.diff(data)))
print("Noise floor:", noise_floor)
from scipy.optimize import curve_fit

def decay(t, a, b):
    return a * np.exp(-t / b)

t = np.arange(len(data))
popt, _ = curve_fit(decay, t, data, p0=[1, 150])
print("Fit parameters:", popt)