import torch
import torch.nn.functional as F
import numpy as np

phase = torch.tensor([1.5739, 5.4386, 0.8792, 5.9354, 6.1567])
freq = torch.tensor([0.1226, 0.5788, 0.1174, 0.8293, 0.4348])
amplitude = torch.tensor([0.8722, 0.8641, 0.6232, 2.6644, 0.7649])
range = torch.tensor([np.pi/2, np.pi, np.pi/2, np.pi, np.pi/2])

print("Phase: ", phase)
print("Frequency: ", freq)
print("Amplitude: ", amplitude)

def inv_softplus(x):
    return torch.log(torch.exp(x) - 1)

transformemd_phase = inv_softplus(phase)
transformemd_freq = inv_softplus(freq)
transformemd_amp = torch.atanh(amplitude / range)

print("Transformemd Phase: ", transformemd_phase)
print("Transformemd Frequency: ", transformemd_freq)
print("Transformemd Amplitude: ", transformemd_amp)