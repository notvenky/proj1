import torch
import torch.nn.functional as F
import numpy as np

phase = torch.tensor([4.0244, 0.0401, 2.4649, 1.6699, 3.4320])
freq = torch.tensor([0.3377, 0.7400, 0.3648, 0.7470, 0.1819])
amplitude = torch.tensor([1.1290, 2.1344, 1.5143, 2.7496, 0.7820])
range = torch.tensor([np.pi/2, np.pi, np.pi/2, np.pi, np.pi/2])

print("Phase: ", phase)
print("Frequency: ", freq)
print("Amplitude: ", amplitude)

def inv_softplus(x):
    return torch.log(torch.exp(x) - 1)

transformed_phase = inv_softplus(phase)
transformed_freq = inv_softplus(freq)
transformed_amp = torch.atanh(amplitude / range)

print("Transformemd Phase: ", transformed_phase)
print("Transformemd Frequency: ", transformed_freq)
print("Transformemd Amplitude: ", transformed_amp)