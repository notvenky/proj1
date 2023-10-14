import torch
import torch.nn.functional as F
import numpy as np

# phase = torch.tensor([1.3694, 1.5156, 1.3102, 3.2705, 2.2954])
# freq = torch.tensor([0.7203, 0.1995, 0.6597, 0.5421, 0.5199])
# amplitude = torch.tensor([1.2339, 2.8989, 1.3453, 1.0498, 0.7762])
range = torch.tensor([np.pi/2, np.pi, np.pi/2, np.pi, np.pi/2])

'''
Max Reward: 941.0858971157184, Frequency: tensor([0.9500, 0.4136, 0.8930, 0.7858, 0.6079]), Amplitude: tensor([0.8669, 2.2125, 0.3930, 2.6756, 0.8622]), Phase: tensor([1.9867, 5.8283, 4.2842, 6.1343, 2.2573])

'''
phase = torch.tensor([1.9867, 5.8283, 4.2842, 6.1343, 2.2573])
freq = torch.tensor([0.9500, 0.4136, 0.8930, 0.7858, 0.6079])
amplitude = torch.tensor([0.8669, 2.2125, 0.3930, 2.6756, 0.8622])

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