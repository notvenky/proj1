############################################################

import hydra
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# From CPG - RL
# r_dot_dot[i] = a * (a/4 * (mu[i] - r[i]) - r_dot[i])


class WrigglyCpg():
    a = 150                 # Convergence factor - can change accordingly
    def __init__(self,
                 r,         # current aplitude of the oscillator
                 mu,        # intrinsic amplitude of the oscillator 
                 freq,      # frequency; can alternate between freq and time period
                 w,         # w[i][j] weights
                 phi        # phi[i][j] phase biases
                 ):
        self.r = r
        self.mu = mu
        self.freq = freq
        self.w = w
        self.phi = phi
        print("Initiating CPG oscillators")

    def oscillate_specfreq():
        pass
        # sample random amplitudes, frequencies, phase biases, and weights in order to see what propels the snake forward the best


# r[i] -               current amplitude of the oscillator
# a > 0 -           convergence factor
# mu[i] -           intrinsic amplitude
############################################################
# theta_dot[i] = freq[i] + summation(j) [r[j] * w[i][j] * sin(theta[j] - theta[i] - phi[i][j])]
# theta[i] -        phase of the oscillator
# freq[i] -         frequency
# w[i][j] -         weights
# phi[i][j] -       phase biases
############################################################
# theta_dot[i] = freq[i]
# phi_dot[i] = psi[i]
############################################################
# Ranges of parameters selected at 100
# mu -              [1, 2]
# freq -            [0, 4.5]
# psi -             [-1.5, 1.5] represents phase bias changes
# a =               150
############################################################
# From Salamander/Lamprey
# 