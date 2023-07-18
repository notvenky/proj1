############################################################
# From CPG - RL
# r_dot_dot[i] = a * (a/4 * (mu[i] - r[i]) - r_dot[i])
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