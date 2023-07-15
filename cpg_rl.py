# r_dot_dot[i] = a * (a/4 * (mu[i] - r[i]) - r_dot[i])
# r_dot_dot -       current amplitude of the oscillator
# a > 0 -           convergence factor
# mu[i] -           intrinsic amplitude
############################################################
# theta_dot[i] = freq[i] + summation(j) [r[j] * w[i][j] * sin(theta[j] - theta[i] - phi[i][j])]
# theta[i] -        phase pf the oscillator
# freq[i] -         frequency
# w[i][j] -         weights
# phi[i][j] -       phase biases
############################################################
# theta_dot[i] = freq[i]
# phi_dot[i] = psi[i]