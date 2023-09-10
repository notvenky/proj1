import numpy as np
import matplotlib.pyplot as plt

class MatsuokaOscillator:
    def __init__(self, num_neurons=10, tau_u=1.0, tau_v=1.0, beta=1.0, u0=1.0, w_ij=1.0, dt=0.01):
        self.num_neurons = num_neurons  # Total number of neurons
        self.tau_u = tau_u  # Time constant for u
        self.tau_v = tau_v  # Time constant for v
        self.beta = beta  # Beta parameter
        self.u0 = u0  # External tonic input
        self.w_ij = w_ij  # Weights connecting neurons
        self.dt = dt  # Time step

        self.u = np.zeros(self.num_neurons)  # Initialize u
        self.v = np.zeros(self.num_neurons)  # Initialize v
        self.y = np.zeros(self.num_neurons)  # Initialize y

    def update(self):
        du = np.zeros(self.num_neurons)
        dv = np.zeros(self.num_neurons)

        for i in range(self.num_neurons):
            total_input = 0
            for j in range(self.num_neurons):
                total_input += self.w_ij * self.y[j]  # Assuming same weights for simplicity
            du[i] = (-self.u[i] - self.beta * self.v[i] + total_input + self.u0) / self.tau_u
            dv[i] = (-self.v[i] + self.y[i]) / self.tau_v

        self.u += du * self.dt
        self.v += dv * self.dt
        self.y = np.maximum(0, self.u)  # Activation function
        return self.y

# Parameters
num_neurons = 10
tau_u = 1.0
tau_v = 1.0
beta = 1.0
u0 = 1.0
w_ij = 1.0
dt = 0.01

# Create the model
model = MatsuokaOscillator(num_neurons, tau_u, tau_v, beta, u0, w_ij, dt)

# Time simulation
T = 1000  # Total time steps
outputs = np.zeros((T, num_neurons))

for t in range(T):
    outputs[t, :] = model.update()

# Plotting the outputs
for i in range(num_neurons):
    plt.plot(outputs[:, i], label=f'Neuron {i+1}')
    plt.legend()
    plt.xlabel('Time Steps')
    plt.ylabel('y (output)')
    plt.title('Matsuoka Half Center Model Outputs')
    plt.show()