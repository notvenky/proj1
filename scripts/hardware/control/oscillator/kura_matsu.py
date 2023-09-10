from config import *

import numpy as np

# Matsuoka-Kuramoto parameters
N = len(DXL_ID_LIST)  # Number of oscillators/motors
alpha = 1.0
beta = 1.0
gamma = 1.0
phi = 1.0
T = 1.0
K = 1.0

# Initial state
u = np.full(N, 0.5)  # Initial conditions for the amplitude dynamics
v = np.full(N, 0.5)
theta = np.random.uniform(0, 2*np.pi, N)  # Random initial phases
omega = np.random.uniform(-1, 1, N)  # Random intrinsic frequencies for oscillators

def sigma(x):
    return 1.0 if x > 0 else 0.0


def RK4_step(y, f, dt):
    k1 = dt * f(y)
    k2 = dt * f(y + 0.5 * k1)
    k3 = dt * f(y + 0.5 * k2)
    k4 = dt * f(y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0


def dynamics(state):
    u, v, theta = state[:N], state[N:2*N], state[2*N:]
    
    du = alpha * (1 - u - beta * v) + K/N * np.sum(np.sin(theta - theta[:, np.newaxis]), axis=1)
    dv = gamma * (1 - u - phi * v)
    dtheta = omega + K/N * np.sum(np.sin(theta - theta[:, np.newaxis]), axis=1)

    return np.concatenate([du, dv, dtheta])


dt = 0.01  # Time step
steps = 1000  # Number of steps

state = np.concatenate([u, v, theta])

for _ in range(steps):
    state = RK4_step(state, dynamics, dt)
    
    # Extract u after integration to determine motor position
    u = state[:N]
    # Normalize u to set motor position within allowable range
    positions = ((u - u.min()) / (u.max() - u.min()) * (dxl_goal_position[1] - dxl_goal_position[0]) + dxl_goal_position[0]).astype(int)
    
    for idx, motor_id in enumerate(DXL_ID_LIST):
        packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_PRO_GOAL_POSITION, positions[idx])