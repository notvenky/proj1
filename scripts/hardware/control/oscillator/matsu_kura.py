from config import *
import numpy as np

# Parameters
alpha, beta, gamma, phi, w, x, T = 3, 1, 2, 1, 1, 2, 1
K = 2  # Kuramoto coupling strength
omega = np.random.rand(len(DXL_ID_LIST))  # Intrinsic frequencies for the oscillators

def matsuoka_kuramoto(u, v, theta):
    du = alpha * (beta * v - u) - w * x + T + K * np.sum(np.sin(theta - theta[0]))
    dv = gamma * v - phi * np.heaviside(u, 0.5)
    dtheta = omega + K * np.sum(np.sin(np.repeat(theta[:, np.newaxis], len(theta), axis=1) - theta), axis=1)
    return du, dv, dtheta

def rk4_step(u, v, theta, dt):
    k1_u, k1_v, k1_theta = matsuoka_kuramoto(u, v, theta)
    k2_u, k2_v, k2_theta = matsuoka_kuramoto(u + 0.5 * dt * k1_u, v + 0.5 * dt * k1_v, theta + 0.5 * dt * k1_theta)
    k3_u, k3_v, k3_theta = matsuoka_kuramoto(u + 0.5 * dt * k2_u, v + 0.5 * dt * k2_v, theta + 0.5 * dt * k2_theta)
    k4_u, k4_v, k4_theta = matsuoka_kuramoto(u + dt * k3_u, v + dt * k3_v, theta + dt * k3_theta)

    u_new = u + dt / 6 * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
    v_new = v + dt / 6 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    theta_new = theta + dt / 6 * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)

    return u_new, v_new, theta_new


# Initialize
u = np.zeros(len(DXL_ID_LIST))
v = np.zeros(len(DXL_ID_LIST))
theta = np.zeros(len(DXL_ID_LIST))

dt = 0.001  # time step

while True:
    u, v, theta = rk4_step(u, v, theta, dt)

    # Map theta (phase) values to Dynamixel angles.
    # This is a hypothetical mapping and might need adjustment.
    angles = MEAN_POSITION + 1024 * np.sin(theta)

    for i, dxl_id in enumerate(DXL_ID_LIST):
        angle = int(np.clip(angles[i], *ANGLE_RANGES[dxl_id]))
        packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_POSITION, angle)