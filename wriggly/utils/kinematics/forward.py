import numpy as np

# Define link lengths and angles
L1 = 1.0
L2 = 1.0
L3 = 1.0
L4 = 1.0
L5 = 1.0

def forward_kinematics(q):
    """
    Calculates the forward kinematics of the snake robot.

    Parameters:
    q (list of floats): Joint angles in radians for each of the 5 degrees of freedom.

    Returns:
    end_effector_pos (numpy array): The 3D position of the end effector of the robot.
    """
    # Define the homogeneous transformation matrices for each joint
    T1 = np.array([[np.cos(q[0]), -np.sin(q[0]), 0, 0],
                   [np.sin(q[0]), np.cos(q[0]), 0, 0],
                   [0, 0, 1, L1],
                   [0, 0, 0, 1]])

    T2 = np.array([[np.cos(q[1]), -np.sin(q[1]), 0, 0],
                   [np.sin(q[1]), np.cos(q[1]), 0, 0],
                   [0, 0, 1, L2],
                   [0, 0, 0, 1]])

    T3 = np.array([[np.cos(q[2]), -np.sin(q[2]), 0, 0],
                   [np.sin(q[2]), np.cos(q[2]), 0, 0],
                   [0, 0, 1, L3],
                   [0, 0, 0, 1]])

    T4 = np.array([[np.cos(q[3]), -np.sin(q[3]), 0, 0],
                   [np.sin(q[3]), np.cos(q[3]), 0, 0],
                   [0, 0, 1, L4],
                   [0, 0, 0, 1]])

    T5 = np.array([[np.cos(q[4]), -np.sin(q[4]), 0, 0],
                   [np.sin(q[4]), np.cos(q[4]), 0, 0],
                   [0, 0, 1, L5],
                   [0, 0, 0, 1]])

    # Calculate the homogeneous transformation matrix from the base to the end effector
    T = np.dot(T1, np.dot(T2, np.dot(T3, np.dot(T4, T5))))

    # Extract the position of the end effector
    end_effector_pos = T[:3, 3]

    return end_effector_pos