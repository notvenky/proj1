import numpy as np
from numpy.linalg import norm
import math

# Define the lengths of the snake robot links
link_lengths = [1.0, 1.0, 1.0, 1.0, 1.0]

# Define the maximum and minimum joint angles for each link
joint_limits = [(0, 180), (0, 180), (0, 180), (0, 180), (0, 180)]

# Define the number of iterations for the inverse kinematics solver
MAX_ITERATIONS = 100
EPSILON = 1e-6


def get_rotation_matrix(axis, angle):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def inverse_kinematics(target_pos, target_orientation):
    """
    Calculate the joint angles for the given target position and orientation.
    """
    target_pos = np.asarray(target_pos)
    target_orientation = np.asarray(target_orientation)
    angles = np.zeros(len(link_lengths))
    link_directions = np.eye(3)
    link_positions = np.zeros((len(link_lengths), 3))
    link_positions[0] = target_pos

    # Start with the last link and work backwards to the first
    for i in range(len(link_lengths)-1, -1, -1):
        # Get the direction vector for the current link
        if i == len(link_lengths)-1:
            link_directions[:, 2] = target_orientation
        else:
            link_directions[:, 2] = link_positions[i+1] - link_positions[i]

        # Get the direction vector for the previous link
        if i == 0:
            link_directions[:, 1] = [0, 0, 1]
        else:
            link_directions[:, 1] = link_positions[i-1] - link_positions[i]

        # Calculate the direction vector for the next link
        link_directions[:, 0] = np.cross(link_directions[:, 1], link_directions[:, 2])

        # Normalize the direction vectors
        link_directions[:, 0] /= norm(link_directions[:, 0])
        link_directions[:, 2] /= norm(link_directions[:, 2])

        # Calculate the target position for the current link
        if i == 0:
            target_position = target_pos
        else:
            target_position = link_positions[i-1] + link_directions[:, 2] * link_lengths[i]

        # Calculate the joint angles for the current link
        x = np.dot(link_directions[:, 0], target_position - link_positions[i])
        y = np.dot(link_directions[:, 1], target_position - link_positions[i])
        z = np.dot(link_directions[:, 1], target_position - link_positions[i])

        if abs(z) < EPSILON:
            joint_angle = 0
        else:
            joint_angle = math.atan2(y, x)

        # Clamp the joint angle to the joint limits
        joint_angle = max(joint_angle, math.radians(joint_limits[i][0]))
        joint_angle = min(joint_angle, math.radians(joint_limits[i][1]))

        # Update the joint angle
        angles[i] = joint_angle

        # Rotate the current link to the correct orientation
        rotation_axis = np.cross(link_directions[:, 0], target_orientation)
        if norm(rotation_axis) < EPSILON:
            continue
        rotation_angle = math.asin(norm(rotation_axis))
        rotation_axis /= norm(rotation_axis)
        rotation_matrix = get_rotation_matrix(rotation_axis, rotation_angle)
        link_directions = np.dot(link_directions, rotation_matrix)

        # Update the position of the current link
        link_positions[i] = target_position - link_directions[:, 2] * link_lengths[i]

    return angles
