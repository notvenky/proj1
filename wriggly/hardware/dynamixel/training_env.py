import numpy as np
import typing as T
import collections
import os
import time
from dxl_client import DynamixelClient, DynamixelReader, DynamixelPosVelCurReader

class WrigglyReal():
    def __init__(self, motor_ids, goal_position, client=DynamixelClient):
        self.client = client
        self.motor_ids = motor_ids
        self.goal_position = goal_position
        self.reader = DynamixelPosVelCurReader(client, motor_ids)
        self.current_position = None
        self.current_velocity = None
        self.current_current = None
        self.start_time = None

    def reset(self):
        # Initialize the environment state and return the initial observation.
        self.start_time = time.time()
        self._read_dynamixel_state()
        return self.get_observation()

    def step(self, action):
        # Apply the action (e.g., motor control signals) and return the new state, reward, done, and info.
        # Assuming action is a desirable position for the motors
        self.client.write_desired_pos(self.motor_ids, action)
        self._read_dynamixel_state()
        observation = self.get_observation()
        reward = self.get_reward()
        done = self.check_done()
        info = {}  # additional data, keep empty
        return observation, reward, done, info
    
    def check_done(self):
        # Check if the task is done based on some condition.
        # Example: If the robot reaches the goal position or a max episode time is achieved.
        if np.allclose(self.current_position, self.goal_position, atol=0.1):
            return True
        if time.time() - self.start_time > 120:  # assuming max 2 minutes per episode
            return True
        return False

    def get_observation(self):
        # Get and return the current state (e.g., motor positions, velocities, currents, etc.)
        return {
            'position': self.current_position,
            'velocity': self.current_velocity,
            'global': self.current_global_position,
            'time': time.time() - self.start_time,
        }

    def get_reward(self):
        # Negative distance between current and goal, for now
        # TODO: Add more dmc-like reward functions
        distance = np.linalg.norm(self.current_global_position - self.goal_position)
        return -distance
    
    def dmc_reward(self, action):
        # TODO: Implement DMC reward function
        pass


    def _read_dynamixel_state(self):
        # Read the current state from the Dynamixel motors using the reader and update attributes.
        pos, vel, cur = self.reader.read()
        self.current_position = pos
        self.current_velocity = vel
        self.current_current = cur

    def _read_global_position(self):
        # Read the current global position of the robot from the camera.
        # TODO: Using Aruco markers
        self.current_global_position = None
        pass

    def render(self):
        # TODO: Plots & visualization
        pass