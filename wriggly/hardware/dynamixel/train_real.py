import argparse
from dxl_client import DynamixelClient
from training_env import RobotEnv
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--motors', required=True, help='Comma-separated list of motor IDs.')
    parser.add_argument('-d', '--device', default='/dev/ttyUSB0', help='The Dynamixel device to connect to.')
    parser.add_argument('-b', '--baud', default=1000000, help='The baudrate to connect with.')
    parsed_args = parser.parse_args()

    motors = [int(motor) for motor in parsed_args.motors.split(',')]
    goal_state = np.array([0.0, 0.0])  # Example goal state.

    with DynamixelClient(motors, parsed_args.device, parsed_args.baud) as dxl_client:
        env = RobotEnv(dxl_client, motors, goal_state)
        
        state = env.reset()
        done = False

        while not done:
            action = np.array([0.1, 0.1])  # Example action: Move to (0.1, 0.1)
            state, reward, done = env.step(action)
            print(f"State: {state}, Reward: {reward}, Done: {done}")
