import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import re
from config import *

import datetime
current_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Create directories if they do not exist
log_dir = os.path.join('logs', f'run_{current_time_str}')
os.makedirs(log_dir, exist_ok=True)

# Set up logging
log_file = os.path.join(log_dir, 'commands.log')
logging.basicConfig(filename=log_file, level=logging.INFO)

plot_dir = os.path.join(log_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

logging.info("Dynamixel %d has been successfully connected" % DXL_ID_LIST[i])

amplitude_conversion_factor = 2048 / 3.14

paste_string = 'Frequency: tensor([0.6443, 0.2451, 0.6229, 0.4360, 0.4590]), Amplitude: tensor([1.5191, 3.1130, 1.5380, 2.7208, 1.1485]), Phase: tensor([1.1488, 1.5353, 1.1200, 2.9399, 2.0592])'

COMMAND_FREQUENCY = 30
COMMAND_PERIOD = 1.0 / COMMAND_FREQUENCY
TIME_INCREMENT = COMMAND_PERIOD/10

tensor_values = re.findall('tensor\((.*?)\)', paste_string)
frequency = eval(tensor_values[0])
amplitude = [round(a * amplitude_conversion_factor) for a in eval(tensor_values[1])]
phase = eval(tensor_values[2])

frequency[0], frequency[1] = frequency[1], frequency[0]
amplitude[0], amplitude[1] = amplitude[1], amplitude[0]
phase[0], phase[1] = phase[1], phase[0]

keys = [22, 21, 20, 12, 11]


FREQUENCIES = dict(zip(keys, frequency))
AMPLITUDES = dict(zip(keys, amplitude))
PHASES = dict(zip(keys, phase))


# Constants
PI = np.pi

# Function to set velocity for the Dynamixel motors
def set_motor_velocity(dxl_id, velocity):
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_SPEED, velocity)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))

positions_log = []

def oscillate_position(dxl_id, t):
    """
    Oscillates the position of the specified Dynamixel.
    """
    omega = 2 * PI * FREQUENCIES[dxl_id]
    A = AMPLITUDES[dxl_id]
    phi = PHASES[dxl_id]

    position = MEAN_POSITION + A * np.sin(omega * t + phi)
    
    # Write the position
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_POSITION, int(position))
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("Dynamixel %d is oscillating at position: %d" % (dxl_id, position))

    # Log the action
    logging.info(f"Oscillating Dynamixel {dxl_id} to position: {position}")

    # Save the position
    positions_log.append({"time": t, "dxl_id": dxl_id, "position": position})

def is_moving(dxl_id):
    """
    Returns True if the specified Dynamixel is moving.
    """
    dxl_present_moving, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(portHandler, dxl_id, ADDR_MOVING)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    return dxl_present_moving != 0

def check_and_issue_next_command(dxl_id, t):
    """
    If the specified Dynamixel is not moving, issue the next command.
    """
    if not is_moving(dxl_id):
        oscillate_position(dxl_id, t)

present_positions_log = []

# Function to get present position
def get_present_position(dxl_id, t):
    dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_POSITION)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    present_positions_log.append({"time": t, "dxl_id": dxl_id, "position": dxl_present_position})

start_time = time.time()

try:
    next_tick = time.time()
    while True:
        next_tick += COMMAND_PERIOD
        current_time = time.time() - start_time
        # Send position commands
        for dxl_id in DXL_ID_LIST:
            oscillate_position(dxl_id, current_time)
            get_present_position(dxl_id, current_time)
        time.sleep(max(0, next_tick - time.time()))

except KeyboardInterrupt:
    pass

finally:
    time.sleep(0.5)
    for i in range(len(DXL_ID_LIST)):
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_LIST[i], ADDR_PRO_TORQUE_ENABLE, 0)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel %d torque has been successfully disabled" % DXL_ID_LIST[i])


    # Plot and save the graph
    plt.figure(figsize=(10, 6))

    '''Plot 5 in one image, concatenated vertically'''
    fig, axs = plt.subplots(5, 1, figsize=(10, 10))
    for i, dxl_id in enumerate(DXL_ID_LIST):
        # Plot commanded positions
        times = [log['time'] for log in positions_log if log['dxl_id'] == dxl_id]
        positions = [log['position'] for log in positions_log if log['dxl_id'] == dxl_id]
        axs[i].plot(times, positions, label=f'Commanded Dynamixel {dxl_id}')
        
        # Plot present positions
        present_times = [log['time'] for log in present_positions_log if log['dxl_id'] == dxl_id]
        present_positions = [log['position'] for log in present_positions_log if log['dxl_id'] == dxl_id]
        axs[i].plot(present_times, present_positions, label=f'Present Dynamixel {dxl_id}', linestyle='dashed')

        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Position')
        axs[i].legend()
        axs[i].set_title(f'Dynamixel {dxl_id} Positions Over Time')
        axs[i].grid(True)
    plt.savefig(os.path.join(plot_dir, f'dynamixel_positions.png'))
    plt.clf()

    # Close the port
    portHandler.closePort()

    # for dxl_id in DXL_ID_LIST:
    #     times = [log['time'] for log in positions_log if log['dxl_id'] == dxl_id]
    #     positions = [log['position'] for log in positions_log if log['dxl_id'] == dxl_id]
    #     plt.plot(times, positions, label=f'Dynamixel {dxl_id}')

    # plt.xlabel('Time (s)')
    # plt.ylabel('Position')
    # plt.legend()
    # plt.title('Dynamixel Positions Over Time')
    # plt.grid(True)
    # plt.savefig(os.path.join(plot_dir, 'dynamixel_positions.png'))

    # for dxl_id in DXL_ID_LIST:
    #     # Plot commanded positions
    #     times = [log['time'] for log in positions_log if log['dxl_id'] == dxl_id]
    #     positions = [log['position'] for log in positions_log if log['dxl_id'] == dxl_id]
    #     plt.plot(times, positions, label=f'Commanded Dynamixel {dxl_id}')
        
    #     # Plot present positions
    #     present_times = [log['time'] for log in present_positions_log if log['dxl_id'] == dxl_id]
    #     present_positions = [log['position'] for log in present_positions_log if log['dxl_id'] == dxl_id]
    #     plt.plot(present_times, present_positions, label=f'Present Dynamixel {dxl_id}', linestyle='dashed')

    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Position')
    #     plt.legend()
    #     plt.title(f'Dynamixel {dxl_id} Positions Over Time')
    #     plt.grid(True)
    #     plt.savefig(os.path.join(plot_dir, f'dynamixel_{dxl_id}_positions.png'))


    # for dxl_id in DXL_ID_LIST:
    #     """ Plot all 5 dynamixel command and present position separately """
    #     # Plot commanded positions
    #     times = [log['time'] for log in positions_log if log['dxl_id'] == dxl_id]
    #     positions = [log['position'] for log in positions_log if log['dxl_id'] == dxl_id]
    #     plt.plot(times, positions, label=f'Commanded Dynamixel {dxl_id}')
        
    #     # Plot present positions
    #     present_times = [log['time'] for log in present_positions_log if log['dxl_id'] == dxl_id]
    #     present_positions = [log['position'] for log in present_positions_log if log['dxl_id'] == dxl_id]
    #     plt.plot(present_times, present_positions, label=f'Present Dynamixel {dxl_id}', linestyle='dashed')

    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Position')
    #     plt.legend()
    #     plt.title(f'Dynamixel {dxl_id} Positions Over Time')
    #     plt.grid(True)
    #     plt.savefig(os.path.join(plot_dir, f'dynamixel_{dxl_id}_positions.png'))
    #     plt.clf()