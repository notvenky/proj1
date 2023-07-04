# import time
# from config import *
# import datetime
# import numpy as np
# import keyboard
# import random

# WRITE_FREQ = 2
# write_counter = 0

# # PID constants
# KP = 2
# KI = 0.1
# KD = 0.1

# # Open log file
# with open('log.txt', 'a') as f:
#     f.write('Timestamp,Dynamixel ID,Direction,Speed,Torque,Angle\n')

# # PID control loop
# try:
#     current_positions = {dxl_id: MEAN_POSITION for dxl_id in DXL_ID_LIST}
#     errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}
#     last_errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}

#     # Goal positions and frequencies
#     goal_positions = {dxl_id: 0 for dxl_id in DXL_ID_LIST}
#     freq_even = 1
#     freq_odd = 1
#     std_dev = 0.02  # Standard deviation of the Gaussian noise

#     while True:
#         for dxl_id in DXL_ID_LIST:
#             if dxl_id in [11, 20, 22]:  # Assuming these are the equivalent of [0, 2, 4] in the simulation
#                 phase = -np.pi if dxl_id == 11 else 0 if dxl_id == 20 else np.pi
#                 goal_positions[dxl_id] = 1024 * np.sin(2*np.pi * freq_even * time.time() + phase) + 2048
#             elif dxl_id in [12, 21]:  # Assuming these are the equivalent of [1, 3] in the simulation
#                 phase = -np.pi/2 if dxl_id == 12 else np.pi/2
#                 goal_positions[dxl_id] = 2048 * np.sin(2*np.pi * freq_odd * time.time() + phase) + 2048

#         num_actuators = 5
#         actuator_ids = (11,12,20,21,22)
#         directions = [random.choice(["clockwise", "anticlockwise"]) for _ in range(num_actuators)]
#         # Randomly select the speed of actuation for each selected Dynamixel
#         speeds = [random.randint(50, 200) for _ in range(num_actuators)]
#         # Randomly select the angle of actuation for each selected Dynamixel
#         angles = [random.randint(ANGLE_RANGES[dxl_id][0], ANGLE_RANGES[dxl_id][1]) for dxl_id in actuator_ids]
#         # Randomly select the torque for each selected Dynamixel
#         torques = [random.randint(10, 100) for _ in range(num_actuators)]
#         print(goal_positions)
#         # Calculate error and apply PID control to set goal position
#         error = goal_positions[dxl_id] - current_positions[dxl_id]
#         errors[dxl_id] += error
#         derivative = error - last_errors[dxl_id]
#         goal_current = KP * error + KI * errors[dxl_id] + KD * derivative
#         #goal_position = current_positions[dxl_id] + KP * error + KI * errors[dxl_id] + KD * derivative
#         goal_position = 1024

#         #TODO: Goal Positions are following the oscillator perfect;y, but pass it
#         # to the dynamixels 
#         # Currently trying to match the code in hardsine
#         # COmnining the two codes can give a good overall sync

#         #for dxl_id in DXL_ID_LIST:
#             # Write goal current, torque and speed
#         dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_CURRENT, int(goal_current))
#         dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_SPEED, speed)
#         dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_POSITION, goal_position)
#         dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_CURRENT, torque)


#         write_counter += 1

#         # Write to log file every WRITE_FREQ iterations
#         if write_counter >= WRITE_FREQ:
#             # Reset the write counter
#             write_counter = 0

#             # Write to log file
#             timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
#             with open('log.txt', 'a') as f:
#                 f.write('{}, {}, {}, {}\n'.format(timestamp, dxl_id, goal_positions, error))

#         # Read present position and current for all Dynamixels
#         for dxl_id in DXL_ID_LIST:
#             dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_POSITION)
#             if dxl_comm_result == COMM_SUCCESS:
#                 current_positions[dxl_id] = dxl_present_position
#                 dxl_present_current, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_CURRENT)
#             if dxl_comm_result == COMM_SUCCESS:
#                 pass

#         # Wait for keyboard interrupt or random time interval
#         try:
#             keyboard.wait('esc', suppress=True)
#             break
#         except:
#             pass
#     time.sleep(random.uniform(0.1, 0.5))

# except KeyboardInterrupt:
#     pass

#     #Disable Dynamixel torque
#     for dxl_id in DXL_ID_LIST:
#         dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_PRO_TORQUE_ENABLE, 0)
#         if dxl_comm_result != COMM_SUCCESS:
#             print(packetHandler.getTxRxResult(dxl_comm_result))
#         elif dxl_error != 0:
#             print(packetHandler.getRxPacketError(dxl_error))
#         else:
#             print("Dynamixel %d torque has been successfully disabled" % DXL_ID_LIST[i])

# #Close port
# portHandler.closePort()




# import time
# from config import *
# import datetime
# import numpy as np
# import keyboard
# import random

# WRITE_FREQ = 2
# write_counter = 0

# # PID constants
# KP = 2
# KI = 0.1
# KD = 0.1

# # Open log file
# with open('log.txt', 'a') as f:
#     f.write('Timestamp,Dynamixel ID,Goal Position,Error\n')

# # PID control loop
# try:
#     current_positions = {dxl_id: MEAN_POSITION for dxl_id in DXL_ID_LIST}
#     errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}
#     last_errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}

#     # Goal positions and frequencies
#     goal_positions = {dxl_id: 0 for dxl_id in DXL_ID_LIST}
#     freq_even = 1
#     freq_odd = 1

#     while True:
#         for dxl_id in DXL_ID_LIST:
#             if dxl_id in [11, 20, 22]:  # Assuming these are the equivalent of [0, 2, 4] in the simulation
#                 phase = -np.pi if dxl_id == 11 else 0 if dxl_id == 20 else np.pi
#                 goal_positions[dxl_id] = 1024 * np.sin(2*np.pi * freq_even * time.time() + phase) + 2048
#             elif dxl_id in [12, 21]:  # Assuming these are the equivalent of [1, 3] in the simulation
#                 phase = -np.pi/2 if dxl_id == 12 else np.pi/2
#                 goal_positions[dxl_id] = 2048 * np.sin(2*np.pi * freq_odd * time.time() + phase) + 2048

#             # Calculate error and apply PID control to set goal position
#             error = goal_positions[dxl_id] - current_positions[dxl_id]
#             errors[dxl_id] += error
#             derivative = error - last_errors[dxl_id]
#             goal_current = KP * error + KI * errors[dxl_id] + KD * derivative
#             last_errors[dxl_id] = error

#             # Write goal current and goal position
#             dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_CURRENT, int(goal_current))
#             dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_POSITION, int(goal_positions[dxl_id]))

#         write_counter += 1

#         # Write to log file every WRITE_FREQ iterations
#         if write_counter >= WRITE_FREQ:
#             # Reset the write counter
#             write_counter = 0

#             # Write to log file
#             timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
#             for dxl_id in DXL_ID_LIST:
#                 with open('log.txt', 'a') as f:
#                     f.write('{}, {}, {}, {}\n'.format(timestamp, dxl_id, goal_positions[dxl_id], errors[dxl_id]))

#         # Read present position and current for all Dynamixels
#         for dxl_id in DXL_ID_LIST:
#             dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_POSITION)
#             if dxl_comm_result == COMM_SUCCESS:
#                 current_positions[dxl_id] = dxl_present_position
#                 dxl_present_current, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_CURRENT)
#             if dxl_comm_result == COMM_SUCCESS:
#                 pass

#         # Wait for keyboard interrupt or random time interval
#         try:
#             keyboard.wait('esc', suppress=True)
#             break
#         except:
#             pass
#     time.sleep(random.uniform(0.1, 0.5))

# except KeyboardInterrupt:
#     pass

#     #Disable Dynamixel torque
#     for dxl_id in DXL_ID_LIST:
#         dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_PRO_TORQUE_ENABLE, 0)
#         if dxl_comm_result != COMM_SUCCESS:
#             print(packetHandler.getTxRxResult(dxl_comm_result))
#         elif dxl_error != 0:
#             print(packetHandler.getRxPacketError(dxl_error))
#         else:
#             print("Dynamixel %d torque has been successfully disabled" % DXL_ID_LIST[i])

# #Close port
# portHandler.closePort()
# --------------------------
import time
import datetime
import numpy as np
import random
import keyboard
from config import *

WRITE_FREQ = 2
write_counter = 0

# PID constants
KP = 2
KI = 0.1
KD = 0.1

# Open log file
with open('log.txt', 'a') as f:
    f.write('Timestamp,Dynamixel ID,Direction,Speed,Torque,Angle\n')

# PID control loop
try:
    current_positions = {dxl_id: MEAN_POSITION for dxl_id in DXL_ID_LIST}
    errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}
    last_errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}

    # Goal positions and frequencies
    goal_positions = {dxl_id: 0 for dxl_id in DXL_ID_LIST}
    freq_even = 1
    freq_odd = 1
    std_dev = 0.02  # Standard deviation of the Gaussian noise

    while True:
        # Generate sinusoidal goal positions for each Dynamixel
        for dxl_id in DXL_ID_LIST:
            if dxl_id in [11, 20, 22]:
                phase = -np.pi if dxl_id == 11 else 0 if dxl_id == 20 else np.pi
                goal_positions[dxl_id] = 1024 * np.sin(2 * np.pi * freq_even * time.time() + phase) + 2048
            elif dxl_id in [12, 21]:
                phase = -np.pi/2 if dxl_id == 12 else np.pi/2
                goal_positions[dxl_id] = 2048 * np.sin(2 * np.pi * freq_odd * time.time() + phase) + 2048

        # Randomly select the number of Dynamixels to actuate
        num_actuators = random.choice([1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
        # Randomly select the Dynamixel IDs to actuate
        actuator_ids = random.sample(DXL_ID_LIST, num_actuators)
        # Randomly select the direction of actuation for each selected Dynamixel
        directions = [random.choice(["clockwise", "anticlockwise"]) for _ in range(num_actuators)]
        # Randomly select the speed of actuation for each selected Dynamixel
        speeds = [random.randint(50, 200) for _ in range(num_actuators)]
        # Randomly select the torque for each selected Dynamixel
        torques = [random.randint(10, 100) for _ in range(num_actuators)]

        # Write goal position, torque, and speed for selected Dynamixels
        for dxl_id, direction, speed, torque in zip(actuator_ids, directions, speeds, torques):
            # Calculate error and apply PID control to set goal position
            error = goal_positions[dxl_id] - current_positions[dxl_id]
            errors[dxl_id] += error
            derivative = error - last_errors[dxl_id]
            goal_current = KP * error + KI * errors[dxl_id] + KD * derivative
            goal_position = current_positions[dxl_id] + KP * error + KI * errors[dxl_id] + KD * derivative

except KeyboardInterrupt:
    pass

    #Disable Dynamixel torque
    for dxl_id in DXL_ID_LIST:
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_PRO_TORQUE_ENABLE, 0)
        if dxl_comm_result != COMM_SUCCESS:
            print(packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print(packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel %d torque has been successfully disabled" % DXL_ID_LIST[i])

#Close port
portHandler.closePort()
