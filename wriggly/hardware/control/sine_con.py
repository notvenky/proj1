import random
import time
import keyboard
from config import *
import datetime
#import rec

WRITE_FREQ = 2
write_counter = 0

# video_file_path = "/home/venky/Desktop/wriggly/robot/videos/video.mp4"

# PID constants
KP = 2
KI = 0.1
KD = 0.1

# Open log file
with open('log.txt', 'a') as f:
    f.write('Timestamp,Dynamixel ID,Direction,Speed,Torque,Angle\n')

#rec.record_video(video_file_path)

# PID control loop
try:
    current_positions = {dxl_id: MEAN_POSITION for dxl_id in DXL_ID_LIST}
    errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}
    last_errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}
    while True:
            # for i in range(num_actuators):
            #     if i in [0, 2, 4]:
            #         pass
            #         # # Generate the sine wave for these actuators
            #         phase = -np.pi if i == 0 else 0 if i == 2 else np.pi
            #         goal_positions[i] = 1.57 * np.sin(2*np.pi * freq_even * data.time + phase)
            #         if data.ctrl[i] < goal_positions[i]:
            #             data.ctrl[i] += 0.3925
            #         else:
            #             data.ctrl[i] -= 0.3925
            #     elif i in [1, 3]:
            #         # Generate the sine wave for these actuators
            #         phase = -np.pi/2 if i == 1 else np.pi/2
            #         goal_positions[i] = 3.14 * np.sin(2*np.pi * freq_odd * data.time + phase)

            #         # Move the actuator towards the goal position
            #         # Separate the data distribution updation for even and odd
            #         if data.ctrl[i] < goal_positions[i]:
            #             data.ctrl[i] += 0.3925
            #         else:
            #             data.ctrl[i] -= 0.3925

            #     # Add Gaussian noise
            #     noise = np.random.normal(0, std_dev)
            #     data.ctrl[i] += noise

            
        # # Randomly select the number of Dynamixels to actuate
        # num_actuators = random.choice([1,2,2,2,3,3,3,4,4,4,5,5,5])
        # # Randomly select the Dynamixel IDs to actuate
        # actuator_ids = random.sample(DXL_ID_LIST, num_actuators)
        # # Randomly select the direction of actuation for each selected Dynamixel
        # directions = [random.choice(["clockwise", "anticlockwise"]) for _ in range(num_actuators)]
        # # Randomly select the speed of actuation for each selected Dynamixel
        # speeds = [random.randint(50, 200) for _ in range(num_actuators)]
        # # Randomly select the angle of actuation for each selected Dynamixel
        # angles = [random.randint(ANGLE_RANGES[dxl_id][0], ANGLE_RANGES[dxl_id][1]) for dxl_id in actuator_ids]
        # # Randomly select the torque for each selected Dynamixel
        # torques = [random.randint(10, 100) for _ in range(num_actuators)]

        # Write goal position, torque and speed for selected Dynamixels
        for dxl_id, direction, speed, angle, torque in zip(actuator_ids, directions, speeds, angles, torques):
            # Ensure that the goal position is within the angle range for the Dynamixel
            goal_position = min(max(current_positions[dxl_id] + angle if direction == "clockwise" else current_positions[dxl_id] - angle, ANGLE_RANGES[dxl_id][0]), ANGLE_RANGES[dxl_id][1])

        # Apply PID control to set goal current
        error = goal_position - current_positions[dxl_id]
        errors[dxl_id] += error
        derivative = error - last_errors[dxl_id]
        goal_current = KP * error + KI * errors[dxl_id] + KD * derivative
        last_errors[dxl_id] = error

        # Write goal current, torque and speed
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_CURRENT, int(goal_current))
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_SPEED, speed)
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_POSITION, goal_position)
        dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_CURRENT, torque)

        write_counter += 1

        # Write to log file every WRITE_FREQ iterations
        if write_counter >= WRITE_FREQ:
            # Reset the write counter
            write_counter = 0

            # Write to log file
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            with open('log.txt', 'a') as f:
                f.write('{},{},{},{},{},{}\n'.format(timestamp, [actuator_ids], [directions], speed, torque, angle))

        # Read present position and current for all Dynamixels
        for dxl_id in DXL_ID_LIST:
            dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_POSITION)
            if dxl_comm_result == COMM_SUCCESS:
                current_positions[dxl_id] = dxl_present_position
                dxl_present_current, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_CURRENT)
            if dxl_comm_result == COMM_SUCCESS:
                pass

        # Wait for keyboard interrupt or random time interval
        try:
            keyboard.wait('q', suppress=True)
            break
        except:
            pass
    time.sleep(random.uniform(0.1, 0.5))

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