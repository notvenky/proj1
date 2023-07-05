import cv2
import os
import json
import random
import time
import keyboard
from config import *
import datetime
import numpy as np
import subprocess

# Check if the media directory exists, if not, create it
if not os.path.exists('media'):
    os.makedirs('media')

# Check if the count file exists, if not, create it and set the count to 0
if not os.path.exists('count.json'):
    with open('count.json', 'w') as f:
        json.dump({'count': 0}, f)

# Load the count from the file
with open('count.json', 'r') as f:
    data = json.load(f)
    count = data['count']

# cap = cv2.VideoCapture(4) # for my laptop

cap1 = cv2.VideoCapture(0)  # for the NUC
cap2 = cv2.VideoCapture(2)  # for the NUC, second

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'media/video_{count}.avi', fourcc, 20.0, (1280, 480))

WRITE_FREQ = 2
write_counter = 0

# PID constants
KP = 2
KI = 0.1
KD = 0.1

# Open log file
with open('log.txt', 'a') as f:
    f.write('Timestamp,Dynamixel ID,Direction,Speed,Torque,Angle\n')

try:
    current_positions = {dxl_id: MEAN_POSITION for dxl_id in DXL_ID_LIST}
    errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}
    last_errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}
    while True:
        # inside the while True: loop
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1 and ret2:
            # Concatenate both frames horizontally
            frame = np.concatenate((frame1, frame2), axis=1)
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Randomly select the number of Dynamixels to actuate
        num_actuators = random.choice([1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
        # Randomly select the Dynamixel IDs to actuate
        actuator_ids = random.sample(DXL_ID_LIST, num_actuators)
        # Randomly select the direction of actuation for each selected Dynamixel
        directions = [random.choice(["clockwise", "anticlockwise"]) for _ in range(num_actuators)]
        # Randomly select the speed of actuation for each selected Dynamixel
        speeds = [random.randint(50, 200) for _ in range(num_actuators)]
        # Randomly select the angle of actuation for each selected Dynamixel
        angles = [random.randint(ANGLE_RANGES[dxl_id][0], ANGLE_RANGES[dxl_id][1]) for dxl_id in actuator_ids]
        # Randomly select the torque for each selected Dynamixel
        torques = [random.randint(10, 100) for _ in range(num_actuators)]

        # Write goal position, torque and speed for selected Dynamixels
        for dxl_id, direction, speed, angle, torque in zip(actuator_ids, directions, speeds, angles, torques):
            # Ensure that the goal position is within the angle range for the Dynamixel
            goal_position = min(
                max(current_positions[dxl_id] + angle if direction == "clockwise" else current_positions[
                                                                                             dxl_id] - angle,
                    ANGLE_RANGES[dxl_id][0]), ANGLE_RANGES[dxl_id][1])

        # Apply PID control to set goal current
        error = goal_position - current_positions[dxl_id]
        errors[dxl_id] += error
        derivative = error - last_errors[dxl_id]
        goal_current = KP * error + KI * errors[dxl_id] + KD * derivative
        last_errors[dxl_id] = error

        # Write goal current, torque and speed
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_CURRENT,
                                                                  int(goal_current))
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_SPEED, speed)
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_POSITION,
                                                                  goal_position)
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
            dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id,
                                                                                           ADDR_PRO_PRESENT_POSITION)
            if dxl_comm_result == COMM_SUCCESS:
                current_positions[dxl_id] = dxl_present_position
                dxl_present_current, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id,
                                                                                               ADDR_PRO_PRESENT_CURRENT)
            if dxl_comm_result == COMM_SUCCESS:
                pass
        # Wait for keyboard interrupt or random time interval
        try:
            keyboard.wait('esc', suppress=True)
            break
        except:
            pass
        time.sleep(random.uniform(0.1, 0.5))

except KeyboardInterrupt:
    # Save the video immediately on keyboard interrupt
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()

    # Increment the count and write it back to the file
    count += 1
    with open('count.json', 'w') as f:
        json.dump({'count': count}, f)

    # Compress the video using ffmpeg
    time.sleep(1)  # Add a delay to allow the video file to be saved
    input_video = f'media/video_{count-1}.avi'
    output_video = f'media/compressed_video_{count-1}.mp4'

    if os.path.exists(input_video):
        ffmpeg_cmd = f'ffmpeg -i {input_video} -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k {output_video}'
        subprocess.run(ffmpeg_cmd, shell=True)

        # Remove the original uncompressed video
        os.remove(input_video)
    else:
        print(f"File '{input_video}' does not exist.")
    
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