import random
import time
import keyboard
from dynamixel_sdk import *

# Dynamixel addresses
ADDR_PRO_TORQUE_ENABLE = 64
ADDR_PRO_GOAL_POSITION = 116
ADDR_PRO_PRESENT_POSITION = 132
ADDR_PRO_GOAL_CURRENT = 102
ADDR_PRO_PRESENT_CURRENT = 126
ADDR_PRO_GOAL_SPEED = 104
ADDR_PRO_PRESENT_SPEED = 128
MEAN_POSITION = 2048

# PID constants
KP = 0.5
KI = 0.1
KD = 0.1

# Dynamixel IDs
DXL_ID_LIST = [11, 12, 20, 21, 22]

# Protocol version
PROTOCOL_VERSION = 2.0

# Default settings
BAUDRATE = 1000000
DEVICENAME = '/dev/ttyUSB0'

# Initialize PortHandler and PacketHandler
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    quit()

# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    quit()

# Enable Dynamixel torque
for dxl_id in DXL_ID_LIST:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_PRO_TORQUE_ENABLE, 1)
    if dxl_comm_result != COMM_SUCCESS:
        print(packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print(packetHandler.getRxPacketError(dxl_error))
    else:
        print("Dynamixel ID {} has been successfully connected".format(dxl_id))

# PID control loop
try:
    current_positions = {dxl_id: MEAN_POSITION for dxl_id in DXL_ID_LIST}
    errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}
    last_errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}
    while True:
        # Randomly select the number of Dynamixels to actuate
        num_actuators = random.choice([1,2,2,2,3,3,3,4,4,4,5,5,5])
        # Randomly select the Dynamixel IDs to actuate
        actuator_ids = random.sample(DXL_ID_LIST, num_actuators)
        # Randomly select the direction of actuation for each selected Dynamixel
        directions = [random.choice(["clockwise", "anticlockwise"]) for _ in range(num_actuators)]
        # Randomly select the speed of actuation for each selected Dynamixel
        speeds = [random.randint(50, 200) for _ in range(num_actuators)]
        # Randomly select the angle of actuation for each selected Dynamixel
        angles = [random.randint(0, 4095) for _ in range(num_actuators)]
        # Randomly select the torque for each selected Dynamixel
        torques = [random.randint(10, 100) for _ in range(num_actuators)]

        # Write goal position, torque and speed for selected Dynamixels
        for dxl_id, direction, speed, angle, torque in zip(actuator_ids, directions, speeds, angles, torques):
            if direction == "clockwise":
                goal_position = current_positions[dxl_id] + angle
            else:
                goal_position = current_positions[dxl_id] - angle

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

        # Read present position and current for all Dynamixels
        for dxl_id in DXL_ID_LIST:
            dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_POSITION)
            if dxl_comm_result == COMM_SUCCESS:
                current_positions[dxl_id] = dxl_present_position
            dxl_present_current, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_CURRENT)
            if dxl_comm_result == COMM_SUCCESS:
                print("Dynamixel ID {} present current: {}".format(dxl_id, dxl_present_current))

        # Wait for keyboard interrupt or random time interval
        try:
            keyboard.wait('esc', suppress=True)
            break
        except:
            pass
        time.sleep(random.uniform(0.1, 0.5))

except KeyboardInterrupt:
    pass

# Disable Dynamixel torque
for dxl_id in DXL_ID_LIST:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_PRO_TORQUE_ENABLE, 0)
    if dxl_comm_result != COMM_SUCCESS:
        print(packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print(packetHandler.getRxPacketError(dxl_error))
    else:
        print("Dynamixel ID {} torque has been successfully disabled".format(dxl_id))

# Close port
portHandler.closePort()