import random
import time
import datetime
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

    # Create log file
    log_file = open("log.txt", "w")
    log_file.write("Timestamp\tActuator IDs\tDirections\tSpeeds\tAngles\tTorques\tDuration\n")

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

        # Compute PID control signal for each selected Dynamixel
        for dxl_id in actuator_ids:
            current_position = current_positions[dxl_id]
            error = angles[actuator_ids.index(dxl_id)] - current_position
            errors[dxl_id] += error
            delta_error = error - last_errors[dxl_id]
            last_errors[dxl_id] = error
            pid_signal = KP * error + KI * errors[dxl_id] + KD * delta_error

            # Set Dynamixel goal current based on PID control signal
            if pid_signal >= 0:
                direction = 1
            else:
                direction = -1
            goal_current = int(abs(pid_signal))
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_CURRENT, goal_current)
            if dxl_comm_result != COMM_SUCCESS:
                print(packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print(packetHandler.getRxPacketError(dxl_error))
            else:
                print("Dynamixel ID {} goal current has been successfully set to {}".format(dxl_id, goal_current))

        # Actuate selected Dynamixels
        for dxl_id, direction, speed in zip(actuator_ids, directions, speeds):
            if direction == "clockwise":
                goal_speed = speed
            else:
                goal_speed = -speed
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_SPEED, goal_speed)
            if dxl_comm_result != COMM_SUCCESS:
                print(packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print(packetHandler.getRxPacketError(dxl_error))
            else:
                print("Dynamixel ID {} goal speed has been successfully set to {}".format(dxl_id, goal_speed))

        # Wait for actuation to finish
        duration = random.uniform(0.1, 1.0)
        time.sleep(duration)

        # Read current positions and torques of all Dynamixels
        current_positions = {}
        current_torques = {}
        for dxl_id in DXL_ID_LIST:
            dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_POSITION)
            dxl_present_current, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_CURRENT)
            if dxl_comm_result != COMM_SUCCESS:
                print(packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print(packetHandler.getRxPacketError(dxl_error))
            else:
                current_positions[dxl_id] = dxl_present_position
                current_torques[dxl_id] = dxl_present_current

        # Write log entry
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        log_entry = "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(timestamp, actuator_ids, directions, speeds, angles, current_torques, duration)
        log_file.write(log_entry)

except KeyboardInterrupt:
    pass

# Disable Dynamixel Torque and close port
for dxl_id in DXL_ID_LIST:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_PRO_TORQUE_ENABLE, 0)
if dxl_comm_result != COMM_SUCCESS:
    print(packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print(packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel ID {} torque has been successfully disabled".format(dxl_id))

portHandler.closePort()
print("Program stopped by Keyboard Interrupt.")

