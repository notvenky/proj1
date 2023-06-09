from dynamixel_sdk import *

# Control table address
ADDR_PRO_TORQUE_ENABLE = 64                  # Address for enabling the torque
ADDR_PRO_GOAL_POSITION = 116                 # Address for goal position
ADDR_PRO_PRESENT_POSITION = 132              # Address for present position
ADDR_PRO_GOAL_CURRENT = 102                  # Address for goal current
ADDR_PRO_PRESENT_CURRENT = 126               # Address for present current
ADDR_PRO_GOAL_SPEED = 104                    # Address for speed
ADDR_PRO_PRESENT_SPEED = 128
ADDR_PRO_VELOCITY_TRAJECTORY = 136
ADDR_PRO_POSITION_TRAJECTORY = 140

# Define angle ranges for each Dynamixel
ANGLE_RANGES = {
    11: (1024, 3072),
    12: (0, 4095),
    20: (1024, 3072),
    21: (1024, 3072),
    22: (0, 4095)
}


MEAN_POSITION = 2048

dxl_goal_position = [0,4095]
JOYSTICK_THRESHOLD = 0.8
DXL_MOVING_STATUS_THRESHOLD = 20

# Protocol version
PROTOCOL_VERSION = 2.0                        # 2.0 for XM430-W210

# Default setting
DXL_ID_LIST = [11, 12, 20, 21, 22]                 # Dynamixel ID list
BAUDRATE = 1000000                            # Dynamixel communication baudrate
DEVICENAME = '/dev/ttyUSB0'                   # U2D2 USB-to-Serial converter device name

# Initialize PortHandler instance
portHandler = PortHandler(DEVICENAME)
print(DXL_ID_LIST)
# Initialize PacketHandler instance
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


for i in range(len(DXL_ID_LIST)):
    # Enable Dynamixel Torque
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_LIST[i], ADDR_PRO_TORQUE_ENABLE, 1)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("Dynamixel %d has been successfully connected" % DXL_ID_LIST[i])

    # Set initial position
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID_LIST[i], ADDR_PRO_GOAL_POSITION, MEAN_POSITION)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("Dynamixel %d has been successfully set to initial position" % DXL_ID_LIST[i])