import time
from dynamixel_sdk import *
from config import *

# Matsuoka Oscillator Parameters
a, b, c, d, k, beta, tau1, tau2, I = 3, 1, 4, 1, 1, 2, 1, 1, 1

# Initialize Oscillator Variables
y1, y2, z1, z2 = 1,2,3,4

# Main Loop
while True:
    # Euler method for integration
    dy1 = (-a * y1 - b * z1 - k * y2 + beta * I) / tau1
    dy2 = (-a * y2 - b * z2 - k * y1 + beta * I) / tau2
    dz1 = (y1 - c - d * z1) / tau1
    dz2 = (y2 - c - d * z2) / tau2
    
    dt = 0.01  # time step
    y1 += dy1 * dt
    y2 += dy2 * dt
    z1 += dz1 * dt
    z2 += dz2 * dt
    
    # Convert y1 and y2 to motor commands
    motor_command1 = int(512 + y1 * 100)  # scale and translate y1
    motor_command2 = int(512 + y2 * 100)  # scale and translate y2
    
    # Send commands to motors
    for dxl_id in DXL_ID_LIST:
        packetHandler.write2ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_POSITION, motor_command1)
        packetHandler.write2ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_POSITION, motor_command2)
    
    time.sleep(dt)
