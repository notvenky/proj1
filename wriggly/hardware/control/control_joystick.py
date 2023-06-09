import pygame
from config import *

# Initialize Pygame
pygame.init()
controller = pygame.joystick.Joystick(0)
controller.init()

# Clockwise
def move_cw(id, present_position):
    new_position = present_position - 512
    return max(0, new_position)

# Anti-Clockwise
def move_ccw(id, present_position):
    new_position = present_position + 512
    return min(4095, new_position)

# Map controller buttons to motor movements
button_to_motor_movement = {
    4: {'motor': DXL_ID_LIST[0], 'direction': move_cw, 'is_pressed': False},  # UP
    6: {'motor': DXL_ID_LIST[0], 'direction': move_ccw, 'is_pressed': False},  # DOWN
    7: {'motor': DXL_ID_LIST[1], 'direction': move_cw, 'is_pressed': False},  # LEFT
    5: {'motor': DXL_ID_LIST[1], 'direction': move_ccw, 'is_pressed': False},  # RIGHT
    3: {'motor': DXL_ID_LIST[2], 'direction': move_cw, 'is_pressed': False},  # TRIANGLE
    1: {'motor': DXL_ID_LIST[2], 'direction': move_ccw, 'is_pressed': False},  # CIRCLE
    0: {'motor': DXL_ID_LIST[3], 'direction': move_cw, 'is_pressed': False},  # SQUARE
    2: {'motor': DXL_ID_LIST[3], 'direction': move_ccw, 'is_pressed': False},  # X
    8: {'motor': DXL_ID_LIST[4], 'direction': move_cw, 'is_pressed': False},  # L1
    9: {'motor': DXL_ID_LIST[4], 'direction': move_ccw, 'is_pressed': False},  # R1
    10: {'motor': DXL_ID_LIST[0], 'direction': move_cw, 'is_pressed': False},  # L2
    11: {'motor': DXL_ID_LIST[0], 'direction': move_ccw, 'is_pressed': False},  # R2
}


# Event handler for controller button press
def on_press(button):
    if button in button_to_motor_movement:
        button_info = button_to_motor_movement[button]
        if not button_info['is_pressed']:
            button_info['is_pressed'] = True
            id = button_info['motor']
            present_position, _, _ = packetHandler.read4ByteTxRx(portHandler, id, ADDR_PRO_PRESENT_POSITION)
            goal_position = button_info['direction'](id, present_position)
            button_info['goal_position'] = goal_position

            # Set goal speed for the motor
            goal_speed = 1023
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, id, ADDR_PRO_GOAL_SPEED, goal_speed)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))

            # Write goal position for all motors that need to move
            goal_positions = {}
            for key, info in button_to_motor_movement.items():
                if info.get('is_pressed') and info.get('motor') == id:
                    goal_positions[id] = info.get('goal_position')
            for goal_id, goal_position in goal_positions.items():
                dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, goal_id, ADDR_PRO_GOAL_POSITION, goal_position)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler.getRxPacketError(dxl_error))


# Event handler for controller button release
def on_release(button):
    if button in button_to_motor_movement:
        button_info = button_to_motor_movement[button]
        button_info['is_pressed'] = False
        button_info['goal_position'] = None

def is_neutral_position(event):
    if event.type == pygame.JOYAXISMOTION:
        return abs(controller.get_axis(event.axis)) <= JOYSTICK_THRESHOLD
    elif event.type == pygame.JOYHATMOTION:
        return controller.get_hat(event.hat) == (0, 0)
    return False



# Event loop
try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                on_press(event.button)
            elif event.type == pygame.JOYBUTTONUP:
                on_release(event.button)
            elif event.type in (pygame.JOYAXISMOTION, pygame.JOYHATMOTION):
                if is_neutral_position(event):
                    for button in button_to_motor_movement.keys():
                        on_release(button)
                else:
                    axis = controller.get_axis(event.axis)
                    if abs(axis) > JOYSTICK_THRESHOLD:
                        if event.axis == 0:  # X axis
                            if axis > 0:
                                on_press(5)  # RIGHT
                            else:
                                on_press(7)  # LEFT
                        elif event.axis == 1:  # Y axis
                            if axis > 0:
                                on_press(6)  # DOWN
                            else:
                                on_press(4)  # UP
                        elif event.axis == 2:  # L2 trigger
                            if axis > 0:
                                on_press(10)  # L2
                        elif event.axis == 5:  # R2 trigger
                            if axis > 0:
                                on_press(11)  # R2
                    if event.axis == 3:  # Right stick X axis
                        if axis > JOYSTICK_THRESHOLD:
                            on_press(1)  # CIRCLE
                        elif axis < -JOYSTICK_THRESHOLD:
                            on_press(0)  # SQUARE
                    elif event.axis == 4:  # Right stick Y axis
                        if axis > JOYSTICK_THRESHOLD:
                            on_press(2)  # X
                        elif axis < -JOYSTICK_THRESHOLD:
                            on_press(3)  # TRIANGLE
            elif event.type == pygame.JOYHATMOTION:
                hat = controller.get_hat(event.hat)
                if hat != (0, 0):
                    if hat == (1, 0):
                        on_press(5)  # RIGHT
                    elif hat == (-1, 0):
                        on_press(7)  # LEFT
                    elif hat == (0, 1):
                        on_press(4)  # UP
                    elif hat == (0, -1):
                        on_press(6)  # DOWN
                    elif hat == (1, 1):
                        on_press(9)  # R1
                    elif hat == (-1, -1):
                        on_press(8)  # L1
                else:
                    for button in button_to_motor_movement.keys():
                        on_release(button)
            elif event.type == pygame.JOYBALLMOTION:
            # Ignore ball motion events
                pass


except KeyboardInterrupt:
    pass


controller.quit()



# Disable Dynamixel Torque and close port
for i in range(len(DXL_ID_LIST)):
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_LIST[i], ADDR_PRO_TORQUE_ENABLE, 0)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("Dynamixel %d torque has been successfully disabled" % DXL_ID_LIST[i])

portHandler.closePort()

# Quit Pygame
pygame.quit()