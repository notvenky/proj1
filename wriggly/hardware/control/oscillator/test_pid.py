from config import *
# from dynamixel.dxl_client import DynamixelClient, DynamixelReader, DynamixelPosVelCurReader
import pickle
import matplotlib.pyplot as plt

# PID Controller Class
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = 0
        self.integral = 0

    def calculate(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output

# Initialize PID Controllers for each joint
pid_controllers = [PIDController(Kp=0.1, Ki=0.01, Kd=0.01) for _ in DXL_ID_LIST]

joint_position_log = []
joint_velocity_log = []
commanded_positions_log = []
last_joint_positions = None

groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_PRO_GOAL_POSITION, 4)

with open('observations_0.pkl', 'rb') as f:
    joint_readings = pickle.load(f)[1:]

def write_joint_currents(joint_currents):
    """
    Write joint currents to the motors using group sync write.
    """
    groupSyncWrite.clearParam()
    for dxl_id, current in zip(DXL_ID_LIST, joint_currents):
        param_goal_current = [DXL_LOBYTE(DXL_LOWORD(current)), DXL_HIBYTE(DXL_LOWORD(current)), 
                              DXL_LOBYTE(DXL_HIWORD(current)), DXL_HIBYTE(DXL_HIWORD(current))]
        dxl_addparam_result = groupSyncWrite.addParam(dxl_id, param_goal_current)
        if not dxl_addparam_result:
            print("[ID:%03d] groupSyncWrite addparam failed" % dxl_id)
            quit()

    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    groupSyncWrite.clearParam()

def unnormalize_and_clip(joint_angles, obs_mean, obs_std):
    """
    Unnormalize and clip joint angles.
    """
    clipped_std = np.clip(obs_std, 0.01, 1000)
    unnormalized = (joint_angles * clipped_std) + obs_mean
    return np.clip(unnormalized, *zip(*CLIPPING_RANGES))

def convert_to_dxl_position(joint_angles):
    """
    Convert joint angles to Dynamixel positions.
    Assuming joint_angles are in radians and need to be mapped to Dynamixel positions.
    """
    dxl_positions = []
    for angle, joint_range in zip(joint_angles, JOINT_RANGES_SIM):
        angle = angle * joint_range
        position = int((angle + np.pi) / (2 * np.pi) * 4095)
        position = np.clip(position, 0, 4095)
        dxl_positions.append(position)
    return dxl_positions

def write_joint_positions(joint_positions):
    """
    Write joint positions to the motors using group sync write.
    """
    global last_joint_positions
    current_time = time.time()
    commanded_positions_log.append((current_time, joint_positions))

    groupSyncWrite.clearParam()
    for dxl_id, position in zip(DXL_ID_LIST, joint_positions):
        param_goal_position = [DXL_LOBYTE(DXL_LOWORD(position)), DXL_HIBYTE(DXL_LOWORD(position)), 
                               DXL_LOBYTE(DXL_HIWORD(position)), DXL_HIBYTE(DXL_HIWORD(position))]
        dxl_addparam_result = groupSyncWrite.addParam(dxl_id, param_goal_position)
        if not dxl_addparam_result:
            print("[ID:%03d] groupSyncWrite addparam failed" % dxl_id)
            quit()

    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    groupSyncWrite.clearParam()

    actual_positions = read_actual_joint_positions()
    joint_position_log.append((current_time, actual_positions))

    if last_joint_positions is not None:
        time_diff = current_time - last_joint_positions[0]
        if time_diff > 0:
            velocities = [(new - old) / time_diff for new, old in zip(actual_positions, last_joint_positions[1])]
            joint_velocity_log.append((current_time, velocities))

    last_joint_positions = (current_time, actual_positions)

def read_actual_joint_positions():
    """
    Reads the actual joint positions from the Dynamixel motors.
    Returns a list of joint positions.
    """
    actual_positions = []
    for dxl_id in DXL_ID_LIST:
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
        else:
            actual_positions.append(dxl_present_position)

    return actual_positions

# Main control loop
last_time = time.time()
for joint_angles in joint_readings:
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    joint_positions = convert_to_dxl_position(joint_angles)
    actual_positions = read_actual_joint_positions()
    joint_currents = []

    for idx, (desired, actual) in enumerate(zip(joint_positions, actual_positions)):
        error = desired - actual
        current = pid_controllers[idx].calculate(error, dt)
        current = np.clip(current, -MAX_CURRENT, MAX_CURRENT)  # Limit the current
        joint_currents.append(int(current))

    write_joint_currents(joint_currents)
    time.sleep(CONTROL_TIME)

for joint_angles in joint_readings:
    joint_positions = convert_to_dxl_position(joint_angles)
    write_joint_positions(joint_positions)
    time.sleep(0.1)

c_times, c_positions = zip(*commanded_positions_log)
a_times, a_positions = zip(*joint_position_log)

plt.figure()
for i in range(len(DXL_ID_LIST)):
    plt.plot(c_times, [pos[i] for pos in c_positions], label=f'Commanded Joint {i+1}')
    plt.plot(a_times, [pos[i] for pos in a_positions], label=f'Actual Joint {i+1}', linestyle='--')
plt.title('Commanded vs Actual Joint Positions Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.legend()
plt.show()