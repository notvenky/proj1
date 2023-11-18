import numpy as np
import time
import re
from config import *


amplitude_conversion_factor = 2048 / 3.14

# paste_string = 'MReward: 4403.559757610102, Frequency: tensor([0.8048, 0.4901, 0.4643, 0.2966, 0.3876]), Amplitude: tensor([0.1598, 2.9545, 1.4238, 0.2111, 1.4200]), Phase: tensor([5.5397, 5.5965, 3.2232, 3.2181, 2.4138])'
# paste_string = 'Reward: 1.1791573039139247, Frequency: tensor([0.0965, 0.8480, 0.0579, 0.4538, 0.2606]), Amplitude: tensor([0.8267, 2.7854, 1.5259, 1.7308, 0.2948]), Phase: tensor([4.6115, 3.6546, 6.1726, 0.2648, 0.8403])'
# paste_string = 'Reward: 3371.8918957098003, Frequency: tensor([0.6077, 0.4636, 0.7526, 0.0612, 0.3925]), Amplitude: tensor([0.5615, 1.5283, 1.2067, 2.2527, 0.8079]), Phase: tensor([3.4202, 5.2558, 3.1959, 4.1053, 4.4613])'
# paste_string = 'MReward: 4403.559757610102, Frequency: tensor([0.1, 0.1, 0.1, 0.1, 0.1]), Amplitude: tensor([0.1598, 2.9545, 1.4238, 0.2111, 1.4200]), Phase: tensor([5.5397, 5.5965, 3.2232, 3.2181, 2.4138])'
'''
Frequency [1.0775284  0.835489   0.72446126 1.0513035  0.49250725]
Amplitude [-0.25858223 -2.8562698   0.5720353  -2.4282863  -0.8014815 ]
Phase [0.34338844 0.6644462  0.6001628  0.3226203  0.9264953 ] 
'''
# paste_string = 'Frequency: tensor([1.0775, 0.8355, 0.7245, 1.0513, 0.4925]), Amplitude: tensor([0.2586, 2.8563,  0.5720, 2.4283, 0.8015]), Phase: tensor([0.3434, 0.6644, 0.6002, 0.3226, 0.9265])'

'''
Time: 2023-09-18 20:09:32.744263, Reward: 937.8601376855665, Frequency: tensor([0.7287, 0.8134, 0.7327, 0.5979, 0.1452]), Amplitude: tensor([0.5174, 1.1551, 1.3948, 1.3612, 0.9059]), Phase: tensor([3.3565, 0.3632, 1.5797, 2.7197, 3.9647])
Time: 2023-09-18 23:58:00.545612, Reward: 985.1769907259068, Frequency: tensor([0.3426, 0.6553, 0.3155, 0.4052, 0.4190]), Amplitude: tensor([0.4440, 2.2106, 1.0004, 0.5180, 0.7033]), Phase: tensor([4.4541, 1.1276, 4.5585, 5.6445, 2.4809])
'''

# Max Reward: 941.0858971157184, Frequency: tensor([0.9500, 0.4136, 0.8930, 0.7858, 0.6079]), Amplitude: tensor([0.8669, 2.2125, 0.3930, 2.6756, 0.8622]), Phase: tensor([1.9867, 5.8283, 4.2842, 6.1343, 2.2573])
# paste_string = 'Frequency: tensor([0.9500, 0.4136, 0.8930, 0.7858, 0.6079]), Amplitude: tensor([0.8669, 2.2125, 0.3930, 2.6756, 0.8622]), Phase: tensor([1.9867, 5.8283, 4.2842, 6.1343, 2.2573])'
'''
# phase = torch.tensor([1.3694, 1.5156, 1.3102, 3.2705, 2.2954])
# freq = torch.tensor([0.7203, 0.1995, 0.6597, 0.5421, 0.5199])
# amplitude = torch.tensor([1.2339, 2.8989, 1.3453, 1.0498, 0.7762])
'''

# paste_string = 'Frequency: tensor([0.7203, 0.1995, 0.6597, 0.5421, 0.5199]), Amplitude: tensor([1.2339, 2.8989, 1.3453, 1.0498, 0.7762]), Phase: tensor([1.3694, 1.5156, 1.3102, 3.2705, 2.2954])'



# paste_string = 'Frequency: tensor([0.5, 0.2, 0.5, 0.2, 0.5]), Amplitude: tensor([1, 2, 1, 2, 1]), Phase: tensor([0, 3.14, 0, 1.57, 1.57])'

'''
Frequency [0.64432    0.24508412 0.62286806 0.4359714  0.45896667]
Amplitude [1.5191154 3.11296   1.537959  2.7207987 1.1485012]
Phase [1.148803  1.5353366 1.120023  2.939859  2.0592115]
'''
paste_string = 'Frequency: tensor([0.6443, 0.2451, 0.6229, 0.4360, 0.4590]), Amplitude: tensor([1.5191, 3.1130, 1.5380, 2.7208, 1.1485]), Phase: tensor([1.1488, 1.5353, 1.1200, 2.9399, 2.0592])'


'''
Frequency [0.6468087  0.26932043 0.67261416 0.40525734 0.45994768]
Amplitude [1.5532279 3.1291852 1.4345068 2.9870205 1.42587  ]
Phase [1.3132219 1.168864  1.1570823 3.5059423 1.9850686]
'''

# paste_string = 'Frequency: tensor([0.6468, 0.2693, 0.6726, 0.4053, 0.4599]), Amplitude: tensor([1.5532, 3.1292, 1.4345, 2.9870, 1.4259]), Phase: tensor([1.3132, 1.1689, 1.1571, 3.5059, 1.9851])'


'''
Frequency [0.46321982 0.3874992  0.40126303 0.7718561  0.4526071 ]
Amplitude [ 1.5707952 -3.141592  -1.5707898  3.1415775 -1.570685 ]
Phase [0.5091987  1.2607336  0.23853761 1.8520186  0.42424145]
'''

# paste_string = 'Frequency: tensor([0.4632, 0.3875, 0.4013, 0.7719, 0.4526]), Amplitude: tensor([1.5708, 3.1416, 1.5708, 3.1416, 1.5707]), Phase: tensor([0.5092, 1.2607, 0.2385, 1.8520, 0.4242])'

'''
        # frequency = torch.tensor([0.7797, 0.8540, 0.7756, 0.8069, 0.7436])
        # amplitude = torch.tensor([0.4550, 1.7164, 1.5673, 2.5919, 0.7335])
        # phase = torch.tensor([0.1001, 2.1576, 3.0394, 1.2417, 3.0698])
'''

# paste_string = 'Frequency: tensor([0.7797, 0.8540, 0.7756, 0.8069, 0.7436]), Amplitude: tensor([0.4550, 1.7164, 1.5673, 2.5919, 0.7335]), Phase: tensor([0.1001, 2.1576, 3.0394, 1.2417, 3.0698])'

'''
Frequency [0.69131047 0.21827517 0.608739   0.5010487  0.5084136 ]
Amplitude [1.3954116 3.0426228 1.4533448 1.9150698 1.1466035]
Phase [1.326706  1.4117506 1.4463348 3.2927694 2.2267342]
'''

# paste_string = 'Frequency: tensor([0.6913, 0.2183, 0.6087, 0.5010, 0.5084]), Amplitude: tensor([1.3954, 3.0426, 1.4533, 1.9151, 1.1466]), Phase: tensor([1.3267, 1.4118, 1.4463, 3.2928, 2.2267])'


'''
Frequency [0.7088309  0.18335757 0.6296178  0.47634    0.5858596 ]
Amplitude [1.5699918 3.1415915 1.4964042 3.120588  1.4250712]
Phase [1.8176943 0.6278016 1.2298567 2.9421692 2.7705061]
'''
# paste_string = 'Frequency: tensor([0.7088, 0.1834, 0.6296, 0.4763, 0.5859]), Amplitude: tensor([1.5700, 3.1416, 1.4964, 3.1206, 1.4251]), Phase: tensor([1.8177, 0.6278, 1.2299, 2.9422, 2.7705])'

'''
Frequency [0.71779186 0.20570067 0.64155173 0.5305246  0.5418944 ]
Amplitude [1.2938102 2.9593925 1.3870306 1.3480613 0.8929089]
Phase [1.321713  1.5314552 1.2659591 3.2502213 2.3125029]
'''
# paste_string = 'Frequency: tensor([0.7178, 0.2057, 0.6416, 0.5305, 0.5419]), Amplitude: tensor([1.2938, 2.9594, 1.3870, 1.3481, 0.8929]), Phase: tensor([1.3217, 1.5315, 1.2660, 3.2502, 2.3125])'


'''
Frequency [0.7167728  0.21945797 0.63716143 0.5033137  0.5226392 ]
Amplitude [1.5353084 3.1414852 1.4991771 2.1380022 0.8210004]
Phase [1.8393664 1.0651913 1.5390184 3.535664  2.947053 ]
'''
# paste_string = 'Frequency: tensor([0.7168, 0.2195, 0.6372, 0.5033, 0.5226]), Amplitude: tensor([1.5353, 3.14, 1.4992, 2.1380, 0.8210]), Phase: tensor([1.8394, 1.0652, 1.5390, 3.5357, 2.9471])'

'''
1.5707833766937256,3.1414740085601807,1.5703097581863403,3.141443967819214,1.5700441598892212,0.5845676064491272,0.20860250294208527,0.5805959105491638,0.2940889298915863,0.4715344309806824,0.982288122177124,0.9325621724128723,1.1868984699249268,5.06200647354126,0.8582013845443726
in order: amplitude, frequency and phase
'''

# paste_string = 



COMMAND_FREQUENCY = 10
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

# frequency[-1], frequency[-2] = frequency[-2], frequency[-1]
# amplitude[-1], amplitude[-2] = amplitude[-2], amplitude[-1]
# phase[-1], phase[-2] = phase[-2], phase[-1]

# keys = [11, 12, 20, 21, 22] # for 4403


FREQUENCIES = dict(zip(keys, frequency))
AMPLITUDES = dict(zip(keys, amplitude))
PHASES = dict(zip(keys, phase))

print('FREQUENCIES =', FREQUENCIES)
print('AMPLITUDES =', AMPLITUDES)
print('PHASES =', PHASES)


# # Define frequencies, amplitudes, and phases for each Dynamixel
# FREQUENCIES = {11: 0.7761, 12: 0.591, 20: 0.9764, 21: 0.5148, 22: 0.6341}
# AMPLITUDES = {11: 877, 12: 985, 20: 802, 21: 1309, 22: 862}
# PHASES = {11: 1.527, 12: 3.419, 20: 5.7299, 21: 3.9914, 22: 1.4527}


# Constants
PI = np.pi

# Function to set velocity for the Dynamixel motors
def set_motor_velocity(dxl_id, velocity):
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_SPEED, velocity)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))

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
    If the specified Dynamixel is not moving, this issues the next command.
    """
    if not is_moving(dxl_id):
        oscillate_position(dxl_id, t)




    # # Write goal speed
    
    # if dxl_comm_result != COMM_SUCCESS:
    #     print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    # elif dxl_error != 0:
    #     print("%s" % packetHandler.getRxPacketError(dxl_error))
    # else:
    #     print("Speed of Dynamixel %d has been changed to: %d" % (dxl_id, speed))

start_time = time.time()

try:
    while True:
        current_time = time.time() - start_time

        # Set velocity for all motors simultaneously
        for dxl_id in DXL_ID_LIST:
            # set to int(330 * position of dxl_id in DXL_ID_LIST/5)
            # velocity = int(330 * (DXL_ID_LIST.index(dxl_id) + 1)/5)
            velocity = int(110)
            set_motor_velocity(dxl_id, velocity)

        # Send position commands
        for dxl_id in DXL_ID_LIST:
            check_and_issue_next_command(dxl_id, current_time)
        time.sleep(COMMAND_PERIOD)

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

    portHandler.closePort()



'''
untested



Frequency [0.59427613 0.16575903 0.50683725 0.3919305  0.41024014]
Amplitude [1.5426279 3.1290264 1.215808  1.7135525 1.46144  ]
Phase [1.432028  1.1149396 1.5218102 3.7017906 2.6132274]

Frequency [0.7261866  0.21808882 0.6629307  0.51908267 0.5394724 ]
Amplitude [1.2946855  2.9582124  1.3893652  1.3500786  0.88106525]
Phase [1.352522  1.5598629 1.2619469 3.2490668 2.3181567]

Frequency [0.71274126 0.21867068 0.64338094 0.5111333  0.5290591 ]
Amplitude [1.4857489  3.1387918  1.5204452  0.92049074 0.66990995]
Phase [1.9859282 1.014937  1.56729   3.6886134 2.852283 ]

Frequency [0.7051032  0.21891251 0.64295584 0.507918   0.52057344]
Amplitude [1.5197158  3.1412315  1.4295807  0.8752577  0.96832186]
Phase [2.0330513 1.1017458 1.6357713 3.6692023 2.8318994]

Frequency [0.7095886  0.21974245 0.641797   0.50014293 0.5065058 ]
Amplitude [1.5117115 3.1413507 1.4635632 1.2663633 0.9451786]
Phase [1.9867971 1.0892963 1.6268532 3.6563911 2.899579 ]

Frequency [0.7128379  0.21921723 0.64327717 0.50292647 0.53410715]
Amplitude [1.5242034 3.1414301 1.4792122 1.7678025 0.889143 ]
Phase [1.9247808 1.0843966 1.5916557 3.6251137 2.99375  ]



Frequency [0.6953954  0.2148201  0.6525503  0.47581226 0.50989026]
Amplitude [1.5680048 3.1415887 1.4867598 3.065154  1.2047837]
Phase [1.7633018 0.8237019 1.1755946 2.8874002 2.9421923]

Frequency [0.6988977  0.18768607 0.6885047  0.47944367 0.5717831 ]
Amplitude [1.5690589 3.1415896 1.4521285 3.0971935 1.3282568]
Phase [1.7346947 0.7129393 1.2257109 2.908001  3.0268404]





'''