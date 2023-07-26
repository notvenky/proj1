import re

amplitude_conversion_factor = 2048 / 3.14
paste_string = 'Frequency: tensor([0.4303, 0.4154, 0.4517, 0.3578, 0.2295]), Amplitude: tensor([1.3330, 1.2507, 0.8577, 2.2365, 0.8378]), Phase: tensor([2.1703, 1.8762, 0.6844, 6.2216, 1.6259])'

tensor_values = re.findall('tensor\((.*?)\)', paste_string)


frequency = eval(tensor_values[0])
amplitude = [round(a * amplitude_conversion_factor) for a in eval(tensor_values[1])]
phase = eval(tensor_values[2])

# Swap second last and last values
frequency[-1], frequency[-2] = frequency[-2], frequency[-1]
amplitude[-1], amplitude[-2] = amplitude[-2], amplitude[-1]
phase[-1], phase[-2] = phase[-2], phase[-1]

keys = [11, 12, 20, 21, 22]
FREQUENCIES = dict(zip(keys, frequency))
AMPLITUDES = dict(zip(keys, amplitude))
PHASES = dict(zip(keys, phase))

print('FREQUENCIES =', FREQUENCIES)
print('AMPLITUDES =', AMPLITUDES)
print('PHASES =', PHASES)