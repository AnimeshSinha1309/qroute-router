x = [
    ("03:24", "05:53", 154, {'basic': 227, 'stochastic': 156, 'sabre': 196}, 145),
    ("38:31", "3:41:20", 1738, {'basic': 2057, 'stochastic': 1879, 'sabre': 1940}, 1743),
    ("44:25", "1:23:31", 1845, {'basic': 2210, 'stochastic': 2013, 'sabre': 2146}, 1707),
    ("40:11", "2:23:53", 1942, {'basic': 2300, 'stochastic': 2145, 'sabre': 2238}, 2036),
    ("48:21", "1:57:41", 2180, {'basic': 2537, 'stochastic': 2475, 'sabre': 2509}, 2204),
    ("1:02:07", "2:40:24", 2712, {'basic': 3365, 'stochastic': 3051, 'sabre': 3163}, 2854),
    ("1:07:02", "2:46:27", 2922, {'basic': 3602, 'stochastic': 3320, 'sabre': 3461}, 3199),
    ("1:09:50", "1:58:22", 3372, {'basic': 4150, 'stochastic': 3862, 'sabre': 3983}, 3706),
    ("1:21:44", "2:48:13", 3776, {'basic': 5409, 'stochastic': 4372, 'sabre': 4404}, 3997),
    ("2:05:02", "5:52:20", 5712, {'basic': 6622, 'stochastic': 6269, 'sabre': 6499}, 5788),
    ("null", "5:37:36",  7450, {'basic': 9178, 'stochastic': 8345, 'sabre': 8746}, 8063),
]

sizes = [154, 1343, 1405, 1498, 1701, 2100, 2319, 2648, 3089, 4459, 5960]

plotables = {
    'qroute': [data[2] for data in x],
    'pytket': [data[4] for data in x],
    'qiskit_basic': [data[3]['basic'] for data in x],
    'qiskit_stochastic': [data[3]['stochastic'] for data in x],
    'qiskit_sabre': [data[3]['sabre'] for data in x]
}

from matplotlib import pyplot as plt
plt.style.use('ggplot')

for key, value in plotables.items():
    plt.plot(sizes[:len(value)], value, 'x-', label=key)

plt.legend()
plt.title('How our methods scale on the realistic test set')
plt.xlabel('Number of Gates in Logical Circuit')
plt.ylabel('Number of Gates in Physical Circuit')
plt.show()


import os
import qroute

device = qroute.environment.device.IBMqx20TokyoDevice()
large_files = ["rd84_142", "adr4_197", "radd_250", "z4_268", "sym6_145", "misex1_241",
               "rd73_252", "cycle10_2_110", "square_root_7", "sqn_258", "rd84_253"]
for e, file in enumerate(large_files):
    cirq = qroute.environment.circuits.circuit_from_qasm(
        os.path.join("./test/circuit_qasm", file + "_onlyCX.qasm"))
    circuit = qroute.environment.circuits.CircuitRepDQN(cirq, len(device))
    print(file)
    print("Qiskit Routing Distance: ", qroute.visualizers.greedy_schedulers.qiskit_routing(circuit, device))
    print("PyTket Routing Distance: ", qroute.visualizers.greedy_schedulers.tket_routing(circuit, device))

for e, file in enumerate(large_files):
    print(file, end=' \t & ')
    for key, value in plotables.items():
        print(value[e], "&", end=' ')
    print("\\\\")
