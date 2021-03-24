import collections
from matplotlib import pyplot as plt
plt.style.use('ggplot')


x = [
    {'qroute': 18, 'cirq': 24, 'basic': 19, 'stochastic': 20, 'sabre': 21, 'pytket': 19},
    {'qroute': 33, 'cirq': 43, 'basic': 36, 'stochastic': 35, 'sabre': 32, 'pytket': 32},
    {'qroute': 39, 'cirq': 38, 'basic': 34, 'stochastic': 35, 'sabre': 36, 'pytket': 28},
    {'qroute': 83, 'cirq': 99, 'basic': 101, 'stochastic': 90, 'sabre': 90, 'pytket': 89},
    {'qroute': 100, 'cirq': 129, 'basic': 120, 'stochastic': 111, 'sabre': 128, 'pytket': 126},
    {'qroute': 25, 'cirq': 26, 'basic': 27, 'stochastic': 23, 'sabre': 26, 'pytket': 22},
    {'qroute': 30, 'cirq': 34, 'basic': 37, 'stochastic': 37, 'sabre': 38, 'pytket': 35},
    {'qroute': 19, 'cirq': 24, 'basic': 25, 'stochastic': 25, 'sabre': 26, 'pytket': 24},
    {'qroute': 94, 'cirq': 133, 'basic': 122, 'stochastic': 113, 'sabre': 106, 'pytket': 101},
    {'qroute': 6, 'cirq': 9, 'basic': 10, 'stochastic': 7, 'sabre': 7, 'pytket': 6},
    {'qroute': 47, 'cirq': 61, 'basic': 55, 'stochastic': 53, 'sabre': 51, 'pytket': 51},
    {'qroute': 49, 'cirq': 53, 'basic': 58, 'stochastic': 30, 'sabre': 49, 'pytket': 25},
    {'qroute': 15, 'cirq': 16, 'basic': 15, 'stochastic': 11, 'sabre': 12, 'pytket': 12},
    {'qroute': 48, 'cirq': 54, 'basic': 46, 'stochastic': 48, 'sabre': 45, 'pytket': 47},
    {'qroute': 25, 'cirq': 29, 'basic': 29, 'stochastic': 24, 'sabre': 27, 'pytket': 25},
    {'qroute': 63, 'cirq': 85, 'basic': 108, 'stochastic': 57, 'sabre': 66, 'pytket': 17},
    {'qroute': 16, 'cirq': 31, 'basic': 24, 'stochastic': 23, 'sabre': 28, 'pytket': 13},
    {'qroute': 18, 'cirq': 24, 'basic': 27, 'stochastic': 24, 'sabre': 24, 'pytket': 22},
    {'qroute': 11, 'cirq': 13, 'basic': 14, 'stochastic': 13, 'sabre': 14, 'pytket': 13},
    {'qroute': 104, 'cirq': 122, 'basic': 104, 'stochastic': 103, 'sabre': 102, 'pytket': 83},
    {'qroute': 13, 'cirq': 17, 'basic': 16, 'stochastic': 16, 'sabre': 15, 'pytket': 17},
    {'qroute': 136, 'cirq': 169, 'basic': 160, 'stochastic': 140, 'sabre': 147, 'pytket': 120},
    {'qroute': 23, 'cirq': 24, 'basic': 24, 'stochastic': 23, 'sabre': 25, 'pytket': 25},
    {'qroute': 34, 'cirq': 42, 'basic': 40, 'stochastic': 43, 'sabre': 39, 'pytket': 36},
    {'qroute': 94, 'cirq': 124, 'basic': 107, 'stochastic': 102, 'sabre': 103, 'pytket': 90},
    {'qroute': 76, 'cirq': 89, 'basic': 72, 'stochastic': 72, 'sabre': 70, 'pytket': 39},
    {'qroute': 17, 'cirq': 15, 'basic': 17, 'stochastic': 14, 'sabre': 14, 'pytket': 13},
    {'qroute': 23, 'cirq': 31, 'basic': 25, 'stochastic': 23, 'sabre': 24, 'pytket': 22},
    {'qroute': 122, 'cirq': 154, 'basic': 161, 'stochastic': 151, 'sabre': 143, 'pytket': 122},
    {'qroute': 23, 'cirq': 38, 'basic': 33, 'stochastic': 32, 'sabre': 34, 'pytket': 36},
    {'qroute': 80, 'cirq': 107, 'basic': 95, 'stochastic': 96, 'sabre': 95, 'pytket': 74},
    {'qroute': 47, 'cirq': 59, 'basic': 60, 'stochastic': 57, 'sabre': 56, 'pytket': 54},
    {'qroute': 46, 'cirq': 58, 'basic': 50, 'stochastic': 45, 'sabre': 49, 'pytket': 45},
    {'qroute': 43, 'cirq': 52, 'basic': 53, 'stochastic': 49, 'sabre': 48, 'pytket': 46},
    {'qroute': 69, 'cirq': 101, 'basic': 84, 'stochastic': 73, 'sabre': 75, 'pytket': 66},
    {'qroute': 84, 'cirq': 119, 'basic': 109, 'stochastic': 105, 'sabre': 101, 'pytket': 100},
    {'qroute': 45, 'cirq': 59, 'basic': 59, 'stochastic': 59, 'sabre': 54, 'pytket': 52},
    {'qroute': 88, 'cirq': 118, 'basic': 106, 'stochastic': 93, 'sabre': 107, 'pytket': 75},
    {'qroute': 108, 'cirq': 144, 'basic': 141, 'stochastic': 128, 'sabre': 138, 'pytket': 118},
    {'qroute': 88, 'cirq': 159, 'basic': 146, 'stochastic': 88, 'sabre': 121, 'pytket': 105},
    {'qroute': 60, 'cirq': 60, 'basic': 59, 'stochastic': 56, 'sabre': 50, 'pytket': 51},
    {'qroute': 21, 'cirq': 28, 'basic': 28, 'stochastic': 26, 'sabre': 27, 'pytket': 22},
    {'qroute': 20, 'cirq': 26, 'basic': 25, 'stochastic': 26, 'sabre': 23, 'pytket': 24},
    {'qroute': 88, 'cirq': 114, 'basic': 104, 'stochastic': 85, 'sabre': 90, 'pytket': 98},
    {'qroute': 93, 'cirq': 102, 'basic': 95, 'stochastic': 95, 'sabre': 88, 'pytket': 67},
    {'qroute': 73, 'cirq': 89, 'basic': 78, 'stochastic': 77, 'sabre': 75, 'pytket': 49},
    {'qroute': 44, 'cirq': 54, 'basic': 47, 'stochastic': 50, 'sabre': 56, 'pytket': 44},
    {'qroute': 17, 'cirq': 17, 'basic': 16, 'stochastic': 16, 'sabre': 18, 'pytket': 11},
    {'qroute': 17, 'cirq': 25, 'basic': 24, 'stochastic': 25, 'sabre': 27, 'pytket': 23},
    {'qroute': 26, 'cirq': 25, 'basic': 23, 'stochastic': 24, 'sabre': 26, 'pytket': 18},
    {'qroute': 90, 'cirq': 119, 'basic': 108, 'stochastic': 78, 'sabre': 82, 'pytket': 76},
    {'qroute': 33, 'cirq': 58, 'basic': 48, 'stochastic': 43, 'sabre': 49, 'pytket': 29},
    {'qroute': 64, 'cirq': 90, 'basic': 75, 'stochastic': 72, 'sabre': 76, 'pytket': 61},
    {'qroute': 21, 'cirq': 25, 'basic': 23, 'stochastic': 25, 'sabre': 25, 'pytket': 18},
    {'qroute': 36, 'cirq': 48, 'basic': 43, 'stochastic': 44, 'sabre': 45, 'pytket': 40},
    {'qroute': 54, 'cirq': 82, 'basic': 64, 'stochastic': 64, 'sabre': 61, 'pytket': 69},
    {'qroute': 19, 'cirq': 26, 'basic': 18, 'stochastic': 20, 'sabre': 18, 'pytket': 28},
    {'qroute': 6, 'cirq': 7, 'basic': 5, 'stochastic': 6, 'sabre': 5, 'pytket': 5},
    {'qroute': 6, 'cirq': 9, 'basic': 10, 'stochastic': 7, 'sabre': 9, 'pytket': 6},
    {'qroute': 47, 'cirq': 50, 'basic': 48, 'stochastic': 44, 'sabre': 49, 'pytket': 46},
    {'qroute': 50, 'cirq': 64, 'basic': 62, 'stochastic': 57, 'sabre': 60, 'pytket': 51},
    {'qroute': 19, 'cirq': 29, 'basic': 24, 'stochastic': 24, 'sabre': 24, 'pytket': 24},
    {'qroute': 27, 'cirq': 31, 'basic': 28, 'stochastic': 29, 'sabre': 26, 'pytket': 25},
    {'qroute': 75, 'cirq': 99, 'basic': 88, 'stochastic': 89, 'sabre': 91, 'pytket': 83},
    {'qroute': 44, 'cirq': 51, 'basic': 53, 'stochastic': 45, 'sabre': 50, 'pytket': 44},
    {'qroute': 74, 'cirq': 110, 'basic': 77, 'stochastic': 79, 'sabre': 83, 'pytket': 80},
]

d = collections.defaultdict(int)

for item in x:
    for key, value in item.items():
        d[key] += value

keys = sorted(d.keys(), key=d.get)
values = [d[key] for key in keys]

plt.figure(figsize=(15, 10))
plt.xlabel('Quantum Compiler')
plt.ylabel('Total Number of Gates in Output')
plt.title('Results on the Small Circuits (<100 gates) benchmark')
plt.bar(keys, values, color=['red', 'blue', 'green', 'orange', 'black', 'purple'])
plt.show()
