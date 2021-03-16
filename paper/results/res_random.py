from matplotlib import pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

data = [
    {'layers': 8, 'size':30, 'time':"01:38", 'qroute':27, 'cirq': 33, 'qiskit-basic': 46, 'qiskit-stochastic': 24, 'qiskit-sabre': 30, 'pytket': 19},
    {'layers': 11, 'size':30, 'time':"02:26", 'qroute':32, 'cirq': 37, 'qiskit-basic': 56, 'qiskit-stochastic': 28, 'qiskit-sabre': 40, 'pytket': 23},
    {'layers': 8, 'size':30, 'time':"02:45", 'qroute':27, 'cirq': 35, 'qiskit-basic': 36, 'qiskit-stochastic': 22, 'qiskit-sabre': 29, 'pytket': 19},
    {'layers': 10, 'size':30, 'time':"02:38", 'qroute':27, 'cirq': 32, 'qiskit-basic': 51, 'qiskit-stochastic': 25, 'qiskit-sabre': 27, 'pytket': 33},
    {'layers': 9, 'size':30, 'time':"02:48", 'qroute':27, 'cirq': 32, 'qiskit-basic': 51, 'qiskit-stochastic': 27, 'qiskit-sabre': 31, 'pytket': 35},
    {'layers': 9, 'size':30, 'time':"01:12", 'qroute':16, 'cirq': 40, 'qiskit-basic': 45, 'qiskit-stochastic': 24, 'qiskit-sabre': 27, 'pytket': 27},
    {'layers': 8, 'size':30, 'time':"03:42", 'qroute':30, 'cirq': 38, 'qiskit-basic': 39, 'qiskit-stochastic': 27, 'qiskit-sabre': 24, 'pytket': 32},
    {'layers': 9, 'size':30, 'time':"03:15", 'qroute':21, 'cirq': 36, 'qiskit-basic': 50, 'qiskit-stochastic': 31, 'qiskit-sabre': 41, 'pytket': 30},
    {'layers': 10, 'size':30, 'time':"04:00", 'qroute':32, 'cirq': 27, 'qiskit-basic': 51, 'qiskit-stochastic': 26, 'qiskit-sabre': 33, 'pytket': 27},
    {'layers': 9, 'size':30, 'time':"02:44", 'qroute':26, 'cirq': 24, 'qiskit-basic': 49, 'qiskit-stochastic': 26, 'qiskit-sabre': 28, 'pytket': 22},
    {'layers': 18, 'size':50, 'time':"04:37", 'qroute':38, 'cirq': 65, 'qiskit-basic': 76, 'qiskit-stochastic': 46, 'qiskit-sabre': 48, 'pytket': 59},
    {'layers': 17, 'size':50, 'time':"05:52", 'qroute':49, 'cirq': 51, 'qiskit-basic': 96, 'qiskit-stochastic': 47, 'qiskit-sabre': 62, 'pytket': 51},
    {'layers': 16, 'size':50, 'time':"08:03", 'qroute':58, 'cirq': 60, 'qiskit-basic': 86, 'qiskit-stochastic': 49, 'qiskit-sabre': 51, 'pytket': 48},
    {'layers': 16, 'size':50, 'time':"04:07", 'qroute':38, 'cirq': 65, 'qiskit-basic': 74, 'qiskit-stochastic': 38, 'qiskit-sabre': 46, 'pytket': 51},
    {'layers': 11, 'size':50, 'time':"05:52", 'qroute':43, 'cirq': 58, 'qiskit-basic': 75, 'qiskit-stochastic': 32, 'qiskit-sabre': 49, 'pytket': 37},
    {'layers': 13, 'size':50, 'time':"05:30", 'qroute':42, 'cirq': 62, 'qiskit-basic': 85, 'qiskit-stochastic': 40, 'qiskit-sabre': 61, 'pytket': 52},
    {'layers': 16, 'size':50, 'time':"08:08", 'qroute':57, 'cirq': 50, 'qiskit-basic': 72, 'qiskit-stochastic': 36, 'qiskit-sabre': 52, 'pytket': 44},
    {'layers': 13, 'size':50, 'time':"07:13", 'qroute':49, 'cirq': 59, 'qiskit-basic': 89, 'qiskit-stochastic': 37, 'qiskit-sabre': 46, 'pytket': 51},
    {'layers': 12, 'size':50, 'time':"04:50", 'qroute':41, 'cirq': 66, 'qiskit-basic': 74, 'qiskit-stochastic': 36, 'qiskit-sabre': 54, 'pytket': 44},
    {'layers': 16, 'size':50, 'time':"07:37", 'qroute':51, 'cirq': 80, 'qiskit-basic': 94, 'qiskit-stochastic': 42, 'qiskit-sabre': 48, 'pytket': 60},
    {'layers': 16, 'size':70, 'time':"06:08", 'qroute':59, 'cirq': 78, 'qiskit-basic': 117, 'qiskit-stochastic': 55, 'qiskit-sabre': 63, 'pytket': 79},
    {'layers': 19, 'size':70, 'time':"09:01", 'qroute':69, 'cirq': 86, 'qiskit-basic': 123, 'qiskit-stochastic': 60, 'qiskit-sabre': 76, 'pytket': 79},
    {'layers': 18, 'size':70, 'time':"08:28", 'qroute':71, 'cirq': 94, 'qiskit-basic': 126, 'qiskit-stochastic': 52, 'qiskit-sabre': 58, 'pytket': 61},
    {'layers': 23, 'size':70, 'time':"07:59", 'qroute':68, 'cirq': 107, 'qiskit-basic': 114, 'qiskit-stochastic': 59, 'qiskit-sabre': 74, 'pytket': 59},
    {'layers': 21, 'size':70, 'time':"06:28", 'qroute':65, 'cirq': 99, 'qiskit-basic': 96, 'qiskit-stochastic': 58, 'qiskit-sabre': 62, 'pytket': 67},
    {'layers': 18, 'size':70, 'time':"06:38", 'qroute':62, 'cirq': 102, 'qiskit-basic': 131, 'qiskit-stochastic': 56, 'qiskit-sabre': 67, 'pytket': 60},
    {'layers': 23, 'size':70, 'time':"05:19", 'qroute':55, 'cirq': 88, 'qiskit-basic': 109, 'qiskit-stochastic': 65, 'qiskit-sabre': 68, 'pytket': 77},
    {'layers': 19, 'size':70, 'time':"12:1", 'qroute':72, 'cirq': 85, 'qiskit-basic': 124, 'qiskit-stochastic': 56, 'qiskit-sabre': 74, 'pytket': 65},
    {'layers': 18, 'size':70, 'time':"11:25", 'qroute':71, 'cirq': 74, 'qiskit-basic': 106, 'qiskit-stochastic': 47, 'qiskit-sabre': 63, 'pytket': 64},
    {'layers': 23, 'size':70, 'time':"04:56", 'qroute':78, 'cirq': 87, 'qiskit-basic': 131, 'qiskit-stochastic': 65, 'qiskit-sabre': 76, 'pytket': 72},
    {'layers': 28, 'size':90, 'time':"03:20", 'qroute':75, 'cirq': 120, 'qiskit-basic': 122, 'qiskit-stochastic': 81, 'qiskit-sabre': 88, 'pytket': 97},
    {'layers': 28, 'size':90, 'time':"04:08", 'qroute':83, 'cirq': 117, 'qiskit-basic': 132, 'qiskit-stochastic': 72, 'qiskit-sabre': 99, 'pytket': 102},
    {'layers': 25, 'size':90, 'time':"03:36", 'qroute':80, 'cirq': 119, 'qiskit-basic': 126, 'qiskit-stochastic': 66, 'qiskit-sabre': 89, 'pytket': 90},
    {'layers': 26, 'size':90, 'time':"03:21", 'qroute':78, 'cirq': 103, 'qiskit-basic': 149, 'qiskit-stochastic': 73, 'qiskit-sabre': 90, 'pytket': 102},
    {'layers': 22, 'size':90, 'time':"04:34", 'qroute':97, 'cirq': 126, 'qiskit-basic': 126, 'qiskit-stochastic': 67, 'qiskit-sabre': 83, 'pytket': 87},
    {'layers': 21, 'size':90, 'time':"03:22", 'qroute':76, 'cirq': 97, 'qiskit-basic': 151, 'qiskit-stochastic': 60, 'qiskit-sabre': 70, 'pytket': 95},
    {'layers': 26, 'size':90, 'time':"04:30", 'qroute':83, 'cirq': 105, 'qiskit-basic': 136, 'qiskit-stochastic': 69, 'qiskit-sabre': 86, 'pytket': 82},
    {'layers': 29, 'size':90, 'time':"04:03", 'qroute':88, 'cirq': 104, 'qiskit-basic': 135, 'qiskit-stochastic': 73, 'qiskit-sabre': 89, 'pytket': 81},
    {'layers': 25, 'size':90, 'time':"03:28", 'qroute':75, 'cirq': 107, 'qiskit-basic': 125, 'qiskit-stochastic': 70, 'qiskit-sabre': 73, 'pytket': 97},
    {'layers': 26, 'size':90, 'time':"02:58", 'qroute':71, 'cirq': 121, 'qiskit-basic': 113, 'qiskit-stochastic': 79, 'qiskit-sabre': 81, 'pytket': 92},
    {'layers': 32, 'size':110, 'time':"05:19", 'qroute':110, 'cirq': 147, 'qiskit-basic': 166, 'qiskit-stochastic': 93, 'qiskit-sabre': 121, 'pytket': 133},
    {'layers': 30, 'size':110, 'time':"04:24", 'qroute':100, 'cirq': 118, 'qiskit-basic': 162, 'qiskit-stochastic': 77, 'qiskit-sabre': 83, 'pytket': 100},
    {'layers': 31, 'size':110, 'time':"04:04", 'qroute':91, 'cirq': 139, 'qiskit-basic': 165, 'qiskit-stochastic': 88, 'qiskit-sabre': 112, 'pytket': 126},
    {'layers': 32, 'size':110, 'time':"04:49", 'qroute':105, 'cirq': 162, 'qiskit-basic': 164, 'qiskit-stochastic': 82, 'qiskit-sabre': 109, 'pytket': 116},
    {'layers': 30, 'size':110, 'time':"04:54", 'qroute':99, 'cirq': 141, 'qiskit-basic': 176, 'qiskit-stochastic': 82, 'qiskit-sabre': 97, 'pytket': 114},
    {'layers': 31, 'size':110, 'time':"04:49", 'qroute':98, 'cirq': 157, 'qiskit-basic': 164, 'qiskit-stochastic': 77, 'qiskit-sabre': 97, 'pytket': 128},
    {'layers': 32, 'size':110, 'time':"07:59", 'qroute':104, 'cirq': 137, 'qiskit-basic': 166, 'qiskit-stochastic': 98, 'qiskit-sabre': 100, 'pytket': 147},
    {'layers': 35, 'size':110, 'time':"04:45", 'qroute':102, 'cirq': 154, 'qiskit-basic': 186, 'qiskit-stochastic': 87, 'qiskit-sabre': 100, 'pytket': 127},
    {'layers': 29, 'size':110, 'time':"04:18", 'qroute':94, 'cirq': 116, 'qiskit-basic': 169, 'qiskit-stochastic': 80, 'qiskit-sabre': 91, 'pytket': 117},
    {'layers': 32, 'size':110, 'time':"04:53", 'qroute':103, 'cirq': 144, 'qiskit-basic': 173, 'qiskit-stochastic': 79, 'qiskit-sabre': 121, 'pytket': 128},
    {'layers': 34, 'size':130, 'time':"05:52", 'qroute':125, 'cirq': 161, 'qiskit-basic': 197, 'qiskit-stochastic': 91, 'qiskit-sabre': 139, 'pytket': 144},
    {'layers': 33, 'size':130, 'time':"09:56", 'qroute':151, 'cirq': 167, 'qiskit-basic': 209, 'qiskit-stochastic': 96, 'qiskit-sabre': 115, 'pytket': 117},
    {'layers': 34, 'size':130, 'time':"07:25", 'qroute':107, 'cirq': 166, 'qiskit-basic': 223, 'qiskit-stochastic': 92, 'qiskit-sabre': 115, 'pytket': 140},
    {'layers': 43, 'size':130, 'time':"05:49", 'qroute':118, 'cirq': 160, 'qiskit-basic': 200, 'qiskit-stochastic': 108, 'qiskit-sabre': 141, 'pytket': 141},
    {'layers': 36, 'size':130, 'time':"06:37", 'qroute':113, 'cirq': 164, 'qiskit-basic': 220, 'qiskit-stochastic': 103, 'qiskit-sabre': 127, 'pytket': 145},
    {'layers': 31, 'size':130, 'time':"07:28", 'qroute':114, 'cirq': 163, 'qiskit-basic': 209, 'qiskit-stochastic': 111, 'qiskit-sabre': 121, 'pytket': 143},
    {'layers': 36, 'size':130, 'time':"11:16", 'qroute':124, 'cirq': 166, 'qiskit-basic': 223, 'qiskit-stochastic': 109, 'qiskit-sabre': 123, 'pytket': 148},
    {'layers': 41, 'size':130, 'time':"15:05", 'qroute':136, 'cirq': 173, 'qiskit-basic': 237, 'qiskit-stochastic': 108, 'qiskit-sabre': 125, 'pytket': 140},
    {'layers': 38, 'size':130, 'time':"12:04", 'qroute':120, 'cirq': 157, 'qiskit-basic': 231, 'qiskit-stochastic': 104, 'qiskit-sabre': 137, 'pytket': 175},
    {'layers': 34, 'size':130, 'time':"06:28", 'qroute':133, 'cirq': 137, 'qiskit-basic': 195, 'qiskit-stochastic': 96, 'qiskit-sabre': 129, 'pytket': 140},
    {'layers': 41, 'size':150, 'time':"07:51", 'qroute':157, 'cirq': 203, 'qiskit-basic': 230, 'qiskit-stochastic': 109, 'qiskit-sabre': 121, 'pytket': 165},
    {'layers': 48, 'size':150, 'time':"06:30", 'qroute':145, 'cirq': 188, 'qiskit-basic': 243, 'qiskit-stochastic': 141, 'qiskit-sabre': 160, 'pytket': 175},
    {'layers': 43, 'size':150, 'time':"06:57", 'qroute':147, 'cirq': 183, 'qiskit-basic': 248, 'qiskit-stochastic': 118, 'qiskit-sabre': 141, 'pytket': 158},
    {'layers': 39, 'size':150, 'time':"05:49", 'qroute':130, 'cirq': 194, 'qiskit-basic': 254, 'qiskit-stochastic': 108, 'qiskit-sabre': 134, 'pytket': 140},
    {'layers': 44, 'size':150, 'time':"06:26", 'qroute':141, 'cirq': 145, 'qiskit-basic': 259, 'qiskit-stochastic': 116, 'qiskit-sabre': 148, 'pytket': 191},
    {'layers': 46, 'size':150, 'time':"06:50", 'qroute':150, 'cirq': 200, 'qiskit-basic': 211, 'qiskit-stochastic': 131, 'qiskit-sabre': 144, 'pytket': 169},
    {'layers': 40, 'size':150, 'time':"06:24", 'qroute':134, 'cirq': 189, 'qiskit-basic': 245, 'qiskit-stochastic': 122, 'qiskit-sabre': 151, 'pytket': 156},
    {'layers': 48, 'size':150, 'time':"07:15", 'qroute':160, 'cirq': 207, 'qiskit-basic': 260, 'qiskit-stochastic': 118, 'qiskit-sabre': 161, 'pytket': 217},
    {'layers': 45, 'size':150, 'time':"07:19", 'qroute':147, 'cirq': 216, 'qiskit-basic': 257, 'qiskit-stochastic': 119, 'qiskit-sabre': 163, 'pytket': 165},
    {'layers': 44, 'size':150, 'time':"06:37", 'qroute':142, 'cirq': 195, 'qiskit-basic': 223, 'qiskit-stochastic': 113, 'qiskit-sabre': 145, 'pytket': 165},
]

if __name__=="__main__":
    labels = ['qroute', 'cirq', 'qiskit-basic', 'qiskit-stochastic', 'qiskit-sabre', 'pytket']
    for key in labels:
        x, y = [], []
        for item in data:
            x.append(item['size'])
            y.append(item[key] / item['layers'])
        plt.plot(x, y)
    plt.show()
