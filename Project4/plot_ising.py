import numpy as np
import matplotlib.pyplot as plt

#results1 = np.load("20x20.npy")
#results1 = np.load("20x20_2.4.npy")/20**2

#energies = results1[0, :30000]
#magnetizations = results1[1, :30000]

#accepted = results1[2]#, 1000:]
#accept2 = results2[2]#, 1000:]

#energies = unordered_1[0]
#magnetizations = unordered_1

results = np.load("WHAT.npy")/40**2

T = np.arange(2, 2.35, 0.05)

fig = plt.figure()

fig.add_subplot(2, 2, 1)
plt.plot(T, results[3])#, "o", markersize=2)
#plt.plot(range(cycles), energies, "o", markersize=3)
#plt.legend(["Energy, random matrix", "Energy, up-matrix"])
plt.xlabel("Temperature")
plt.ylabel("Energy")


fig.add_subplot(2, 2, 2)
plt.plot(T, results[1])#, "o", markersize=2)
#plt.plot(range(cycles), magnetizations[1], "o", markersize=3)
#plt.legend(["Mean magnetization, random matrix", "Energy, up-matrix"])
plt.xlabel("Temperature")
plt.ylabel("Magnetization")


fig.add_subplot(2, 2, 3)
plt.plot(T, results[2])  # , "o", markersize=2)
# plt.plot(range(cycles), magnetizations[1], "o", markersize=3)
# plt.legend(["Mean magnetization, random matrix", "Energy, up-matrix"])
plt.xlabel("Temperature")
plt.ylabel("Heat Capacity")

fig.add_subplot(2, 2, 4)
plt.plot(T, results[3])  # , "o", markersize=2)
# plt.plot(range(cycles), magnetizations[1], "o", markersize=3)
# plt.legend(["Mean magnetization, random matrix", "Energy, up-matrix"])
plt.xlabel("Temperature")
plt.ylabel("Susceptibility")


plt.tight_layout()
plt.show()