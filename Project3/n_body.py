import numpy as np
import matplotlib.pyplot as plt
from planet import Planet
import time
from mpl_toolkits.mplot3d import Axes3D
plt.style.use("ggplot")

class SolarSystem():

    def __init__(self, total_time, dt):
        self.solver = Solver(total_time, dt)
        self.planets = np.array([Planet(self.solver, 1, [-2.2e-4, 7.27e-3, -7.08e-5], [-7.61e-6*365, 2.5e-6*365, 1.90e-7*365]),
                                 Planet(self.solver, 1.6e-7, [6.74e-2, -4.4e-1, -4.32e-2], [2.21e-2*365, 5.61e-3*365, -1.57e-3*365]),
                                 Planet(self.solver, 2.4e-6, [6.34e-1, 3.55e-1, -3.19e-2], [-9.79e-3*365, 1.77e-2*365, 8.07e-4*365]),
                                 Planet(self.solver, 3e-6, [8.59e-1, 5.09e-1, -9.35e-5], [-8.96e-3*365, 1.48e-2*365, -3.64e-7*365]),
                                 Planet(self.solver, 3.3e-7, [1.39, 3.96e-3, -3.43e-2], [5.6e-4*365, 1.52e-2*365, 3.05e-4*365]),
                                 Planet(self.solver, 9.5e-4, [-2.6, -4.7, 7.75e-2], [6.51e-3*365, -3.29e-3*365, -1.32e-4*365]),
                                 Planet(self.solver, 2.7e-4, [1.6, -9.92, 1.09e-1], [5.2e-3*365, 8.71e-4*365, -2.22e-4*365]),
                                 Planet(self.solver, 4.4e-5, [1.71e1, 1.0e1, -1.85e-1], [-2.01e-3*365, 3.21e-3*365, 3.8e-5]),
                                 Planet(self.solver, 5.18e-5, [2.89e1, -7.69, -5.08e-2], [7.85e-4*365, 3.05e-3*365, -8.09e-5*365]),
                                 Planet(self.solver, 6.5e-9, [1.17e1, -3.16e1, 1.46e-3], [3.01e-3*365, 4.34e-4*365, -9.05e-4*365])])

    def run(self):
        self.solver.velocity_verlet(self.planets)


class Solver():

    def __init__(self, total_time, dt):
        self.total_time = total_time
        self.dt = dt
        self.time_steps = int(self.total_time / self.dt)

    def forward_euler(self, planets):
        for t in range(self.time_steps - 1):
            for planet in planets:
                planet.positions[t + 1, :] = planet.positions[t, :] + planet.velocities[t, :] * self.dt
                planet.velocities[t + 1, :] = planet.velocities[t, :] + planet.get_acceleration(planets, t) * self.dt

    def velocity_verlet(self, planets):
        for t in range(self.time_steps - 1):
            for planet in planets:
                planet.positions[t + 1, :] = planet.positions[t, :] + planet.velocities[t, :] * self.dt + 0.5 * planet.accelerations[t, :] * self.dt ** 2
            for planet in planets:
                planet.accelerations[t + 1, :] = planet.get_acceleration(planets, t + 1)
            for planet in planets:
                planet.velocities[t + 1, :] = planet.velocities[t, :] + 0.5 * (planet.accelerations[t, :] + planet.accelerations[t + 1, :]) * self.dt

start_time = time.time()
solarsystem = SolarSystem(100, 1e-4)
solarsystem.run()
print("timenag: ", time.time()-start_time)
#print(np.arctan2(solarsystem.planets[1].perihelions[-1][1], solarsystem.planets[1].perihelions[-1][0]))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(solarsystem.planets)):
    ax.plot(solarsystem.planets[i].positions[:, 0], solarsystem.planets[i].positions[:, 1], solarsystem.planets[i].positions[:, 2])
ax.set_xlabel("x [AU]")
ax.set_ylabel("y [AU]")
ax.legend(["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"])
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
plt.tight_layout()
plt.show()
