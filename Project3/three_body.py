import numpy as np
import matplotlib.pyplot as plt
from planet import Planet
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
plt.style.use("ggplot")

class SolarSystem():

    def __init__(self, total_time, dt):
        self.solver = Solver(total_time, dt)
        self.planets = np.array([Planet(self.solver, 1, [0, 0], [0, 0], True), Planet(self.solver, 3e-6, [1, 0], [0, 2*np.pi]), Planet(self.solver, 9.5e-4, [5.2, 0], [0, 2.75])])

    def run(self):
        self.solver.velocity_verlet(self.planets)
        self.animate_plot()

        



class Solver():

    def __init__(self, total_time, dt):
        self.total_time = total_time
        self.dt = dt
        self.time_steps = int(self.total_time / self.dt)

    def forward_euler(self, planets):
        for t in range(self.time_steps - 1):
            for planet in planets:
                if planet.sun != True:
                    planet.positions[t + 1, :] = planet.positions[t, :] + planet.velocities[t, :] * self.dt
                    planet.velocities[t + 1, :] = planet.velocities[t, :] + planet.get_acceleration(planets, t) * self.dt

    def velocity_verlet(self, planets):
        for t in range(self.time_steps - 1):
            for planet in planets:
                if planet.sun != True:
                    planet.positions[t + 1, :] = planet.positions[t, :] + planet.velocities[t, :] * self.dt + 0.5 * planet.accelerations[t, :] * self.dt ** 2
            for planet in planets:
                if planet.sun != True:
                    planet.accelerations[t + 1, :] = planet.get_acceleration(planets, t + 1)
            for planet in planets:
                if planet.sun != True:
                    planet.velocities[t + 1, :] = planet.velocities[t, :] + 0.5 * (planet.accelerations[t, :] + planet.accelerations[t + 1, :]) * self.dt



solarsystem = SolarSystem(10, 1e-3)
solarsystem.run()

plt.scatter(0, 0, c="yellow")
for i in range(len(solarsystem.planets)):
    if i != 0:
        plt.plot(solarsystem.planets[i].positions[:, 0], solarsystem.planets[i].positions[:, 1])

plt.xlabel("x [AU]")
plt.ylabel("y [AU]")
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.legend(["Earth", "Jupiter", "Sun"])
plt.tight_layout()
#plt.savefig("3_body_jupiter.png")
plt.show()