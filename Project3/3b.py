import numpy as np
import matplotlib.pyplot as plt
from planet import Planet
import time
plt.style.use("ggplot")

class SolarSystem():

    def __init__(self, total_time, dt):
        self.solver = Solver(total_time, dt)
        self.planet = Planet(self.solver, 3e-6, [1, 0], [0, 2*np.pi])
        self.sun = Planet(self.solver, 1, [0, 0], [0, 0])
        self.planet.accelerations[0, :] = self.planet.get_acceleration([self.sun], 0)

    def run(self):
        self.solver.velocity_verlet(self.planet, self.sun)
        #self.solver.forward_euler(self.planet, self.sun)
        plt.plot(self.planet.positions[:, 0], self.planet.positions[:, 1])
        plt.xlabel("x [AU]")
        plt.ylabel("y [AU]")
        plt.legend(["Earth"])
        plt.tight_layout()
        plt.savefig("circular_orbit.png")
        plt.show()

    def test_energy_conservation(self):
        kinetic_energies = 0.5*self.planet.mass*(self.planet.velocities[:, 0]**2 + self.planet.velocities[:, 1]**2)
        potential_energies = -4*np.pi**2*self.sun.mass*self.planet.mass/np.sqrt(self.planet.positions[:, 0]**2 + self.planet.positions[:, 1]**2)
        angular_momentum = self.planet.mass*np.cross(self.planet.positions, self.planet.velocities)

        x = np.linspace(0, self.solver.total_time, self.solver.time_steps)
        plt.plot(x, kinetic_energies, x, potential_energies, x, angular_momentum)
        plt.xlabel("Time[Years]")
        plt.ylabel('Energy[$M_\odot AU^2/yr^2$]')
        plt.legend(["Kinetic energy", "Potential energy", "Angular momentum"])
        plt.tight_layout()
        plt.savefig("energy_conservation_elliptic.png")
        plt.show()


class Solver():

    def __init__(self, total_time, dt):
        self.total_time = total_time
        self.dt = dt
        self.time_steps = int(self.total_time / self.dt)

    def forward_euler(self, planet, sun):
        start_time = time.time()
        for t in range(self.time_steps - 1):
            planet.positions[t + 1, :] = planet.positions[t, :] + planet.velocities[t, :] * self.dt
            planet.velocities[t + 1, :] = planet.velocities[t, :] + planet.get_acceleration([sun], t) * self.dt
        print("Euler time: ", start_time - time.time())

    def velocity_verlet(self, planet, sun):
        start_time = time.time()
        for t in range(self.time_steps - 1):
            planet.positions[t + 1, :] = planet.positions[t, :] + planet.velocities[t, :] * self.dt + 0.5 * planet.accelerations[t, :] * self.dt ** 2
            planet.accelerations[t + 1, :] = planet.get_acceleration([sun], t + 1)
            planet.velocities[t + 1, :] = planet.velocities[t, :] + 0.5 * (planet.accelerations[t, :] + planet.accelerations[t + 1, :]) * self.dt
        print("Verlet time: ", start_time - time.time())

solarsystem = SolarSystem(1, 1e-3)
solarsystem.run()