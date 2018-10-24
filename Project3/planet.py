import numpy as np
class Planet():

    def __init__(self, solver, m, init_pos, init_vel, sun=False):
        self.mass = m
        self.sun = sun
        self.positions = np.zeros((solver.time_steps, 3))
        self.velocities = np.zeros((solver.time_steps, 3))
        self.accelerations = np.zeros((solver.time_steps, 3))
        self.positions[0, :] = init_pos
        self.velocities[0, :] = init_vel
        self.perihelions = []
        self.going_down = False


    def get_acceleration(self, planets, t):
        #skalle
        if np.linalg.norm(self.positions[t, :]) < np.linalg.norm(self.positions[t-1, :]):
            self.going_down = True
        else:
            if self.going_down == True:
                self.perihelions.append(self.positions[t])
            self.going_down = False

        G = 4*np.pi**2
        acc = np.array([0, 0, 0])
        for planet in planets:
            if planet != self and self.sun == False:
                r_vec = planet.positions[t, :] - self.positions[t, :]
                r = np.linalg.norm(r_vec)
                l = planet.positions[t, 0]*self.velocities[t, 1] - planet.positions[t, 1]*self.velocities[t, 0]
                acc = acc + (G * planet.mass * r_vec/r**3)#*(1 + (3*l**2)/(r**2 * 63198**2))
        return acc