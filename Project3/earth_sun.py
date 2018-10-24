import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

total_time = 1      #year
dt = 1e-3
time_steps = int(total_time/dt)

positions = np.zeros((time_steps, 2))
velocities = np.zeros((time_steps, 2))
accelerations = np.zeros((time_steps, 2))

def get_acceleration(t):
    GM = 4 * np.pi ** 2  # [AU^3/yr^2]
    r_vec = np.array([0, 0]) - positions[t, :]
    r = np.sqrt(r_vec[0] ** 2 + r_vec[1] ** 2)
    # r_unit_vec = r_vec/r
    acc = GM * r_vec / r ** 3
    return acc

def forward_euler():
    for t in range(time_steps-1):
        positions[t+1, :] = positions[t, :] + velocities[t, :]*dt
        velocities[t+1, :] = velocities[t, :] + get_acceleration(t)*dt

def velocity_verlet():
    for t in range(time_steps-1):
        positions[t+1, :] = positions[t, :] + velocities[t, :]*dt + 0.5*accelerations[t, :]*dt**2
        accelerations[t+1, :] = get_acceleration(t+1)
        velocities[t+1, :] = velocities[t, :] + 0.5*(accelerations[t, :] + accelerations[t+1, :])*dt


positions[0, :] = [1, 0]
velocities[0, :] = [0, 2*np.pi]
accelerations[0, :] = get_acceleration(0)

#forward_euler()
velocity_verlet()

plt.plot(positions[:, 0], positions[:, 1])
plt.show()