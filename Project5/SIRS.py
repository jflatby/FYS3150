import numpy as np
from ODEsolver import RungeKutta4
import matplotlib.pyplot as plt
plt.style.use("ggplot")


class SIR_equation:
    """
    Returns
    S'(t), I'(t), R'(t)
    when called
    """
    def __init__(self, dt, T, N, a, b, c, S0, I0, R0, birthrate = 0, deathrate = 0, infected_deathrate = 0, seasonal_variation = False, vaccination_rate = 0):
        """
        dt: timestep
        T: total time [days]
        N: total population
        a: transmission rate
        b: recovery rate
        c: rate of immunity loss
        S0: initial susceptible population
        I0: initial infected population
        R0: initial resistant population

        Optionals: The rest have self-explainatory names and will
                   not affect the simulation unless you change them.
        """

        self.dt, self.T, self.N, self.a, self.b, self.c = dt, T, N, a, b, c
        self.S0, self.I0, self.R0 = S0, I0, R0

        self.e, self.d, self.dI = birthrate, deathrate, infected_deathrate

        self.w, self.A, self.a_0 = (4 * np.pi) / T, 2, a

        self.seasonal = seasonal_variation

        self.f = vaccination_rate

    def __call__(self, u, t):
        """
        Called by the ODEsolver class
        u:
        t:
        """
        S, I, R = u     # S, I, R are components of the vector u
        self.N = S+I+R  # Make sure N is always correct for when vital dynamics is in play

        if self.seasonal: # Change transmission rate over time if seasonal variations are enabled
            self.a = self.a_0 + self.A*np.cos(self.w*t)


        return np.array([self.c*R - self.a*S*I/self.N - self.d*S + self.e*self.N - self.f,             #S'(t)
                self.a * S * I/self.N - self.b * I - self.d*I - self.dI*I,                              #I'(t)
                 self.b*I - self.c*R - self.d*R + self.f ])                                             #R'(t)

def monte_carlo(dt, T, N, a, b, c, S0, I0, R0, birthrate = 0, deathrate = 0, infected_deathrate = 0, seasonal_variation = False, vaccination_rate = 0):
    """
    See docstring in the init function of the SIR_equation class.
    This function takes the exact same parameters.

    Instead of treating SIR as continuous paramaters we change
    our dt to a number where at max one person would transition
    from one group to the next. This way we can use discrete values.

    The first if-statement updates a and dt if seasonal variations are
    enabled. The next three are the transition probabilities derived
    from the basic SIRS model. The next 5 handle vital dynamics and the
    last one vaccinations. These are all optional arguments and if they
    are not set they default to 0 and will not affect the simulation.
    """
    time_steps = int(T/dt)
    SIR = np.zeros((time_steps, 3))

    SIR[0, :] = [S0, I0, R0]

    e = birthrate
    d = deathrate
    d_I = infected_deathrate

    a_0 = a
    w = 4*np.pi/T
    A = 2

    f = vaccination_rate

    # time loop
    for t in range(time_steps-1):
        SIR[t+1, :] = SIR[t, :]
        N = np.sum(SIR[t, :]) #Update N in case of vital dynamics

        if seasonal_variation:
            #dt = np.min([4 / (a * N), 1 / (b * N), 1 / (c * N)])
            a = a_0 + A*np.cos(w*(t*dt))

        # S -> I
        if np.random.random() < a * SIR[t, 0] * SIR[t, 1] * dt / N:
            SIR[t+1, 0] -= 1
            SIR[t+1, 1] += 1

        # I -> R
        if np.random.random() < b * SIR[t, 1] * dt:
            SIR[t+1, 1] -= 1
            SIR[t+1, 2] += 1

        # R -> S
        if np.random.random() < c * SIR[t, 2] * dt:
            SIR[t+1, 2] -= 1
            SIR[t+1, 0] += 1

        # S Death
        if np.random.random() < d*SIR[t, 0] * dt:
            SIR[t+1, 0] -= 1

        # S Birth
        if np.random.random() < e*SIR[t, 0] * dt:
            SIR[t+1, 0] += 1

        # I Death(regular deathrate)
        if np.random.random() < d*SIR[t, 1] * dt:
            SIR[t+1, 1] -= 1

        # I Death(infected deathrate)
        if np.random.random() < d_I * SIR[t, 1] * dt:
            SIR[t + 1, 0] -= 1

        # R Death
        if np.random.random() < d*SIR[t, 2] * dt:
            SIR[t+1, 2] -= 1

        # Vaccination
        if np.random.random() < f * dt:
            SIR[t + 1, 0] -= 1
            SIR[t + 1, 2] += 1


    return SIR

def get_analytical(N, a, b, c):
    """
    N: Total amount of people
    a: rate of transmission
    b: rate of recovery
    c: rate og immunity loss

    Returns: steady state values for the given set-up.
    """
    return[N*b/a, N*(1 - b/a)/(1 + b/c), N * b / c * (1 - b/a)/(1 + b/c)]

def simulate_SIRS(runge_kutta = False):
    """
    Main funciton, creates an array of populations containing
    the values of a, b and c for each one, loops over the
    different populations and simulates over time T using the
    Monte Carlo method unless runge_kutta == True
    """

    populations = [[4, 1, 0.5]]#, [4, 2, 0.5]]#, [4, 3, 0.5], [4, 4, 0.5]]

    fig = plt.figure()
    for i in range(len(populations)):
        a, b, c = populations[i]
        N = 400
        S0, I0, R0 = 300, 100, 0
        T = 30  # days


        """
        Change Values here to enable different functions to the model.
        If 0 and False, simulation will work without those functions.
        """
        ## Vital dynamics
        e, d, d_I = 0, 0, 0  # 0.001, 0.001, 0.002
        ## Seasonal variation
        seasons = False
        ## Vaccination (f)
        vaccine = 0

        dt = np.min([4/(a*N), 1/(b*N), 1/(c*N)])

        if runge_kutta:
            f = SIR_equation(dt, T, N, a, b, c, S0, I0, R0, e, d, d_I, seasonal_variation=seasons, vaccination_rate=vaccine)
            solver = RungeKutta4(f, [S0, I0, R0])

            u, t = solver.solve(np.arange(0, T, dt))

            S, I, R = u[:, 0], u[:, 1], u[:, 2]

        else: #Monte Carlo

            u = monte_carlo(dt, T, N, a, b, c, S0, I0, R0, e, d, d_I, seasonal_variation=seasons, vaccination_rate=vaccine)
            S, I, R = u[:, 0], u[:, 1], u[:, 2]
            t = np.linspace(0, T, len(S))


        fig.add_subplot(1, 1, i+1)
        plt.plot(t, S,
                 t, I,
                 t, R)

        plt.xlabel('Days after outbreak')
        plt.ylabel('No. of people')
        titles = ["A", "B", "C", "D"]
        plt.title(titles[i])
        print(titles[i])
        print(f"Numerical: {S[-1]} & {I[-1]} & {R[-1]}\\\\")
        analytical = get_analytical(N, a, b, c)
        print(f"Analytical: {analytical[0]} & {analytical[1]} & {analytical[2]}\\\\")

    plt.legend(['Susceptible', 'Infected', 'Resistant'])
    plt.tight_layout()
    plt.show()



simulate_SIRS(runge_kutta=False)