import numpy as np

class ODESolver(object):
    """
    t: array of time values
    u: array of solution values (at time points t)
    k: step number of the most recently computed solution
    f: callable object implementing f(u, t)
    """
    def __init__(self, f, U0):
        self.f = f
        U0 = np.asarray(U0)  # assuming U0 is a sequence
        self.neq = U0.size   # number of equations
        self.U0 = U0

    def solve(self, time_points, terminate=None):
        """
        Computes u for t values in the array
        time_points
        """
        self.t = np.asarray(time_points)

        n = self.t.size
        self.u = np.zeros((n, self.neq))
        self.u[0] = self.U0

        # Time loop
        for k in range(n-1):
            self.k = k
            self.u[k+1] = self.advance()

        return self.u[:k+2], self.t[:k+2]


class RungeKutta4(ODESolver):
    def advance(self):
        u, f, k, t = self.u, self.f, self.k, self.t
        dt = t[k+1] - t[k]
        dt2 = dt/2.0
        K1 = dt*f(u[k], t[k])
        K2 = dt*f(u[k] + 0.5*K1, t[k] + dt2)
        K3 = dt*f(u[k] + 0.5*K2, t[k] + dt2)
        K4 = dt*f(u[k] + K3, t[k] + dt)
        u_new = u[k] + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)
        return u_new

