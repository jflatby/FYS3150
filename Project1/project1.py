import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.linalg


plt.style.use("ggplot")

def general_algorithm(n):
    a = np.array([-1 for i in range(n - 1)])
    b = np.array([2 for i in range(n)])
    c = np.array([-1 for i in range(n - 1)])

    b_tilde = get_b_tilde(n)

    b_prime = np.zeros(n)
    b_prime_tilde = np.zeros(n)
    v = np.zeros(n)

    b_prime[0] = b[0]
    b_prime_tilde[0] = b_tilde[0]

    # Forward substitution
    for i in range(1, n - 1):
        b_prime[i] = b[i] - (a[i] / b_prime[i - 1]) * c[i - 1]
        b_prime_tilde[i] = b_tilde[i] - (a[i] / b_prime[i - 1]) * b_prime_tilde[i - 1]

    # Backward substitution

    for i in range(n-2, 0, -1):
        v[i] = (b_prime_tilde[i] - c[i] * v[i + 1]) / b_prime[i]
    return v

def specialized_algorithm(n):
    b_tilde = get_b_tilde(n)
    b_prime_tilde = np.zeros(n)
    v = np.zeros(n)

    b_prime = [(i+1)/i for i in range(1, n)]
    b_prime_tilde[0] = b_tilde[0]

    # Forward substitution
    for i in range(1, n - 1):
        b_prime_tilde[i] = b_tilde[i] - (-1 / b_prime[i - 1]) * b_prime_tilde[i - 1]

    # Backward substitution

    for i in range(n-2, 0, -1):
        v[i] = (b_prime_tilde[i] + v[i + 1]) / b_prime[i]
    return v


def get_b_tilde(n):
    h = 1 / (n+1)
    x = np.array([i * h for i in range(n)])
    f = 100*np.exp(-10*x)
    return h*h * f

def analytical(n):
    h = 1 / (n + 1)
    x = np.array([i * h for i in range(n)])
    return u(x)

def u(x):
    """
    Closed form solution for 1D Poisson Equation
    given our source term: f(x) = 100e^{-10x}
    """
    value = 1 - (1 - np.exp(-10)) * x - np.exp(-10 * x)
    return value

def plot(v, n):
    h = 1 / (n + 1)
    x = np.array([i * h for i in range(n)])
    plt.plot(x, v)

def relative_error(v, n):
    return np.max(np.log10(np.abs((v[1:] - analytical(n)[1:])/analytical(n)[1:])))

def LU_decomposition(n):
    #Create matrix
    matrix = np.array([[0 for i in range(n)] for j in range(n)])

    for i in range(n):
        matrix[i][i] = 2
        if(i!=0):
            matrix[i][i-1] = -1
        if (i!=n-1):
            matrix[i][i+1] = -1

    return scipy.linalg.lu_solve(scipy.linalg.lu_factor(matrix), get_b_tilde(n))

def compare_with_LU(v, n):
    t0_v = time.time()
    v_error = relative_error(v, n)
    comp_time_v = time.time()-t0_v

    t0_LU = time.time()
    LU_error = relative_error(LU_decomposition(n), n)
    comp_time_LU = time.time() - t0_LU
    print(str(n) + ":")
    print("Error: " + str(v_error) + " " + str(LU_error))
    print("Time: " + str(comp_time_v) + " " + str(comp_time_LU))

if __name__ == "__main__":

    ns = [10, 100, 1000, 10**4]
    for n in ns:
        plot(general_algorithm(n), n)
        #plot(LU_decomposition(n), n)
        #compare_with_LU(general_algorithm(n), n)

        """
        t0_gen = time.time()
        kartong = general_algorithm(n)
        error_gen = relative_error(general_algorithm(n), n)
        comp_time_gen = time.time() - t0_gen

        t0_spe = time.time()
        hei = specialized_algorithm(n)
        error_spe = relative_error(specialized_algorithm(n), n)
        comp_time_spe = time.time() - t0_spe

        t0_LU = time.time()
        robert = specialized_algorithm(n)
        error_LU = relative_error(LU_decomposition(n), n)
        comp_time_LU = time.time() - t0_LU

        print("N = ", n)
        print("Relative Error:", error_gen, error_spe)#, error_LU)
        print("Computation Time:", comp_time_gen, comp_time_spe)#, comp_time_LU)
        """

    plot(analytical(100), 100)

    plt.legend(["Thomas(n=10)", "Thomas(n=$10^2$)", "Thomas(n=$10^3$)", "Thomas(n=$10^4$)", "Analytical"])
    plt.xlabel("x")
    plt.ylabel("$u(x)$, $v_{100}(x)$")
    plt.savefig("thomas.png")

    plt.show()