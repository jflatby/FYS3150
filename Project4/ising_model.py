import numpy as np
from numba import jit
import matplotlib.pyplot as plt
#np.random.seed(0)
import time
from multiprocessing import Pool
plt.style.use("ggplot")
start_time = time.time()

@jit
def periodic(i, lim, add):
    return (i+lim+add)%lim

@jit
def metropolis(n_spins, spin_matrix, E, M, w):
    num_accepted = 0
    for y in range(n_spins):
        for x in range(n_spins):
            ix = int(np.random.uniform() * n_spins)
            iy = int(np.random.uniform() * n_spins)
            dE = 2*spin_matrix[ix, iy]*(spin_matrix[periodic(ix, n_spins, -1), iy] + spin_matrix[ix, periodic(iy, n_spins, -1)] + spin_matrix[periodic(ix, n_spins, 1), iy] + spin_matrix[ix, periodic(iy, n_spins, 1)])

            if np.random.uniform() <= w[dE+8] or dE <= 0:
                spin_matrix[ix, iy] *= -1
                M += 2*spin_matrix[ix, iy]
                E += dE
                num_accepted += 1

    return spin_matrix, E, M, num_accepted


@jit
def ising(T):
    E = 0
    M = 0
    n_spins = 100
    temperature = T
    cycles = 15000
    ordered = False

    #energies = np.zeros(cycles)
    #magnetizations = np.zeros(cycles)
    #accepteds = np.zeros(cycles)
    #energies = []#np.zeros(cycles)

    cycle_start_time = time.time()

    spin_matrix = np.random.choice([1, -1], size=(n_spins, n_spins))
    w = np.zeros(17)
    for de in np.arange(-8, 9, 4):
        w[de + 8] = np.exp(-de / temperature)

    values = np.zeros(6)
    M = np.sum(spin_matrix)

    ## Initial magnetization and energy
    for y in range(n_spins):
        for x in range(n_spins):
            if ordered:
                spin_matrix[x, y] = 1
            E -= spin_matrix[x, y] * (
                    spin_matrix[x, periodic(y, n_spins, -1)] + spin_matrix[periodic(x, n_spins, -1), y])

    ## main loop
    for cycle in range(1, cycles):
        spin_matrix, E, M, num_accepted = metropolis(n_spins, spin_matrix, E, M, w)
        # Update expectation values
        values[0] += E
        values[1] += E**2
        values[2] += M
        values[3] += M**2
        values[4] += np.abs(M)
        #values[5] += num_accepted
        #energies.append(E)

        #energies[cycle] = values[0]/cycle
        #magnetizations[cycle] = values[4]/cycle
        #accepteds[cycle] = values[5]
        teit = time.time() - start_time
        if cycle%1000 == 0:
            print("{} cycles done for T={}. {}s elapsed.".format(cycle, temperature, teit))

    #values = np.sum(values, axis=1)/cycles
    values = values/(cycles-1)
    energy = values[0]
    energy2 = values[1]
    magnetization = values[2]
    magnetization2 = values[3]
    abs_magnetization = values[4]

    energy_variance = (energy2 - energy ** 2)  # / (dim**2 * temperature**2)
    magnetization_variance = (magnetization2 - abs_magnetization ** 2)  # / (dim**2 * temperature)

    heat_capacity = energy_variance / temperature ** 2
    susceptibility = magnetization_variance / temperature

    return np.array([energy, abs_magnetization, heat_capacity, susceptibility])


if __name__ == "__main__":
    T = np.arange(2, 2.35, 0.05)

    with Pool() as pool:
        results = pool.map(ising, T)
        pool.close()
        pool.join()

    results  = np.transpose(results)
    np.save("WHAT.npy", results)

    print(f"Runtime: {time.time()-start_time}")

    """
    T = np.arange(2, 2.35, 0.05)
    results = np.zeros((4, len(T)))
    for t in range(len(T)):
        results[:, t] = ising(T[t])

    print(results)
    np.save("SKALLEskall.npy", results)
    """
    #np.save("20x20_ordered.npy", results)

    #for cycle in range(1, cycles):
     #   avg = ising(temp, cycle, spin_dim)
      #  energies[cycle] = avg[0]/cycle
       # magnetizations[cycle] = avg[4]/cycle


    """
    T = temperature
    print("Expected Values: (T="+str(temperature)+")")
    cosh_fac = (3.0 + np.cosh(8 / T))
    print("E: " + str((-8.0 * np.sinh(8 / T) / cosh_fac)))
    print("M: " + str((2.0 * np.exp(8.0 / T) + 4.0) / cosh_fac))
    print("Cv: " + str((8.0 / T) * (8.0 / T) * (1.0 + 3 * np.cosh(8 / T)) / (cosh_fac * cosh_fac)))
    print("Suscept: " + str(1 / T * (12.0 + 8.0 * np.exp(8.0 / T) + 8 * np.cosh(8.0 / T)) / (cosh_fac * cosh_fac)))
    print("Real Vlaues:")
    print(energy)
    print(abs_magnetization)
    print(heat_capacity)
    print(susceptibility)
    #print(abs_magnetization)
    
    skalle
    
        for cycle in range(1, cycles):
        spin_matrix, E, M = metropolis(n_spins, spin_matrix, E, M, w)
        # Update expectation values
        avg[0] += E
        avg[1] += E ** 2
        avg[2] += M
        avg[3] += M ** 2
        avg[4] += np.abs(M)
    
    
    print(f"Runtime: {time.time()-start_time}")
    fig = plt.figure()

    fig.add_subplot(2, 1, 1)
    plt.plot(range(len(Es)), Es)#, "o", markersize=2)
    #plt.plot(range(cycles), energies, "o", markersize=3)
    #plt.legend(["Energy, random matrix", "Energy, up-matrix"])
    plt.xlabel("Temperature")
    plt.ylabel("Energy")

    fig.add_subplot(2, 1, 2)
    plt.plot(range(len(Ms)), Ms)#, "o", markersize=2)
    #plt.plot(range(cycles), magnetizations[1], "o", markersize=3)
    #plt.legend(["Mean magnetization, random matrix", "Energy, up-matrix"])
    plt.xlabel("Temperature")
    plt.ylabel("Magnetization")
    """
    """
    fig.add_subplot(2, 2, 3)
    plt.plot(range(len(results[2])), results[2])  # , "o", markersize=2)
    # plt.plot(range(cycles), magnetizations[1], "o", markersize=3)
    # plt.legend(["Mean magnetization, random matrix", "Energy, up-matrix"])
    plt.xlabel("Temperature")
    plt.ylabel("Heat Capacity")

    fig.add_subplot(2, 2, 4)
    plt.plot(range(len(results[3])), results[3])  # , "o", markersize=2)
    # plt.plot(range(cycles), magnetizations[1], "o", markersize=3)
    # plt.legend(["Mean magnetization, random matrix", "Energy, up-matrix"])
    plt.xlabel("Temperature")
    plt.ylabel("Susceptibility")

    """
    #plt.tight_layout()
    #plt.show()