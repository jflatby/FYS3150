import numpy as np

def max_offdiag(A, n):
    #Requires symmetric matrix - only checks above the diagonal
    max = 0
    k, l = 0, 0
    for i in range(n):
        for j in range(i+1, n):
            if abs(A[i][j]) >= max:
                max = abs(A[i][j])
                k, l = i, j
    #print(l, k)
    return max, k, l



def rotate(A, S, k, l, n):
    if(A[k][l] != 0):
        tau = (A[l][l] - A[k][k])/(2*A[k][l])
        if(tau > 0):
            t = 1/(tau + np.sqrt(1 + tau**2))
        else:
            t = -1/(-tau + np.sqrt(1 + tau**2))

        c = 1/np.sqrt(1+t**2)
        s = c*t
    else:
        c = 1
        s = 0

    a_kk = A[k][k]
    a_ll = A[l][l]

    A[k][k] = c**2*a_kk - 2*c*s*A[k][l] + s**2*a_ll
    A[l][l] = s**2*a_kk + 2*s*c*A[k][l] + c**2*a_ll
    A[k][l] = 0
    A[l][k] = 0

    for i in range(n):
        if(i != k and i != l):
            a_ik = A[i][k]
            a_il = A[i][l]
            A[i][k] = c*a_ik - s*a_il
            A[k][i] = A[i][k]
            A[i][l] = c*a_il + s*a_ik
            A[l][i] = A[i][l]

        #r_ik = S[i][k]
        #r_il = S[i][l]
        #S[i][k] = c*r_ik - s*r_il
        #S[i][l] = c*r_il + s*r_ik

    return A, S

def jacobi_method(A, n):

    S = np.zeros((n, n))
    for i in range(n):
        S[i][i] = 1

    epsilon = 1e-2
    max_iterations = n**3
    iterations = 0
    max, k, l = max_offdiag(A, n)
    while(np.abs(max**2)>epsilon and iterations<max_iterations):
        max, k, l = max_offdiag(A, n)
        #print(max)
        #print(k, l)
        A, S = rotate(A, S, k, l, n)
        iterations += 1
    #print(A)
    return iterations, np.sort(np.diagonal(A))

def buckling_beam_tridiagonal(dim):
    # Setting up A

    r_max = 1
    step = r_max / (dim+1)

    A = np.zeros((dim, dim))

    for i in range(1, dim-1):
        a = -1 / step ** 2
        d = 2 / step ** 2
        A[i, i-1] = a
        A[i, i] = d
        A[i, i+1] = a

    A[0, 0] = d
    A[0, 1] = a

    A[dim-1, dim-2] = a
    A[dim-1, dim-1] = d

    return A

def single_electron_tridiagonal(dim):
    r_min = 0
    r_max = 5
    h = r_max / (dim)

    A = np.zeros((dim, dim))

    for i in range(1, dim - 1):
        rho = (i + 1) * h
        a = -1 / h ** 2
        d = 2 / h ** 2 + rho** 2
        A[i, i - 1] = a
        A[i, i] = d
        A[i, i + 1] = a

    A[0, 0] = d
    A[0, 1] = a

    A[dim - 1, dim - 2] = a
    A[dim - 1, dim - 1] = d

    return A

def double_electron_tridiagonal(dim, omega_r):
    r_max = 5
    h = r_max / (dim)
    omega_r = omega_r

    A = np.zeros((dim, dim))

    for i in range(1, dim - 1):
        rho = (i + 1) * h
        a = -1 / h ** 2
        d = 2 / h ** 2 + omega_r*rho** 2 + 1/rho
        A[i, i - 1] = a
        A[i, i] = d
        A[i, i + 1] = a

    A[0, 0] = d
    A[0, 1] = a

    A[dim - 1, dim - 2] = a
    A[dim - 1, dim - 1] = d

    return A

def compare_iterations(dims):
    for dim in dims:
        A = buckling_beam_tridiagonal(dim)
        iterations, B = jacobi_method(A, dim)
        print(dim, iterations)

#compare_iterations(np.arange(10, 110, 10))

def test_max_offdiag():
    A = np.array([[23, 34, 43, 23, 5], [3422,45,6,78,43],[534,45,67,7,23], [34,65,78,3,700], [1, 454,78,342, 35]])
    max, k, l = max_offdiag(A, 5)

    assert max == 700 #Should return 700 and not 3422 because the function we are testing only looks above the diagonal.

def test_jacobi():
    tolerance = 1e-2
    A = buckling_beam_tridiagonal(20)
    iterations, B = jacobi_method(A, 20)
    eigvals, eigvecs = np.linalg.eig(A)
    analytical = np.sort(eigvals)
    for i in range(len(B)):
        assert (np.abs(B[i] - analytical[i])) < tolerance

dim = 100
#A = buckling_beam_tridiagonal(dim)
#print(A)
#iterations, B = jacobi_method(A, dim)
#print(iterations)
#test_max_offdiag()
#test_jacobi()

#A = single_electron_tridiagonal(dim)

#A = buckling_beam_tridiagonal(dim)
#test_max_offdiag()
#test_jacobi()
#omegas = [0.01, 0.5, 1, 5]
#for omega_r in omegas:
#    print(omega_r, ":")
#    A = double_electron_tridiagonal(dim, omega_r)
#    iterations, B = jacobi_method(A, dim)
#    print(B[:4])
#print(np.average(np.abs(B - np.sort(eigenvalues))))

A = single_electron_tridiagonal(dim)
eigvals, eigvecs = np.linalg.eig(A)
iterations, B = jacobi_method(A, dim)
analytical = np.sort(eigvals)
print(eigvals[:4])
print(B[:4])
print(np.abs(eigvals[:4] - B[:4]))