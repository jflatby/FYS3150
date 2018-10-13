import numpy as np
import matplotlib.pyplot as plt

dim = 100

r_min = 0
r_max = 5
step = r_max/(dim+1)

a = -1/step**2
d = 2/step**2

A = np.zeros((dim, dim))
A[0, 0] = d
A[0, 1] = a

for i in range(1, dim-1):
    A[i, i-1] = a
    A[i, i] = d
    A[i, i+1] = a

A[dim-1, dim-2] = a
A[dim-1, dim-1] = d

eigenvalues, eigenvectors = np.linalg.eig(A)

permute = eigenvalues.argsort()
eigenvalues = eigenvalues[permute]
eigenvectors = eigenvectors[:, permute]

for i in range(dim):
    lambda_i = d + 2 * a * np.cos((i+1) * np.pi * step)
    print(np.abs(eigenvalues[i]-lambda_i))