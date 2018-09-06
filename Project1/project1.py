import numpy as np
import matplotlib.pyplot as plt

def get_b_tilde(n):
    h = 1/(n+1)
    x = np.linspace(1, n, n-1)
    f = 100*np.exp(-10*x)

    return h**2 * f


n = 20

#a = np.zeros(n-1)
#c = np.zeros(n-1)
#b = np.zeros(n)

a = [-1 for i in range(n-1)]#np.array([12,34,46,2,57,23,12,34,67,45,76,8,53,23,67,54,34,56,76])
b = [2 for i in range(n)]#np.array([45,57,34,78,89,24,7,45,23,31,12,34,46,2,57,23,12,34,67,45])
c = [-1 for i in range(n-1)]#np.array([45,76,8,53,23,67,54,34,56,45,57,34,78,89,24,7,45,23,34])

a = np.array([12,34,46,2,57,23,12,34,67,45,76,8,53,23,67,54,34,56,76])
b = np.array([45,57,34,78,89,24,7,45,23,31,12,34,46,2,57,23,12,34,67,45])
c = np.array([45,76,8,53,23,67,54,34,56,45,57,34,78,89,24,7,45,23,34])

b_tilde = get_b_tilde(n)

b_prime = np.zeros(n)
b_prime_tilde = np.zeros(n)
v = np.zeros(n)


b_prime[0] = b[0]
b_prime_tilde[0] = b_tilde[0]

#Forward substitution
for i in range(1, n-1):
    b_prime[i] = b[i] - (a[i-1]/b_prime[i-1])*c[i-1]
    b_prime_tilde[i] = b_tilde[i] - (a[i-1]/b_prime[i-1])*b_prime_tilde[i-1]

#Backward substitution
for i in range (n-2, 1, -1):
    v[i] = (b_prime_tilde[i] -c[i]*v[i+1])/b_prime[i]

print(v)
x = np.linspace(0, 1, n)
plt.plot(x, v)
plt.show()

