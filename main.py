import numpy as np
import matplotlib.pyplot as plt

hbar = 1973  # eV
m = 0.511E6  # eV
e = 3.795  # Two electrons with a separation of 1 angstrom have an electrostatic energy of 3.795 eV.

rmin = 1E-10  # Lower limit
rmax = 10  # Upper limit
N = 1000  # Step size

r = np.linspace(rmin, rmax, N)  # Graph grid / distance from nucleus
d = r[1] - r[0]  # Distance between two consecutive points

T = np.zeros((N - 2) ** 2).reshape(N - 2, N - 2)  # Creating the KE matrix
for i in range(N - 2):
    for j in range(N - 2):
        if i == j:
            T[i, j] = -2
        elif np.abs(i - j) == 1:
            T[i, j] = 1


def Vpot(x):                                      # Different methods for PE calculation
    return -(e ** 2) / x


def Vconstant(x):
    return 1


def Vlinear(x):
    return x


def Vquadratic(x):
    return x ** 2


V = np.zeros((N - 2) ** 2).reshape(N - 2, N - 2)  # Creating the PE matrix
for i in range(N - 2):
    for j in range(N - 2):
        if i == j:
            V[i, j] = Vpot(r[i + 1])

H = (-hbar ** 2 / (2 * m * d ** 2)) * T + V  # Creating the H matrix

values, vectors = np.linalg.eig(H)  # Finding eigenvalues(energies) and vectors(wavefunctions) and sorting them
energies = np.sort(values)
probability = vectors ** 2
states = 3
z = np.argsort(values)
z = z[0:states]

plt.figure(figsize=(12, 8))  # Plotting
for i in range(len(z)):
    y = []
    y = np.append(y, probability[:, z[i]])
    y = np.append(y, 0)
    y = np.insert(y, 0, 0)
    plt.plot(r, y, lw=2, label=f"State: {i + 1}        Energy = {energies[i]}")
    plt.xlabel('Distance from nucleus r', size=14)
    plt.ylabel('Electron probability $\psi^{2}$ ', size=14)

plt.legend()
plt.show()
