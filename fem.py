import numpy as np
import matplotlib.pyplot as plt

# Number of elements
n_elements = 3
n_nodes = n_elements + 1

# Mesh
x = np.linspace(0, 1, n_nodes)
# We split the line from 0 to 1 into 3 elements:
# 0 ---- 1 ---- 2 ---- 3
# So we have 4 nodes
#
h = 1 / n_elements
# FEM solves:
#
# Ku=F
#
# K starts empty.
# F starts empty.

# Global matrix and vector
K = np.zeros((n_nodes, n_nodes))
F = np.zeros(n_nodes)


# Local element matrix and vector
# Create one element matrix
Ke = (1 / h) * np.array([[1, -1],
                         [-1, 1]])
# This is the stiffness matrix for one element.
#
# It connects two neighboring nodes:
# node e ---- node e+1
# Create one element load vector
Fe = (h / 2) * np.array([1, 1])
# The load is shared equally between the two nodes of the element.
# Assembly
# This adds each small element into the big matrix.
#
# For 3 elements:
# element 0 → nodes [0, 1]
# element 1 → nodes [1, 2]
# element 2 → nodes [2, 3]
# So each element contributes to the correct positions in K and F.
for e in range(n_elements):
    nodes = [e, e + 1]

    K[np.ix_(nodes, nodes)] += Ke
    F[nodes] += Fe

print("Global stiffness matrix K:")
print(K)

print("\nGlobal load vector F:")
print(F)

# Apply boundary conditions: u(0)=0, u(1)=0
free = np.arange(1, n_nodes - 1)
# The boundary nodes are fixed:
#
# u(0)=0,u(1)=0
#
# So nodes 0 and 3 are fixed.
# Only nodes 1 and 2 are unknown.
K_reduced = K[np.ix_(free, free)]
F_reduced = F[free]

# Solve
u = np.zeros(n_nodes)
u[free] = np.linalg.solve(K_reduced, F_reduced)
# This solves for the unknown middle node values.
#
# The boundary values stay zero.

print("\nSolution u:")
print(u)

# Plot
plt.plot(x, u, "o-", label="FEM solution")
plt.xlabel("x")
plt.ylabel("u")
plt.grid(True)
plt.legend()
plt.show()