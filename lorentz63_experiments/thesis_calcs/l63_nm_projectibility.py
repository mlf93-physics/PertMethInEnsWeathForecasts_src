import numpy as np
import matplotlib.pyplot as plt


def gamma(alpha, delta):
    return alpha - 1 / 2 * np.arctan(1 / np.tan(delta))


delta = np.linspace(0.01, np.pi - 0.01)
alpha = 0

gamma_values = gamma(alpha, delta)

l63_angles = np.array([0.37, 0.2, 0.18])
l63_angles = np.arccos(l63_angles)
print("l63_angles", l63_angles * 180 / np.pi)

print("gamma(0, l63_angles[0])", gamma(0, l63_angles[0]) * 180 / np.pi)

print("projection value for BV-EOF2 on eigenvector 1", np.cos(gamma(0, l63_angles[0])))

# plt.plot(delta * 180 / np.pi, gamma_values * 180 / np.pi)
# plt.xlabel("delta")
# plt.ylabel("gamma")
# plt.show()
