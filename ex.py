import numpy as np

a = np.array([0, 1])
b = np.array([[0], [1]])

print("a:", a, "shape:", a.shape)
print("a.T:", a.T, "shape:", a.T.shape)
print("b:", b, "shape:", b.shape)
print("b.T:", b.T, "shape:", b.T.shape)
print("b.T.T:", b.T.T, "shape:", b.T.T.shape)
