import numpy as np

# Softmax for vector
v = np.random.randn(5)
print("random vector v:", v)
exp_v = np.exp(v)
print("exponential of v:", exp_v)
softmax_v = exp_v / sum(exp_v)
print("softmax of v:", softmax_v)
print()

# Softmax for Matrix
# Create a random matrix shape 2x3
A = np.random.randn(2, 3)
print("random matrix:\n", *A)
exp_A = np.exp(A)
print("exponential of A:\n", exp_A)
softmax_A = exp_A / exp_A.sum(axis=1, keepdims=True)
print("softmax of A:\n", softmax_A)
print("sum of each row of softmax of A:", softmax_A.sum(axis=1))
print()
