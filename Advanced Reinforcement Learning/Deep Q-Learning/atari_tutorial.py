from skimage.transform import resize
import gymnasium as gym
import matplotlib.pyplot as plt
import ale_py
import warnings
warnings.filterwarnings('ignore')

env = gym.make("Breakout-v0")
A, info = env.reset()

plt.imshow(A)
plt.show()

B = A[31:195]
plt.imshow(B)
plt.show()

C = resize(B, (105, 80, 3))
plt.imshow(C)
plt.show()
