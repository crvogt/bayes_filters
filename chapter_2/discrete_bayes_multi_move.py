import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1)

def update(likelihood, prior):
    return normalize(likelihood * prior)

def normalize(value):
    return value/sum(value)

#def predict_move_convolution

def lh_hallway(hall, z, z_prob):
    try:
        scale = z_prob / (1.0 - z_prob)
    except ZeroDivisionError:
        scale = 1e8

    likelihood = np.ones(len(hall))
    likelihood[hall==z] *= scale

    return likelihood



hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])

prior = np.array([0.1] * 10)

likelihood = lh_hallway(hallway, z=1, z_prob=0.75)

posterior = update(likelihood, prior)
print(posterior)

x = np.linspace(0, len(hallway) - 1, len(hallway))

axs[0].bar(x, prior)
axs[0].set_ylim(0, 1)
axs[1].bar(x, posterior)
axs[1].set_ylim(0, 1)

plt.show()
