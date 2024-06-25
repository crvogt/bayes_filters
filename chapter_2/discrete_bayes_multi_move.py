import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1)

def update(likelihood, prior):
    return normalize(likelihood * prior)

def normalize(value):
    return value/sum(value)

# Convolution prediction
def predict(pdf, offset, kernel):
    N = len(pdf)
    kN = len(kernel)
    print(N)
    print(kN)

    width = int((kN - 1) / 2)
    print(width)

    prior = np.zeros(N)
    for i in range(N):
        for k in range(kN):
            index = (i + (width - k) - offset % N)
            prior[i] += pdf[index] * kernel[k]
    return prior

def lh_hallway(hall, z, z_prob):
    try:
        scale = z_prob / (1.0 - z_prob)
    except ZeroDivisionError:
        scale = 1e8

    likelihood = np.ones(len(hall))
    likelihood[hall==z] *= scale

    return likelihood


# Setup 
hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
prior = np.array([0.1] * 10)
# At position 0
# z is our measurement, not our movement
likelihood = lh_hallway(hallway, z=1, z_prob=0.75)
posterior = update(likelihood, prior)

kernel = [0.1, 0.8, 0.1]
# Now we have a new prior, based on a move to the right
prior = predict(posterior, 0, kernel)
# We take a measurement, and it shows there's a door
likelihood = lh_hallway(hallway, z=1, z_prob=0.75)
posterior = update(likelihood, prior)

# Next we will sense the wall
prior = predict(posterior, 1, kernel)
likelihood = lh_hallway(hallway, z=0, z_prob=0.75)
posterior = update(likelihood, prior)

# Once more, where we will see the wall again
prior = predict(posterior, 1, kernel)
likelihood = lh_hallway(hallway, z=0, z_prob=0.75)
posterior = update(likelihood, prior)

# Once more, where we will see the wall again
prior = predict(posterior, 1, kernel)
likelihood = lh_hallway(hallway, z=0, z_prob=0.75)
posterior = update(likelihood, prior)

x = np.linspace(0, len(hallway) - 1, len(hallway))

axs[0].bar(x, prior)
axs[0].set_ylim(0, 0.5)
axs[1].bar(x, posterior)
axs[1].set_ylim(0, 0.5)

plt.show()
