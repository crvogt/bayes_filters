import numpy as np
import matplotlib.pyplot as plt

def normalize(belief):
    return belief/sum(belief)

def update_belief(hall, belief, z, correct_scale):
    for i, val in enumerate(hall):
        if val == z:
            belief[i] *= correct_scale

def scaled_update(hall, belief, z, z_prob):
    scale = z_prob / (1 - z_prob)
    belief[hall==z] *= scale
    return normalize(belief)

def scaled_update_terms(hall, belief, z, z_prob):
    scale = z_prob / (1 - z_prob)
    likelihood = np.ones(len(hall))
    likelihood[hall==z] *= scale
    return normalize(likelihood * belief)

def update(likelihood, prior):
    # This function is meant to be more generalized than the previous two
    return normalize(likelihood * prior)

def lh_hallway(hall, z, z_prob):
    """
    Compute likelihood that a measurement matches
    positions in the hallway
    """
    try:
        scale = z_prob / (1. - z_prob)
    except ZeroDivisionError:
        scale = 1e8

    likelihood = np.ones(len(hall))
    likelihood[hall==z] *= scale
    return likelihood

############# Code Body ############
# This is called the prior, our initial belief
belief = np.array([0.1] * 10)

# This is the structure of the enivornment
hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])

# This is a measurement
reading = 1

likelihood = lh_hallway(hallway, reading, 0.75)

posterior = update(likelihood, belief)
print(posterior)

# We add correct scale as 3, because "through testing"
# we assume a door positive reading is 3x more likely to
# be right than wrong
#update_belief(hallway, belief, reading, correct_scale=3.)

# We assign a probability to each of the doorways,
# this is now a probability distribution
# door_belief = np.array([1/3, 1/3, 0, 0, 0, 0, 0, 0, 1/3, 0])

# scaled_update updates our belief and normalizes
# belief = scaled_update(hallway, belief, reading, 0.75)
likelihood = scaled_update_terms(hallway, belief, reading, 0.75)

print(sum(belief))

########## Setup for plotting ##########
fig, axs = plt.subplots(2, 1)
x = np.linspace(0, len(hallway) - 1, len(hallway)).astype(int)
axs[0].set_ylim(0, 0.3)
axs[0].set_xlim(-1, 10)
axs[0].grid()

# This plot can be called a categorical distribution, or multimodal 
# distribution
axs[0].bar(x, belief)
axs[1].bar(x, likelihood)
plt.show()
