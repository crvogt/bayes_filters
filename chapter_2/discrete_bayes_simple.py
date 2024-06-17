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

############# Code Body ############
# This is called the prior, our initial belief
belief = np.array([0.1] * 10)

# This is the structure of the enivornment
hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])

# This is a measurement
reading = 1

# We add correct scale as 3, because "through testing"
# we assume a door positive reading is 3x more likely to
# be right than wrong
#update_belief(hallway, belief, reading, correct_scale=3.)

# We assign a probability to each of the doorways,
# this is now a probability distribution
# door_belief = np.array([1/3, 1/3, 0, 0, 0, 0, 0, 0, 1/3, 0])

# scaled_update updates our belief and normalizes
belief = scaled_update(hallway, belief, reading, 0.75)

print(sum(belief))

########## Setup for plotting ##########
fig, axs = plt.subplots(1, 1)
x = np.linspace(0, len(hallway) - 1, len(hallway)).astype(int)
axs.set_ylim(0, 0.3)
axs.set_xlim(-1, 10)
axs.grid()

# This plot can be called a categorical distribution, or multimodal 
# distribution
axs.bar(x, belief)
plt.show()
