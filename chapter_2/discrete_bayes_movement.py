import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1)

def predict_move_convolution(pdf, offset, kernel):
    N = len(pdf)
    kN = len(kernel)
    width = int((kN - 1) / 2)

    prior = np.zeros(N)
    for i in range(N):
        for k in range(kN):
            index = (i + (width-k) - offset) % N
            prior[i] += pdf[index] * kernel[k]
    return prior

def predict_move(belief, move, p_under, p_correct, p_over):
    """
    Now we are adding noise to our predicted movement
    p_under: Probability we undershoot our movement
    p_over: Probability we overshoot our movement
    """
    n = len(belief)
    prior = np.zeros(n)
    for i in range(n):
        prior[i] = (
                belief[(i-move) % n] * p_correct +
                belief[(i-move-1) % n] * p_over + 
                belief[(i-move+1) % n] * p_under)
    return prior

def perfect_predict(belief, move):
    """
    move the position by 'move' positive to
    the right, and negative to the left
    """
    n = len(belief)
    result = np.zeros(n)
    for i in range(n):
        result[i] = belief[(i-move) % n]
    return result

"""
# For perfect_predict
belief = np.array([0.35, 0.1, 0.2, 0.3, 0, 0, 0, 0, 0, 0.05])
x = [val for val in range(len(belief))]
"""
# For predict move, assuming we know exactly where the
# dog started
belief = [0., 0., 1.0, 0.0, 0., 0., 0., 0., 0., 0.]
x = [val for val in range(len(belief))]

axs[0].bar(x, belief)
axs[0].set_title("before")
axs[0].set_ylim(0, 1)

for i in range(200):
    print(i)
    prior = predict_move(belief, 2, 0.1, 0.8, 0.1)
    belief = prior

    axs[1].bar(x, prior)
    axs[1].set_title("after")
    axs[1].set_ylim(0, 1)

    plt.show(block=False)
    plt.pause(0.1)
    axs[1].clear()
