import numpy as np
import matplotlib.pyplot as plt

def predict_move_convolution(pdf, offset, kernel):
    N = len(pdf)
    kN = len(kernel)
    width = int((kN - 1) / 2)

    prior = np.zeros(N)
    for i in range(N):
        for k in range(kN):
            index = (i + (width-k) - offset % N)
            prior[i] += pdf[index] * kernel[k]
    return prior

belief = [0.05, 0.05, 0.05, 0.05, 0.55, 0.05, 0.05, 0.05, 0.05, 0.05]
x = np.linspace(0, len(belief)-1, len(belief)).astype(int)
print(x)

prior = predict_move_convolution(belief, offset=2, kernel=[0.1, 0.8, 0.1])

fig, axs = plt.subplots(2, 1)

axs[0].bar(x, belief)
axs[0].set_ylim(0, 0.6)
axs[1].bar(x, prior)
axs[1].set_ylim(0, 0.6)

plt.show()

