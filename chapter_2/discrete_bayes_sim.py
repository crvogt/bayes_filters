import matplotlib.pyplot as plt
import numpy as np

"""
Write functions to perform the filtering and to plot the results at any step.
Doorways in black. Prior in orange, posterior in blue.
Thick vertical line to indicate dog position - which is not a product of the simulator
but simulated.
"""

def discrete_bayes_sim(prior, kernel, measurements, z_prob, hallway):
    posterior = np.array([0.1] * 10)
    priors, posteriors = [], []
    for i, z in enumerate(measurements):
        prior = predict(posterior, i, kernel)
        priors.append(prior)

        likelihood = lh_hallway(hallway, z, z_prob)
        posterior = update(likelihood, prior)
        posteriors.append(posterior)
    return priors, posteriors

def normalize(value):
    return value / sum(value)

def update(prior, likelihood):
    return normalize(prior*likelihood)

def lh_hallway(hall, z, z_prob):
    try:
        scale = z_prob / (1 - z_prob)
    except ZeroDivisionError:
        scale = 1e8

    likelihood = np.ones(len(hall))
    likelihood[hall==z] *= scale

    return likelihood

def predict(pdf, offset, kernel):
    N = len(pdf)
    kN = len(kernel)

    width = int((kN - 1) / 2)

    prior = np.zeros(N)
    for i in range(N):
        for k in range(kN):
            index = (i + (width - k) - offset) % N
            prior[i] += pdf[index] * kernel[k]
    return prior

def plot_posterior(hallway, posteriors, i):
    plt.title('Posterior')
    x = np.linspace(0, len(hallway)-1, len(hallway))
    #plt.bar(x, hallway, color='k')
    plt.bar(x, posteriors[i], color='k')
    plt.axvline(i % len(hallway), lw=5)

def plot_prior(hallway, priors, i):
    plt.title('Prior')
    x = np.linspace(0, len(hallway)-1, len(hallway))
    plt.bar(x, hallway, color='k')
    plt.bar(x, priors[i], color='#ff8015')
    plt.axvline(i % len(hallway), lw=5)

def animate_discrete_bayes(hallway, priors, posteriors):
    def animate(step):
        step -= 1
        i = step // 2
        if step % 2 == 0:
            plot_prior(hallway, priors, i)
        else:
            plot_posterior(hallway, posteriors, i)

    return animate

kernel = [0.1, 0.8, 0.1]
prior = [0.1]*10
z_prob = 1.0
hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])

# Measurements with no noise
zs = [hallway[i % len(hallway)] for i in range(50)]

priors, posteriors = discrete_bayes_sim(prior, kernel, zs, z_prob, hallway)

plot_posterior(hallway, posteriors, 0)
plot_prior(hallway, priors, 0)
plt.show()

