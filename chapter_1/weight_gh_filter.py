import matplotlib.pyplot as plt
import numpy as np

# Initial guesses
weight = 160.0
gain_rate = -1.0

# Measured weights
weights = [158.0,
           164.2,
           160.3,
           159.9,
           163.1,
           164.6,
           169.6,
           167.4,
           166.4,
           171.0,
           171.2,
           172.6]
t = [x for x in range(len(weights))]

time_step = 1.0
weight_scale = 4.0/10
gain_scale = 1.0/3
estimates = [weight]
predictions = []

# z is a measurement
for z in weights:
    # prediction step
    weight = weight + gain_rate*time_step
    predictions.append(weight)

    # update step
    # residual = measurement - prediction
    residual = z - weight

    gain_rate = -1#gain_rate + gain_scale * (residual/time_step)
    weight = weight + weight_scale * residual

    estimates.append(weight)

# lazy
t_est = [x for x in range(len(estimates))]
t_pred = [x for x in range(len(estimates))]
del(t_pred[0])

plt.scatter(t, weights, s=100, linewidth=3, facecolors='none', edgecolors='k',
        label="Measurements")
plt.plot([t[0],t[-1]],[160, 172], 'k', label="Actual")
plt.plot(t_est, estimates, 'rs-', label="Estimates")
plt.plot(t_pred, predictions, 'bd-', label="Predictions")
plt.grid(linestyle="--")
plt.xlabel("Time (days)")
plt.ylabel("Weight (lbs)")

plt.title("Weight gh filter")
plt.legend(loc="lower right")

plt.show()
