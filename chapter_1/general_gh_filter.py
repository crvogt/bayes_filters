import numpy as np
import matplotlib.pyplot as plt

def gen_data(x0, dx, n, noise_factor):
    return [x0 + dx*t + noise_factor*np.random.randn() for t in range(n)]

def g_h_filter(data, x0, dx, g, h, dt):
    """
    Performs g-h filter on 1 state variable with a fixed g and h
    
    'data' contains the data to be filtered
    'x0' is the initial value for our state variable
    'dx' is the initial change rate for our state variable
    'g' is the g-h's g scale factor
    'h'h is the g-h's h scale factor
    'dt' is the length of the time step
    """
    # Initialization
    # Initialize the state of the filter
    x_est = x0
    estimates = []

    for measurement in data:
        # prediction
        x_pred = x_est + (dx * dt) 
        dx = dx

        # Compute residual between estimated state and measurement
        residual = measurement - x_pred
        dx = dx + h * (residual) / dt
        x_est = x_pred + g * residual
        estimates.append(x_est)

    return np.array(estimates)

weights = [158.0,
           164.2,
           160.3,
           159.9,
           162.1,
           164.6,
           169.6,
           167.4,
           166.4,
           171.0,
           171.2,
           172.6]

"""
t = [x for x in range(len(weights))]

estimates_out = g_h_filter(data=weights, x0=160, dx=1, g=6./10, h=2./3, dt=1)
print(estimates_out)

fig, axs = plt.subplots(1, 1, sharex=True, figsize=(15, 5))

axs.plot(t, estimates_out, label="Estimate")
axs.scatter(t, weights, label="Measurements")
axs.plot([t[0], t[-1]], [160, 172], 'k--', label="Actual")
axs.legend(loc="lower right")
plt.show()
"""

# Using the noisy data generation
measurements = gen_data(5, 2, 100, 100)
estimates_out = g_h_filter(measurements, 5.0, 2.0, 0.2,0.02, 1.0)

t = [x for x in range(len(measurements))]

fig, axs = plt.subplots(1, 1, sharex=True, figsize=(15, 5))
axs.plot(t, estimates_out, label="Estimates")
axs.scatter(t, measurements, label="Measurements")
plt.show()
