# Bridget Andersen, 9/19/18
# PHYS 641, HW 1, Problem 5
# This script generates random Gaussian noise and adds it to a Gaussian template
# T times. The script estimates the noise according to the scattering of the
# data (including the Gaussian template) and does a least-squares fit to
# determine the amplitude for each iteration. We then calculate an overall
# amplitude/error from our individual values and plot the amplitude vs
# the noise to view the bias.
# Note that much of this is based on the Gaussian fitting example that we
# did in class.

import numpy as np
from matplotlib import pyplot as plt

# Setting up the noise parameters
noise = 0.5
Ninv = 1.0 / noise**2

# Setting up the actual data parameters
dx = 0.01
x = np.arange(-10, 10, dx)
n = len(x)
x0 = 0 # mean
amp_true = 2.0 # true amplitude
sig = 0.4 # true standard deviation
true_template = np.exp(-0.5 * (x - x0)**2 / sig**2)

# Now we iterate through and generate the data T times
T = 100
# Create arrays to save the amplitude and error data
amp = np.zeros(T)
err = np.zeros(T)
for i in range(0,T):
    # The actual data is given by:
    dat = true_template * amp_true + np.random.randn(n) * noise
    # Calculate the standard deviation of the data
    sig_est = np.std(dat)
    Ninv = 1.0 / sig_est**2
    err[i] = sig_est
    # Our fitting template (for simplicity of fit, we assume we know x0 and sig)
    fit_template = np.exp(-0.5 * (x - x0)**2 / sig**2)
    # Calculate the denominator
    denom = np.dot(fit_template, Ninv * fit_template)
    # Calculate the numerator
    numer = np.dot(fit_template, Ninv * dat)
    # Save the amplitudes
    amp[i] = numer / denom

# Take the weighted average to determine the overall amplitude and error
# where ``weight'' means w_i = 1 / sig**2
total_amp = 0.
total_err = 0.
for i in range(0,T):
    w_i = 1. / err[i]**2
    total_amp = total_amp + w_i * amp[i]
    total_err = total_err + w_i
total_amp = total_amp / total_err

print amp
    
# Plot the data and an example fit
plt.clf();
#plt.plot(x, dat);
plt.scatter(x, dat, c='red', marker='o', s=10, linewidth=0.5, edgecolor='black', label="Data Points")
plt.plot(x, amp[-1] * np.exp(-0.5 * (x - x0)**2 / sig**2), c='blue', label="Example Fit")
plt.title('Raw Data Plot')
plt.xlim((-10, 10))
plt.xlabel("x Value")
plt.ylabel("Amplitude")
plt.gca().legend()
plt.savefig('rawdat_plot.pdf', format='pdf', dpi=1000)

# Plot the amplitudes vs the errors
plt.clf();
plt.scatter(err, amp, c='blue', marker='o', s=10, linewidth=0.5, edgecolor='black');
plt.title('Amplitude vs Error Plot')
plt.xlabel('Error')
plt.ylabel('Amplitude')
plt.savefig('amperr_plot.pdf', format='pdf', dpi=1000)

# Print out the results
print "Actual Amplitude: " + str(amp_true)
print "Averaged Amplitude: " + str(total_amp)
