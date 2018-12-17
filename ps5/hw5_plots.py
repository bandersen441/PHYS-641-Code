# Bridget Andersen, 11/15/18
# PHYS 641, Problem Set #5
####################### PROBLEM 1 #######################
# This script creates a bunch of useful plots for this
# assignment.
#########################################################
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc,rcParams
from matplotlib.ticker import ScalarFormatter
import corner
import hw5 as main

rc('text', usetex=True)
fs=12
params_label = ["$\Omega_{b}h^2$", "$\Omega_{c}h^2$", "$\\tau$", "$H_{0}$", "$n_{s}$", "$A_{s}$"]

# Given a chain, this function plots the correlation function, power spectrum,
# and chain.
def plot_diagnostics(chain, iters, accept_rate, save_label):
    print "Plotting diagnostics"
    global params_label

    n_params = chain.shape[1] - 1
    
    # Plot the correlation function
    figsize = (10,6)
    fontsize = 10
    n_samps = []
    corr_ls = []
    rows = 2
    cols = 3
    f, axarr = plt.subplots(rows, cols, figsize=figsize)
    for index in range(n_params):
        # First, extract the fit parameter value from the chain
        mean_val = np.mean(chain[:,index+1])
        # Calculate the autocorrelation of the chain values
        remains = chain[:,index+1] - mean_val
        n = len(remains)
        remainsft = np.fft.rfft(remains)
        autocorr = np.fft.irfft(remainsft * np.conj(remainsft))
        autocorr = autocorr / autocorr[0]
        # autocorr = np.correlate(chain[:,index+1], chain[:,index+1], mode='full')
        # autocorr = autocorr[len(autocorr)/2:]
        # Find the minimum length at which the autocorrelation goes to zero (the samples are uncorrelated)
        less_than_zero = np.where(autocorr < 0)[0]
        if len(less_than_zero) is not 0:
            uncorr_length = np.min(less_than_zero)
        else:
            uncorr_length = 1.
        # The number of independent samples is then just the total number of samples divided
        # by the length between uncorrelated samples
        num_indep_samps = len(chain[:,index+1]) / uncorr_length
        corr_ls.append(uncorr_length)
        n_samps.append(num_indep_samps)
        # Now actually plot things
        r = index / cols
        c = index % cols
        axarr[r,c].plot(range(len(autocorr[:len(autocorr)/2])), autocorr[:len(autocorr)/2], lw=1, color='b', alpha=0.7)
        axarr[r,c].set_title(params_label[index] + "\n corr\_l=" + str(int(uncorr_length)) + ", n\_indep=" + str(int(num_indep_samps)))
        # plt.plot(range(len(autocorr)), autocorr, lw=1, color='b', alpha=0.7)
        axarr[r,c].set_xlabel("Correlation Length")
        axarr[r,c].set_ylabel("Power")
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    f.suptitle("Correlation Functions for " + save_label + " iteration " + str(iters) + ", accept rate=" + "{0:.2f}".format(accept_rate))
    plt.savefig('./saved_chains/' + save_label + '_corrfuncs_iter_' + str(iters) + '.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    
    # Plot the power spectrum
    rows = 2
    cols = 3
    f, axarr = plt.subplots(rows, cols, figsize=figsize)
    for index in range(n_params):
        num_indep_samps = n_samps[index]
        corr_l = corr_ls[index]
        fft_ps = np.abs(np.fft.rfft(chain[:,index+1]))**2
        r = index / cols
        c = index % cols
        axarr[r,c].plot(range(len(fft_ps)), fft_ps, lw=1, color='r', alpha=0.7)
        # plt.plot(range(len(fft_ps)), fft_ps, lw=1, color='b', alpha=0.7)
        axarr[r,c].set_xscale("log")
        axarr[r,c].set_yscale("log")
        axarr[r,c].set_xlabel("Frequency Bin")
        axarr[r,c].set_ylabel("Power")
        axarr[r,c].set_title(params_label[index] + "\n corr\_l=" + str(int(corr_l)) + ", n\_indep=" + str(int(num_indep_samps)))
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    f.suptitle("Power Spectra for " + save_label + " iteration " + str(iters) + ", accept rate=" + "{0:.2f}".format(accept_rate))
    plt.savefig('./saved_chains/' + save_label + '_ps_iter_' + str(iters) + '.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
    plt.clf()

    # Plot the chain itself
    rows = 2
    cols = 3
    f, axarr = plt.subplots(rows, cols, figsize=figsize)
    for index in range(n_params):
        num_indep_samps = n_samps[index]
        corr_l = corr_ls[index]
        r = index / cols
        c = index % cols
        axarr[r,c].plot(range(len(chain[:,index+1])), chain[:,index+1], lw=1, color='purple', alpha=0.7)
        axarr[r,c].set_xlabel("Iteration")
        axarr[r,c].set_ylabel("Parameter value")
        axarr[r,c].set_title(params_label[index] + "\n corr\_l=" + str(int(corr_l)) + ", n\_indep=" + str(int(num_indep_samps)))
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    f.suptitle("Chains for " + save_label + " iteration " + str(iters) + ", accept rate=" + "{0:.2f}".format(accept_rate))
    plt.savefig('./saved_chains/' + save_label + '_chains_iter_' + str(iters) + '.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
    plt.clf()

# Given an input WMAP power spectrum and input parameters, this function plots the input and predicted
# power spectrum on the same plot
def plot_WMAP_power_spectrum(wmap_power_spectrum, parameters):
    print "Plotting the wmap power spectrum vs expected parameters"
    # Create a new CAMBparams() object
    pars_object = camb.CAMBparams()
    pars_object.set_for_lmax(len(wmap_power_spectrum[:,0]))
    predicted_power_spectrum = get_predicted_power_spectrum(parameters, pars_object)
    
    ls = np.array(wmap_power_spectrum[:,0], dtype=int)
    plt.plot(ls, wmap_power_spectrum[:,1], lw=1.5, color='black', label="Input WMAP Power Spectrum")
    plt.plot(ls, predicted_power_spectrum, lw=1.5, color='red', label="Predicted Power Spectrum")
    params_label = "$[\Omega_{b}h^2, \Omega_{c}h^2, \\tau , H_{0}, n_{s}, A_{s}]=$" + str(parameters)
    plt.title("WMAP vs Predicted Power Spectrum \n" + params_label, fontsize=fs)
    plt.xlabel("$\ell$")
    plt.ylabel("$\ell (\ell + 1) C_{\ell} / 2 \pi$")
    plt.legend(loc='lower left')
    plt.savefig('power_spectrum.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0.2)
    plt.clf()

# Given a chain of parameter values and chisqs, plot the 1D curvature
def plot_1D_curvature(index, parameter_vals, error, chisqs, fn='', fit_params=None, b=None):
    print "Plotting 1D curvature"
    
    params_label = ["$\Omega_{b}h^2$", "$\Omega_{c}h^2$", "$\\tau$", "$H_{0}$", "$n_{s}$", "$A_{s}$"]
    fig = plt.figure(figsize=(6,6))
    plt.scatter(parameter_vals, chisqs, c='b', s=20, alpha=0.5, marker='o', edgecolor='', label="MCMC Results")
    plt.title("1D curvature for " + params_label[index] + ", Error=" + str(error), fontsize=fs)
    plt.xlabel(params_label[index])
    plt.ylabel("$\chi^2$")

    # If fit parameters are given, then plot the fit
    if fit_params is not None:
        x = np.linspace(np.min(parameter_vals), np.max(parameter_vals), 100)
        fit = fit_params[0] * x**2 + fit_params[1] * x + fit_params[2]
        plt.plot(x, fit, lw=1.5, color='red', alpha=0.5, label="Parabolic Fit")
    plt.legend(loc='best')
    
    if fn == '':
        filename = './1D_curvatures/1D_curvature_' + str(index) + '.pdf'
    else:
        filename = fn
    plt.savefig(filename, format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0.2)
    plt.clf()

# Given a chain of acceptance rates, plot them vs iteration
def plot_acceptance_rate(accept_rate, save_label, iters):
    plt.plot(range(len(accept_rate)), accept_rate, lw=1.5, color='red')
    plt.title("Acceptance Rate for " + save_label + " iteration " + str(iters), fontsize=fs)
    plt.xlabel("Iteration")
    plt.ylabel("Acceptance Rate")
    plt.savefig('./saved_chains/' + save_label + '_ar_iter_' + str(iters) + '.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
    plt.clf()

# Given a chain, plot the corner plot!
def plot_corner(chain, save_label):
    print "Plotting corner plot"
    fig = corner.corner(chain[:,1:], labels=[r"$\Omega_{b}h^2$", r"$\Omega_{c}h^2$", r"$\\tau$", r"$H_{0}$", r"$n_{s}$", r"$A_{s}$"], plot_contours=False)
    plt.savefig('./' + save_label + '_corner.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0.2)
    plt.clf()

if __name__ == '__main__':
    # Testing out the parabola fitting function
    index = 1
    error = 0.00393667333803
    parameter_vals = np.load('./1D_curvatures/1D_curvature_1.npz')['param_vals']
    chisqs = np.load('./1D_curvatures/1D_curvature_1.npz')['chisqs']

    a, b, c = main.parabola_fit(parameter_vals, chisqs)
    print a, b, c
    
    plot_1D_curvature(index, parameter_vals, error, chisqs, fit_params=[a,b,c], fn='save.pdf')
