# Bridget Andersen, 11/15/18
# PHYS 641, Problem Set #5
####################### PROBLEM 1 #######################
# This script completes an MCMC simuation of the CMB power
# spectrum first using uncorrelated
#########################################################
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc,rcParams
from matplotlib.ticker import ScalarFormatter
import camb
import signal
import hw5_plots as plotter

rc('text', usetex=True)
fs=12
generate_plots = False

# Set a condition variable to catch a Ctrl-C
condition = True

# This function takes in the current CAMBparams() object and updates it with new values
# according to our input cosmology parameters. The new CAMBparams() object is returned.
# This function is very similar to what was shown in class.
# Parameter order: [Omega_bh^2, Omega_ch^2, tau, H0, n_s, A_s]
def update_model(parameters, pars_object):
    new_pars_object = pars_object.copy()
    new_pars_object.set_cosmology(ombh2=parameters[0], omch2=parameters[1], tau=parameters[2], H0=parameters[3])
    new_pars_object.InitPower.set_params(ns=parameters[4], As=parameters[5])
    return new_pars_object

# This function predicts the power spectrum given CMB parameters
def get_predicted_power_spectrum(parameters, pars_object):
    new_pars_object = update_model(parameters, pars_object)
    predictions = camb.get_results(new_pars_object)
    predicted_power_spectrum = predictions.get_cmb_power_spectra(new_pars_object, CMB_unit='muK')['total']
    # Keep only the actual power spectrum, and remove the l=0,1 measurements to be consistent with the
    # spectrum input from the WMAP 9yr data.
    predicted_power_spectrum = predicted_power_spectrum[:,0][2:]
    return predicted_power_spectrum

# This function reads in the input power spectrum data and optionally plots it against the
# predicted power spectrum using reasonable parameters and CAMB
def read_input_power_spectrum(filename='wmap_tt_spectrum_9yr_v5.txt', parameters=[0.02264, 0.1138, 0.089, 70., 0.972, 2.41*10**(-9)]):
    # Note that column 0 are the l's and column 1 is the power spectrum, l(l + 1) / 2pi * C_l, in uK units
    wmap_power_spectrum = np.loadtxt(filename, dtype=float)

    if generate_plots:
        plotter.plot_WMAP_power_spectrum(wmap_power_spectrum, parameters)
    # Return just the power spectrum and errors, not the l's
    return wmap_power_spectrum[:,1], wmap_power_spectrum[:,2]

# This function takes in an MCMC chain and calculates the number of independent samples
def calc_n_indep_samples(chain):
    # First, extract the fit parameter value from the chain
    mean_val = np.mean(chain)
    # Calculate the autocorrelation of the chain values
    remains = chain - mean_val
    remainsft = np.fft.rfft(remains)
    autocorr = np.fft.irfft(remainsft * np.conj(remainsft))
    # Find the minimum length at which the autocorrelation goes to zero (the samples are uncorrelated)
    less_than_zero = np.where(autocorr < 0)[0]
    if len(less_than_zero) is not 0:
        uncorr_length = np.min(less_than_zero)
    else:
        uncorr_length = 1.
    # The number of independent samples is then just the total number of samples divided
    # by the length between uncorrelated samples
    n_indep_samples = len(remains) / uncorr_length
    return n_indep_samples

# This function takes in the input WMAP power spectrum, the associated WMAP errors, and the predicted
# power spectrum and calculates the chisq
def calc_chisq(wmap_power_spectrum, wmap_errors, predicted_power_spectrum):
    chisq = np.sum((wmap_power_spectrum - predicted_power_spectrum)**2 / wmap_errors**2)
    return chisq

# This function takes in the stepsize array and generates steps for each parameter.
# If, instead, a covariance matrix is supplied, then the matrix is used to generate step values
def get_step(stepsize_arr, cov_matrix):
    npar = len(stepsize_arr)
    rand = np.random.randn(npar)
    if cov_matrix is not None:
        # Make random numbers correlated
        L = np.linalg.cholesky(cov_matrix)
        rand = np.dot(L, rand)
    steps = rand * stepsize_arr
    return steps

# This function updates the global condition variable to exit the MCMC loop gracefully
def intHandler(signum, frame):
    print "Ctrl-C detected"
    global condition
    condition = False

# Given a matrix of the chains of each parameter, this function generates a correlation matrix
def generate_corr_matrix(chain):
    corr_chain = chain[:,1:].copy()
    for i in range(corr_chain.shape[1]):
        corr_chain[:,i] = corr_chain[:,i] - corr_chain[:,i].mean()
        corr_chain[:,i] = corr_chain[:,i] / corr_chain[:,i].std()
    corr_matrix = np.dot(corr_chain.transpose(), corr_chain) / corr_chain.shape[0]
    return corr_matrix

# Given a matrix of the chains of each parameter, this function generates a covariance matrix
def generate_cov_matrix(chain):
    cov_chain = chain[:,1:].copy()
    for i in range(cov_chain.shape[1]):
        cov_chain[:,i] = cov_chain[:,i] - cov_chain[:,i].mean()
    cov_matrix = np.dot(cov_chain.transpose(), cov_chain) / cov_chain.shape[0]
    return cov_matrix

# This function runs an MCMC simulation, saving the chain as an npz and plotting the FFT power spectrum of the chain
# every n_save steps (plot includes the acceptance ratio and the number of independent samples currently calculated).
# If n_steps is defined, then the simulation runs for n_steps samples. Otherwise, it keeps running until it is
# manually interrupted with Ctrl-C. Before exiting, saves final chain as npz.
def run_MCMC(parameters, pars_object, wmap_power_spectrum, wmap_errors, stepsize_arr, cov_matrix=None, n_steps=-1, save=False, save_label='norm', n_save=50):
    # Set up code to catch a Ctrl-C
    global condition
    signal.signal(signal.SIGINT, intHandler)

    # Useful labels for each parameter
    params_label = ["$\Omega_{b}h^2$", "$\Omega_{c}h^2$", "$\\tau$", "$H_{0}$", "$n_{s}$", "$A_{s}$"]

    # Initialize the chain. We won't specify the size right now in case we are doing a very long chain
    # Chain will have chisq as column 0 and the other parameters as the other columns
    # chain = np.array([np.zeros(len(parameters)+1)])
    chain = None

    # Calculate the initial chisq
    predicted_power_spectrum = get_predicted_power_spectrum(parameters, pars_object)
    chisq = calc_chisq(wmap_power_spectrum, wmap_errors, predicted_power_spectrum)

    # Global ``condition'' variable controls when the MCMC loop stops iterating
    # (after a Ctrl-C event or after n_steps iterations, if n_steps is not left as -1)
    # Set up variable i to keep track of the number of iterations
    i = 0
    # Set up variable n_accepted to keep track of how many steps have been accepted
    n_accepted = [0]
    accept_rate = []
    while condition:
        # Determine the possible next step
        new_parameters = parameters + get_step(stepsize_arr, cov_matrix)
        new_predicted_power_spectrum = get_predicted_power_spectrum(new_parameters, pars_object)
        # Calculate the chisq for this new possible step
        new_chisq = calc_chisq(wmap_power_spectrum, wmap_errors, new_predicted_power_spectrum)

        # Calculate the likelihood that this step will be accepted
        likelihood = np.exp(-0.5 * (new_chisq - chisq))
        # Determine if the step is, in fact, accepted
        accepted = np.random.rand() < likelihood
        # Check if tau is negative
        if new_parameters[2] < 0:
            accepted = False

        # Calculate the acceptance rate based on the past n_samps
        n_samps = 20
        lower_index = 0 if (i - n_samps < 0) else i - n_samps
        upper_index = len(n_accepted)
        # print n_accepted
        accept_rate.append(np.sum(np.array(n_accepted)[lower_index:upper_index]) / float(upper_index - lower_index))

        # Print out some diagnostic information
        print i, chisq, new_chisq, likelihood, accepted, accept_rate[-1]
        # print accept_rate
        # print n_accepted
        print new_parameters

        # If the step is accepted, then set the new parameter values
        if accepted:
            parameters = new_parameters
            chisq = new_chisq
            # n_accepted.append(n_accepted[-1] + 1)
            n_accepted.append(1)
        else:
            n_accepted.append(0)
        # Either way, update our chain
        next_chain_values = np.zeros(len(parameters) + 1)
        next_chain_values[0] = chisq
        next_chain_values[1:] = parameters
        # chain = np.append(chain, [next_chain_values], axis=0)
        if chain is None:
            chain = [next_chain_values]
        else:
            chain = np.append(chain, [next_chain_values], axis=0)

        # If save option is chosen and enough iterations have passed, then save plots of the chain and the chain itself
        if (save == True) and (i % n_save == 0) and (i != 0):
            # Save the entire chain
            print "Saving chain to npz..."
            np.savez('./saved_chains/' + save_label + '_iteration_' + str(i) + '.npz', chain=chain, accept_rate=accept_rate)
            # Plot the FFT power spectrum of the chain of each parameter so far
            if True:
                print "Plotting chain and power spectra..."
                plotter.plot_diagnostics(chain, i, accept_rate[-1], save_label)
                plotter.plot_acceptance_rate(accept_rate, save_label, i)
        # Increment the number of iterations
        i = i + 1
        if (i > n_steps) and (n_steps != -1):
            condition = False
        # Check if condition has been set False either by Ctrl-C or by n_steps iterations
        if condition == False:
            print "Exiting MCMC loop"
            break

    print "Saving final plots and chain..."
    # Save the entire chain
    np.savez('./saved_chains/' + save_label + '_iteration_' + str(i) + '.npz', chain=chain, accept_rate=accept_rate)
    if True:
        plotter.plot_diagnostics(chain, i, accept_rate[-1], save_label)
        plotter.plot_acceptance_rate(accept_rate, save_label, i)

    # Set the global condition variable to True for the next MCMC run
    condition = True

    # Return the final chain
    return chain

# This function takes in the A matrix and noise matrix (N), and fits the data (y)
# using the QR decomposition linear least squares framework.
# Returns the derived parameters (m).
def QR(A, N, d):
    Q, R = np.linalg.qr(A)
    Qt = Q.transpose()
    N1 = np.linalg.inv(N)
    # Calculate the left hand side
    Qt_N1_Q_R = np.dot(Qt, np.dot(N1, np.dot(Q, R)))
    # Calculate the right hand side
    Qt_N1_d = np.dot(Qt, np.dot(N1, d))
    # Calculate and return the parameters
    m = np.dot(np.linalg.inv(Qt_N1_Q_R), Qt_N1_d)
    return m

# This function takes in the parameter value vs chisq for a given 1D curvature and fits a parabola to it using QR
# linear least squares fitting.
# Returns the parabola parameters
def parabola_fit(parameter_vals, chisqs):
    n = len(parameter_vals)
    
    # Assume Gaussian noise with unit variance
    N = np.eye(n)
    # The data is just given by the chisqs
    d = np.array(chisqs)
    x = np.array(parameter_vals)
    # Construct the A matrix
    A = np.zeros([n, 3])
    A[:,0] = 1.
    A[:,1] = x
    A[:,2] = x**2

    # Complete QR fitting
    m = QR(A, N, d)
    c = m[0]
    b = m[1]
    a = m[2]
    
    return a, b, c 

# This function takes in the index of the parameter that we are currently analyzing
# and completes a short chain keeping all other parameters fixed.
# It plots and saves the 1D curvature of the parameter (as an npz), and returns the error
# on the parameter from the 1D curvature. We will use these errors to determine our stepsize
def calc_1D_curvature(index, parameters, pars_object, wmap_power_spectrum, wmap_errors, stepsize_arr):
    # Run a short chain
    chain = run_MCMC(parameters, pars_object, wmap_power_spectrum, wmap_errors, stepsize_arr, n_steps=100, save=False, save_label="1Dcurvature" + str(index))
    # Calculate the error from the 1D curvature
    # parameter_vals = np.load('./1D_curvatures/1D_curvature_' + str(index) + '.npz')['param_vals']
    # chisqs = np.load('./1D_curvatures/1D_curvature_' + str(index) + '.npz')['chisqs']
    parameter_vals = chain[:,index+1]
    chisqs = chain[:,0]

    # Calculate the errors (to be used in stepsize) by fitting the parameter vs chisq values
    a, b, c = parabola_fit(parameter_vals, chisqs)
    error = 1. / np.sqrt(a)
    
    # Plot the 1D curvature
    if True:
        plotter.plot_1D_curvature(index, parameter_vals, error, chisqs, fit_params=[a,b,c])
    # Save the 1D curvature
    np.savez('./1D_curvatures/1D_curvature_' + str(index) + '.npz', param_vals=parameter_vals, chisqs=chisqs)
    # Finally, return just the error
    return error

if __name__ == '__main__':
    print "Problem #1\n"

    # Define "reasonable" estimates of the cosmological parameters. We get these values from
    # the WMAP 9yr papers.
    # Order: [Omega_bh^2, Omega_ch^2, tau, H0, n_s, A_s]
    init_parameters = [0.02264, 0.1138, 0.089, 70., 0.972, 2.41*10**(-9)]
    params_label = ["$\Omega_{b}h^2$", "$\Omega_{c}h^2$", "$\\tau$", "$H_{0}$", "$n_{s}$", "$A_{s}$"]

    # Define new initial parameters to get a better estimate of the errors while calculating 1D curvatures
    # init_parameters = [0.02264, 0.131, 0.132, 73.5, 1.055, 2.21*10**(-9)]

    # Read in the power spectrum data and associated errors (will use them when calculating chisq)
    wmap_power_spectrum, wmap_errors = read_input_power_spectrum(parameters=init_parameters)

    # Create a CAMBparams() object. Note that we will pass this object around to all of our functions
    # and modify it as we determine new cosmological parameters
    pars_object = camb.CAMBparams()
    pars_object.set_for_lmax(len(wmap_power_spectrum))

    # Set the initial stepsizes according to the errors from the WMAP 9yr paper
    init_stepsize_arr = np.array([0.0005, 0.0045, 0.014, 2.2, 0.013, 0.1*10**(-9)])

    # First, we will calculate 1D curvatures (parameter value vs chisq) and errors for each
    # parameter while keeping all others fixed
    # errors = []
    # for i in range(len(init_parameters)):
    #     print "Calculating 1D curvature for " + params_label[i]
    #     # Set all other stepsizes to zero except for the parameter that we are currently considering
    #     stepsize_arr = np.zeros(len(init_stepsize_arr))
    #     stepsize_arr[i] = init_stepsize_arr[i]
    #     # Run a short MCMC to find 1D curvature
    #     error = calc_1D_curvature(i, init_parameters, pars_object, wmap_power_spectrum, wmap_errors, stepsize_arr)
    #     errors.append(error)
    #     print
    # errors = np.array([0.00015999637874012678, 0.0045655314925601674, 0.0056600039920650753, 0.99372115598657096, 0.019388114135765005, 3.2009126626716649e-11])
    # errors = np.array([0.00019458797491677169, 0.0039366733380255441, 0.010951728573540617, 0.80547944526000703, 0.023474380868976213, 6.2502964434176527e-11])
    errors = np.array([0.00020597391377840101, 0.00076809786966665523, 0.0017536326892554259, 0.6070446905977589, 0.0047273323836420512, 8.0616144496052335e-12])
    print "Errors Estimates: " + str(errors)
    
    # Now run a short MCMC (n_steps = 500) with all the parameters varying according to the estimated errors
    # print "Running a chain to determine the covariance matrix"
    stepsize_arr = errors # * 2.38 / len(init_parameters)
    # save_label = "covrun4"
    # chain = run_MCMC(init_parameters, pars_object, wmap_power_spectrum, wmap_errors, stepsize_arr, n_steps=500, save=True, n_save=100, save_label=save_label)

    chain = np.load('./saved_chains/covrun4_iteration_501.npz')['chain']
    save_label = "final"
    # accept_rate = np.load('./saved_chains/covrun4_iteration_501.npz')['accept_rate']

    # Generate a correlation matrix
    corr_matrix = generate_corr_matrix(chain)
    # # Generate a covariance matrix
    cov_matrix = generate_cov_matrix(chain)

    print corr_matrix
    print cov_matrix

    # Plot the corner plot using the corner python module
    plotter.plot_corner(chain, save_label)

    # Now begin a final chain
    print "Running the final chain"
    chain = run_MCMC(init_parameters, pars_object, wmap_power_spectrum, wmap_errors, stepsize_arr, cov_matrix=cov_matrix, save=True, n_save=500, save_label="finalrun")
