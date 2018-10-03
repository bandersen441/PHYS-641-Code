# Bridget Andersen, 10/2/18
# PHYS 641, Problem Set #2
####################### PROBLEM 4 #######################
# This script generates random correlated data by taking the Cholesky
# decomposition of the particular noise matrix defined in the question
# and multiplying uncorrelated data according to that decomposition.
#########################################################
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc,rcParams
from matplotlib.ticker import ScalarFormatter

rc('font',**{'family':'serif','serif':['Times']})
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rc('text', usetex=True)
fs=18
ms = 2
lw = 2

# Generates the noise matrix according to the assignment
def N_generator(a, sig, n):
    N = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            N[i][j] = a * np.exp(-0.5 * (i-j)**2 / sig**2)
    return N + (1. - a) * np.eye(n)

# This function generates an array of uncorrelated gaussian random noise
def uncorr_data_generator(n):
    return np.random.randn(n)

# This function takes in the A matrix and noise matrix (N), and fits the data (d)
# using the classical linear least squares framework.
# Returns the derived parameters (m).
def classical(A, N, d):
    At = A.transpose()
    N1 = np.linalg.inv(N)
    # Calculate the left hand side
    At_N1_A = np.dot(At, np.dot(N1, A))
    # Calculate the right hand side
    At_N1_d = np.dot(At, np.dot(N1, d))
    # Calculate and return the parameters
    m = At_N1_d / At_N1_A
    return m

# This function takes in matrix N and returns the eigenvalue/vector decomposition
# where N = VLV^-1. The columns of V are the eigenvectors of N and L is a diagonal
# matrix with the eigenvalues of N on the diagonal
def e_decomp(N):
    e, v = np.linalg.eig(N)
    e = e.real
    v = v.real
    # Check to make sure that the generated noise matrix is positive semi-definite
    print "Minimum eigenvalue: " + str(e.min())
    print "Maximum eigenvalue: " + str(e.max())
    psd = (e.min() > 0) and (e.max() > 0)
    print "N is positive semi-definite?", psd, "\n"
    if not psd:
        print "Noise matrix N is not positive semi-definite!"
        exit()
    # Create V matrix
    V = np.array(v)
    # Create the L matrix
    L = np.diag(e)
    return np.dot(V, np.sqrt(L))

# Calculates the error bar for a given gaussian amplitude fit
# Use: <m m^T> = 1 / (A^T N^{-1} A)
def error_bar(A, N):
    At = A.transpose()
    N1 = np.linalg.inv(N)
    return np.sqrt(1. / (np.dot(At, np.dot(N1, A))))

if __name__ == '__main__':
    print "Problem #4\n"
    # Set the size of the data
    n = 1000
    # Create a ``true'' dataset
    x = np.arange(0, n, 1)
    amp_true = 1.0
    x0 = 500
    sig = 50
    d_true = np.exp(-0.5 * (x - x0)**2 / sig**2)
    # Noise matrix parameters
    a_arr = [0.1, 0.5, 0.9]
    sig_arr = [5, 50, 500]

    # Derive an A matrix for gaussian amplitde fitting
    # (for simplicity of fit, we assume we know x0 and sig of source signal)
    A = np.exp(-0.5 * (x - x0)**2 / sig**2)

    err_bars = np.zeros([3, 3])
    # Iterate through each pair of possible noise matrix values
    for i in range(len(a_arr)):
        for j in range(len(sig_arr)):
            a = a_arr[i]
            sig = sig_arr[j]
            # Generate the noise matrix for these values
            N = N_generator(a, sig, n)
            # Fit the data with this noise matrix
            m = classical(A, N, d_true)
            # Calculate the error of this fit
            err = error_bar(A, N)
            # Print out results
            print "Result for a = " + str(a) + ", sig = " + str(sig)
            print "Amplitude = " + str(m)
            print "Error bar = " + str(err)
            err_bars[i][j] = err

            # Now generate correlated noise with and without the signal added in
            print "Generating correlated data for a = " + str(a) + ", sig = " + str(sig)
            uncorr_data = uncorr_data_generator(n)
            L = np.linalg.cholesky(N) #e_decomp(N)
            noise = np.dot(L, uncorr_data)
            # Plot the noisy data
            print "Plotting data for a = " + str(a) + ", sig = " + str(sig) + "\n"
            plt.plot(x, noise, color='red', linestyle='-', linewidth=lw, zorder=1, alpha=0.5, label='Without Signal')
            plt.plot(x, noise + d_true, color='blue', linestyle='-', linewidth=lw, zorder=1, alpha=0.5, label='With Signal')
            plt.title(r"a = " + str(a) + ", $\sigma$ = " + str(sig) + ", err\_bar = " + str(np.round(err_bars[i][j], 2)),fontsize=fs,fontweight='bold')
            plt.gca().set_xlabel(r'x',fontsize=fs,fontweight='bold')
            plt.gca().set_ylabel(r'y',fontsize=fs,fontweight='bold')
            plt.gca().set_xlim([0.0,len(x)])
            plt.gca().legend(loc='upper right')
            plt.tight_layout()
            plt.savefig('prob4_a' + str(a)[-1]  + '_sig' + str(sig) + '.pdf', format='pdf', dpi=1000)
            plt.clf()

    # Print out results nicely
    print '\ta\t\t\tsig'
    print ' \t' + ' \t' + str(sig_arr[0]) + '\t\t' + str(sig_arr[1]) + '\t\t' + str(sig_arr[2])
    print ' \t' + str(a_arr[0]) + '\t' + str(err_bars[0][0]) + '\t' + str(err_bars[0][1]) + '\t' + str(err_bars[0][2])
    print ' \t' + str(a_arr[1]) + '\t' + str(err_bars[1][0]) + '\t' + str(err_bars[1][1]) + '\t' + str(err_bars[1][2])
    print ' \t' + str(a_arr[2]) + '\t' + str(err_bars[2][0]) + '\t' + str(err_bars[2][1]) + '\t' + str(err_bars[2][2])
