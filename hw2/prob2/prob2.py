# Bridget Andersen, 10/2/18
# PHYS 641, Problem Set #2
####################### PROBLEM 2 #######################
# This script uses the classical linear least squares scheme to fit a Chebyshev
# polynomial solution to data.
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
    m = np.dot(np.linalg.inv(At_N1_A), At_N1_d)
    return m

# Calculates the rms error given the true data (d_true) and the predicted values (pred)
def rms_err(d_true, pred):
    return np.std(pred - d_true)

# Calculates the maximum error given the true data (d_true) and the predicted values (pred)
def max_err(d_true, pred):
    return np.max(np.abs(pred - d_true))

if __name__ == '__main__':
    print "Problem #2\n"
    # Set the number of points in our data set
    n = 1000
    # Create a ``true'' dataset
    x = np.linspace(-1, 1, n)
    d_true = np.exp(x)
    # Add noise to the data
    # noise = 0.1
    # d = d_true + np.random.randn(n) * noise
    d = d_true
    # Set the noise matrix to the identity
    N = np.eye(n)

    # Iterate over many orders of Chebyshev polynomials and evaluate the rms at each order
    rms_Cheb = []
    rms_simple = []
    ords = range(1, 100)
    for ord in ords:
        # Create A matrix for a Chebyshev polynomial fit
        A = np.zeros([n, ord + 1])
        A[:,0] = 1.
        A[:,1] = x
        for i in range(1,ord):
            A[:,i+1] = 2 * x * A[:,i] - A[:,i-1]
        # Create B matrix for a simple polynomial fit
        B = np.zeros([n, ord + 1])
        B[:,0] = 1.
        for i in range(ord):
            B[:,i+1] = B[:,i] * x
        # Calculate the fit parameters using the classical method and Chebychev polynomials
        m_Cheb = classical(A, N, d)
        # Calculate the fit parameters using the classical method and simple polynomials
        m_simple = classical(B, N, d)
        # Find the rms error for both
        rms_Cheb.append(rms_err(d_true, np.dot(A, m_Cheb)))
        rms_simple.append(rms_err(d_true, np.dot(B, m_simple)))
    # Plot the rms vs order
    plt.plot(ords, rms_Cheb, color='blue', linestyle='-', linewidth=lw, zorder=1, alpha=0.5, label='Chebyshev')
    plt.plot(ords, rms_simple, color='red', linestyle='-', linewidth=lw, zorder=1, alpha=0.5, label='Simple')
    plt.gca().set_xlabel(r'Order of Polynomial Fit',fontsize=fs,fontweight='bold')
    plt.gca().set_ylabel(r'RMS Error',fontsize=fs,fontweight='bold')
    plt.gca().legend(loc='upper left')
    plt.gca().set_xlim([1.0,len(ords)])
    plt.gca().set_ylim([-0.5,55])
    plt.tight_layout()
    plt.savefig('prob2_rms.pdf', format='pdf', dpi=1000)
    plt.clf()

    # Fit a 6th order Chebyshev polynomial to exp(x) on (-1,1)
    ord = 6
    # Create A matrix for a Chebyshev polynomial fit
    A_6 = np.zeros([n, ord + 1])
    A_6[:,0] = 1.
    A_6[:,1] = x
    for i in range(1,ord):
        A_6[:,i+1] = 2 * x * A_6[:,i] - A_6[:,i-1]
    m_Cheb = classical(A_6, N, d)
    print "6th Order Chebyshev fit to exp(x) on (-1,1)"
    rms_6 = rms_err(d_true, np.dot(A_6, m_Cheb))
    max_6 = max_err(d_true, np.dot(A_6, m_Cheb))
    print "RMS error =", rms_6
    print "Maximum error =", max_6, "\n"

    # Fit a 40th order Chebyshev polynomial to exp(x) on (-1,1)
    ord = 40
    # Create A matrix for a Chebyshev polynomial fit
    A_40 = np.zeros([n, ord + 1])
    A_40[:,0] = 1.
    A_40[:,1] = x
    for i in range(1,ord):
        A_40[:,i+1] = 2 * x * A_40[:,i] - A_40[:,i-1]
    m_Cheb = classical(A_40, N, d)
    rms_40 = rms_err(d_true, np.dot(A_40, m_Cheb))
    max_40 = max_err(d_true, np.dot(A_40, m_Cheb))
    print "40th Order Chebyshev fit to exp(x) on (-1,1)"
    print "RMS error =", rms_40
    print "Maximum error =", max_40, "\n"
    # Truncate this fit to only use the first 7 terms
    m_Cheb_trunc = m_Cheb[:7]
    pred = np.dot(A_6, m_Cheb_trunc)
    rms_trunc = rms_err(d_true, pred)
    max_trunc = max_err(d_true, pred)
    print "40th Order Chebyshev fit to exp(x) on (-1,1), truncated to 7 parameters"
    print "RMS error =", rms_trunc
    print "Maximum error =", max_trunc
    print m_Cheb[7:]
    print "Predicted maximum error =", np.sum(np.abs(m_Cheb[7:]))

    print np.abs(rms_6 - rms_trunc)/(0.5 * (np.abs(rms_trunc) + np.abs(rms_6)))
    print max_6 / max_trunc
