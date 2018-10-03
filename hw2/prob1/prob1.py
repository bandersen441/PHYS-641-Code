# Bridget Andersen, 10/2/18
# PHYS 641, Problem Set #2
####################### PROBLEM 1 #######################
# This script fits polynomials using the classical expression as well as
# QR decomposition.
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

# This function takes in the A matrix and noise matrix (N), and fits the data (y)
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

if __name__ == '__main__':
    print "Problem #1\n"
    # Set the number of points in our data set
    n = 1000
    # Create a ``true'' dataset
    x = np.linspace(-1, 1, n)
    d_true = x**3 - 0.5 * x + 0.2
    # Add noise to the data
    # noise = 0.5
    # d = d_true + np.random.randn(n) * noise
    d = d_true
    # Set the noise matrix to the identity
    N = np.eye(n)
    
    # Iterate over many orders of polynomials and evaluate the rms at each
    # order for both the classical and QR decomposition methods
    classical_rms = []
    QR_rms = []
    ords = range(1, 100)
    for ord in ords:
        # Create A matrix for a simple polynomial fit
        A = np.zeros([n, ord + 1])
        A[:,0] = 1.
        for i in range(ord):
            A[:,i+1] = A[:,i] * x
        # Calculate the fit parameters using the classical method
        m = classical(A, N, d)
        # Find the rms error for classical
        classical_rms.append(np.std(np.dot(A, m) - d_true))
        # Calculate the fit parameters using the QR method
        m = QR(A, N, d)
        # Find the rms error for QR
        QR_rms.append(np.std(np.dot(A, m) - d_true))
    # Plot the rms vs order for both methods to compare
    plt.plot(ords, classical_rms, color='blue', linestyle='-', linewidth=lw, zorder=1, alpha=0.5, label='Classical')
    plt.plot(ords, QR_rms, color='red', linestyle='-', linewidth=lw, zorder=1, alpha=0.5, label='QR')
    plt.gca().set_xlabel(r'Order of Polynomial Fit',fontsize=fs,fontweight='bold')
    plt.gca().set_ylabel(r'RMS Error',fontsize=fs,fontweight='bold')
    plt.gca().set_xlim([0.0,len(ords)])
    plt.gca().set_ylim([-0.5,27])
    plt.gca().legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('prob1_rms.pdf', format='pdf', dpi=1000)
    plt.clf()

    # Plot the classical and QR fits compared to the data for an order of 40
    # (where the classical fit goes wild and the QR fit remains stable)
    ord = 40
    A = np.zeros([n, ord + 1])
    A[:,0] = 1.
    for i in range(ord):
        A[:,i+1] = A[:,i] * x
    # Calculate the fit parameters using the classical method
    m_classical = classical(A, N, d)
    pred_classical = np.dot(A, m_classical)
    # Calculate the fit parameters using the QR method
    m_QR = QR(A, N, d)
    pred_QR = np.dot(A, m_QR)
    # Plot the data and each of the resulting curves
    plt.scatter(x, d, color='black', marker='.')
    plt.title("Order 40 Polynomial Fit",fontsize=fs,fontweight='bold')
    plt.plot(x, pred_classical, color='blue', linestyle='-', linewidth=lw, zorder=1, alpha=0.5, label='Classical')
    plt.plot(x, pred_QR, color='red', linestyle='-', linewidth=lw, zorder=1, alpha=0.5, label='QR')
    plt.gca().set_xlabel(r'x',fontsize=fs,fontweight='bold')
    plt.gca().set_ylabel(r'y',fontsize=fs,fontweight='bold')
    plt.gca().set_xlim([-1.0,1.0])
    plt.gca().set_ylim([-2,2])
    plt.gca().legend(loc='upper left')
    plt.subplots_adjust(left=0.15)
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(top=0.88)
    plt.savefig('prob1_fit_compare_ord40.pdf', format='pdf', dpi=1000)
    plt.clf()

    ord = 5
    A = np.zeros([n, ord + 1])
    A[:,0] = 1.
    for i in range(ord):
        A[:,i+1] = A[:,i] * x
    # Calculate the fit parameters using the classical method
    m_classical = classical(A, N, d)
    pred_classical = np.dot(A, m_classical)
    # Calculate the fit parameters using the QR method
    m_QR = QR(A, N, d)
    pred_QR = np.dot(A, m_QR)
    # Plot the data and each of the resulting curves
    plt.scatter(x, d, color='black', marker='.')
    plt.title("Order 5 Polynomial Fit",fontsize=fs,fontweight='bold')
    plt.plot(x, pred_classical, color='blue', linestyle='-', linewidth=lw, zorder=1, alpha=0.5, label='Classical')
    plt.plot(x, pred_QR, color='red', linestyle='-', linewidth=lw, zorder=1, alpha=0.5, label='QR')
    plt.gca().set_xlabel(r'x',fontsize=fs,fontweight='bold')
    plt.gca().set_ylabel(r'y',fontsize=fs,fontweight='bold')
    plt.gca().set_xlim([-1.0,1.0])
    plt.gca().set_ylim([-2,2])
    plt.gca().legend(loc='upper left')
    plt.subplots_adjust(left=0.15)
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(top=0.88)
    plt.savefig('prob1_fit_compare_ord5.pdf', format='pdf', dpi=1000)
