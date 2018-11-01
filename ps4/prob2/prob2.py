# Bridget Andersen, 10/24/18
# PHYS 641, Problem Set #4
####################### PROBLEM 2 #######################
# This script completes several computations related to
# testing different properties of some given spherical
# power spectrum data.
#########################################################
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc,rcParams
from matplotlib.ticker import ScalarFormatter
import healpy

rc('text', usetex=True)
fs=18
generate_plots = False

# This function reads in the example power spectrum, optionally plots it,
# stores it in an array, and returns the array and matching l values
def read_power_spectrum(filename='./example_ps.txt', plot=False):
    # Read in the data
    data = np.loadtxt(filename)
    # Keep just the power spectrum
    ps = data[:,0]
    # Generate matching l values
    ls = np.arange(len(ps))
    if plot:
        print "Plotting the power spectrum"
        # Plot the power spectrum
        plt.plot(ls, ps, lw=1.5, color='r')
        plt.xlabel("$\ell$")
        plt.ylabel("$\ell (\ell + 1) C_{\ell} / 2 \pi$")
        plt.savefig('power_spectrum.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0.2)
        plt.clf()
    return ls, ps

if __name__ == '__main__':
    print "Problem #2\n"

    # Read in the data
    print "Reading in the data"
    ls, ps = read_power_spectrum(plot=generate_plots)
    l_max = int(np.max(ls))

    # Convert to C_l values
    # Note that the original data was given in terms of:
    # l (l + 1) C_l / 2PI
    C_ls = 2 * np.pi * ps / (ls * (ls + 1))
    C_ls[0] = 0. # Change l=0 value to something reasonable
    C_ls[1] = 0. # Change l=1 value to something reasonable
    
    # Our grid size in degrees
    x = 20.
    # Find the k_max corresponding to the l_max of our data
    k_max = x * l_max / 360.
    # Create an array of ks
    k_len = int(k_max / np.sqrt(2.)) + 1
    k_x = np.arange(2 * k_len)
    # Make the kx's and ky's symmetric about (0,0) so that the  2DFFT behaves nicely later (still only up to k_max)
    n = 2 * k_len - 1
    k_x[k_x>n/2] = k_x[k_x>n/2] - n
    print k_x
    k_x = np.repeat([k_x], len(k_x), axis=0)
    k_y = k_x.transpose()
    ks = np.sqrt(k_x**2 + k_y**2)
    # Calculate the l-values that would fit a complete period within 20 deg
    l_fit = 360. * ks / x
    # Pick out the closest l-values that we have C_ls for by rounding our previous answer
    l_closest = np.array(np.round(l_fit), dtype=int)
    # Pick out the corresponding C_ls to get our C_ks
    C_ks = C_ls[l_closest]
    C_ks = C_ks * l_max / float(k_max)

    # Now decompose our matrix into a 1D array so we can plot the power spectrum
    if generate_plots:
        ks_arr = ks.flatten()
        C_ks_arr = C_ks.flatten()
        ks_zip = zip(ks_arr, C_ks_arr)
        # Take only the unique elements
        unique_ks_zip = set(ks_zip)
        # Sort them
        sorted_ks_zip = sorted(unique_ks_zip)
        # Unzip them for plotting
        ks_arr, C_ks_arr = zip(*sorted_ks_zip)
        print "Plotting C_k values"
        # Plot the C_k values
        plt.plot(ks_arr, C_ks_arr, lw=1.5, color='r')
        plt.xlabel("$k$")
        plt.ylabel("$C_{k}$")
        plt.yscale('log')
        plt.savefig('Ck.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0.2)
        plt.clf()
    if generate_plots:
        # Plot the C_l values
        plt.plot(ls, C_ls, lw=1.5, color='r')
        for l in np.unique(l_closest.flatten()):
            plt.axvline(x=l, lw=0.02, color='b', alpha=0.1)
        plt.xlabel("$\ell$")
        plt.ylabel("$C_{\ell}$")
        plt.yscale('log')
        plt.savefig('Cl_choose.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0.2)
        plt.clf()

    # Now use C_ks to simulate F(k)'s to generate data that we can 2D FFT into a map
    # Note that we can simulate these F(k)'s in the same way that we simulated alm's since:
    # <|F(k)|^2> = C_k like <|a_lm|^2> = C_l
    # Generate Gaussian random numbers with appropriate phase relations to be the transform
    # of a real field by taking the fourier transform of a real field
    map = np.random.normal(size=C_ks.shape)
    mapft = np.fft.fft2(map)
    
    # Scale by sqrt(C_ks)
    mapft_scale = mapft * 1. / 2.**0.5 * np.sqrt(C_ks)
    # Calculate the maximum number of pixels to use
    pix_size = 360. / l_max
    n_pix = int(20. / pix_size)
    print l_max
    print pix_size
    print n_pix
    # 2D FFT to produce map!
    map_back = np.real(np.fft.ifft2(mapft_scale))
    map_back = map_back * map_back.shape[0] # Multiply by number of pixels in one dimension to get uK units
    print map_back.shape
    print k_max
    print k_len
    if generate_plots:
        print "Plotting the map"
        # Plot the generated map
        im_map = plt.imshow(map_back, extent=[0,20,20,0], vmin=np.min(map_back), vmax=np.max(map_back))
        plt.xlabel("degrees")
        plt.ylabel("degrees")
        cbar = plt.gcf().colorbar(im_map, ax=plt.gca())
        cbar.set_label("$\mu$K", rotation=0)
        plt.savefig('map.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0.2)
        plt.clf()
    map_var = np.var(map_back)
    print "Generated map variance: " + str(map_var)

    # Calculate the expected variance
    pred_var = np.sum(C_ks) / 2.
    print "Predicted map variance: " + str(pred_var)
    # Calculate the error between the predicted variance and the generated variance in this problem
    print "Error between predicted and generated variance: " + "{0:.2f}".format(abs(map_var - pred_var) / abs((map_var + pred_var) / 2.) * 100) + "%"

    # Calculate the error between the predicted variance from Problem #1 and the generated variance in this problem
    pred_var_prob1 = 12077.8265144
    print "Problem #1 predicted map variance: " + str(pred_var_prob1)
    print "Error between predicted from Problem #1 and generated variance: " + "{0:.2f}".format(abs(map_var - pred_var_prob1) / abs((map_var + pred_var_prob1) / 2.) * 100) + "%"
