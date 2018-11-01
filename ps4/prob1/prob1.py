# Bridget Andersen, 10/24/18
# PHYS 641, Problem Set #4
####################### PROBLEM 1 #######################
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
def read_power_spectrum(filename='./example_ps.txt', plot=True):
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

# This function takes in series of C_l values and returns the predicted variance
# of the corresponding map
def calc_map_variance(ls, C_ls):
    var_map = np.sum(C_ls * (2. * ls + 1))
    return var_map / (4. * np.pi)

# This function contains a crappy (space-wise, but reasonable time-wise) way of generating the a_lms
# to feed into healpy.alm2map() for the input power series
def generate_alms(ls, C_ls):
    print "Generating alm values"
    # Create a matrix containing a row for each l value
    # Each row is populated by corresponding m values, padded by zeros
    # (I know this is inefficient in space, but I'm gonna go for it and see how long it takes. Update, not too long.)
    num_ls = len(ls)
    alm_matrix = np.full((num_ls, num_ls), fill_value=-10., dtype=np.complex_)
    for l in ls:
        #print "Generating for l=" + str(l)
        # Generate real and imaginary components for the given a_lm by sampling from a
        # Gaussian distribution with standard deviation is sqrt(C_l)
        std = np.sqrt(C_ls[l])
        m_max = l
        if (m_max == 0) or (m_max == 1):
            real = np.zeros(m_max+1)
            imaginary = np.zeros(m_max+1)
        else:
            real = 1. / 2.**0.5 * np.random.normal(loc=0., scale=std, size=m_max+1)
            imaginary = 1. / 2.**0.5 * np.random.normal(loc=0., scale=std, size=m_max+1)
        # Make the m=0 term real
        imaginary[0] = 0.
        # Input generated numbers into
        alm_matrix[l,:m_max+1] = real + imaginary * 1j
    # Put a_lm values into the order that healpy.alm2map() wants
    # WHO DECIDED ON THIS ORDERING???
    l_max = int(np.max(ls))
    n_lm = (l_max + 1) * (l_max + 2) / 2
    input_alms = np.zeros(n_lm, dtype=np.complex_)
    # Slice through each m value, clip off extra filler values and add them to the input_alms array
    begin_index = 0
    for m in ls:
        #print "Slicing off values for m=" + str(m)
        ms = alm_matrix[:,m]
        indices = np.argwhere(ms == -10.)
        ms = np.delete(ms, indices)
        num_ms = len(ms)
        input_alms[begin_index:begin_index + num_ms] = ms
        begin_index = begin_index + num_ms
    return input_alms

if __name__ == '__main__':
    print "Problem #1\n"
    
    # Read in the data
    print "Reading in the data"
    ls, ps = read_power_spectrum(plot=generate_plots)
    l_max = int(np.max(ls))
    print "l_max = " + str(l_max)
    
    # Convert to C_l values
    # Note that the original data was given in terms of:
    # l (l + 1) C_l / 2PI
    C_ls = 2 * np.pi * ps / (ls * (ls + 1))
    C_ls[0] = 0. # Change l=0 value to something reasonable
    C_ls[1] = 0. # Change l=1 value to something reasonable
    if generate_plots:
        print "Plotting C_l values"
        # Plot the C_l values
        plt.plot(ls, C_ls, lw=1.5, color='r')
        plt.xlabel("$\ell$")
        plt.ylabel("$C_{\ell}$")
        plt.yscale('log')
        plt.savefig('Cl.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0.2)
        plt.clf()

    # Now calculate the map variance based on the power spectrum
    predicted_map_var = calc_map_variance(ls, C_ls)
    print "Predicted map variance based on the power spectrum: " + str(predicted_map_var)

    # Calculate the number of a_lm values that we need to generate
    # Generate a_lm values
    input_alms = generate_alms(ls, C_ls)
    # Input values into healpy.alm2map() to make a map and plot it!
    print "Creating a map from generated alm values"
    nside = l_max / 4
    map = healpy.alm2map(input_alms, nside)
    if generate_plots:
        print "Plotting generated map"
        healpy.mollview(map, unit="$\mu$K")
        plt.title("Generated Map with Non-Healpy $a_{\ell m}$'s")
        plt.savefig('generated_map.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0.2)
        plt.clf()
    generated_map_var = np.var(map)
    print "Generated map variance: " + str(generated_map_var)
    var_err = np.abs(generated_map_var - predicted_map_var) / np.abs((generated_map_var + predicted_map_var) / 2.) * 100. 
    print "Error between predicted and generated variance: " + "{0:.2f}".format(var_err) + "%"

    # Convert the generated map into a power spectrum (C_ls) using healpy.anafast()
    print "Convert the generated map into power spectrum using healpy.anafast()"
    generated_Cls = healpy.anafast(map,lmax=l_max)
    if generate_plots:
        # Compare to the original input C_ls
        plt.plot(ls, C_ls/np.max(C_ls), lw=1.5, color='r', alpha=0.5, label="Original")
        plt.plot(ls, generated_Cls/np.max(generated_Cls), lw=1.5, color='b', alpha=0.5, label="Generated")
        plt.xlabel("$\ell$")
        plt.ylabel("Normalized $C_{\ell}$")
        plt.yscale('log')
        plt.gca().legend(loc='upper right')
        plt.savefig('Cl_compare.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0.2)
        plt.clf()

    # Now complete the same analysis but using healpy.synfast() to generate the map from the input Cl's
    print "Creating a map from healpy.synfast()"
    map = healpy.synfast(C_ls, nside)
    if generate_plots:
        print "Plotting generated map"
        healpy.mollview(map, unit="$\mu$K")
        plt.title("Generated Map with healpy.synfast()")
        plt.savefig('generated_map_2.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0.2)
        plt.clf()
    generated_map_var = np.var(map)
    print "Generated map variance: " + str(generated_map_var)
    var_err = np.abs(generated_map_var - predicted_map_var) / np.abs((generated_map_var + predicted_map_var) / 2.) * 100. 
    print "Error between predicted and generated variance: " + "{0:.2f}".format(var_err) + "%"

    # Convert the generated map into a power spectrum (C_ls) using healpy.anafast()
    print "Convert the generated map into power spectrum using healpy.anafast()"
    generated_Cls = healpy.anafast(map,lmax=l_max)
    if generate_plots:
        # Compare to the original input C_ls
        plt.plot(ls, C_ls/np.max(C_ls), lw=1.5, color='r', alpha=0.5, label="Original")
        plt.plot(ls, generated_Cls/np.max(generated_Cls), lw=1.5, color='b', alpha=0.5, label="Generated")
        plt.xlabel("$\ell$")
        plt.ylabel("Normalized $C_{\ell}$")
        plt.yscale('log')
        plt.gca().legend(loc='upper right')
        plt.savefig('Cl_compare_2.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0.2)
        plt.clf()
