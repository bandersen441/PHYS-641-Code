# Bridget Andersen, 10/24/18
# PHYS 641, Problem Set #3
#########################################################
# This script completes a simple LIGO analysis on one
# input dataset and makes a bunch of useful plots.
#########################################################
import numpy as np
import simple_read_ligo as rl
import json
from matplotlib import pyplot as plt
from matplotlib import rc,rcParams
from matplotlib.ticker import ScalarFormatter

rc('text', usetex=True)
generate_plots = True

# This function creates a Tukey or tapered cosine window function of length
# window_length, and width controled by alpha
# Note that I stole this function from: https://leohart.wordpress.com/2006/01/29/hello-world/
def tukeywin(window_length, alpha=0.5):
    # Special cases
    if alpha <= 0:
        return np.ones(window_length) #rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)
 
    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)
 
    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))
 
    # second condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2))) 
 
    return w

# This function makes a Gaussian smoothing kernel of a given fwhm
def make_gaussian_kernel(n, fwhm):
    # Convert the fwhm into a sigma value
    # See https://brainder.org/2011/08/20/gaussian-kernels-convert-fwhm-to-sigma/) for a derivation
    # of the hardcoded numbers here
    sig = fwhm / np.sqrt(8. * np.log(2.))
    # Make normalized Gaussian smoothing kernel
    x = np.arange(n)
    # Make kernel symmetric
    x[n/2:] = x[n/2:] - n
    gauss = np.exp(-0.5 * x**2 / sig**2)
    gauss = gauss / np.sum(gauss)
    return gauss

# This function makes a boxcar smoothing kernel of a given width
def make_boxcar_kernel(n, width):
    boxcar = np.zeros(n)
    half_width = width / 2
    boxcar[:half_width] = 1.
    boxcar[-half_width:] = 1.
    boxcar = boxcar / np.sum(boxcar)
    return boxcar

# This function takes in the noise estimate vector and smooths it by convolution
# with a Gaussian kernel with the given fwhm
def smooth_noise(noise_estimate, n, width, kernel_type='gauss'):
    # n = len(noise_estimate)
    if kernel_type == 'boxcar':
        kern = make_boxcar_kernel(n, width)
    else:
        kern = make_gaussian_kernel(n, width)
    # Convolve the smoothing kernel with the noise estimate
    noise_estimateft = np.fft.rfft(noise_estimate)
    kernft = np.fft.rfft(kern)
    noise_estimate_smooth = np.fft.irfft(kernft * noise_estimateft, n)
    return noise_estimate_smooth

if __name__ == '__main__':
    print "Problem #1\n"
    # In this plots file we examine only GW150914
    eventname = "GW150914"
    data_loc = "./data/" # location of the data files
    
    # Read in the data
    fn_json = "BBH_events_v3.json"
    events = json.load(open(data_loc + fn_json,"r"))
    event = events[eventname]
    fn_H1 = event['fn_H1']
    fn_L1 = event['fn_L1']
    fn_template = event['fn_template']
    # fs = event['fs']
    # tevent = event['tevent']
    # fband = event['fband']
    
    # fname='H-H1_LOSC_4_V2-1126259446-32.hdf5'
    print 'Reading data file ' + fn_H1
    strain, dt, utc=rl.read_file(data_loc + fn_H1)
    
    # Read in the template
    # template_name='GW150914_4_template.hdf5'
    print 'Reading template ' + fn_template
    tl, template = rl.read_template(data_loc + fn_template)
    
    # Create a window to smooth the data
    n = len(strain)
    win_tukey = tukeywin(n, alpha=0.5)
    
    # Multiply the strain by the window to reduce the periodic discontinuity at the
    # edges of the data
    print "Window the data to avoid ringing"
    windowed_strain = strain * win_tukey
    if generate_plots:
        # Plot the non-windowed and windowed strain
        print "Plotting the non-windowed and windowed strain"
        plt.plot(np.arange(len(strain)) * dt, strain, lw=1, color='b', alpha=0.5, label='original')
        plt.plot(np.arange(len(windowed_strain)) * dt, windowed_strain, lw=1, color='r', alpha=0.5, label='windowed')
        plt.plot(np.arange(len(win_tukey)) * dt, win_tukey * np.max(strain), lw=1, ls='--', color='black', label='window function')
        #plt.gca().set_xlim([10**(1),2.*10**(3)])
        #plt.gca().set_ylim([10**(-42),10**(-28)])
        plt.gca().legend(loc='upper right')
        plt.xlabel("Time (s)")
        plt.ylabel("Strain")
        plt.title("Non-windowed and Windowed Strain for " + eventname + " at Hanover")
        plt.savefig('./explanation_plots/' + 'windowed_strain.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
        plt.clf()

    # Calculate and plot the fft of the windowed strain
    print "Calculate the power spectrum of the data to estimate the noise"
    Fk = np.abs(np.fft.rfft(strain * win_tukey))**2
    #check = np.fft.rfft(strain * win_tukey)
    fft_freqs = np.fft.rfftfreq(n, d = 1. / 4096.)
    if generate_plots:
        # Plot the power spectrum of the data
        print "Plotting the power spectrum of the data"
        plt.plot(fft_freqs, Fk, lw=1, color='b', alpha=0.7)
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        #plt.gca().set_xlim([10**(1),2.*10**(3)])
        #plt.gca().set_ylim([10**(-42),10**(-28)])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Strain Power Spectrum")
        plt.title("Power spectrum for " + eventname + " at Hanover")
        plt.savefig('./explanation_plots/' + 'strain_power_spectrum.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
        plt.clf()

    # Compare the smoothed noise using the boxcar and the Gaussian by zooming into a
    # particular section of the noise to examine how it smooths
    N_boxcar = smooth_noise(Fk, len(Fk), 20, kernel_type='boxcar')
    if generate_plots:
        print "Plot the smoothed noise for boxcar and gaussian to compare"
        N_gauss = smooth_noise(Fk, len(Fk), 20, kernel_type='gauss')
        plt.plot(fft_freqs, N_boxcar, lw=1, color='b', alpha=0.5, label='boxcar')
        plt.plot(fft_freqs, N_gauss, lw=1, color='r', alpha=0.5, label='gaussian')
        plt.plot(fft_freqs, Fk, lw=1, color='black', alpha=0.5, label='original')
        #plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.gca().set_xlim([327.5,337.5])
        plt.gca().set_ylim([10**(-41),10**(-32)])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Strain Power Spectrum")
        plt.title("Smoothed Noise Using Boxcar and Gaussian for " + eventname + " at Hanover")
        plt.gca().legend(loc='upper right')
        plt.savefig('./explanation_plots/' + 'noise_smooth_compare.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
        plt.clf()

    # The boxcar seems to broaden spectral features less than the gaussian, so we will go with the
    # boxcar-smoothed noise model
    N = N_boxcar
    Ninv = 1 / N
    Ninv_sqrt = 1 / np.sqrt(N)
    if generate_plots:
        # Plot the boxcar-smoothed inverse noise so we can identify and excise odd turnovers
        print "Plot the boxcar-smoothed inverse noise to identify odd turnovers"
        plt.plot(np.arange(len(Ninv_sqrt)), Ninv_sqrt, lw=1, color='b', alpha=0.5)
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        #plt.gca().set_xlim([327.5,337.5])
        #plt.gca().set_ylim([10**(-41),10**(-32)])
        plt.xlabel("Array Index")
        plt.ylabel("Boxcar Noise Spectrum")
        plt.title("Boxcar Smoothed Inverse Noise for " + eventname + " at Hanover")
        plt.savefig('./explanation_plots/' + 'Ninv_sqrt_plot.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
        plt.clf()
    # Excise the weird turnovers
    Ninv_sqrt[:200]=0
    Ninv_sqrt[53000:]=0
    if generate_plots:
        # Re-plot the boxcar-smoothed inverse noise
        print "Re-plotting the boxcar-smoothed inverse noise"
        plt.plot(np.arange(len(Ninv_sqrt)), Ninv_sqrt, lw=1, color='b', alpha=0.5)
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.gca().set_xlim([200,53000])
        # plt.gca().set_ylim([10**(-41),10**(-32)])
        plt.xlabel("Array Index")
        plt.ylabel("Boxcar Noise Spectrum")
        plt.title("Boxcar Smoothed Inverse Noise for " + eventname + " at Hanover")
        plt.savefig('./explanation_plots/' + 'excised_Ninv_sqrt_plot.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
        plt.clf()

    # Calculate the fft of the windowed strain and prewhiten it
    print "Calculate the whitened strain"
    strainft = np.fft.rfft(windowed_strain)
    strainft_white = Ninv_sqrt * strainft
    strain_white = np.fft.irfft(strainft_white, n)
    if generate_plots:
        # Plot the fft of the strain before and after whitening
        print "Plotting the fft of the strain before and after whitening"
        plt.plot(fft_freqs[200:53000], np.abs(strainft[200:53000])**2 / np.max(np.abs(strainft[200:53000])**2), lw=1, color='b', alpha=0.5, label='before whitening')
        plt.plot(fft_freqs[200:53000], np.abs(strainft_white[200:53000])**2 / np.max(np.abs(strainft_white[200:53000])**2), lw=1, color='r', alpha=0.5, label='after whitening')
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.gca().legend(loc='lower right')
        #plt.gca().set_xlim([327.5,337.5])
        #plt.gca().set_ylim([10**(-41),10**(-32)])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Normalized Strain Data Power Spectrum")
        plt.title("Power Spectrum of Strain Before and After Whitening for " + eventname + " at Hanover")
        plt.savefig('./explanation_plots/' + 'strain_whitening_compare.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
        plt.clf()

    # Now prewhiten the template
    print "Calculate the whitened template"
    templateft = np.fft.rfft(template * win_tukey)
    templateft_white = Ninv_sqrt * templateft
    #templateft_white[:200] = 0.
    #templateft_white[7000:] = 0.
    template_white = np.fft.irfft(templateft_white, len(template))
    if generate_plots:
        # Plot the whitened template
        print "Plot the whitened template"
        plt.plot(np.arange(len(template)) * dt, template_white, lw=1, color='r', alpha=0.5)
        # plt.gca().set_xscale("log")
        # plt.gca().set_yscale("log")
        plt.gca().set_xlim([0,17])
        #plt.gca().set_ylim([10**(-41),10**(-32)])
        plt.xlabel("Time (s)")
        plt.ylabel("Strain")
        plt.title("Whitened Template for " + eventname)
        plt.savefig('./explanation_plots/' + 'template_white.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
        plt.clf()

    if generate_plots:
        # Plot the fft of the template before and after whitening
        print "Plot the fft of the template before and after whitening"
        plt.plot(fft_freqs[200:7000], np.abs(templateft[200:7000])**2 / np.max(np.abs(templateft[200:7000])**2), lw=1, color='b', alpha=0.5, label='before whitening')
        plt.plot(fft_freqs[200:7000], np.abs(templateft_white[200:7000])**2 / np.max(np.abs(templateft_white[200:7000])**2), lw=1, color='r', alpha=0.5, label='after whitening')
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.gca().legend(loc='lower right')
        #plt.gca().set_xlim([327.5,337.5])
        #plt.gca().set_ylim([10**(-41),10**(-32)])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Normalized Strain Template Power Spectrum")
        plt.title("Whitened Template for " + eventname)
        plt.savefig('./explanation_plots/' + 'template_whitening_compare.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
        plt.clf()
        
        
    # Now actually calculate the matched filter
    print "Now actually calculate the matched filter"
    matched_filterft = strainft_white * np.conj(templateft_white)
    matched_filter = np.fft.irfft(matched_filterft, n)

    # Determine the signal to noise
    # Calculate denominator in fft space and then invert
    print "Finally, determine the signal to noise of any detection"
    denom = np.mean(template_white * template_white)
    snr = np.abs(matched_filter / np.sqrt(np.abs(denom)))
    if generate_plots:
        # Plot the resulting signal to noise
        print "Plot the resulting signal to noise"
        plt.plot(dt * np.arange(len(snr)), np.roll(snr, len(snr)/2), lw=1, color='b', alpha=0.5)
        #plt.gca().set_xlim([327.5,337.5])
        #plt.gca().set_ylim([10**(-41),10**(-32)])
        plt.xlabel("Time (s)")
        plt.ylabel("SNR")
        plt.title("SNR for " + eventname + " at Hanover")
        plt.savefig('./explanation_plots/' + 'SNR.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
        plt.clf()
