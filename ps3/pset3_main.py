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
generate_plots = False

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

    data_loc = "./data/" # The directory containing all the data files
    fn_json = "BBH_events_v3.json"
    events = json.load(open(data_loc + fn_json,"r"))

    # First, generate noise models for Livingston and Hanford separately by averaging
    # over all the noise realizations for each event
    noise_models_H = []
    noise_models_L = []
    event_names = [e for e in events]
    for en in event_names:
        event = events[en]
        fn_H1 = event['fn_H1']
        fn_L1 = event['fn_L1']

        # Read in the data
        print 'Reading Hanford data file ' + fn_H1
        strain_H, dt_H, utc_H = rl.read_file(data_loc + fn_H1)
        print 'Reading Livingston data file ' + fn_L1
        strain_L, dt_L, utc_L = rl.read_file(data_loc + fn_L1)

        # Create a window to smooth the data
        n = len(strain_H)
        win_tukey = tukeywin(n, alpha=0.5)

        # Multiply the strain by the window to reduce the periodic discontinuity at the
        # edges of the data
        print "Window the data to avoid ringing"
        windowed_strain_H = strain_H * win_tukey
        windowed_strain_L = strain_L * win_tukey

        # Calculate the power spectrum of the data to estimate the noise
        print "Calculate the power spectrum of the data to estimate the noise"
        Fk_H = np.abs(np.fft.rfft(windowed_strain_H))**2
        Fk_L = np.abs(np.fft.rfft(windowed_strain_L))**2

        fft_freqs = np.fft.rfftfreq(n, d = 1. / 4096.)
        if generate_plots:
            # Plot the power spectrum of the data
            print "Plotting the power spectrum of the data"
            plt.plot(fft_freqs, Fk_H, lw=1, color='b', alpha=0.7)
            plt.gca().set_xscale("log")
            plt.gca().set_yscale("log")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Strain Power Spectrum")
            plt.title("Unsmoothed Noise Hanford " + en)
            plt.savefig('./noise_plots/' + fn_H1[:-5] + '.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
            plt.clf()
            
            plt.plot(fft_freqs, Fk_L, lw=1, color='g', alpha=0.7)
            plt.gca().set_xscale("log")
            plt.gca().set_yscale("log")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Strain Power Spectrum")
            plt.title("Unsmoothed Noise Livingston " + en)
            plt.savefig('./noise_plots/' + fn_L1[:-5] + '.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
            plt.clf()
        noise_models_H.append(Fk_H)
        noise_models_L.append(Fk_L)
    
    # Now average over all the noise realizations
    avg_noise_H = np.average(noise_models_H, axis=0)
    avg_noise_L = np.average(noise_models_L, axis=0)
    # Use a boxcar convolution to smooth each noise model
    print "Use a boxcar convolution to smooth each noise model"
    N_H = smooth_noise(avg_noise_H, len(avg_noise_H), 20, kernel_type='boxcar')
    N_L = smooth_noise(avg_noise_L, len(avg_noise_L), 20, kernel_type='boxcar')
    if generate_plots:
        # Plot the power spectrum of the data
        print "Plotting the smoothed average noise realizations"
        plt.plot(fft_freqs, N_H, lw=1, color='b', alpha=0.5)
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Strain Power Spectrum")
        plt.gca().set_xlim([10,1700])
        plt.title("Smoothed Average Noise Model Hanover")
        plt.savefig('./noise_plots/avg_noise_H.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
        plt.clf()
            
        plt.plot(fft_freqs, N_L, lw=1, color='g', alpha=0.5)
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Strain Power Spectrum")
        plt.gca().set_xlim([10,1700])
        plt.title("Smoothed Average Noise Model Hanover")
        plt.savefig('./noise_plots/avg_noise_L.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
        plt.clf()
    print

    # Compute the inverses of the noise models
    Ninv_sqrt_H = 1 / np.sqrt(N_H)
    Ninv_sqrt_L = 1 / np.sqrt(N_L)
    if generate_plots:
        # Plot the boxcar-smoothed inverse noise so we can identify and excise odd turnovers
        print "Plot the boxcar-smoothed inverse noise to identify odd turnovers"
        plt.plot(np.arange(len(Ninv_sqrt_H)), Ninv_sqrt_H, lw=1, color='b', alpha=0.5)
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.xlabel("Array Index")
        plt.ylabel("Noise Spectrum")
        plt.title("Smoothed Average Noise Model Hanover")
        plt.savefig('./noise_plots/' + 'Ninv_sqrt_H_plot.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
        plt.clf()
        
        plt.plot(np.arange(len(Ninv_sqrt_L)), Ninv_sqrt_L, lw=1, color='g', alpha=0.5)
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.xlabel("Array Index")
        plt.ylabel("Noise Spectrum")
        plt.title("Smoothed Average Noise Model Hanover")
        plt.savefig('./noise_plots/' + 'Ninv_sqrt_L_plot.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
        plt.clf()
    # Excise the weird turnovers
    Ninv_sqrt_H[:200] = 0.
    Ninv_sqrt_H[53000:] = 0.
    Ninv_sqrt_L[:200] = 0.
    Ninv_sqrt_L[53000:] = 0.

    snr_save_H = []
    snr_save_L = []
    noise_save_H = []
    noise_save_L = []
    halffreq_save_H = []
    halffreq_save_L = []
    # Now we can use our noise models to try and detect events in each dataset
    for en in event_names:
        event = events[en]
        fn_H1 = event['fn_H1']
        fn_L1 = event['fn_L1']
        
        # Read in the data
        print 'Reading Hanford data file ' + fn_H1
        strain_H, dt_H, utc_H = rl.read_file(data_loc + fn_H1)
        print 'Reading Livingston data file ' + fn_L1
        strain_L, dt_L, utc_L = rl.read_file(data_loc + fn_L1)

        # Create a window to smooth the data
        n = len(strain_H)
        win_tukey = tukeywin(n, alpha=0.5)

        # Multiply the strain by the window to reduce the periodic discontinuity at the
        # edges of the data
        print "Window the data to avoid ringing"
        windowed_strain_H = strain_H * win_tukey
        windowed_strain_L = strain_L * win_tukey

        print "Calculate the whitened strain"
        strainft_H = np.fft.rfft(windowed_strain_H)
        strainft_white_H = Ninv_sqrt_H * strainft_H
        strain_white_H = np.fft.irfft(strainft_white_H, n)
        strainft_L = np.fft.rfft(windowed_strain_L)
        strainft_white_L = Ninv_sqrt_L * strainft_L
        strain_white_L = np.fft.irfft(strainft_white_L, n)

        fn_template = event['fn_template']
        print 'Reading template ' + fn_template
        tl, template = rl.read_template(data_loc + fn_template)

        print "Calculate the whitened template"
        templateft = np.fft.rfft(template * win_tukey)
        templateft_white_H = Ninv_sqrt_H * templateft
        template_white_H = np.fft.irfft(templateft_white_H, len(template))
        templateft_white_L = Ninv_sqrt_L * templateft
        template_white_L = np.fft.irfft(templateft_white_L, len(template))

        print "Now actually calculate the matched filter"
        matched_filterft_H = strainft_white_H * np.conj(templateft_white_H)
        matched_filter_H = np.fft.irfft(matched_filterft_H, n)
        matched_filterft_L = strainft_white_L * np.conj(templateft_white_L)
        matched_filter_L = np.fft.irfft(matched_filterft_L, n)

        print "Finally, determine the signal to noise of the detection"
        denom_H = np.mean(template_white_H * template_white_H)
        snr_H = np.abs(matched_filter_H / np.sqrt(np.abs(denom_H)))
        denom_L = np.mean(template_white_L * template_white_L)
        snr_L = np.abs(matched_filter_L / np.sqrt(np.abs(denom_L)))

        snr_save_H.append(np.max(snr_H))
        snr_save_L.append(np.max(snr_L))
        noise_save_H.append(denom_H)
        noise_save_L.append(denom_L)
        
        if generate_plots:
            # Plot the resulting signal to noise
            print "Plot the resulting signal to noise"
            plt.plot(dt_H * np.arange(len(snr_H)), np.roll(snr_H, len(snr_H)/2), lw=1, color='b', alpha=0.5)
            plt.xlabel("Time (s)")
            plt.ylabel("SNR")
            plt.title(en + " Hanover, SNR=" + "{0:.2f}".format(snr_save_H[-1]))
            plt.savefig('./noise_plots/' + en + '_SNR_H.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
            plt.clf()

            plt.plot(dt_L * np.arange(len(snr_L)), np.roll(snr_L, len(snr_L)/2), lw=1, color='g', alpha=0.5)
            plt.xlabel("Time (s)")
            plt.ylabel("SNR")
            plt.title(en + " Livingston, SNR=" + "{0:.2f}".format(snr_save_L[-1]))
            plt.savefig('./noise_plots/' + en + '_SNR_L.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0)
            plt.clf()

        # For each event, find the frequency where half the weight comes from above that frequency and half below
        # I will assume that "weight" means that we have prewhitened the template with the noise model, and that
        # we look at the power spectrum of the whitened template to find the half-frequency
        ps_template_white_H = np.abs(templateft_white_H)**2 / np.max(np.abs(templateft_white_H)**2)
        total_power_H = np.sum(ps_template_white_H)
        iter_sum_H = 0.
        for i in range(len(ps_template_white_H)):
            iter_sum_H = iter_sum_H + ps_template_white_H[i]
            if iter_sum_H >= total_power_H / 2.:
                halffreq_save_H.append(fft_freqs[i])
                break
        ps_template_white_L = np.abs(templateft_white_L)**2 / np.max(np.abs(templateft_white_L)**2)
        total_power_L = np.sum(ps_template_white_L)
        iter_sum_L = 0.
        for i in range(len(ps_template_white_L)):
            iter_sum_L = iter_sum_L + ps_template_white_L[i]
            if iter_sum_L >= total_power_L / 2.:
                halffreq_save_L.append(fft_freqs[i])
                break

    print
    for i in range(len(event_names)):
        print "Event  " + event_names[i]
        print "SNR for Hanford: " + str(snr_save_H[i])
        print "SNR for Livingston: " + str(snr_save_L[i])
        print "Combined SNR: " + str(np.sqrt(snr_save_H[i]**2 + snr_save_L[i]**2)) # str(snr_save_H[i]**2 + snr_save_L[i]**2)
        print "Half-weight frequency for Hanford: " + str(halffreq_save_H[i]) + " Hz"
        print "Half-weight frequency for Livingston: " + str(halffreq_save_L[i]) + " Hz"
