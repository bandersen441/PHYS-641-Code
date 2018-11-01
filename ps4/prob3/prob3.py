# Bridget Andersen, 10/24/18
# PHYS 641, Problem Set #4
####################### PROBLEM 3 #######################
# This script generates a few plots to help explain
# problem 3.
#########################################################
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc,rcParams
from matplotlib.ticker import ScalarFormatter
import healpy

rc('text', usetex=True)
fs=18

N = .6*10**(12)
#N = 10**(10)
v = np.linspace(10**(-5), N, 500)

h = 6.626*10**(-27)
c = 2.99*10**(10)
k = 1.38*10**(-16)
T = 2.725

Bv = (2. * h * v**3 / c**2) / (np.exp(h * v / (k * T)) - 1)
RJ = 2* k * T * v**2 / c**2

plt.plot(v, Bv, lw=1.5, color='b', alpha=0.5, label='Blackbody')
plt.plot(v, RJ, lw=1.5, color='g', alpha=0.5, label='Rayleigh-Jeans')
plt.xlabel("$v$")
plt.ylabel("$B_{v}$")
#plt.axvline(x=150*10**9, lw=1.5, color='r', alpha=0.5, label='150 GHz')
plt.gca().axvspan(150*10**9 - 30*10**9, 150*10**9 + 30*10**9, alpha=0.5, color='red', label='150GHz Detector Bandwidth')
plt.gca().set_ylim([0,4*10**(-15)])
plt.gca().legend(loc='upper right')
plt.savefig('BB_plot.pdf', format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0.2)
