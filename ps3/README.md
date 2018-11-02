# PHYS 641 - Problem Set #4

This directory contains all my code for Problem Set #4

## Directory contents
* **hw4.pdf** : contains a discussion of my methods with illustrative plots
* **pset4_main.py** : the main code script for this problem
  * To run **pset3_main.py** and get the results, you first need to set the variable *data_loc* to the location of all of the example datasets, templates, and the *BBH_events_v3.json* file
  * **pset3_main.py** also requires that the *simple_read_ligo.py* script is in the same directory
  * **pset3_main.py** reports the SNR of each event detection at each detector, the combined SNR from both detectors, and the frequency from each event where half the weight comes from above that frequency and half below
* **pset4_plots.py** : an addition script that completes analysis for a single event and outputs many useful plots that we present in our report

## Authors

* **Bridget Andersen**

## Acknowledgments

* Much of the derivations were expanded upon the class notes.
* We got our Tukey function code from: https://leohart.wordpress.com/2006/01/29/hello-world/