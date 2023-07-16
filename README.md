# Analytical Calculations of Halo Mass Function and Age 
This project allows to calculate a variety of dynamical properties. Different files and functions allow contain 
analytical predictions of a variety of cosmological properties, including the power-spectrum, growth 
factor, distribution as a function of mass and redshift and formation time.
 
The code can be fully autonomous or use CAMB power spectrum. It calculates the power spectrum from Eisenstein & Hu approximation, 
then the rms of the smoothed overdensity field and then uses extended press schechter formalism to calculate different age properties. 
It can also use Colossus mass function, it requires to downland Colossus.

All codes are parallelized as much as possible.

# Files 
## cosmo_parameters.py 
All necessary cosmological parameters and basic functions. This will be the fiducial cosmology. 
Modify this file for any change in cosmological parameter. 

## power_spectrum_analytic,py 
Eisenstein % Hu fitting formula for the transfer function. 

## fluctuation_rms.py 
Generates both the power_spectrum using the transfer function from power_spectrum_analytic,py and spectral index ns, and the rms of the smoothed overdensity field sigma(R). 
The file fluctuation_rms_camb.py does the exact same thing but using the CAMB generated power spectrum 

## halo_mass_function.py 
Contains functions related to the halo mass function, and the halo mass function itself for any cosmological parameter and redshift needed. Possible to use the publicly available code Colossus to do that as well or CAMB for the power spectrum too. 

## formation_time.py 
Functions that calculate the probability distributions of formation times as well as median and average ages. 

