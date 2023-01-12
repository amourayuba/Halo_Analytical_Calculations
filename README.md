# halo_formation_time 
This project allows to calculate a variety of dynamical properties of dark matter halos, most notably their median formation time as a function of mass and redshift. 
The code can be fully autonomous or use CAMB power spectrum. It calculates the power spectrum from Eisenstein & Hu approximation, then the rms of the smoothed overdensity field and then uses extended press schechter formalism to calculate different age properties. 

All codes are parallelized as much as possible.

# Files 
## cosmo_parameters.py 
All necessary cosmological parameters and basic functions. This will be the fiducial cosmology. 
## power_spectrum_analytic,py 
Eisenstein % Hu fitting formula for the transfer function 
## fluctuation_rms.py 
Generates both the power_spectrum using the transfer function from power_spectrum_analytic,py and spectral index ns, and the rms of the smoothed overdensity field sigma(R). 
The file fluctuation_rms_camb.py does the exact same thing but using the CAMB generated power spectrum 

## halo_mass_function.py 
Contains functions related to the halo mass function, and the halo mass function itself for any cosmological parameter and redshift needed. Possible to use the publicly available code Colossus to do that as well or CAMB for the power spectrum too. 

## formation_time.py 
The key file of the project.
