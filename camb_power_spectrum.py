from __future__ import division
import camb
from camb import model


def camb_power_spectrum(z=[0], H0=67.5, ombh2=0.022, omch2=0.12, omk=0.0, cosmomc_theta=None,
                   thetastar=None, neutrino_hierarchy='degenerate',
                   num_massive_neutrinos=1, mnu=0.06, nnu=3.046, YHe=None,
                   meffsterile=0.0, standard_neutrino_neff=3.046, TCMB=2.7255,
                   tau=None, deltazrei=None, Alens=1.0, bbn_predictor=None,
                   theta_H0_range=[10, 100], ns=0.965, kmax=2, nonlinear=True,
                   linear=False, npoints=10000):
    """Provides the power spectrum using CAMB code
    Variables :
    redshits : list. list of redshifts of wanted Power spectra
    cosmological parameters in camb.set_cosmo (see camp.set_cosmo for reference
    ns : float initial power spectrum index
    kmax : max value of wave number for calculating the power spectrum
    nonlinear/linear : Boolean, if wanting the nonlinear/linear power spectrum for result
    npoints : number of desired points for the power spectrum restult"""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0, ombh2, omch2, omk, cosmomc_theta, thetastar, neutrino_hierarchy,
                       num_massive_neutrinos, mnu, nnu, YHe, meffsterile, standard_neutrino_neff, TCMB,
                       tau, deltazrei, Alens, bbn_predictor, theta_H0_range)

    pars.InitPower.set_params(ns=ns)
    pars.set_matter_power(redshifts=z, kmax=kmax)
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    if not nonlinear:
        kh, z, pk = results.get_matter_power_spectrum(minkh=5e-5, maxkh=kmax, npoints=npoints)
        return kh, z, pk
    elif not linear:
        pars.NonLinear = model.NonLinear_both
        results.calc_power_spectra(pars)
        kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=5e-5, maxkh=kmax, npoints=npoints)
        return kh_nonlin, z_nonlin, pk_nonlin
    else:
        kh, z, pk = results.get_matter_power_spectrum(minkh=5e-5, maxkh=kmax, npoints=npoints)

        pars.NonLinear = model.NonLinear_both
        results.calc_power_spectra(pars)
        kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=5e-5, maxkh=kmax, npoints=npoints)
        return ([kh, z, pk], [kh_nonlin, z_nonlin, pk_nonlin])
