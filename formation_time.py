from halo_mass_function import *
from scipy.integrate import quad


def upcrossing(M1, M2, z1, z2, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000, om0=om,
               ol0=oml, omb=omb, camb=False):
    w1 = delta_c(z1, om0, ol0)
    w2 = delta_c(z2, om0, ol0)
    S1 = sigma(M1, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb)**2
    S2 = sigma(M2, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb)**2
    dw = w1-w2
    dS = S1-S2
    return np.exp(-dw**2/(2*dS))*dw/np.sqrt(2*np.pi*dS**3)


def formation_time(M2, z1, z2, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000, acc = np.int(1e4), om0=om,
                   ol0=oml, omb=omb, camb=False):
    #Mass = np.logspace(np.log10(M2/2), np.log10(M2), acc)
    Mass = np.linspace(M2/2, M2, acc)
    sigs = sigma(Mass, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb)**2

    #M1 = np.sqrt(Mass[2:]*Mass[:-2])
    M1 = (Mass[2:]+Mass[:-2])/2
    dsig = -(sigs[2:] - sigs[:-2])
    fS = upcrossing(M1, M2, z1, z2, sig8, h, kmax, window, prec, om0, ol0, omb, camb)

    return M2*np.sum(fS*dsig/M1)



def K(ds, dw, model='sheth', A=0.27, a = 0.707, p=0.3):
    if model == 'sheth' :
        ndw = np.sqrt(a)*dw
        return A*(1 + (ds/ndw**2)**p)*ndw*np.exp(-ndw**2/(2*ds))/np.sqrt(2*np.pi*ds**3)
    else:
        return dw*np.exp(-dw**2/(2*ds))/np.sqrt(2*np.pi*ds**3)
def mu(St, a):
    return (1 + (2**a -1)*St)**(1/a)

def f_ec(S1, S0, w1, w0):
    dw = w1-w0
    dS = S1-S0
    nu0 = w0**2/S0
    A0 = 0.8661*(1-0.133*nu0**(-0.615))
    A1 = 0.308*nu0**(-0.115)
    A2 = 0.0373*nu0**(-0.115)
    Sbar = dS/S0
    A3 = A0**2 + 2*A0*A1*np.sqrt(dS*Sbar)/dw
    return A0*(2*np.pi)**(-0.5)*dw*dS**(-1.5)*\
           np.exp(-0.5*A1**2*Sbar)*(np.exp(-0.5*A3*dw**2/dS)+A2*Sbar**1.5*(1+2*A1*np.sqrt(Sbar/np.pi)))
    #return 2*A0*(2*np.pi)**(-0.5)*\
    #       np.exp(-0.5*A1**2*Sbar)*(np.exp(-0.5*A3*dw**2/dS)+A2*Sbar**1.5*(1+2*A1*np.sqrt(Sbar/np.pi)))
def f_sc(S1, S0, w1, w0):
    dw = w1-w0
    ds = S1-S0
    return (2*np.pi)**-0.5*dw*ds**(-1.5)*np.exp(-0.5*dw**2/ds)
def proba(M,  zf,  frac=0.5, acc=1000, zi=0.0, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000, om0=om,
          ol0=oml, omb=omb, camb=False, model='press', colos=False, A=0.322, a=0.707, p=0.3):
    """

    :param M: float. mass of the halo considered
    :param zf: float or ndarray. formation redshift(s) at which we want to calculate the probability
    :param frac: float between 0 and 1. Fraction of mass to define formation redshift. Default :0.5
    :param acc: int. Number redshift steps. Default : 1000.
    :param zi: float. Observed redshift of considered halo. Default :0.
    :param sig8: float : sigma 8 cosmo parameter
    :param h: float : H0/100 cosmo parameter
    :param kmax: float or int : maximum wavenumber for CAMB power spectrum.
    :param window:  str : type of smoothing window function. either "TopHat", "Gauss" or k-Sharp'.
    :param prec: int : number of bins for integral calculations.
    :param om0: float : fraction matter density
    :param ol0: float : fraction dark energy density
    :param omb: float : fraction baryon density
    :param camb:  boolean : if using camb spectrum or analytical version of Eisenstein and Hu.
    :param model: if Press&Schechter mass function "press" or sheth &tormen "sheth"
    :param colos: :param Colos : boolan : using Colossus halo mass function or not
    :param A: float. normalisation in Sheth & Tormen fonctional
    :param a: float. multiplication scaling of peak height in sheth & tormen fonctional
    :param p: float. Power of peak heigh in Sheth & Tormen halo mass function
    :return: Probability density function of redshift at which halos had x fraction of their mass
    """
    S0 = sigma(M, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)**2 #variance of the field at mass M
    Sh = sigma(M*frac, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)**2 # variance at mass frac*M
    w0 = delta_c(zi, om0, ol0)  #critical density at observed redshift
    if type(zf) == np.ndarray:  #for probability distribution
        mass = np.logspace(np.log10(M*frac), np.log10(M), acc)  #masses to calculate the integral
        l = len(zf)   #number of steps in PDF
        mat_zf = np.array([zf]*acc)   #(acc, l)
        S = sigma(mass, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)**2 #variance of all masses (acc,)
        # masses for integral
        mat_mass = np.array([mass]*l).transpose() #duplicating mass array to vectorize calculations  (acc, l)
        mat_S = np.array([S]*l).transpose()  #duplicating var array to vectorize calculations  (acc, l)
        mat_wf = delta_c(mat_zf, om0, ol0)   # critical density for all z_f
        mat_wt = (mat_wf-w0)/np.sqrt(Sh-S0)  #normalisation following lacey&Cole eq 2.27
        #mat_S = sigma(mat_mass, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)**2
        mat_St = (mat_S - S0)/(Sh - S0)   # normalisation following lacey&Cole eq 2.27 (acc, l)
        mat_St[-1, :] = 1e-10             # to avoid integration problems and deviding by 0
        mat_Ks = K(mat_St, mat_wt, model, A, a, p)   #function as defined in lacey & cole eq (2.29)
        mat_ds = (mat_St[2:, :] - mat_St[:-2,:])*0.5 #differential of variance S(M)
        return -M*np.sum(mat_ds*mat_Ks[1:-1,:]/mat_mass[1:-1,:], axis=0)
    else:
        mass = np.logspace(np.log10(M*frac), np.log10(M), acc) #masses to calculate the integral
        wf = delta_c(zf, om0, ol0)  # critical density for all z_f
        wt = (wf-w0)/np.sqrt(Sh-S0)  #normalisation following lacey&Cole eq 2.27
        S = sigma(mass, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)**2 #variance of all masses (acc,)
        St = (S - S0)/(Sh - S0) # normalisation following lacey&Cole eq 2.27
        #St[-1] = 1e-10
        Ks = K(St, wt, model, A, a, p) #function as defined in lacey & cole eq (2.29)
        ds = (St[2:] - St[:-2])*0.5 #differential of variance S(M)
        return -M*np.sum(ds*Ks[1:-1]/mass[1:-1])


def new_proba(M,  zf,  frac=0.5, acc=10000, zi=0.0, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000, om0=om,
          ol0=oml, omb=omb, camb=False, model='sheth', colos=False):
    S0 = sigma(M, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)**2
    Sh = sigma(M*frac, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)**2
    w0 = delta_c(zi, om0, ol0)
    if type(zf) == np.ndarray:
        mass = np.logspace(np.log10(M*frac), np.log10(M), acc) #size (0, acc)
        l = len(zf)
        mat_zf = np.array([zf]*acc)   #(acc, l)
        mat_mass = np.array([mass]*l).transpose()  #(acc, l)
        mat_wf = delta_c(mat_zf, om0, ol0)-w0     # (acc, l)
        #mat_wt = (mat_wf-w0)/np.sqrt(Sh-S0)
        mat_S = sigma(mat_mass, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)**2-S0  # (acc, l)
        mat_S[-1, :] = 1e-10
        mat_nu = mat_wf/np.sqrt(mat_S)  # (acc, l)
        #mat_St = (mat_S - S0)/(Sh - S0)   # (acc, l)
        #mat_St[-1, :] = 1e-10
        if model=='EC':
            mat_f = f_ec(mat_S[:,:]+S0, S0, mat_wf[:,:]+w0, w0)
            mat_ds = 0.5*(mat_S[2:,:]-mat_S[:-2,:])
            #mat_dnu = (mat_nu[2:, :] - mat_nu[:-2,:])*0.5  #(acc-3, l)
            #mat_dm = 0.5*(mat_mass[2:, :] - mat_mass[:-2, :])
            return -M*np.sum(mat_ds*mat_f[1:-1,:]/mat_mass[1:-1,:], axis=0)
        else:
            mat_f = fps(mat_nu[:-1,:])/mat_nu[:-1,:] # (acc-1, l)
            #mat_f = fps(mat_nu[:-1, :])
            mat_dnu = (mat_nu[2:-1, :] - mat_nu[:-3,:])*0.5  #(acc-3, l)
            #mat_Ks = K(mat_S, mat_wf, model, A, a, p)
            #mat_ds = (mat_S[2:, :] - mat_S[:-2,:])*0.5
            return M*np.sum(mat_dnu*mat_f[1:-1,:]/mat_mass[1:-2,:], axis=0) #(acc-3, l)
    else:
        mass = np.logspace(np.log10(M*frac), np.log10(M), acc)
        wf = delta_c(zf, om0, ol0) - w0
        #St = np.logspace(-10, 0, acc)
        S = sigma(mass, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)**2 -S0
        S[-1] = 1e-10
        nu = wf/np.sqrt(S)
        if model == 'EC':
            f = f_ec(S + S0, S0, wf+w0, w0)
            ds = 0.5*(S[2:] - S[:-2])
            return -M*np.sum(ds*f[1:-1]/mass[1:-1])
        else:
            f = fps(nu)/nu
            dnu = (nu[2:] - nu[:-2])*0.5
            return M*np.sum(dnu*f[1:-1]/mass[1:-1])

def M_integ_proba(masses, weights=None, zf=np.linspace(0, 7, 20),  frac=0.5, acc=10000, zi=0.0, sig8=sigma8, h=h, kmax=30,
                  window='TopHat', prec=1000, om0=om,
          ol0=oml, omb=omb, camb=False, model='sheth', colos=False, A=0.5, a=1, p=0):
    res = []
    if not( type(weights) == np.ndarray or type(weights) == list):
        for mass in masses:
            res.append(new_proba(mass, zf, frac, acc, zi, sig8, h, kmax, window, prec, om0, ol0, omb, camb,
                         model, colos, A, a, p))
        ares = np.array(res)
        return np.sum(ares, axis=0) / len(masses)
    else:
        for i in range(len(masses)):
            mass = masses[i]
            w = weights[i]/np.sum(weights)
            res.append(new_proba(mass, zf, frac, acc, zi, sig8, h, kmax, window, prec, om0, ol0, omb, camb,
                                 model, colos, A, a, p)*w)
        ares = np.array(res)
        return np.sum(ares, axis=0)


def proba2(wf, a, man=False, acc=1000):
    if man == True:
        xs = np.logspace(-10, 0, acc)
        Ks = K(xs, wf)
        mus = mu(xs, a)
        dx = (xs[2:] - xs[:-2])*0.5
        return -np.sum(Ks[1:-1]*mus[1:-1]*dx)
    else :
        def f(x):
            return K(x, wf)*mu(x, a)
        return -quad(f, 0, 1)[0]

def dpdw(wf, a, man=False, acc=1000):
    if man == True:
        xs = np.logspace(-10, 0, acc)
        newpar = 1/wf - wf/xs
        Ks = K(xs, wf)
        mus = mu(xs, a)
        dx = (xs[2:] - xs[:-2])*0.5
        return -np.sum(Ks[1:-1]*mus[1:-1]*newpar*dx)
    else:
        def f(x):
            return K(x, wf)*mu(x, a)*(1/wf - wf/x)
        return -quad(f, 0, 1)[0]


def median_formation(M, z, frac=0.5, acc=1000, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000,
                      om0=om, ol0=oml, omb=omb, camb=False, colos=False, outc=False):
    zs = np.linspace(z+0.1, 6+z, acc)
    res = proba(M, zs, frac, acc, z, sig8, h, kmax, window, prec, om0, ol0, omb, camb, 'press', colos)
    zf = np.max(zs[res>0.5])
    if outc:
        return 0.7 + 0.77*np.log10(zf)
    else:
        return zf

def average_formation(M, z, frac=0.5, acc=1000, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000,
                      om0=om, ol0=oml, omb=omb, camb=False, colos=False, outc=False):
    zs = np.linspace(z + 1.2*sig8/(2.7+ 0.2*np.log10(M) + 0.1*om0), z+8, acc)
    res = proba(M, zs, frac, acc, z, sig8, h, kmax, window, prec, om0, ol0, omb, camb, 'press', colos)
    dens = (res[2:]-res[:-2])/(zs[2:]-zs[:-2])
    dz = zs[1]-zs[0]
    lower = -dz*np.sum(zs[1:-1]*dens)
    deltap = (zs[0]-z)*dens[0]
    upper = lower - deltap
    if outc:
        return 0.7 + 0.77 * np.log10(lower)
    else:
        return [lower, upper]

def peak_formation(M, z, frac=0.5, acc=1000, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000,
                      om0=om, ol0=oml, omb=omb, camb=False, colos=False, outc=False):
    zs = np.linspace(z + 0.1, z+6, acc)
    res = proba(M, zs, frac, acc, z, sig8, h, kmax, window, prec, om0, ol0, omb, camb, 'press', colos)
    dens = -(res[2:] - res[:-2]) / (zs[2:] - zs[:-2])
    zf = zs[np.argmax(dens)]
    if outc:
        return 0.7 + 0.77 * np.log10(zf)
    else:
        return zf

def slope_age(z, which_formation= 'median', frac=0.5, acc=1000, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000,
                      om0=om, ol0=oml, omb=omb, camb=False, colos=True):
    if which_formation == 'median':
        zf1 = median_formation(1e8, z, frac, acc, sig8, h, kmax, window, prec, om0, ol0, omb,
                               camb, colos)
        zf2 = median_formation(1e14, z, frac, acc, sig8, h, kmax, window, prec, om0, ol0, omb,
                               camb, colos)
        return (zf1 - zf2)/6
    elif which_formation == 'peak':
        zf1 = peak_formation(1e8, z, frac, acc, sig8, h, kmax, window, prec, om0, ol0, omb,
                               camb, colos)
        zf2 = peak_formation(1e14, z, frac, acc, sig8, h, kmax, window, prec, om0, ol0, omb,
                               camb, colos)
        return (zf1 - zf2)/6
    elif which_formation == 'average':
        zf1 = average_formation(1e8, z, frac, acc, sig8, h, kmax, window, prec, om0, ol0, omb,
                               camb, colos)[0]
        zf2 = average_formation(1e14, z, frac, acc, sig8, h, kmax, window, prec, om0, ol0, omb,
                               camb, colos)[0]
        return (zf1 - zf2)/6

