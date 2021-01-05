from __future__ import division
from colossus.lss import mass_function
from colossus.lss import peaks
from fluctuation_rms import *


my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om, 'Ode0': oml, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)




########################################################################################################################

########################----------------------- PEAK HEIGHT AND MULTIPLICITY FUNCTION--------------#####################

########################################################################################################################


def fps(nu):
    """
    Press and Schechter Multiplicty function
    :param nu: float or array of floats : peak height
    :return: float or array of floats.
    """
    return nu*np.sqrt(2/np.pi)*np.exp(-nu**2/2)

def nufnu_st(nu, A=0.322, a=0.707, p=0.3):
    """
    Multiplicity function times nu generalised using Sheth and Tormen. For a regular Press and Chechter A=1/2, a=1 p=0
    :param nu: Peak height, delta_c/sigma
    :param A: sheth and tormen parameter
    :param a: sheth and tormen parameter
    :param p: sheth and tormen parameter
    :return:
    :return:
    """
    nup = np.sqrt(a)*nu
    return A*(1 + 1/nup**(2*p))*fps(nup)

def nu(M, z, om0=om, ol0=oml, omb=omb, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000, camb=False, Colos=False):
    """
    peak height from press and schechter 74. delta_c/sigma
    :param M: float or array of floats : mass
    :param z: float or array of floats : redshifts
    :param window: str : type of smoothing window function. either "TopHat", "Gauss" or k-Sharp'
    :param sig8: float : sigma 8 cosmo parameter
    :param om0: float : fraction matter density
    :param ol0: float : fraction dark energy density
    :param omb: float : fraction baryon density
    :param h: float : H0/100 cosmo parameter
    :param kmax: float or int : maximum wavenumber for CAMB power spectrum.
    :param prec: int : number of bins for integral calculations
    :param camb: boolean : if using camb spectrum or analytical version of Eisenstein and Hu

    :return: peak height
    """
    if Colos:
        my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om0, 'Ode0': ol0, 'Ob0': omb, 'sigma8': sig8, 'ns': ns}
        cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
        return peaks.peakHeight(M, z)
    else:
        del_c = delta_c(z, om0, ol0)  #critical overdensity
        sig = sigma(M, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb)  #fluctuation rms
        if (type(z)==np.ndarray) and (type(M)==np.ndarray):   #case multiple redshifts and mass
            l = len(z)
            n = len(M)
            mat_del_c = np.array([del_c]*n)   #duplicating arrays to do vectorial calculations
            mat_sig = np.array([sig]*l).transpose() #duplicating arrays to do vectorial calculations
            return mat_del_c/mat_sig
        else:
            return del_c/sig


########################################################################################################################

########################----------------------- PRESS AND SCHECHTER HALO MASS FUNCTION--------------###################

########################################################################################################################


def hmf(M, z=0, window='TopHat', sig8=sigma8, om0=om, ol0=oml, omb=omb, h=h, kmax=30, prec=1000,
        out='hmf', model = 'sheth', A=0.322, a=0.707, p=0.3, camb=False):
    """

    :param M: float or array: mass or array of mass. If array, minimum size = 3.
    :param z: float : redshift or array of redshifts. default = 0.
    :param window: str : type of smoothing window function. either "TopHat", "Gauss" or k-Sharp'
    :param sig8: float : sigma 8 cosmo parameter
    :param om0: float : fraction matter density
    :param ol0: float : fraction dark energy density
    :param omb: float : fraction baryon density
    :param h: float : H0/100 cosmo parameter
    :param kmax: float or int : maximum wavenumber for CAMB power spectrum.
    :param prec: int : number of bins for integral calculations
    :param out: str : type of output. Either "hmf" for number density per unit mass. or "dn/dlnM" or "dimensionless"
    :param camb: boolean : if using camb spectrum or analytical version of Eisenstein and Hu
    :param model: string : Halo Mass Function model. Either Sheth & Tormen 2001 or Press & Schechter 1973
    :return: float, array of floats.
    """
    if type(z)==np.ndarray:   #case multiple redshifts
        l = len(z)
        del_c = delta_c(z, om0, ol0)  #critical density array shape (l, )

        if type(M) == np.ndarray or type(M) == list:   #case M is an array
            n = len(M)-2                               # n - 2 because we will do a derivative
            sig = sigma(M, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb) # array of fluctiation rmsshape : (n,)
            dlsig = np.log(sig[2:] / sig[:-2])            # differential of sigma. shape : (n-2, )
            dlM = np.log(M[2:] / M[:-2])        # differential of mass shape : (n-2, )
            new_sig = (sig[2:] + sig[:-2]) * 0.5    #averaging to get the same size as dlsig. shape : (n-2, )
            new_m = (M[2:] + M[:-2]) * 0.5        # averaging to get the same size as dlM shape : (n-2, )

            mat_new_m = np.array([new_m]*l)  #duplicating array of mass shape (l, n-2)
            mat_new_sig = np.array([new_sig] * l)  #duplicating array of sigma shape (l, n-2)
            mat_del_c = np.array([del_c]*n).transpose()  #duplicating array of critical density shape (l, n-2)

            ra1 = rho_m(z=0, om0=om0) / mat_new_m ** 2        #part 1 of Press and Schechter hmf shape : (l, n-2 )
            #part 2 of PS : multiplicity function shape : (l, n-2)
            nu = mat_del_c/mat_new_sig
            #ra2 = np.exp(-mat_del_c ** 2 / (2 * mat_new_sig ** 2)) * np.sqrt(2 / np.pi) * mat_del_c / mat_new_sig
            if model == 'sheth':
                ra2 = nufnu_st(nu, A, a, p)  #using sheth and tormen multiplicity function
            else:
                ra2 = fps(nu)
            ra3 = np.array([dlsig / dlM]*l)  #part 3 of Press and Schechter hmf
            if out == 'hmf':
                return -ra1 * ra2 * ra3  #number density of halos per unit M
            elif out == 'dndlnM':
                return -mat_new_m * ra1 * ra2 * ra3  #number density of halos per unit logM
            elif out == 'dimensionless':
                return -ra2 * ra3    #multiplicity function
        else:    #case of unique M
            nM = np.array([0.99999 * M, 1.00001 * M])     #to get a derivative at M
            sig = sigma(nM, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb=camb) #get sigma for values close to M
            dlsig = np.log(sig[1:] / sig[:-1])    #differential of log sigma at M
            dlM = np.log(nM[1:] / nM[:-1])        # differential of log M at M
            new_sig = (sig[1:] + sig[:-1]) * 0.5  # taking the average as the value for sigma(M) to be symetric wr M
            new_m = (nM[1:] + nM[:-1]) * 0.5      #same. To be symmetric
            nu = del_c/new_sig                   #peak height
            if model == 'sheth':

                ra2 = nufnu_st(nu, A, a, p)    #using sheth and tormen multiplicity function
            else:
                ra2 = fps(nu)         #Press and schechter multiplicity function
            ra1 = rho_m(z=0, om0=om0)/new_m ** 2     #density normalisation of the halo mass function
            ra3 = dlsig / dlM  #part 3 of PS halo mass function

            if out == 'hmf':
                return -ra1 * ra2 * ra3 #number density of halos per unit M
            elif out == 'dndlnM':
                return -new_m * ra1 * ra2 * ra3  #number density of halos per unit logM
            elif out == 'dimensionless':
                return -ra2 * ra3   #multiplicity function
    else: #case of unique z
        del_c = delta_c(z, om0, ol0)
        if type(M) == np.ndarray or type(M) == list:
            sig = sigma(M, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb=camb)  #fluctuation rms
            dlsig = np.log(sig[2:]/sig[:-2])  #differential of log sigma
            dlM = np.log(M[2:]/M[:-2])   # differential of log M
            new_sig = (sig[2:]+sig[:-2])*0.5 #averaged sigma. Difference of 2 to be symmetric and avoid boundary pbs
            new_m = (M[2:]+M[:-2])*0.5  #same. To be symmetric

            nu = del_c/new_sig      #Peak height
            if model == 'sheth':
                ra2 = nufnu_st(nu, A, a, p) #using sheth and tormen multiplicity function
            else:
                ra2 = fps(nu) #Press and schechter multiplicity function
            ra1 = rho_m(z=0, om0=om0)/new_m**2 #density normalisation of the halo mass function
            ra3 = dlsig/dlM  #part 3 of PS halo mass function
            if out=='hmf':
                return -ra1*ra2*ra3  #number density of halos per unit M
            elif out =='dndlnM':
                return -new_m*ra1*ra2*ra3 #number density of halos per unit logM
            elif out == 'dimensionless':
                return -ra2*ra3   #multiplicity function
        else:
            nM = np.array([0.99999*M, 1.00001*M]) #to get a derivative at M
            sig = sigma(nM, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb=camb) #get sigma for values close to M
            dlsig = np.log(sig[1:] / sig[:-1]) #differential of log sigma at M
            dlM = np.log(nM[1:] / nM[:-1]) # differential of log M at M
            new_sig = (sig[1:] + sig[:-1]) * 0.5 # taking the average as the value for sigma(M) to be symetric wr M
            new_m = (nM[1:] + nM[:-1]) * 0.5  #same. To be symmetric
            nu = del_c/new_sig       #peak height
            if model == 'sheth':
                ra2 = nufnu_st(nu, A, a, p)  #using sheth and tormen multiplicity function
            else:
                ra2 = fps(nu)     #Press and schechter multiplicity function

            ra1 = rho_m(z=0, om0=om0)/new_m ** 2 #density normalisation of the halo mass function
            ra3 = dlsig / dlM   #part 3 of PS halo mass function

            if out=='hmf':
                return -ra1*ra2*ra3 #number density of halos per unit M
            elif out =='dndlnM':
                return -new_m*ra1*ra2*ra3 #number density of halos per unit logM
            elif out == 'dimensionless':
                return -ra2*ra3 #multiplicity function


#######################-------------------------------Halo Mass Function plot----------------###########################


'''import matplotlib.pyplot as plt
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plt.rcParams.update(params)
M = np.logspace(7,15, 100)
z = np.array([0, 0.5, 2, 4])
oms = [0.15, 0.3, 0.5]
s8 = [0.6, 0.8, 1]
#for el in s8:
for el in oms:
    y1 = hmf(M, z=0, om0=el, sig8=0.8, kmax=50, prec=100, model='press', out='dndlnM')
    y2 = hmf(M, z=0, om0=el, sig8=0.8, kmax=50, prec=100, model='sheth', out='dndlnM')
    #y1 = hmf(M, z=0, om0=0.3, sig8=el, kmax=50, prec=100, model='press', out='dimensionless')
    #y2 = hmf(M, z=0, om0=0.3, sig8=el, kmax=50, prec=100, model='sheth', A=0.5, a=1, p=0, out='dndlnM')
#    for i in range(4):
    #plt.loglog(M[1:-1], y1, label='$\sigma_8$='+str(el), linewidth = 3)
    #plt.loglog(M[1:-1], y1, label='$\Omega_m$='+str(el)+' PS', linewidth = 2)
    #plt.loglog(M[1:-1], y2, '--', label='$\Omega_m$=' + str(el)+ ' ST', linewidth=1)
    plt.loglog(M[1:-1], y1, label='$\Omega_m$='+str(el)+' PS', linewidth = 2)
    plt.loglog(M[1:-1], y2, '--', label='$\Omega_m$=' + str(el)+ ' ST', linewidth=2)
#    plt.loglog(M[1:-1], mass_function.massFunction(M[1:-1], z[i], model='press74', q_out='dndlnM'), '--', label='Colossus')
plt.xlabel('M [$h^{-1}M_\odot$]', size = 25)
plt.ylabel('dn/dlnM [$h^3/Mpc^{3}$]', size = 25)
plt.xticks(size=20)
plt.yticks(size=20)
#plt.title('Press and Schechter halo mass function', size = 15)
#plt.ylim(1e-6, 0.1)
#plt.xlim(1.5e5, 1e8)

plt.legend()
plt.show()'''

'''import matplotlib.pyplot as plt
params = {'legend.fontsize': 10,
          'legend.handlelength': 2}
plt.rcParams.update(params)
M = np.logspace(8,15, 100)
z = np.array([0, 0.5, 2, 4])
oms = [0.15, 0.3, 0.5]
s8 = [0.6, 0.8, 1]
#vols = [(1248*h)**3, (571*h)**3, (3571*h)**3, (2048*h)**3]
#vols = [(1428*h)**3,(110.7*h)**3, (302.6*h)**3, (1714*h)**3]
#vols = [(256*h)**3,(549*h)**3, (1097*h)**3, (1714*h)**3, (2194*h)**3, (4389*h)**3, (6429*h)**3, (10971*h)**3]
#volnames = ['Quijote', 'Illustris small', 'Illustris big', 'TianNU']
#volnames = ['MICE179', 'MICE389', 'MICE768', 'MICE1200','MICE1536', 'MICE3072', 'MICE4500', 'MICE7680']
#volnames = ['MDPL', 'Small MDPL', 'Big MDPL', 'Bolshoi']
for i in range(len(vols)):
    vol = vols[i]
    y1 = hmf(M, z=0, sig8=0.8, kmax=50, prec=100, model='sheth', out='hmf')*vol
#    for i in range(4):
    plt.loglog(M[1:-1], y1, label=volnames[i], linewidth = 3)
#    plt.loglog(M[1:-1], mass_function.massFunction(M[1:-1], z[i], model='press74', q_out='dndlnM'), '--', label='Colossus')
plt.xlabel('M [$h^{-1}M_\odot$]', size = 25)
plt.ylabel('N [$h/M_{\odot}$]', size = 18)
plt.xticks(size=20)
plt.yticks(size=20)
#plt.title('Press and Schechter halo mass function', size = 15)
#plt.ylim(1e-6, 0.1)
#plt.xlim(1.5e11, 1e15)

plt.legend()
plt.show()'''

##############################"---------------------HMF sig8 evolution---------------------#############################

'''import matplotlib.pyplot as plt
M = np.logspace(11,16, 100)
sigma8 = [0.6, 0.8, 1]
for el in sigma8:
    my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om, 'Ode0': oml, 'Ob0': omb, 'sigma8': el, 'ns': ns}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    y1 = hmf(M, z=0, sig8=el, model='Sheth-Tormen', out='dndlnM')
    plt.loglog(M[1:-1], y1, label='Analytic  $\sigma_8=$'+str(el))
    plt.loglog(M[1:-1], mass_function.massFunction(M[1:-1], 0, model='sheth99', q_out='dndlnM'), '--', label='Colossus')

plt.xlabel('M [$h^{-1}M_\odot$]', size = 15)
plt.ylabel('dn/dlnM [$h^3/Mpc^{3}$]', size = 15)
plt.xlim(5e12, 6e15)
plt.ylim(0.004, 1e-7)
plt.title('Press and Schechter multiplicity function', size = 15)
plt.legend()
plt.show()'''

######################------------------- HMF Omega_m evolution------------------------################################

'''import matplotlib.pyplot as plt
M = np.logspace(11,16, 100)
omv = [0.1,0.3,0.5,0.7]
for el in omv:
    my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': el, 'Ode0': 1-el, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    y1 = hmf(M, z=0, om0=el, ol0=1-el, out='dndlnM')
    plt.loglog(M[1:-1], y1, label='Analytic  $\Omega_m=$'+str(el))
    plt.loglog(M[1:-1], mass_function.massFunction(M[1:-1], 0, model='press74', q_out='dndlnM'), '--', label='Colossus')
plt.xlabel('M [$h^{-1}M_\odot$]', size = 15)
plt.ylabel('dn/dlnM [$h^3/Mpc^{3}$]', size = 15)
plt.xlim(5e12, 6e15)
plt.ylim(0.004, 1e-7)
plt.title('Press and Schechter multiplicity function', size = 15)
plt.legend()
plt.show()'''






################-------------------Peak Heigh CAMB-COLOSSUS COMPARISON------------###################################
'''import matplotlib.pyplot as plt
M = np.logspace(9, 17, 1000)
z = [0, 1, 2, 4]
my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om, 'Ode0': oml, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
for el in z:
    nu1 = nu(M, z=el)
    nu2 = peaks.peakHeight(M, z=el)
    plt.loglog(M, nu1, label='Analytic  z='+str(el))
    plt.loglog(M, nu2,'--', label='COLOSSUS')
plt.xlabel('M [$M_\odot/h$]', size=15)
plt.ylabel(r'$\nu$', size=15)
plt.legend()
plt.show()'''


'''import matplotlib.pyplot as plt
M = np.logspace(9, 17, 1000)
z = [0, 1, 2, 4]
my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om, 'Ode0': oml, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
for el in z:
    nu1 = nu(M, z=el)
    nu2 = peaks.peakHeight(M, z=el)
    plt.loglog(M, nu1, linewidth = 3, label='Analytic  z='+str(el))
    plt.loglog(M, nu2, linewidth = 3, label='COLOSSUS')
plt.xlabel('M [$M_\odot/h$]', size=25)
plt.ylabel(r'$\nu$', size=25)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend()
plt.show()'''

######################--------------- peak height z dependence----------################################################
'''import matplotlib.pyplot as plt
zs = np.linspace(0,4, 100)
s8 = [0.6, 0.8, 1, 1.2]
omv = [0.15, 0.3, 0.5, 0.7]
for el in s8:
#    res1 = nu(4e13, zs, om0=el, ol0=1-el, sig8=sigma8, Colos=True)
    res2 = nu(4e13, zs, om0=om, ol0=oml, sig8=el, Colos=False)
#    plt.plot(zs, res1, '-.', label='Colossus $\Omega_m =$'+str(el))
#    plt.plot(zs, res2, '-', label='$\Omega_m =$'+str(el))
    plt.plot(zs, res2, '-', label='$\sigma_8 =$'+str(el))
plt.legend(fontsize='x-large')
plt.xlabel('z', size=20)
plt.ylabel(r'$\nu$', size=20)
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlim(0, 2)
plt.ylim(1.2, 3.7)
#plt.title('Peak Heght at M=4e13 $M_\odot /h$')
plt.show()'''


'''import matplotlib.pyplot as plt
zs = np.linspace(0,2, 100)
#s8 = [0.6, 0.8, 1, 1.2]
omv = [0.15, 0.3, 0.5, 0.7]

for el in omv:
    res = nu(4e13, zs, om0=el, ol0=1-el, sig8=sigma8)
    plt.plot(zs, res, label='$\Omega_m =$'+str(el))
plt.legend()
plt.xlabel('z', size=15)
plt.ylabel(r'$\nu$', size=15)
plt.title('Peak Heght at M=4e13 $M_\odot /h$')
plt.show()'''

######################-------------------------sigma 8---------------------------------#################################
'''import matplotlib.pyplot as plt
M = np.logspace(9, 17, 1000)
sig8 = [0.4, 0.6, 0.8, 1, 1.2]
for el in sig8:
    nu1 = nu(M, z = 0, sig8=el)
    my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om, 'Ode0': oml, 'Ob0': omb, 'sigma8': el, 'ns': ns}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    nu2 = peaks.peakHeight(M, z=0)
    plt.loglog(M, nu1, label='Analytic  $\sigma_8$='+str(el))
    plt.loglog(M, nu2,'--', label='COLOSSUS')
plt.xlabel('M [$M_\odot/h$]', size=15)
plt.ylabel(r'$\nu$', size=15)
plt.legend()
plt.show()'''

# ######################################-----------------Omega m---------------------------###############################
# import matplotlib.pyplot as plt
# #M = np.logspace(14, 16, 1000)
# M = 1e15
# zs= np.linspace(0, 2, 1000)
# omegam = [0.15, 0.3, 0.5]
# for el in omegam:
#     nu1 = nu(M, z=zs, om0=el, ol0=1-el)
#     my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': el, 'Ode0': 1-el, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
#     cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
#     nu2 = []
#     for z in zs:
#         nu2.append(peaks.peakHeight(M, z=z))
#     plt.loglog(zs, nu1, label='Analytic  $\Omega_m$='+str(el))
#     plt.loglog(zs, nu2,'--', label='COLOSSUS')
# plt.xlabel('z', size=15)
# plt.ylabel(r'$\nu$', size=15)
# plt.legend()
# plt.show()


'''import matplotlib.pyplot as plt
omegam = np.linspace(0.15, 0.6, 30)
Ms = [1e8, 1e10, 1e12, 1e13, 1e14]
for mass in Ms:
    nu2=[]
    for el in omegam:
        my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': el, 'Ode0': 1-el, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
        cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
        nu2.append(nufnu_st(peaks.peakHeight(mass, z=0)))
    plt.loglog(omegam, nu2, label='Mass = '+str(np.log10(mass)))
plt.xlabel('$\Omega_m$', size=15)
plt.ylabel(r'$\nu$', size=15)
plt.legend()
plt.show()'''


######################----------------------- Multiplicity function plot------------------##############################


######################---------------------- Comparison with collossus---------------###################################
'''import matplotlib.pyplot as plt
M = np.logspace(11,16, 100)
z = [0, 2, 4]
for el in z:
    y1 = fps(nu(M, z=el, kmax=50, prec=200))
    y2 = mass_function.massFunction(M, z=el, model='press74')
    plt.loglog(M, y1, label = 'z='+str(el))
    plt.loglog(M, y2, '--', label = ' Colossus')
plt.xlabel('M [$h^{-1}M_\odot$]', size = 15)
plt.ylabel('f', size = 15)
plt.xlim(2e11, 8e15)
plt.ylim(1e-14, 1)
plt.title('Press and Schechter multiplicity function', size = 15)
plt.legend()
plt.show()'''

###############"-------------------Multiplicity compaarison with colossus sig8 evolution----------------################

'''import matplotlib.pyplot as plt
M = np.logspace(7,16, 100)
sigma8 = [0.6, 0.8, 1]
for el in sigma8:
    y1 = fps(nu(M, z=0, sig8 = el))
    my_cosmo = {'flat': True, 'H0': 100*h, 'Om0': om, 'Ob0': omb, 'sigma8': el, 'ns': ns}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    y2 = fps(peaks.peakHeight(M, z= 0))
    #mfunc = mass_function.massFunction(M, z=0, mdef='fof', model='press74', q_out='f')
    plt.loglog(M, y1, linewidth=3, label = 'sigma='+str(el))
    #plt.loglog(M, y2, '--', label = ' Colossus')
plt.xlabel('M [$h^{-1}M_\odot$]', size = 15)
plt.ylabel('f', size = 15)
plt.xlim(2e8, 1e15)
plt.ylim(1e-1, 7e-1)
plt.title('Press and Schechter multiplicity function', size = 15)
plt.legend()
plt.show()'''


##########################------------  same : omega_m evolution--------------------------##############################

'''import matplotlib.pyplot as plt
M = np.logspace(7,15, 100)
om1 = [0.15, 0.3, 0.5]
for el in om1:
    y1 = fps(nu(M, z=0, om0=el, ol0=1-el,  sig8 = sigma8))
    my_cosmo = {'flat': True, 'H0': 100*h, 'Om0': el, 'Ode0' : 1-el, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    y2 = fps(peaks.peakHeight(M, z= 0))
    #mfunc = mass_function.massFunction(M, z=0, mdef='fof', model='press74', q_out='f')
    plt.loglog(M, y1, linewidth=3, label = '$\Omega_m=$'+str(el))
    #plt.loglog(M, y2, '--', label = ' Colossus')
plt.xlabel('M [$h^{-1}M_\odot$]', size = 20)
plt.ylabel('f', size = 20)
plt.xlim(2e7, 1e15)
plt.ylim(1e-1, 5e-1)
plt.title('Press and Schechter multiplicity function', size = 20)
plt.legend()
plt.show()'''






########################################################################################################################

########################----------------------- PS Caracteristic non linear Mass M^star--------------###################

########################################################################################################################


def Mstar(lMmin=6, lMmax=15, npoints = 10000, z=0, h=h, om0=om, ol0=oml, omb=omb, sigma8 = sigma8,
               prec = 1000, kmax=100, window='TopHat', camb=False, Colossus=False):
    """
    Caracteristic non-linear mass
    :param lMmin: float : log lower mass limit
    :param lMmax: float : log lower mass limit
    :param npoints: int : number of mass subdivision
    :param z: float : redshift
    :param window: str : type of smoothing window function. either "TopHat", "Gauss" or k-Sharp'
    :param sigma8: float : sigma 8 cosmo parameter
    :param om0: float : fraction matter density
    :param ol0: float : fraction dark energy density
    :param omb: float : fraction baryon density
    :param h: float : H0/100 cosmo parameter
    :param kmax: float or int : maximum wavenumber for CAMB power spectrum.
    :param prec: int : number of bins for integral calculations
    :param camb: boolean : if using camb spectrum or analytical version of Eisenstein and Hu
    :return: float : caracteristic non linear mass
    """
    if Colossus:
        my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om0, 'Ode0': ol0, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
        cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
        return peaks.nonLinearMass(z)
    else:
        mass = np.logspace(lMmin, lMmax, npoints)
        res = nu(mass, z, om0, ol0, omb, sigma8,h, kmax, window, prec, camb)
        if type(z) == np.ndarray:
            l = len(z)           #res shape (n x l)
            mat_mass = np.array([mass]*l).transpose()
            new_mass = np.where(res>1, mat_mass, np.inf)
            return np.min(new_mass, axis=0)
        else:
            return np.min(mass[res>1])




########################---------------------------Plotting M_\star----------------#####################################

'''import matplotlib.pyplot as plt
zs = np.linspace(0, 4, 50)
my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om, 'Ode0': oml, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
Ms_co = peaks.nonLinearMass(zs)
res = Mstar(z=zs, npoints=1000, kmax=100)
plt.loglog(1+zs, res, label='Analytic')
plt.loglog(1+zs, Ms_co, label='Colossus')
plt.xlabel('z', size = 15)
plt.ylabel('$M^\star$[$h^{-1}M_\odot$]', size = 15)
plt.xlim(0, 4)
plt.ylim(1e8, 1e13)
plt.title('Press and Schechter caracteristic non linear mass', size=15)
plt.legend()
plt.show()'''


'''import matplotlib.pyplot as plt
om = np.linspace(0.1, 0.8, 50)
res = []
res2 = []
for el in om:
    my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': el, 'Ode0': 1-el, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    res.append(Mstar(z=0, om0=el, ol0=1-el, npoints=300, kmax=5))
    res2.append(peaks.nonLinearMass(0))
plt.plot(om, res, '-g', label='Analytic')
plt.plot(om, res2, '--k', label='Colossus')
plt.yscale('log')
plt.xlabel('$\Omega_m$', size = 15)
plt.ylabel('$M^\star$[$h^{-1}M_\odot$]', size = 15)
plt.xlim(0.1, 0.8)
plt.ylim(1e11, 1e14)
plt.legend()
plt.title('Press and Schechter caracteristic non linear mass', size=15)
plt.show()'''


'''import matplotlib.pyplot as plt
s8 = np.linspace(0.1, 1.5, 50)
res = []
res2 = []
for el in s8:
    my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om, 'Ode0': oml, 'Ob0': omb, 'sigma8': el, 'ns': ns}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    res.append(Mstar(z=0, sigma8=el, om0=om, ol0=oml, npoints=300, kmax=50))
    res2.append(peaks.nonLinearMass(0))
plt.plot(s8, res, '-g', label='Analytic')
plt.plot(s8, res2, '--k', label = 'Colossus')
plt.yscale('log')
plt.xlabel('$\sigma_8$', size = 15)
plt.ylabel('$M^\star$[$h^{-1}M_\odot$]', size = 15)
plt.xlim(0.1, 1.5)
plt.ylim(2e10, 5e14)
plt.title('Press and Schechter caracteristic non linear mass', size=15)
plt.legend()
plt.show()'''


#####################------------------------omega_m vs sigma8 Mstar=const-----------------------######################"

'''import matplotlib.pyplot as plt
sze = 15
omv = np.linspace(0.1, 0.7, sze)
sig8 = np.linspace(0.4, 1.5, sze)
nom = np.zeros((sze,sze))
x = np.array([omv]*sze).transpose()
y = np.array([sig8]*sze)
for i in range(sze):
    for j in range(sze):
        nom[i,j] = Mstar(lMmin=1, lMmax=18,  z=0, om0 = omv[i], ol0=1-omv[i], sigma8=sig8[j],
                              npoints=1000, prec=100, kmax=100)
plt.contourf(x, y, np.log10(nom), levels=100, cmap='jet')
plt.xlabel('$\Omega_m$', size = 15)
plt.ylabel('$\sigma_8$', size = 15)
plt.colorbar()
plt.title('$\log M^\star$')
plt.show()'''

'''import matplotlib.pyplot as plt
sze = 15
omv = np.linspace(0.15, 0.5, sze)
sig8 = np.linspace(0.6, 1.1, sze)
x = np.array([omv]*sze).transpose()
y = np.array([sig8]*sze)
nom = np.zeros((sze,sze))
zs = [0, 0.5, 1, 2]
for el in zs:
    for i in range(sze):
        for j in range(sze):
            my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': omv[i], 'Ode0': 1-omv[i], 'Ob0': omb,
                        'sigma8': sig8[j], 'ns': ns}
            cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
            nom[i,j] = peaks.nonLinearMass(el)
    plt.contourf(x, y, np.log10(nom), levels=100, cmap='jet')
    plt.xlabel('$\Omega_m$', size = 25)
    plt.ylabel('$\sigma_8$', size = 25)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.title('z = '+str(el), size = 20)
    plt.show()'''


########################################################################################################################

########################----------------------- Integrated HMF N(>M) -----------------------############################

########################################################################################################################



def nofm(M, lMmax=18, z=0, window='TopHat', sigma8=sigma8, om0=om, ol0=oml, omb=omb, h=h, kmax=30,
         prec=300, Colos = False, camb=False):
    """
    Number density of halos more massive than M. Integrated using scipy quad
    :param lMmax: float : upper bound limit for integration
    :param M: float or array of floats : mass
    :param z: float or array of floats : redshifts
    :param window: str : type of smoothing window function. either "TopHat", "Gauss" or k-Sharp'
    :param sigma8: float : sigma 8 cosmo parameter
    :param om0: float : fraction matter density
    :param ol0: float : fraction dark energy density
    :param omb: float : fraction baryon density
    :param h: float : H0/100 cosmo parameter
    :param kmax: float or int : maximum wavenumber for CAMB power spectrum.
    :param prec: int : number of bins for integral calculations
    :param Colos : boolan : using Colossus halo mass function or not
    :param camb: boolean : if using camb spectrum or analytical version of Eisenstein and Hu

    :return:
    """
    from scipy.integrate import quad
    if Colos :
        my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om0, 'Ode0': ol0, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
        cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
        def dn(x):
            return np.exp(x)*mass_function.massFunction(np.exp(x), z, model='press74', q_out='dndlnM')
    else :
        def dn(x):       #define function to integrate
            return np.exp(x)*hmf(np.exp(x), z, window, sigma8, om0, ol0, omb, h, kmax, prec, out='hmf', camb=camb)
    return quad(dn, np.log(M), lMmax*np.log(10))[0]


def nofm_man(M, lMmax=20, z=0, window='TopHat', sigma8=sigma8, om0=om, ol0=oml, omb=omb, h=h, kmax=30,
                     prec=100, acc=np.int(1e4), Colos=False, camb=False, out='hmf'):
    """
    Number density of halos more massive than M. Integrated using simple sum.
    :param lMmax: float : upper bound limit for integration
    :param M: float or array of floats : mass
    :param z: float or array of floats : redshifts
    :param window: str : type of smoothing window function. either "TopHat", "Gauss" or k-Sharp'
    :param sigma8: float : sigma 8 cosmo parameter
    :param om0: float : fraction matter density
    :param ol0: float : fraction dark energy density
    :param omb: float : fraction baryon density
    :param h: float : H0/100 cosmo parameter
    :param kmax: float or int : maximum wavenumber for CAMB power spectrum.
    :param prec: int : number of bins for integral calculations in fluctuation rms
    :param acc: int : number of bins for integral calculations in halo mass function
    :param Colos : boolan : using Colossus halo mass function or not
    :param camb: boolean : if using camb spectrum or analytical version of Eisenstein and Hu
    """
    Ms = np.logspace(np.log10(M), lMmax, acc)   #array of mass
    dlM = np.log(10)*(lMmax-np.log10(M))/acc    #differential in log mass

    if Colos :
        my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om0, 'Ode0': ol0, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
        cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
        y = mass_function.massFunction(Ms[1:-1], z, model='press74', q_out='dndlnM')
        return np.sum(y*dlM)
    elif type(z) == np.ndarray:
        y = hmf(Ms, z, window, sigma8, om0, ol0, omb, h, kmax, prec, out='hmf', camb=camb)
        l, m = y.shape
        mat_Ms = np.array([Ms[1:-1]]*l)
        return np.sum(mat_Ms*dlM*y, axis=1)
    else:
        y = hmf(Ms, z, window, sigma8, om0, ol0, omb, h, kmax, prec, out=out, camb=camb)
        return np.sum(dlM*y)



#########################------------------------PLOTS OF VARIOUS QUANTITIES----------------############################


'''import matplotlib.pyplot as plt
Ms = np.logspace(8, 14, 100)
lmMax = 17
y1 = []
y2 = []
y3 = []
for el in Ms:
    y1.append(nofm_man(el, lmMax, acc=100, out='hmf'))
    y2.append(nofm_man(el, lmMax, acc=100, out='dndlnM'))
    y3.append(nofm_man(el, lmMax,  acc=100,Colos=True))

plt.loglog(Ms, y1, label ='Analytic hmf')
plt.loglog(Ms, y2, label ='Analytic dndlnM')
plt.loglog(Ms, y3, label ='Colossus')

plt.legend()
plt.show()'''
#########################----------------omega_m vs sigma8 at n = const--------------------############################

##########################---------------Varying z ----------------------------------###################################

'''import matplotlib.pyplot as plt
import matplotlib as mpl
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}

plt.rcParams.update(params)
#mpl.rcParams["figure.dpi"] = 50
#onepluszs = np.logspace(0, np.log10(3), 10)
onepluszs = [1, 1.5, 2, 3, 4]
masses = [1e8, 1e10, 1e12, 1e14]
size = 15
#sig8 = np.logspace(np.log10(0.6), np.log10(1.1), size)
#omv = np.logspace(np.log10(0.2), np.log10(0.4), size)
sig8 = np.linspace(0.6, 1.1, size)
omv = np.linspace(0.15, 0.5, size)
x = np.array([omv]*size).transpose()
y = np.array([sig8]*size)
olv = 1 - omv

nom = np.zeros((size, size))
fig, axs = plt.subplot(4, 4)
for k in range(len(onepluszs)):
    el = onepluszs[k]
    for l in range(len(masses)):
        mt = masses[l]
        for i in range(size):
            for j in range(size):
                nom[i,j] = np.log10(hmf(mt, z=el-1, sig8=sig8[j], om0=omv[i], ol0=olv[i],
                                                     kmax=5, prec=100, camb=False, model='sheth',
                                        out='dndlnM'))
        plt.contourf(x, y, nom, levels=100, cmap='jet')
        plt.xlabel('$\Omega_m$', size = 25)
        plt.ylabel('$\sigma_8$', size = 25)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=15)
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.title('dN/dlogM log M='+str(round(np.log10(mt), 2))+' z='+str(round(el-1, 2)), size=20)
        plt.savefig('om_s8_n_M'+str(round(np.log10(mt), 2))+'z'+str(round(el-1, 2))+'.png',
                    bbox_inches ='tight')
        plt.show()'''
'''import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
params = {'legend.fontsize': 10, 'legend.handlelength': 1}

#plt.rcParams.update(params)
#mpl.rcParams["figure.dpi"] = 50
#onepluszs = np.logspace(0, np.log10(3), 10)
onepluszs = [1, 2, 3]
masses = [5e12, 5e13, 5e14]
size = 15
#sig8 = np.logspace(np.log10(0.6), np.log10(1.1), size)
#omv = np.logspace(np.log10(0.2), np.log10(0.4), size)
sig8 = np.linspace(0.6, 1.1, size)
omv = np.linspace(0.15, 0.5, size)
x = np.array([omv]*size).transpose()
y = np.array([sig8]*size)
olv = 1 - omv

nom = np.zeros((size, size))
fig, axs = plt.subplots(3, 3, dpi=500)

for k in range(len(onepluszs)):
    el = onepluszs[k]
    for l in range(len(masses)):
        mt = masses[l]
        for i in range(size):
            for j in range(size):
                nom[i,j] = np.log10(hmf(mt, z=el-1, sig8=sig8[j], om0=omv[i], ol0=olv[i],
                                                     kmax=5, prec=100, camb=False, model='sheth',
                                        out='dndlnM'))
        ax = axs[k,l]
        im = ax.contourf(x, y, nom, levels=100, cmap='jet')
        #cbar = fig.colorbar(im, ax=ax)
        #tick_locator = ticker.MaxNLocator(nbins=5)
        #cbar.locator = tick_locator
        #cbar.update_ticks()
        #cbar.ax.tick_params(labelsize=6)
        #plt.xlabel('$\Omega_m$', size = 25)
        #plt.ylabel('$\sigma_8$', size = 25)
#cbar = plt.colorbar()
#cbar.ax.tick_params(labelsize=15)
        #plt.xticks(size=15)
        #plt.yticks(size=15)
        #plt.title('dN/dlogM log M='+str(round(np.log10(mt), 2))+' z='+str(round(el-1, 2)), size=20)
for ax in axs.flat:
    ax.set(xlabel='$\Omega_m$', ylabel='$\sigma_8$')
    ax.label_outer()

#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#cbar = fig.colorbar(im, ax=axs[:,:])

#cbar.ax.tick_params(labelsize=15)
plt.savefig('om_s8_n_M'+str(round(np.log10(mt), 2))+'z'+str(round(el-1, 2))+'.png',bbox_inches ='tight', dpi=1100)
plt.show()'''

########################################-----------------sigma8 =const----------------##################################
'''import matplotlib.pyplot as plt
size = 15
onepluszs = np.logspace(0, np.log10(3), size)
omv = np.logspace(np.log10(0.1), np.log10(0.6), size)
olv = 1 - omv
# ombv = 0.13*omv
nom = np.zeros((size, size))
mt = 1e10
x = np.zeros((size,size))
y = np.zeros((size,size))

for j in range(len(onepluszs)):
    el = onepluszs[j]
    for i in range(size):
        x[i,j] = omv[i]
        y[i, j] = el-1
        nom[i,j] = np.log10(nofm_man(mt, z=el-1, lMmax= 18, om0=omv[i], ol0=olv[i],
                                             kmax=5, prec=100, Colos=True, camb=False))
plt.contour(x, y, nom, levels=100, cmap='jet')
plt.xlabel('$\Omega_m$', size = 15)
plt.ylabel('z', size = 15)
plt.colorbar()
plt.title(r'$n(>4\times10^{13})$')
plt.show()'''


########################################################################################################################

##################################--------------------N(z) for a given volume-----------################################

########################################################################################################################


def N(z, M, solid_angle, lMmax=20, window='TopHat', sigma8=sigma8, om0=om, ol0=oml, omb=omb, h=h, kmax=30,prec=100,
      acc=np.int(1e4), Colos=False, differential=True, z2=None, prec2=None, camb=False):
    """
    Number of structures of mass greater than M detected at redshift z within a given solid angle
    :param z: float : redshift of detection
    :param M: float : Mass limit of structures
    :param solid_angle: float : solid angle in steradian
    :param lMmax: float: Maximim log mass. Parameter for the integration of the Halo Mass Function
    :param window: str: type of filter. Either 'TopHat', 'Gaussian' or 'k-Sharp'
    :param sigma8: float : sigma 8 cosmo parameter
    :param om0: float : fraction matter density
    :param ol0: float : fraction dark energy density
    :param omb: float : fraction baryon density
    :param h: float : H0/100 cosmo parameter
    :param kmax: float or int : maximum wavenumber for CAMB power spectrum.
    :param prec: int : number of bins for integral calculations in fluctuation rms
    :param acc: int : number of bins for integral calculations in halo mass function
    :param Colos : boolan : using Colossus halo mass function or not
    :param camb: boolean : if using camb spectrum or analytical version of Eisenstein and Hu
    :param differential: bool. If out is dN/dz.
    :param z2: float: if differential = False. upper limit redshift for integration
    :param prec2: int:  number of bins for integration over redshift
    :return: float. Number of structures (/per unit redshift)
    """
    import astropy as aspy
    from astropy.cosmology import LambdaCDM
    cosmo = LambdaCDM(H0=100 * h, Om0=om0, Ode0=ol0, Ob0=omb)

    if differential:
        vol = cosmo.differential_comoving_volume(z).value * h ** 3
        Ntot = nofm_man(M, lMmax, z, window, sigma8, om0, ol0, omb, h, kmax, prec, acc, Colos, camb)
        return Ntot*solid_angle*vol
    elif Colos:
        zs = np.linspace(z, z2, prec2)
        res = []
        vol = cosmo.differential_comoving_volume(zs).value * h ** 3
        dz = (z2 - z) / prec2
        for el in zs:
            res.append(nofm_man(M, lMmax, el, window, sigma8, om0, ol0, omb, h, kmax, prec, acc, Colos, camb))
        res = np.array(res)
        return np.sum(res*vol)*dz*solid_angle
    else:
        zs = np.linspace(z, z2, prec2)
        vol = cosmo.differential_comoving_volume(zs).value * h ** 3
        dz = (z2 - z) / prec2
        Nofz = nofm_man(M, lMmax, zs, window, sigma8, om0, ol0, omb, h, kmax, prec, acc, Colos, camb)
        return np.sum(Nofz*vol)*dz*solid_angle




'''import matplotlib.pyplot as plt
size = 15
omv = np.linspace(0.15, 0.5, size)
olv = 1-omv
#ombv = 0.13*omv
#sig8 = np.linspace(0.6, 1.1, size)
S8 = np.linspace(0.4, 1.4, size)
nom = np.zeros((size,size))
mt = 3e13
ang = 1000*np.pi**2/180**2
zs = [0.1, 0.5, 1, 2]
x = np.array([omv]*size).transpose()
y = np.array([S8]*size)
for el in zs:
    for i in range(size):
        for j in range(size):
            s8 = S8[j]*(0.3/omv[i])**0.46
            nom[i,j] = np.log10(N(z=el, M=mt, solid_angle=ang,  lMmax= 18, sigma8=s8, om0=omv[i], ol0=olv[i],kmax=5,
                                  prec=100, Colos=True, differential=True, z2=el+0.05, prec2=100))
    plt.contourf(x, y, nom, levels=60, cmap='jet')
    plt.xlabel('$\Omega_m$', size = 25)
    plt.ylabel('$\sigma_8$', size = 25)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    plt.title('z = '+str(el), size=20)
#    plt.title(str(el) +' < z < ' + str(el+0.05), size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()'''

