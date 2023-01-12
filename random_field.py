from fluctuation_rms import *


def sig(sig8=sigma8, h=h, omb=omb, om0=om, ns=ns, test= False, prec=1000):


    prec=10*prec   # having more bins is less expensive than in the camb power spectrum case
    k = np.logspace(-7, 5, prec)  #creating a k array in log space for the integration
    pk = power_spectrum(k, sig8, h, om0, omb, ns, test)  #corresponding power spectrum values

    # In units of Mpc/h
    m = prec  # size of wavenumbers imput

    dlk = np.log(np.max(k) / np.min(k)) / len(k)  # element of k for approximating the integral
    res = pk * k ** 3  # Values inside the integral foreach k
    integ = np.sum(res) * dlk  # approximate evaluation of the integral through k.
    return np.sqrt(integ / (2 * np.pi ** 2))

size = 1000
si = sig()

Rmax = 10
dr = Rmax/size
def ksharp(r, R):
    x = r/R
    ra1 = 3*(np.sin(x) - x*np.cos(x))/x**3
    ra2 = 4*np.pi*R**3/3
    return ra1/ra2
def gaussian(r, R):
    return np.exp(-r**2/(2*R**2))*(2*np.pi*R**2)**(-1.5)

radius = np.logspace(-5, np.log10(Rmax), size)
dlr = (np.log(Rmax) - np.log(1e-5))/size
#radius = np.linspace(1e-5, Rmax, size)
def delta_R(R):
    dels = delta[radius<R]
    rads = radius[radius<R]
    flis = ksharp(rads, R)
#    flis = gaussian(rads, R)
#    drs = radius[1:]-radius[:-1]
#    dr = drs[radius<R]
    return np.sum(rads*dels*flis*dlr)
scales = np.logspace(-1, 0, 100)
Mass = 6*np.pi**2*rho_m(0)*scales**3
import matplotlib.ticker as tkr
Niter = 10
fig, ax = plt.subplots()
for iter in range(Niter):
    delta = np.random.normal(0, si, size = size)
    res = []
    for R in scales:
        res.append(delta_R(R))
    #ax.plot(1/scales[50:], np.array(res[50:]), color='black', linewidth=0.5)
    ax.plot(1/Mass[50:], np.array(res[50:]), color='black', linewidth=0.5)

    plt.xscale('log')
plt.xlabel('1/M', size=20)
plt.ylabel('$\delta$', size=20)
#plt.xlim(1/scales[-1], 1/scales[50])
y1 = delta_c(0)
y2 = delta_c(1)
y3 = delta_c(2)
#plt.hlines(y1, 1/scales[-1], 1/scales[50], colors= 'green', label= 'z = 0')
#plt.hlines(y2, 1/scales[-1], 1/scales[50], colors= 'blue', label= 'z = 1')
#plt.hlines(y3, 1/scales[-1], 1/scales[50], colors= 'red', label= 'z = 2')
plt.hlines(y1, 1/Mass[-1], 1/Mass[50], colors= 'green', label= 'z = 0')
plt.hlines(y2, 1/Mass[-1], 1/Mass[50], colors= 'blue', label= 'z = 1')
plt.hlines(y3, 1/Mass[-1], 1/Mass[50], colors= 'red', label= 'z = 2')
#ax.xaxis.set_minor_formatter(tkr.NullFormatter())
ax.xaxis.set_major_formatter(tkr.NullFormatter())

plt.legend()
plt.show()
