from formation_time import *
from scipy import special
from scipy.integrate import quad


def cole2000(zi, Mi, Mres, zf, dz=1e-3, frac=0.5, acc=10000, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000,
             om0=om,
             ol0=oml, omb=omb, camb=False, model='sheth', colos=True, A=0.5, a=1, p=0):
    zs = np.arange(zi, zf, dz)
    mass_tree = [[Mi]]
    for i in range(len(zs) - 1):
        mass_tree_M = []
        for M in mass_tree[i]:
            if M > 5 * Mres:
                R = np.random.uniform(0, 1)
                S0 = sigma(M, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb,
                           colos) ** 2  # variance of the field at mass M
                w0 = delta_c(zs[i], om0, ol0)  # critical density at observed redshift
                dw = delta_c(zs[i + 1], om0, ol0) - w0
                M1s = np.logspace(np.log10(Mres), np.log10(frac * M), acc)
                S1s = sigma(M1s, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb,
                            colos) ** 2  # variance of the field at mass M
                dS1 = (S1s[2:] - S1s[:-2]) * 0.5
                dM1 = (M1s[2:] - M1s[:-2]) * 0.5
                # dlsig = np.log(S1s[2:]) - np.log(S1s[:-2])*0.5
                # dlM =  np.log(M1s[2:]) - np.log(M1s[:-2])*0.5
                # dNdM1 = -M*S1s[1:-1]*(S1s[1:-1]-S0)**(-1.5)*dw*dlsig/(np.sqrt(2*np.pi)*dz*dlM*M1s**2)
                # dM1 = M1s[2:]-M1s[:-2]
                dNdM1 = -M * (S1s[1:-1] - S0) ** (-1.5) * dw * dS1 / (np.sqrt(2 * np.pi) * dM1 * M1s[1:-1])
                P = np.sum(dNdM1 * dM1)
                Pcum = np.cumsum(dNdM1 * dM1)
                M2s = np.logspace(-5, np.log10(Mres), acc)
                S2s = sigma(M2s, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos) ** 2
                # F = special.erfc(dw/np.sqrt(2*(S2s[0]-S2s[-1])))
                dS2 = (S2s[2:] - S2s[:-2]) * 0.5
                dM2 = (M2s[2:] - M2s[:-2]) * 0.5
                dNdM2 = -(S2s[1:-1] - S0) ** (-1.5) * dw * dS2 / (np.sqrt(2 * np.pi) * dM2)
                F = np.sum(dNdM2 * dM2)
                if R > P:
                    mass_tree_M.append(M * (1 - F))
                else:
                    mass = np.min(M1s[1:-1][Pcum > R])
                    mass_tree_M.append(mass)
                    mass_tree_M.append(-mass + M * (1 - F))
            else:
                mass_tree_M.append(M)
        mass_tree.append(mass_tree_M)
        # print(np.sum(np.array(mass_tree[i+1]))/Mi)
    return mass_tree, zs


def S(q, sigma2, sigmah, alphah, w, dwdz, B=0.1, eta=1, G0=1, mu=1, gamma1=1, gamma2=1):
    ra1 = np.sqrt(2 / np.pi) * B * alphah * q ** (eta - 1) * G0 / 2 ** (mu * gamma1)
    ra2 = (w / sigma2) ** gamma2 * (sigmah / sigma2) ** gamma1 * dwdz
    return ra1 * ra2


def parkinson08(zi, Mi, qres, zf, dz=1e-3, frac=0.5, acc=10000, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000,
                om0=om,
                ol0=oml, omb=omb, camb=False, colos=True, B=0.1, beta=0.1, eta=1, G0=1, mu=1, gamma1=1, gamma2=1):
    sigma2 = sigma(Mi, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)
    sigmares = sigma(Mi * qres, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)
    sigmah = sigma(Mi / 2, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)
    dlMh = np.log(Mi / 2 + 1e-10) - np.log(Mi / 2 - 1e-10)
    dlSigh = np.log(sigma(Mi / 2 + 1e1 - 10, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb,
                          colos) / sigma(Mi / 2 - 1e1 - 10, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb,
                                         colos))
    alphah = -dlSigh / dlMh
    deltaz = 0.01
    w = delta_c(zi, om0, ol0)
    dwdz = (delta_c(zi + 1e-10, om0, ol0) - delta_c(zi - 1e-10, om0, ol0)) / 2e-10
    epsilon1 = 0.1
    epsilon2 = 0.1

    # Step 1 choosing Delta z

    integral_sdq = (S(0.5, sigma2, sigmah, alphah, w, dwdz, B=B, eta=eta, G0=G0, mu=mu, gamma1=gamma1,
                      gamma2=gamma2) * 0.5 - S(qres,
                                               sigma2, sigmah, alphah, w, dwdz, B=B, eta=eta, G0=G0, mu=mu,
                                               gamma1=gamma1, gamma2=gamma2) * qres) / eta
    delz1 = epsilon2 / integral_sdq
    delz2 = epsilon1 * np.sqrt(2) * (sigmah ** 2 - sigma2 ** 2) ** 0.5 / dwdz
    delz = np.min([delz1, delz2])

    # step 2 evaluation F
    ures = sigma2 * (sigmares ** 2 - sigma2 ** 2) * (-0.5)
    y = lambda x: (1 + 1 / x ** 2) ** gamma1 / 2
    Jures = quad(y, 0, ures)
    F = np.sqrt(2 / np.pi) * Jures * G0 * (w / sigma2) ** gamma2 * dwdz * delz / sigma2

    # step 3
    r1 = np.random.uniform(0, 1)
    if r1 > integral_sdq * delz:
        Mreturn = Mi * (1 - F)
    else:
        r2 = np.random.uniform(0, 1)
        q = (qres ** eta + r2 * (2 ** (-eta) - qres ** eta)) ** 1 / eta
        r3 = np.random.uniform(0, 1)
        sigmaq = sigma(Mi * q, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)
        dlSigq = np.log(sigma(q * Mi + 1e1 - 10, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb,
                              colos) / sigma(q * Mi - 1e1 - 10, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb,
                                             colos))
        dlMq = np.log(q * Mi + 1e-10) - np.log(q * Mi - 1e-10)
        alphaq = -dlSigq / dlMq
        Vq = sigmaq ** 2 * (sigmaq ** 2 - sigma2 ** 2) ** (-1.5)
        Rq = alphaq * Vq * ((2 * q) ** mu * sigmaq / sigmah) ** gamma1 / (alphah * B * q ** beta)

        if r3 < Rq:
            M1 = q * Mi
            M2 = Mi(1 - F - q)
    return


'''masses = []
Niter = 1000000
M0 = 1e13
for i in range(Niter):
    tree, reds = cole2000(0.1, 1e13, 1e10, zf=0.106, dz=3e-3, acc=10000)
    for el in tree[1]:
        masses.append(el)

y, x = np.histogram(np.array(masses), density=True)

Ms = np.logspace(10, 13, 10000)
S0 = sigma(M0, Colos=True) ** 2
S1 = sigma(Ms, Colos=True) ** 2
dS = S1 - S0
dS[-1] = 1e-10
dw = delta_c(0.15) - delta_c(0.1)
fsc = dw * np.exp(-dw ** 2 / (2 * dS)) * dS ** (-1.5) / np.sqrt(2 * np.pi)
dsdM = - (S1[2:] - S1[:-2]) / (Ms[2:] - Ms[:-2])

neps = fsc[1:-1] * dsdM * M0 / Ms[1:-1]
plt.plot(x[1:], 100*y, label='MC')
plt.plot(Ms[1:-1], neps, label='EPS')
plt.yscale('log')
plt.legend()
plt.show()'''
'''
zi = 0.1
zf = 2
M = 1e13
Mres = 1e9
dz = 1e-3
mass_tree = [[M]]
i=0
mass_tree_M = []
for M in mass_tree[i]:
    S0 = sigma(M, Colos=True)**2 #variance of the field at mass M
    w0 = delta_c(zi, om0=om0)  #critical density at observed redshift
    dw = delta_c(zi+dz, om0=om0) - w0
    M1s = np.logspace(np.log10(Mres), np.log10(0.5*M), 10000)
    S1s = sigma(M1s, Colos=True)**2
    #dlsig = (np.log(S1s[2:]) - np.log(S1s[:-2]))*0.5
    dS1 = (S1s[2:] - S1s[:-2])*0.5
    dM1 = (M1s[2:]-M1s[:-2])*0.5
    #dlM =  (np.log(M1s[2:]) - np.log(M1s[:-2]))*0.5
    dNdM1 = -M*(S1s[1:-1]-S0)**(-1.5)*dw/(np.sqrt(2*np.pi)*M1s[1:-1])
    M2s = np.logspace(-22, np.log10(Mres), 1000000)
    S2s = sigma(M2s, Colos=True)**2
    dS2 = (S2s[2:] - S2s[:-2])*0.5
    dM2 = (M2s[2:] - M2s[:-2])*0.5
    dNdM2 = -(S2s[1:-1]-S0)**(-1.5)*dw/(np.sqrt(2*np.pi))
    #dNdM2 = -np.exp(-0.5*dw**2/(S2s[1:-1]-S0))*(S2s[1:-1]-S0)**(-1.5)*dw/(np.sqrt(2*np.pi))
    #F = np.sum(dNdM2*dS2)
    #F = special.erfc(dw/np.sqrt(2*))
    F = np.sqrt(2/np.pi)*(S2s[-1] - S0)**(-0.5)*dw
    P = np.sum(dNdM1*dS1)
    Pcum = np.cumsum(dNdM1*dS1)
    R = np.random.uniform(0,1)
    #mass_tree = [[M]]
    if R > P:
        mass_tree_M.append(M*(1-F))
    else:
        mass = np.min(M1s[1:-1][Pcum>R])
        mass_tree_M.append(mass)
        mass_tree_M.append(-mass + M*(1-F))
mass_tree.append(mass_tree_M)
#print(len(mass_tree[i+1]))
#plt.plot(M1s[1:-1], dNdM1)
#plt.xscale('log')
#plt.show()'''

'''def cond(ds, dw):
    return dw*(np.pi*2)**(-0.5)*np.exp(-0.5*dw**2/ds)*ds**(-1.5)

def dfdz(ds, dw):
    return dw*(np.pi*2)**(-0.5)*ds**(-1.5)
sh = sigma(1, Colos=True)
s2 = sigma(1e14, Colos=True)

s1 = np.linspace(s2, sh, 100000)
deltas = s1-s2
ds = s1[1]-s1[0]
dw = np.logspace(-10, 0, 100)
res = []
res2 = []
y = special.erfc(dw/np.sqrt(2*(sh-s2)))
for el in dw:
    res.append(np.sum(cond(deltas[1:], el)*ds))
    res2.append(np.sum(dfdz(deltas[1:], el)*ds))
plt.plot(dw, res, label='non approx')
#plt.plot(dw, res2, label='approx')
plt.plot(dw, y, label = 'erfc')
plt.legend()
plt.show()'''
