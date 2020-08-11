import numpy as np
import scipy.integrate

def spikedWishart(N, M, spikes):
    Z = np.random.normal(size = (N, M))
    std = np.append(np.sqrt(spikes), np.ones(shape = M-len(spikes)))
    Z = Z*std / np.sqrt(N)
    U, S, V = np.linalg.svd(Z)
    return U, S*S, V

def mp(x, gam):
    #gamma between 0<gamma <1
    gp = 1 + np.sqrt(gam)
    gp = gp*gp
    gm = 1 - np.sqrt(gam)
    gm = gm*gm

    p = np.sqrt((gp-x)*(x-gm)) / ( gam*x)
    p = p / (2*np.pi)
    return p

def mz(E,gam):

    gp = 1+ np.sqrt(gam)
    gp = gp*gp
    gm = 1 - np.sqrt(gam)
    gm = gm*gm

    f = lambda x : mp(x, gam) / (x-E)
    m = scipy.integrate.quad(f, gm, gp)[0]
    return m

def mtilz(E, gam):

    m = mz(E, gam)
    return gam*m+(gam-1) / E


def mpz(E, gam):
    gp = 1 + np.sqrt(gam)
    gp = gp * gp
    gm = 1 - np.sqrt(gam)
    gm = gm * gm

    f = lambda x : mp(x, gam) / ((x-E)*(x-E))

    mder = scipy.integrate.quad(f, gm, gp)[0]
    return mder

def mptilz(E, gam):
    mder = mpz(E, gam)
    return gam*mder+(1-gam) / (E*E)
