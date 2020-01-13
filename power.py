import numpy as np

def power(N, beta, p):
    d = np.arange(N) +1.0
    d = np.power(d, beta)
    d = 1 / d
    d = d / np.sum(d)
    dd = np.reshape(d, (N, 1))
    prob = N*p* np.dot(dd, dd.transpose())
    U = np.random.uniform(size=(N, N))
    X = (prob > U) + 0.0
    Y = np.random.normal(size=(N, N))
    H = X*Y
    H = H / np.sqrt(p)
    a = N * np.sum(d*d)
    H = H / np.sqrt(a)

    H = np.triu(H)+np.triu(H, k=1).transpose()
    D, V = np.linalg.eigh(H)

    return D, V

def wig(d=400):
    x = 4.0*np.arange(d+1)/d - 2.0
    s = 4 - x*x
    s = np.power(s, 0.5)/ (2 * np.pi )
    return x, s


def chung(N, beta, m, d, diag = True):
    inot = N*np.power(d*(beta-2) / (m*(beta-1)), beta-1)
    inot = np.int(inot)+1

    pp = np.arange(N+inot)+1.0
    w = pp[inot:N+inot]
    w = np.power(w, (1.0)/ (beta-1))
    w = 1 / w
    c = d * (beta-2) * np.power(N, 1 / (beta-1)) / (beta-1)
    w = c * w
    rho = np.sum(w)
    ww = np.reshape(w, (N, 1))
    prob = np.dot(ww, ww.transpose()) / rho
    U = np.random.uniform(size=(N, N))
    X = (prob > U)+0.0
    Y = np.random.normal(size=(N, N))
    H = X*Y
    ###NORMALIZE!!!!
    a = np.sum(w*w) / rho
    H = H / np.sqrt(a)


    H = np.triu(H) + np.triu(H, k=1).transpose()
    if diag:
        D, V = np.linalg.eigh(H)
        return D, V, H
    else:
        return H

def addit(N, lamb, beta, m, d):
    X = chung(N, beta, m, d, diag=False)
    Y = np.random.normal(size=(N, N))
    Y = (Y+ Y.transpose()) / np.sqrt(2)
    Y = Y / np.sqrt(N)

    H = np.sqrt(lamb)*Y + np.sqrt(1-lamb)*X

    D, V = np.linalg.eigh(H)
    return D, V, H


