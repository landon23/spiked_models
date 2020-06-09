
import numpy as np
import matplotlib.pyplot as plt

def marcenko_pastur(N, M, d=400):
    phi = N / M
    gammap = np.power(phi, 0.5)+ 1 / np.power(phi, 0.5)+2
    gammam = gammap-4
    x = 4*np.arange(d+1)/d+gammam
    mp = np.power(phi, 0.5)*np.power((x-gammam)*(gammap-x), 0.5)/x
    mp = mp  / (2*np.pi)
    print('check')
    return x, mp

def wishart(N, M, norm = False):
    X = np.random.normal(size=(N, M))
    if norm:
        X = X - np.mean(X, axis=0)
        X = X / np.power(np.sum(np.power(X, 2), axis=0), 0.5)

    U, S, V = np.linalg.svd(X)
    S = np.power(S, 2)
    if norm:
        S = S*np.power(N/M, 0.5)
    else:
        S = S / np.power(N*M, 0.5)
    return X, U, S, V

def spiked(N, M, lamb, u, v, norm=False):
    X = np.random.normal(size=(N, M))
    X = X / np.power(N*M, 0.25)
    u = np.reshape(u, (N, 1))
    v = np.reshape(v, (1, M))
    X = X + lamb*np.matmul(u, v)
    if norm:
        X = X - np.mean(X, axis=0)
        X = X / np.power(np.sum(np.power(X, 2), axis=0), 0.5)
        X = X * np.power(N / M, 0.25)
    U, S, V = np.linalg.svd(X)
    S = np.power(S, 2)
    return X, U, S, V

def spiked2(N, M):
    X = np.random.normal(size=(N, M))
    X = X / np.power(N*M, 0.25)
    u = np.random.normal(size=(N, 1))
    u = u / np.linalg.norm(u)
    v = np.random.normal(size=(N, 1))
    v = v / np.linalg.norm(v)

    I = np.random.permutation(N)
    v = u[I]
    print('It changed!')


    e1 = np.array([0, 1])
    e2 = np.array([1, 0])
    e1 = np.concatenate((e1, np.zeros(M-2)))
    e2 = np.concatenate((e2, np.zeros(M-2)))
    e1 = np.reshape(e1, (1, M))
    e2 = np.reshape(e2, (1, M))
    X = X + 4*np.matmul(u, e1)+4* np.matmul(v, e2)
    U, S, V = np.linalg.svd(X)
    S = np.power(S, 2)

    return X, U, S, V



def number(N, M, S):
    phi = N/M
    gammap = np.power(phi, 0.5) + 1 / np.power(phi, 0.5)+2
    n = np.where(S > gammap+1)[0].shape[0]
    return n
def edge_dist(N, M, S):
    lam = np.max(S)
    phi = N/ M
    gammap = np.power(phi, 0.5) + 1 / np.power(phi, 0.5)+2
    return lam - gammap

def delocalized(d):
    u = np.random.normal(size=d)
    u = u / np.linalg.norm(u)
    return u
def localized(n, d):
    u = np.random.normal(size=n)
    v = np.zeros(d-n)
    w = np.concatenate((u, v))
    w = w / np.linalg.norm(w)
    return w



def shuffled_spike(N, M, lamb, u, v, norm=False):


    X = np.random.normal(size=(N, M))
    X = X / np.power(N * M, 0.25)
    u = np.reshape(u, (N, 1))
    v = np.reshape(v, (1, M))
    X = X + lamb * np.matmul(u, v)
    for i in range(0, M):
        np.random.shuffle(X[:, i])
    if norm:
        X = X - np.mean(X, axis=0)
        X = X / np.power(np.sum(np.power(X, 2), axis=0), 0.5)
        X = X *np.power(N/M, 0.25)
    U, S, V = np.linalg.svd(X)
    S = np.power(S, 2)
    return X, U, S, V


def inhomogWishart(N, M, a, b, diag=True, verb= False, white = True):

    p = np.random.beta(a, b, size = (M, ))
    #p = 1-np.zeros(shape=(M, ))
    A = np.random.uniform(size=(N, M))
    B = np.zeros(shape=(N, M))
    for i in range(M):
        I = np.where(A[:, i] < p[i])
        B[I, i] = 1
    nnz = np.sum(B, axis=0)
    I = np.where(nnz < 1.5)
    if verb:
        print('Number of zero or one columns:', len(I[0]))
    if white:
        X = np.random.normal(size=(N, M))
        H = X*B
    else:
        #v = np.random.uniform(size=4)
        #X = np.random.choice(v, size=(N, M))
        X= np.random.standard_cauchy(size=(N, M))
        X = X / np.power((np.abs(X)), 0.4)

        H = X*B

    H =np.delete(H, I, axis=1)
    H = H - np.mean(H, axis=0)

    H = H / np.sqrt(np.sum(H*H, axis=0))
    if diag:
        U, S, V = np.linalg.svd(H)
        S = np.power(S, 2)
        return H, U, S, V
    else:
        return H, B

def MPwiki(x, lam, sig):
    lamMinus = np.power(sig*(1-np.sqrt(lam)), 2)
    lamPlus = np.power(sig*(1+np.sqrt(lam)), 2)
    val = np.sqrt((x-lamMinus)*(lamPlus-x))
    val = val / x
    val = val / (2 * np.pi * sig*sig*lam)
    return val

def plotWiki(n, lam, sig):
    lamMinus = np.power(sig * (1 - np.sqrt(lam)), 2)
    lamPlus = np.power(sig * (1 + np.sqrt(lam)), 2)
    x = np.arange(0, n+1)/ (n)* (lamPlus - lamMinus)*0.999+lamMinus

    y = MPwiki(x, lam, sig)
    return x, y

def params(mu, second):
    sig = np.sqrt(mu)
    lam = second / np.power(sig, 4) - 1
    return lam, sig

def heavy(N, M, p):
    X = np.random.standard_cauchy(size=(N, M))
    X = X / np.power(np.abs(X), 1-p)
    X = X / np.sqrt( np.sum(X*X, axis=0))
    #X = X / np.sqrt(N)
    U, S, V = np.linalg.svd(X)
    return U, S, V

def power(N, M, p0, p1=0.5, k=1, emp=False):
    #p = (np.arange(M)+1.0)/M
    #p = np.log(1 / p0)*(p - 1 )
    #p = np.exp(p)/2

    Z = np.log(p1)- np.log(p0)
    p = (np.arange(M)+1.0)/M
    p = Z*p + np.log(p0)
    p = np.exp(p)/2



    e = np.ones(shape=(N, 1))
    pMat = np.dot(e, np.reshape(p, (1, M)))
    X = np.random.uniform(size=(N, M))
    Y = np.random.uniform(size=(N, M))
    H = (pMat >X) + 0.0
    H = H + (pMat > Y)
    #plt.hist(p, bins=50)
    #plt.show()
    #empirical = False
    empirical = emp

    if empirical:
        nnz = np.sum(H, axis=0)
        I = np.where(nnz < 1.5)
        H =np.delete(H, I, axis=1)

        p = np.mean(H, axis=0)
    else:
        nnz = np.count_nonzero(H, axis=0)
        I = np.where(nnz < (k -0.5))


    H = H - 2*p
    std = np.sqrt(2 *p * (1-p))
    H = H / std
    if not empirical:
        H = np.delete(H, I, axis=1)

    H = H / np.sqrt(N)
    S = np.linalg.svd(H, compute_uv=False)
    if empirical:
        return S*S, H.shape[1]
    else:
        return S*S


def generalWishart(N, M, variance):
    X = np.random.normal(size=(N, M))
    X = X*variance
    X = X / np.sqrt(N)
    S = np.linalg.svd(X, compute_uv=False)
    return S*S

def powerTailWishart(N, M):
    X = np.random.normal(size=(N, M))
    gam = np.arange(M)/M
    gam = np.power(gam, 0.25)
    gam = gam /2
    gam = 1 - gam
    X = X*np.sqrt(gam)
    X = X / np.sqrt(N)
    U, S, V = np.linalg.svd(X)
    S2 = np.linalg.svd(X[:, 1:M], compute_uv=False)
    #columns of U are eigenvectors of XX^T and rows of V are eigenvectors of X^T X.
    return U, S*S, V,S2*S2

def powerLawSpiked(N, M, lamb):
    X = np.random.normal(size=(N, M))
    gam = np.arange(M) / M
    gam = np.power(gam, 0.25)
    gam = gam / 2
    gam = 1 - gam
    gam[0] = lamb + gam[0]
    X = X * np.sqrt(gam)
    X = X / np.sqrt(N)
    U, S, V = np.linalg.svd(X)
    S2 = np.linalg.svd(X[:, 1:M], compute_uv=False)
    # columns of U are eigenvectors of XX^T and rows of V are eigenvectors of X^T X.
    return U, S * S, V

def Hemp(x):
    M=500
    N = 2500
    gamma = (M-1)/N
    gam = np.arange(M) / M
    gam = gam[1:500]
    gam = np.power(gam, 0.25)
    gam = gam / 2
    gam = 1 - gam
    integ = np.mean(gam / (x-gam))
    psi = x + gamma*x*integ
    return psi

def psider(x):
    M = 500
    N = 2500
    gamma = (M-1)/N
    gam = np.arange(M) / M
    gam = gam[1:500]
    gam = np.power(gam, 0.25)
    gam = gam / 2
    gam = 1 - gam
    integ = np.mean(gam*gam / ((x-gam)*(x-gam)))
    return 1 - gamma*integ

def eta(x):
    M = 500
    N =2500
    gamma = (M-1)/N
    gam = np.arange(M) / M
    gam = gam[1:M]
    gam = np.power(gam, 0.25)
    gam = gam/2
    gam = 1 - gam

    psip = 1 - gamma*np.mean(gam*gam/ ((x-gam)*(x-gam)))
    psi = x+gamma*x*np.mean(gam / (x-gam))
    return x*psip / psi

