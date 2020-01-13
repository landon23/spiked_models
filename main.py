
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



