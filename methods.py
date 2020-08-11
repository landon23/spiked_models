import numpy as np
import scipy.special
import scipy.integrate
import scipy.stats
import matplotlib.pyplot as plt

class fitter:

    def __init__(self, eig = None, N = None, M = None):
        if eig is None:
            self._data_loaded = False
        else:
            self._data_loaded = True
        self.eig = eig
        self.N = N
        self.M = M

    def return_fit(self, xx, nbins, nout, k):
        xbulk = xx[nout:]
        bulkmin, bulkmax = xbulk[0], xbulk[-1]
        print(bulkmin, bulkmax)
        xbulk = xbulk - xbulk[0]
        xbulk = xbulk / xbulk[-1]

        b, a = fit_cdf_power(xbulk, nbins, k=k)

        G, Gd = get_func(a, 0.5)

        rescal = lambda x: (x - bulkmin) / (bulkmax - bulkmin)
        F = lambda x: G(rescal(x) - b) / G(1 - b)
        Fd = lambda x: Gd(rescal(x) - b) / (G(1 - b) * (bulkmax - bulkmin))

        return F, Fd, b * (bulkmax - bulkmin) + bulkmin, bulkmax

    def set_params(self, N, M):
        self.N = N
        self.M = M

    def load_eigenvalues(self, eig):
        self.eig = eig
        self._data_loaded = True

    def square_root_est(self, eigs, n, k, nout):
        if not (np.diff(eigs) <= 0).all():
            print('Eigenvalues not sorted!')
            eigs = np.sort(eigs)[::-1]

        biggest = eigs[0]
        xx = eigs[0:n]
        xx = xx[0]-xx
        bulkev = eigs[nout:]

        G, Gd, lh, rh = self.return_fit(xx, 50, nout, k)

        Fd = lambda x : Gd(biggest-x)
        self.lh = lh
        self.rh = rh


        r = biggest-lh
        l = biggest-rh
        self.density = Fd
        self.l = l
        self.r = r


        #so we think of Fd as a density that integrates to 1 on the interval [l, r]

        gamma = self.M / self.N

        for i in range(nout):
            print('Outlier: ',i+1)
            print('Sample eigenvalue: ', eigs[i])
            f = lambda x : 1.0 / (x-eigs[i])
            m = integrator(bulkev,n,Fd, l,r,f)
            mtil = gamma*m +(gamma-1) / eigs[i]
            print('Estimated population eigenvalue:', -1 / mtil)
            mp = integrator(bulkev, n, Fd,l, r, lambda x : f(x)*f(x))
            mptil = gamma*mp + (1-gamma) / (eigs[i]*eigs[i])
            print('Sample eigenvalue standard deviation: ', np.sqrt(2.0 / (self.N*mptil)))
            print('Overlap:', -1.0*mtil / (eigs[i]*mptil))
            mppp = integrator(bulkev, n, Fd, l, r, lambda x : f(x)*f(x)*f(x)*f(x)*6)
            mppptil = gamma*mppp+(1-gamma)*6 / np.power(eigs[i], 4.0)

            print('Relative overlap std: ', eigs[i]*np.sqrt(mppptil / (3*self.N)))



class outlier:

    def __init__(self, N, M, sample = None, population = None, samp_std = None, overlap = None, over_norm_std = None, index = None):
        self.N = N
        self.M = M
        self.gam = M/ N
        self.sample = sample
        self.population = population
        self.samp_std = samp_std
        self.overlap = overlap
        self.over_norm_std = over_norm_std
        self.index = index
        self.inside_spec = False
    def report(self, give_over_std = True):
        # if self.inside_spec:
        #     print('Outlier inside fitted DOS')
        #     return
        # if self.index is not None:
        #     print('Sample eigenvalue index ', self.index, ' value:', self.sample)
        # else:
        #     print('Sample eigenvalue: ', self.sample)
        # if self.population is not None:
        #     print('Estimated population eigenvalue: ', self.population)
        # if self.samp_std is not None:
        #     print('Estimated sample eigenvalue std:, ', self.samp_std)
        # if self.overlap is not None:
        #     print('Estimated overlap:', self.overlap)
        # if self.over_norm_std is not None:
        #     print('Normalized overlap std:', self.over_norm_std)
        # if (self.over_norm_std is not None) and (self.overlap is not None) and give_over_std:
        #     print('overlap std:', self.over_norm_std*self.overlap)
        return (self.inside_spec, self.sample, self.population, self.samp_std, self.overlap, self.over_norm_std)


    def report_edge_dist(self, b, L, tw=True):
        if tw:
            print('Distance from TW-mean normalized by TW-std:', b/L)
        else:
            print('Distance from edge normalized by density inter-particle distance', b/L)

    def calculate(self, data):
        self.population, self.samp_std, self.overlap, self.over_norm_std = calc_outlier_quantities(self.N, self.M, data)

class density:
    def __init__(self, p, l, r, F=None, scaling_factor = None, sq=True, alpha = None):
        self.p =p
        self.l = l
        self.r = r
        self.F = F #optional CDF
        self.scaling_factor = scaling_factor
        self.sq = sq
        if not sq:
            self.alpha = alpha
        else:
            self.alpha = 0.5

class spectrum:
    def __init__(self, eigenvalues, N, M, nout = None, fit_power_law = False):
        self.eigenvalues = eigenvalues
        self.N = N
        self.M = M
        self.gam = M/N
        self.nout = nout
        if not fit_power_law:
            self.alpha =0.5
        else:
            self.alpha = None
        self.sq = not fit_power_law
    def fit(self,  nbins, n, k, nout=None, edge = None):
        #nout = number of outliers
        #n = number of ev to fit density
        #k degree of polynomial to use in fit.
        # if edge is passed, then it will be used, and an edge will not be fit
        self.nfit = n
        if nout is None:
            nout  = self.nout

        if nout is None:
            print('Supply number of outliers')
        else:
            if self.sq:
                #self.edge_density = square_root_fit(self.eigenvalues[nout:nout+n], nbins, k)
                self.edge_density = density_fit(self.eigenvalues[nout:nout+n], nbins, k, alpha = 0.5, sq=True, edge= edge)
                self.appr_esd = approximate_esd(self.edge_density, self.eigenvalues[nout:], n)
                self.tw_mean = self.appr_esd.dens.r -1.2065336*self.appr_esd.sf / np.power(self.M, 2/3)
                self.tw_std = np.sqrt(1.60778)*self.appr_esd.sf / np.power(self.M, 2/3)
            else:
                if self.alpha is None:
                    print('Please fit or set self.alpha')
                    return
                self.edge_density = density_fit(self.eigenvalues[nout:nout+n], nbins, k, alpha=self.alpha, sq=False, edge=edge)
                self.appr_esd=approximate_esd(self.edge_density, self.eigenvalues[nout:], n)
                self.edge = self.edge_density.r
                self.ip = self.appr_esd.ip


    def calc_outlier_quants(self, nout=None):
        if nout is None:
            nout = self.nout
        if nout is None:
            print('Supply number of outliers')
        else:
            self.outliers = dict()
            for i in range(nout):
                s = self.eigenvalues[i]

                self.outliers[i] = outlier(self.N, self.M, sample = s)
                if s < self.appr_esd.dens.r:
                    self.outliers[i].inside_spec = True
                else:
                    self.outliers[i].inside_spec = False
                    m = self.appr_esd.calc_m(s, 0)
                    mp = self.appr_esd.calc_m(s, 1)
                    mppp = self.appr_esd.calc_m(s, 3)
                    data = (s, m, mp, mppp)
                    self.outliers[i].calculate(data)

    def report(self, verbose = True):
        x = []
        for i in range(self.nout):
            x.append(self.outliers[i].report())
            if verbose:
                print_outlier_quants(x[i])
        return x


    def outlier_diagnostics(self, verbose = True):
        x = []
        for i in range(self.nout):
            outlier = self.outliers[i]
            if i == 0:
                d = self.eigenvalues[0]-self.eigenvalues[1]
            else:
                d = np.minimum(self.eigenvalues[i-1] - self.eigenvalues[i], self.eigenvalues[i]-self.eigenvalues[i+1])
            s = outlier.sample
            nearest  = d / outlier.samp_std
            edge_dist = (outlier.sample-self.appr_esd.dens.r) / outlier.samp_std

            if self.sq:
                q = (outlier.sample - self.tw_mean)/self.tw_std
            else:
                q = (outlier.sample - self.appr_esd.dens.r)/ self.appr_esd.ip
            tup = (s, nearest, edge_dist, q)
            x.append(tup)
            if verbose:
                print('Outlier with index', i, ' at location: ', s)
                print('Distance to nearest neighbouring eigenvalue normalized by outlier std: ', nearest)
                print('Distance to spectral edge normalized by outlier std: ', edge_dist)
                if self.sq:
                    print('Distance to TW mean normalized by TW std:', q)
                else:
                    print('Distance to spectral edge normalized by density interparticle distance', q)

        return x

    def fit_alpha(self, n, m, alphas):
        if self.nout is None:
            print('please set number of outliers')
            return
        x = self.eigenvalues[self.nout:self.nout+n]
        x = x[0]-x
        resid = power_grid(x, m, 1+alphas)
        i = np.argmin(resid)
        self.alpha = alphas[i]

    def plot_density(self, grid_size=1000, nbins=30):
        x = (np.arange(grid_size)/ grid_size)*(self.edge_density.r - self.edge_density.l)+self.edge_density.l
        y = self.edge_density.p(x)
        x = np.append(x, np.array([self.edge_density.r]))
        y = np.append(y, np.array([0.0]))
        plt.plot(x, y, color='red')
        plt.hist(self.eigenvalues[self.nout:self.nfit+self.nout], bins = nbins, density = True, alpha = 0.5)
        plt.show()

    def auto_sq_fit(self, nbins, n, k, edge_thresh = 4.0, over_thresh = 0.1, nmax = None, supplied_density = None):
        if nmax is None:
            nmax = int(self.M/4)
        cont = True
        self.nfit = n
        old_dens = square_root_fit(self.eigenvalues[0:n], nbins, k)
        old_esd = approximate_esd(old_dens, self.eigenvalues[0:], n)


        nout = 1
        calc_density = (supplied_density is None)

        while cont:
            if calc_density:
                edge_density = square_root_fit(self.eigenvalues[nout:nout + n], nbins, k)
                appr_esd = approximate_esd(edge_density, self.eigenvalues[nout:], n)
            else:
                edge_density = supplied_density.dens
                appr_esd = supplied_density
            tw_mean = appr_esd.dens.r - 1.2065336 * appr_esd.sf / np.power(self.M, 2 / 3)
            tw_std = np.sqrt(1.60778) * appr_esd.sf / np.power(self.M, 2 / 3)

            for i in range(nout):
                s = self.eigenvalues[i]
                if cont:
                    if s <= edge_density.r:
                        cont = False
                        ifail = i
                        message = 'Eigenvalue within spectral distribution'

                    elif nout >= nmax:
                        cont = False
                        ifail = -1
                        message = 'Max outliers reached'


                    else:
                        possible_outlier = outlier(self.N, self.M, sample = s)
                        is_outlier, message = test_outlier(possible_outlier, appr_esd, edge_thresh, over_thresh, tw_mean, tw_std)
                        if is_outlier == False:
                            cont = False
                            ifail = i

            if cont:
                nout = nout+1
                old_dens = edge_density
                old_esd = appr_esd
            else:
                nout = nout-1
                print('Number of outliers found: ', nout)
                print('Index of eigenvalue that failed test: ', ifail)
                print('Reason: '+message)
                self.nout = nout
                self.edge_density = old_dens
                self.appr_esd = old_esd
                self.tw_mean = old_esd.dens.r - 1.2065336 * old_esd.sf / np.power(self.M, 2 / 3)
                self.tw_std = np.sqrt(1.60778) * old_esd.sf / np.power(self.M, 2 / 3)



def test_outlier(out, asd, edge_thresh, over_thresh, tw_mean, tw_std):
    s = out.sample
    m = asd.calc_m(s, 0)
    mp = asd.calc_m(s, 1)
    mppp = asd.calc_m(s, 3)
    data = (s, m, mp, mppp)
    out.calculate(data)

    if s <= asd.dens.r:
        is_outlier = False
        message = 'Eigenvalue inside fitted spectral distribution'
    else:
        d = s - asd.dens.r
        if (d / out.samp_std) < edge_thresh:
            is_outlier = False
            message = 'Eigenvalue within threshold sample stds of spectral edge'
        elif (s - tw_mean) / tw_std < edge_thresh:
            is_outlier = False
            message = 'Eigenvalue within threshold TW stds of spectral edge'
        elif out.over_norm_std > over_thresh:
            is_outlier = False
            message = 'Normalized overlap std above threshold'
        else:
            is_outlier = True
            message = 'Passed tests'
    return is_outlier, message









class approximate_esd:

    def __init__(self, dens, bulkev, n):
        self.dens = dens
        self.bulkev = bulkev
        self.n = n
        self.wr = n / (len(bulkev.flatten()))
        self.wl = 1 - self.wr
        if dens.sq:
            if self.dens.scaling_factor is not None:
                self.sf = self.dens.scaling_factor / np.power(self.wr, 2/3)
        else:
            M = len(bulkev.flatten())
            alpha = self.dens.alpha
            sf = self.wr*self.dens.scaling_factor

            self.ip = np.power((1+alpha)/(sf*M), 1 / (1+alpha))
    def calc_m(self, E, k=0):
        f = lambda x : 1 / (x-E)
        F = lambda x : np.power(f(x), k+1)*scipy.special.factorial(k)
        p = self.dens.p
        g = lambda x : p(x) * F(x)
        cr = self.wr*scipy.integrate.quad(g, self.dens.l, self.dens.r)[0]
        cl = self.wl*np.mean(F(self.bulkev[self.n:]))
        return cl+cr





def calc_outlier_quantities(N, M, data):
    (s, m, mp, mppp) = data
    gam = M / N
    mtil = gam*m +(gam-1) / s
    population = - 1 / mtil

    if mp is not None:
        mptil = gam*mp + (1-gam) / np.power(s, 2)
        samp_std = np.sqrt(2 / (N*mptil))
        overlap = -1.0 * mtil / (s *mptil)
    else:
        samp_std = None
        overlap = None

    if mppp is not None:
        mppptil = mppp*gam + 6*(1-gam) / np.power(s, 4.0)
        over_norm_std = s * np.sqrt(mppptil / (3*N))
    else:
        over_norm_std = None

    return population, samp_std, overlap, over_norm_std




def square_root_fit(xx, nbins, k, edge = None):
    fit_edge = (edge is None)
    if fit_edge:
        xbulk = xx[0]-xx
    else:
        xbulk = edge - xx

    bulkmin, bulkmax = xbulk[0], xbulk[-1]

    xbulk = xbulk / xbulk[-1]

    b, a = fit_cdf_power(xbulk, nbins,  fit_edge=fit_edge, k=k)
    sf = 1 / ( np.power(3/2, 2/3)*a[1]/ (bulkmax-bulkmin))
    sf = sf / np.power(np.pi, 2/3)

    G, Gd = get_func(a, 0.5)

    #rescal = lambda x: (x - bulkmin) / (bulkmax - bulkmin)
    rescal = lambda x: x / bulkmax
    F = lambda x: G(rescal(x) - b) / G(1 - b)
    #Fd = lambda x: Gd(rescal(x) - b) / (G(1 - b) * (bulkmax - bulkmin))

    Fd = lambda x: Gd(rescal(x) - b) / (G(1 - b) * (bulkmax))
    if fit_edge:
        p = lambda x : Fd(xx[0]-x)
        cdf = lambda x : F(xx[0]-x)
        r = xx[0] -b * (bulkmax - bulkmin) + bulkmin
        l = xx[0] - bulkmax
    else:
        p = lambda x: Fd(edge - x)
        cdf = lambda x: F(edge - x)
        r = edge
        l = edge - bulkmax

    dens = density(p, l, r, F=cdf, scaling_factor = sf)
    return dens

def density_fit(xx, nbins, k, edge = None, sq = True, alpha = None):
    fit_edge = (edge is None)
    if sq:
        alpha = 0.5
    if (not sq) and (alpha is None):
        print('Fit an alpha first!')
        return
    if fit_edge:
        xbulk = xx[0] - xx
    else:
        xbulk = edge - xx
    bulkmin, bulkmax = xbulk[0], xbulk[-1]

    xbulk = xbulk / xbulk[-1]

    b, a = fit_cdf_power(xbulk, nbins, fit_edge=fit_edge, k=k, alpha=alpha)
    if sq:
        sf = 1 / (np.power(3 / 2, 2 / 3) * a[1] / (bulkmax-bulkmin))
        sf = sf / np.power(np.pi, 2 / 3)
    else:
        #some code about getting ip here
        sf = (alpha+1)*np.power(a[1], alpha+1) #F' = sf*(x-b)^alpha

    G, Gd = get_func(a, alpha)

    rescal = lambda x: x / bulkmax
    F = lambda x: G(rescal(x) - b) / G(1 - b)
    Fd = lambda x: Gd(rescal(x) - b) / (G(1 - b) * (bulkmax))

    if fit_edge:
        p = lambda x : Fd(xx[0]-x)
        cdf = lambda x : F(xx[0]-x)
        r = xx[0] -b * bulkmax
        l = xx[0] - bulkmax
    else:
        p = lambda x: Fd(edge - x)
        cdf = lambda x: F(edge - x)
        r = edge
        l = edge - bulkmax
    dens = density(p, l, r, F=cdf, scaling_factor = sf, sq=sq, alpha=alpha)
    return dens

def fit_cdf_power(x, n, k=1, alpha=0.5, fit_edge= True, verbose = False):
    #assume x is increasing
    #n is number of bins
    #k is order of polynomial
    #alpha is power of density, p(x) ~ x^\alpha

    F, right = get_cdf(x, n)
    F = F / F[-1]

    y = np.power(F, 1 / (1 + alpha))

    #out = np.polyfit(right, y, k)
    if fit_edge:
        P = np.polynomial.polynomial.Polynomial.fit(right, y, k)
        P.convert(domain=np.array([-1, 1]))
        ro = P.roots()
        if not np.isreal(ro).any():
            if verbose:
                print('no real roots found for initial fit of k= ', k,'. Instead fit k=1')
            k=1
            P = np.polynomial.polynomial.Polynomial.fit(right, y, k)
            P.convert(domain=np.array([-1, 1]))
            ro = P.roots()
        I = np.where(np.isreal(ro))[0]
        rero = ro[I]

        b = rero[np.argmin(np.abs(rero))]
        b = np.real(b)

    #fit y = a[0] (x-b) + a[1](x-b)^2 + ...
        a = np.zeros(shape=k)

        for i in range(k):
            a[i] = (P.deriv(i+1)).__call__(b) / scipy.special.factorial(i+1)
        return b, np.append(np.array([0]), a)
    else:
        P = np.polynomial.polynomial.Polynomial.fit(right, y, np.arange(1, k+1))
        P.convert(domain=np.array([-1, 1]))
        a = P.coef
        return 0.0, a



def get_cdf(x, n):
    #x is increasing
    #n+1 is number of points to fit F
    rang = x[-1] - x[0]
    right = rang * (np.arange(n + 1) - 0.5) / n + x[0]
    F = np.zeros(shape=(n + 1))
    j = 0
    current = x[0]
    for i in range(n + 1):
        edge = right[i]
        if i > 0:
            F[i] = F[i - 1]

        while current < edge:
            F[i] += 1
            j += 1
            if j < len(x):
                current = x[j]
            else:
                current = edge[n] + 10
    return F, right

def get_func(a, alpha):
    #let p be the polynomial p(x) = \sum_{n>0} a_{n} x^n
    #returns function objects for x -> F(x-b) = (\max p(x), 0 )^{alpha+1} and
    # F'(x) = p'(x) (p(x) )^alpha (again up to taking maxes)
    p = np.polynomial.polynomial.Polynomial(a)
    F = lambda x : np.power(np.maximum(0.0, p(x)), 1+alpha)
    pd = p.deriv(1)
    Fd = lambda x : (1+alpha)*np.power(np.maximum(0.0, p(x)), alpha)*np.maximum(0.0, pd(x))
    return F, Fd



def integrator(eigs, n, p, l, r, f):
    lefteigs = eigs[n:]
    wl = len(lefteigs) / len(eigs)
    wr = 1 - wl

    F = lambda x: wr*f(x)*p(x)
    cr = scipy.integrate.quad(F, l, r)[0]
    cl = wl*np.mean(f(lefteigs))
    return cr+cl


def power_law_fit(x, n, k, tol=0.001, max_iter =100):
    # in increasing order
    F, right = get_cdf(x, n)
    F = (F+0.001) / F[-1]
    iter=0
    cont = True
    replace = True
    b = x[0]

    bbest = 0
    abest = np.zeros(k+1)
    alphabest = 0.5
    resid = 0

    while cont:
        anew = np.zeros(k+1)
        bnew = np.zeros(k+1)

        iter +=1
        if iter > max_iter:
            print('Max iterations reached')
            cont = False
            return bbest, abest, alphabest

        I = np.where(right > b )[0]
        xx = right[I]-b
        yy = F[I]
        J = np.where(yy>0)[0]

        xx= np.log(xx[J])
        yy = np.log(yy[J])
        #xx = np.log(right[I]-b)
        #yy = np.log(F[I])
        regress = scipy.stats.linregress(xx, yy)
        slope, intercept = regress[0], regress[1]
        alphanew = slope
        print(slope)
        yy = np.power(F, 1 / slope)

        xx = right

        result = np.polynomial.polynomial.Polynomial.fit(xx, yy, k, full=True)
        P =result[0]
        print(P)
        new_resid = result[1][0]
        P.convert(domain=np.array([-1, 1]))

        ro = P.roots()
        if not np.isreal(ro).any():
            print('no real roots, returning best iteration')
            cont = False
            if iter == 1:
                print('first iteration, sorry')
                return bnew, anew, alphanew, bnew, anew, alphanew
            else:
                return bnew, anew, alphanew, bbest, abest, alphabest



        else:
            I = np.where(np.isreal(ro))[0]
            rero = ro[I]
            bnew = np.real(rero[np.argmin(np.abs(rero))])
            anew = np.zeros(k + 1)
            for i in range(1, k + 1):
                anew[i] = (P.deriv(i + 1)).__call__(b) / scipy.special.factorial(i + 1)

            if iter ==1:
                bbest, abest, alphabest = bnew, anew, alphanew
                residbest= new_resid

            elif new_resid < residbest:

                bbest, abest, alphabest = bnew, anew, alphanew
                residbest = resid
            if iter >1:
                if np.max(np.abs(aold-anew)) < tol and np.max(np.abs(bold-bnew))< tol and np.max(np.abs(alphanew-alphaold))<tol:
                    print('Tolerance achieved')
                    cont = False
                    return bnew, anew, alphanew, bbest, abest, alphabest

            aold = anew
            bold = bnew
            alphaold = alphanew
            b = bnew


        print('iteration:', iter)
        print('a:', aold)
        print('b:', bold)
        print('alpha:', alphaold)


def power_grid(x, n, betas):
    F, right = get_cdf(x, n)
    I = np.where(F>0)[0]
    F = F[I]
    F = F / F[-1]
    right = right[I]
    resids = np.zeros(len(betas))

    for i in range(len(betas)):
        f = np.power(F, 1 / betas[i])
        result = scipy.stats.linregress(right, f)
        sl = result[0]
        inte = result[1]
        pr = inte + sl*right
        resids[i] = np.mean(np.power(f-pr, 2.0)) /np.var(f)
    return resids




def print_outlier_quants(tup):
    if tup[0]:
        print('Eigenvalue at: ', tup[1], 'inside spectrum')
    else:
        print('Location: ', tup[1])
        print('Population eigenvalue estimate: ', tup[2])
        print('Sample eigenvalue std:', tup[3])
        print('Estimated overlap:', tup[4])
        print('Overlap standard error:', tup[5])


def make_outlier_dist_sv(appr_esd, N, M):
    #returns the function s --> (s-R) / outlier std
    R = appr_esd.dens.r
    gam = M / N
    mp = lambda x : appr_esd.calc_m(x, k=1)
    mptil = lambda x : gam*mp(x) + (1-gam) / np.power(x, 2.0)
    std = lambda x : np.sqrt( 2 / (N*mptil(x)))

    return lambda x : (x-R) / std(x)

def make_overlap_sterror_sv(appr_esd, N, M):
    gam = M/N
    mppp = lambda x : appr_esd.calc_m(x, k=3)
    mppptil = lambda x : gam*mppp(x) + (1-gam)*6 / np.power(x, 4.0)
    return lambda x : x * np.sqrt(mppptil(x) / (3*N))

def wrapper(p):

    def f(x):
        sh = x.shape

        x = x.flatten()
        y = np.zeros(x.shape)
        for i in range(len(x)):
            y[i] = p(x[i])
        y = y.reshape(sh)
        return y
    return f




