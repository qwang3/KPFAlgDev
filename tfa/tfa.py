import numpy as np
import scipy.interpolate as ip
import matplotlib.pyplot as plt

from common import macro as mc
from common.primitive import process as pc
from common.argument import spec as sp
from common.argument import tfa_res as tr
from common.primitive import rm_outlier as rmo

# Opterations necessary for the entire template fitting algorithm
def prob_for_prelim(flist: list) -> str:
    ''' 
    find the file with the highest mean flux
    this file is used as the preliminary template
    '''
    best_file = None, 
    best_val = 0
    for f in flist:
        S = sp.Spec(filename=f)
        mean = np.mean(S._spec)
        if mean > best_val:
            best_val = mean
            best_file = f
    return best_file

def make_template(prelim: str, flist: list, m:int) -> sp.Spec:
    '''
    based on the files given in flist, create a tempkate 
    as specified in the HAPRS-TERRA paper. 
    [prelim] becomes the preliminary template 
    '''
    # Initilaize data processing methods
    P = pc.ProcessSpec()
    # Initialize the preliminary as the template
    SP = sp.Spec(filename=prelim)
    SP = P.run(SP) 

    n_files = len(flist)
    # get the wavelength and specs of the preliminary  
    # as foundation to the template
    twave = SP._wave
    tspec = SP._spec

    # Currently just a average of all spectrum
    # should also be taking care of the outliers (3-sigma clipping)
    for file in flist:
        S = sp.Spec(filename=file)
        SS = P.run(S)
        T = TFA(SP, SS, m)
        res = T.run()

        for i, order in enumerate(mc.ord_range):
            a = res.get_alpha()
            success = res.get_success()

            if success[i] == True:
                flamb, fspec = SS.get_order(order)
                fspec2 = np.interp(twave[order, :], flamb, fspec)
                tspec[order, :] += fspec2
            else: 
                n_files -= 1
    tspec = np.divide(tspec, n_files)
    return sp.Spec(data=(twave, tspec))

class TFA:

    def __init__(self, temp: sp.Spec, obs: sp.Spec, m: int):
        '''
        Constructor
        '''
        self.temp = temp
        self.obs = obs

        self.m = m
        self.res = tr.TFAResult(m, obs.julian_day)
        self.outlier = rmo.RemoveOutlier()

    def run(self):
        '''
        Run the template fitting algorithm 
        '''
        for i, order in enumerate(mc.ord_range): 
            a, err, s, it = self.solve_order(order)
            self.res.append(a, err, s, it)
        return self.res
        

    def common_range(self, x, y, w) -> mc.EchellePair_TYPE:
        '''
        
        '''
        idx = np.where(np.logical_and(
            x < np.amax(y),
            x > np.amin(y)
        ))
        return x[idx], w[idx]

    def correct(self, w):
        '''any nonvalid weight is set to zero'''
        w[np.where(np.isfinite(w) == 0)] = 0
        w[np.where(w <= 0)] = 0
        return w

    def solve_step(self, order: int,
                           a: mc.ALPHA_TYPE,
                           w0: np.ndarray) -> mc.ALPHA_TYPE:
        '''

        '''
        # Reference data
        tlamb, tspec = self.temp.get_order(order)
        # Observed data
        flamb ,fspec = self.obs.get_order(order)

        av_lamb = np.multiply(a[0], tlamb)
        # overlapping interval between tspec (F) and observed(f)
        # we can only compare the two in this interval
        # print(av_lamb.size, flamb.size)
        lamb, w= self.common_range(flamb, av_lamb, w0)
        tckF = ip.splrep(av_lamb, tspec)
        tckf = ip.splrep(flamb,fspec)
        F_av = ip.splev(lamb, tckF)
        f = ip.splev(lamb, tckf)


        # create f[lamb]*sum(a_m * (lamb - lamb_c)) in eqn 1
        am = a[1:]            # polynomial coefficients [a_0, a_1, ... a_m]
        c = int(lamb.size/2)  # index for center wavelength of each order
        px = lamb - lamb[c]
        # np.polyval is setup as:
        #    polyval(p, x) = p[0]*x**(N-1) + ... + p[N-1]
        # Thus we need to reverse am
        amf = np.flip(am)

        # f_coor = f[lamb]*sum(a_m * (lamb - lamb_c)) 
        poly = np.polyval(amf, px)
        f_corr = np.multiply(f, poly)
        
        # Final form of eqn 1
        R = F_av - f_corr

        ## calculate partial derivatives
        # eqn 3-4
        dR_dm = []
        grad = np.gradient(F_av,lamb)
        grad = np.nan_to_num(grad, nan=0)

        dF = np.multiply(lamb, grad)
        dR_dm.append(dF)
        for i in np.arange(self.m+1):
            dR_dm.append(-np.multiply(f, np.power(px, i)))

        ## setup hessian and eqn 8:
        #  summing all of pixels (res * 4096 * 76) in matrix
        A_lk = np.zeros((self.m+2, self.m+2))
        b_l = np.zeros((self.m+2, 1))
        # eqn 6 & 15
        for i in np.arange(0, self.m+2):
            for j in np.arange(0, self.m+2):
                A_lk[i, j] = np.sum(np.multiply(w,
                             np.multiply(dR_dm[i], dR_dm[j])))
            b_l[i] = -np.sum(np.multiply(w, np.multiply(dR_dm[i], R)))

        da = np.linalg.solve(A_lk, b_l)
        return da, R, A_lk

    def solve_order(self, order: int): 
        '''

        '''
        wave, flux = self.temp.get_order(order)
        flux = self.correct(flux)

        w = np.sqrt(flux)
        w = self.correct(w)
        # average flux of yje prder
        f_mean = np.sqrt(np.mean(flux))
        success = True

        # Initial alpha
        a = np.asarray([1,1] + [0] *self.m, dtype=np.float64)
        da = np.asarray([np.inf]*(self.m+2), dtype=np.float64)
        # Keep track of number of iterations to convergence
        iteration = 0
        # Convergence criteria
        converge = False
        err_v = 1

        while not converge: # convergence criteria specified in 2.1.5
            # solve
            w = self.correct(w)
            da, R, A_lk = self.solve_step(order, a, w)
            if len(R) == 1:
                # This happens when a initial guess is too far from the minimum
                success = False
                break

            # update
            # not removing outliers as in the IDL file
            da = np.reshape(da, a.shape)
            da[0] *= -1.0 # still have no idea why we negate this
            a += da

            R_sig = np.std(R)
            k = np.divide(R_sig, np.sqrt(f_mean))  # kappa in 2.1.2
            w = np.reciprocal(np.multiply(np.square(k), flux))

            error = np.sqrt(np.linalg.inv(A_lk).diagonal())
            err_v = np.multiply(error[0], mc.C_SPEED)

            # remove outlier
            bad = self.outlier.sigma_clip(R, 4)
            w[bad] = 0

            # record:
            # --TODO--
            # print('[{}] a={}, da={}, k={}'.format(iteration, a, da, k))
            iteration += 1
            if iteration > 50:
                # print("[{}] Failed: Infinite loop".format(order))
                success = False
                break

            converge = abs(da[0]*mc.C_SPEED) < 1e-6
            for i in range(0, self.m+1):
                converge &= abs(da[i+1]*mc.C_SPEED) < 1e-6

        return a, err_v, success, iteration


if __name__ == '__main__':
    pass
