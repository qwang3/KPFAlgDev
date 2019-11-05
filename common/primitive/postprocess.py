import typing as tp
# Using Python3's PEP 484 typing

from astropy.io import fits
import numpy as np
import scipy.ndimage as img

import matplotlib.pyplot as plt

# Local dependencies
from common import macro as mc
from common.argument import spec as sp
from common.primitive import rm_outlier as rmo

class PostProcess:
    '''
    Action Object class that process the input spec
    '''
    def __init__(self):
        '''

        '''
        self.correct = rmo.RemoveOutlier()
    def average(self, result: list) -> np.ndarray:
        '''
        input should be a list of tfa_res class
        sort
        '''
        earliest = min(result, key=lambda x: x.julian_day)
        jd0 = earliest.julian_day

        # [julian_day, rv, err]
        final_result = np.zeros([len(result), 3])

        for i, res in enumerate(result):
            rv = res.rv
            bad = self.correct.sigma_clip(res.rv, 2)
            rv[bad] = 0
            mu_e = np.mean(rv)
            err = np.mean(res.get_error())
            # sec acc correct
            year = np.divide(res.julian_day - jd0, 365.25)
            offset = np.multiply(year, mc.SEC_ACC)
            mu_e += offset

            final_result[i] = [res.julian_day, -mu_e, err]
        return final_result
    
    def weighted(self, result: list) -> np.ndarray:
        '''
        input should be a list of tfa_res class
        sort
        '''
        earliest = min(result, key=lambda x: x.julian_day)
        jd0 = earliest.julian_day

        # [julian_day, rv, err]
        final_result = np.zeros([len(result), 3])

        for i, res in enumerate(result):
            # eqn 10, 11
            rv = res.rv
            bad = self.correct.sigma_clip(res.rv, 2)
            rv[bad] = 0
            
            sig2 = np.square(res.get_error())
            Z = np.sum(np.reciprocal(sig2))
            mu_e = np.sum(np.divide(rv, sig2))/Z
            err = np.mean(res.get_error())


            # sec acc correct
            year = np.divide(res.julian_day - jd0, 365.25)
            offset = np.multiply(year, mc.SEC_ACC)
            mu_e += offset

            final_result[i] = [res.julian_day, -mu_e, err]
        return final_result

    def run(self, result: list) -> np.ndarray:
        '''
        
        '''
        final_result = self.sec_acc(result)
        return final_result


if __name__ == '__main__':
    pass
