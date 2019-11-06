import common.macro as mc

import copy
import typing as tp
from astropy.io import fits
import numpy as np
import scipy.ndimage as img

import matplotlib.pyplot as plt

class TFAResult:
    # An Argument class used to store results from the 
    # template fitting algorithms 

    def __init__(self, m, jd):
        '''
        constructor
        initialize relevant result members
        '''
        
        # File specific information
        self.m = m
        self.julian_day = jd
        self.ran = False

        # order specific results from TFAs
        # [alpha, err, success, iteration]
        self.result = None
        self.rv = None

        # intermediate calculation results 
        self.R = None

    
    def append(self, a, e, s, ith):
        '''
        add to the bottom of the result
        '''
        if self.ran: 
            res = [a, e, s, ith]
            self.result = np.vstack([self.result, res])
            self.rv = np.vstack([self.rv, (1-a[0])*mc.C_SPEED])
        else: 
            self.result = [a, e, s, ith]
            self.rv = (1-a[0])*mc.C_SPEED
            self.ran = True
    
    def get_alpha(self):
        return self.result[:, 0]
    
    def get_error(self):
        return self.result[:, 1]
    
    def get_success(self):
        return self.result[:, 2]
    
    def get_iter(self):
        return self.result[:, 3]

    def print(self):
        '''
        print the average to terminal
        '''
        a = (1 - np.mean(self.result[:, 0], axis=0)[0]) * mc.C_SPEED * 1000,
        err = np.mean(self.result[:, 1], axis=0),
        ith = np.mean(self.result[:, 3])
        msg = '[{:6d}] RV={:7.3f}, sig={:.3f}, mean converge {:.3f}'.format(
            int(self.julian_day), a[0], err[0], ith)
        return msg
    
