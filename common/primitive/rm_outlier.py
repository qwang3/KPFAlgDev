import common.macro as mc

import copy
import typing as tp
from astropy.io import fits
import numpy as np
import scipy.ndimage as img

import matplotlib.pyplot as plt

class RemoveOutlier:

    def __init__(self):
        '''

        '''
    def sigma_clip(self, x, factor):
        '''
        perform a sigma clipping on the given data
        outlier is set to val, default 0
        '''
        x_mu = np.mean(x)
        x_sig = np.std(x)
        # boundaries
        up = x_mu + np.multiply(factor, x_sig)
        down = x_mu - np.multiply(factor, x_sig)

        too_high = (x - x_mu) > up
        too_low = (x - x_mu) < down
        bad = np.where(np.logical_or(too_high, too_low))
        return bad