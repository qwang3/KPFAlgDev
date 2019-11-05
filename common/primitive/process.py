import typing as tp
# Using Python3's PEP 484 typing

from astropy.io import fits
import numpy as np
import scipy.ndimage as img

import matplotlib.pyplot as plt

# Local dependencies
from common import macro as mc
from common.argument import spec as sp

class ProcessSpec:
    '''
    Action Object class that process the input spec
    '''
    def __init__(self):
        '''

        '''
    
    def bary_correct(self, spec: sp.Spec) -> sp.Spec: 
        '''

        '''
        berv = spec.header['eso drs berv']
        dlamb = np.sqrt((1+berv/mc.C_SPEED)/(1-berv/mc.C_SPEED))
        for order in range(spec.NOrder): 
            spec.shift_wave(dlamb, order)

        return spec
    
    def run(self, spec: sp.Spec) -> sp.Spec:
        '''
        
        '''
        spec = self.bary_correct(spec)
        return spec


if __name__ == '__main__':
    pass
