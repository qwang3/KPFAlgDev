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
    def __init__(self, spec: type(sp.Spec)):

        self._spec = spec
        self._ret_spec = spec.copy()
    
    def bary_correct(self) -> None: 
        '''

        '''
        berv = self._spec.header['eso drs berv']
        dlamb = np.sqrt((1+berv/mc.C_SPEED)/(1-berv/mc.C_SPEED))
        for order in range(self._ret_spec.NOrder): 
            self._ret_spec.shift_wave(dlamb, order)
    
    def run(self) -> type(sp.Spec):
        '''
        
        '''
        self.bary_correct()
        return self._ret_spec


if __name__ == '__main__':
    S = sp.Spec()
    P = ProcessSpec(S)
    pass
