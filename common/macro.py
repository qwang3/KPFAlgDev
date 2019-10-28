import typing as tp 
import numpy as np

import os
import sys

ECHELLE_SHAPE = [72, 4096]
C_SPEED = 2.99792458e5 # [km/s] speed of light

# Data type
EchelleData_TYPE = tp.NewType('EchelleData_TYPE', np.ndarray)
EchellePair_TYPE = tp.Tuple[EchelleData_TYPE, EchelleData_TYPE]

# Result type
ALPHA_TYPE = tp.NewType('alpha_TYPE', np.ndarray)

ord_range = [i for i in range(71) if i > 20 and i != 57 and i != 66 ]
# Some helpful general purpose function
def findfiles(fpath, extension):
    '''
    find all the files in the sub directories with relevant extension
    '''
    lst_fname = []
    for dirpath,_, filenames in os.walk(fpath):
        for filename in [f for f in filenames if f.endswith(extension)]:
            lst_fname.append(os.path.join(dirpath, filename))
    return lst_fname