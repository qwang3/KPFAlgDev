import typing as tp 
import numpy as np

ECHELLE_SHAPE = [72, 4096]
C_SPEED = 2.99792458e5 # [km/s] speed of light

# Data type
EchelleData_TYPE = tp.NewType('EchelleData_TYPE', np.ndarray)
EchellePair_TYPE = tp.Tuple[EchelleData_TYPE, EchelleData_TYPE]

# Result type
ALPHA_TYPE = tp.NewType('alpha_TYPE', np.ndarray)