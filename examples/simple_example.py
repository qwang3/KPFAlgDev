import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import common.macro as mc
import common.argument.spec as sp
import common.primitive.process as pc

if __name__ == '__main__':
    order = 23
    fpath = 'data/HARPS_Barnards_Star'
    flist = mc.findfiles(fpath, '_e2ds_A.fits')
    
    S = sp.Spec(filename='template.fits')
    S.plot(order, color='r')
    SS = sp.Spec(filename=flist[10])
    SS.plot(order, color='b')
    plt.savefig('template {}'.format(order))
