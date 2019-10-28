import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import common.macro as mc
import common.argument.spec as sp
import common.primitive.process as pc

import tfa.tfa as tf
if __name__ == '__main__':
    fpath = 'data/HARPS_Barnards_Star'
    flist = mc.findfiles(fpath, '_e2ds_A.fits')

    print(flist[0])
    SP = sp.Spec(filename=flist[0])
    P = pc.ProcessSpec(SP)
    SSP = P.run()

    for file in flist[1:]:
        S = sp.Spec(filename=file)
        p = pc.ProcessSpec(S)
        SS = p.run()

        T = tf.TFA(SSP, SS)
        a_res, err_v, success, iteration = T.run(3)
        day = T.obs.julian_day

        print('[{:.0f}], RV={:7.3f}. sig={:.3f}. mean convergence {:.3f}'.format(day, 
              (1 - np.mean(a_res, axis=0)[0]) * mc.C_SPEED * 1000,
              np.mean(err_v, axis=0)[0],
              np.mean(iteration)))


