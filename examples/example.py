import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import warnings
import time

import common.macro as mc
import common.argument.spec as sp
import common.primitive.process as pc

import tfa.tfa as tf
if __name__ == '__main__':
    t0 = time.time()

    warnings.filterwarnings('ignore', category=RuntimeWarning)

    fpath = 'data/HARPS_Barnards_Star'
    flist = mc.findfiles(fpath, '_e2ds_A.fits')
    temp_name = 'template.fit'

    print('Reading file from: {}'.format(fpath))

    prelim = tf.prob_for_prelim(flist)

    print('Preliminary template: \n{}'.format(prelim))
    print('making template ...')
    temp = tf.make_template(prelim, flist)
    temp.write_to('template.fits', 3)
    print('template saved to: {}'.format(temp_name))

    for file in flist[1:]:
        S = sp.Spec(filename=file)
        p = pc.ProcessSpec(S)
        SS = p.run()

        T = tf.TFA(temp, SS)
        a_res, err_v, success, iteration = T.run(3)
        day = T.obs.julian_day

        print('[{:7.0f}] RV={:7.3f}, sig={:.3f}, mean converge {:.3f}, success rate:{:.2f}'.format(day, 
              (1 - np.mean(a_res, axis=0)[0]) * mc.C_SPEED * 1000,
              np.mean(err_v, axis=0)[0],
              np.mean(iteration),
              np.mean(success)))
    t1 = time.time()
    print('Finished in {:.3f}seconds'.format(t1-t0))

