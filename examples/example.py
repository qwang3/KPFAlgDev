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
    temp = tf.make_template(prelim, flist, 3)
    temp.write_to('template.fits', 3)
    print('template saved to: {}'.format(temp_name))

    pre = pc.ProcessSpec()
    res = []
    for file in flist:
        S = sp.Spec(filename=file)
        SS = pre.run(S)

        T = tf.TFA(temp, SS, 3)
        r = T.run()
        res.append(r)
        r.print()

    t1 = time.time()
    print('Finished in {:.3f} seconds'.format(t1-t0))

