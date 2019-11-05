import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import warnings
import time

import common.macro as mc
import common.argument.spec as sp
import common.primitive.process as pc
import common.primitive.postprocess as ppc

import tfa.tfa as tf


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    fpath = 'data/HARPS_Barnards_Star'
    ref_path = 'data/HARPS_Barnards_Star/reference.dat'
    flist = mc.findfiles(fpath, '_e2ds_A.fits')
    temp_name = 'template.fit'

    # print('Reading file from: {}'.format(fpath))

    # prelim = tf.prob_for_prelim(flist)

    # print('Preliminary template: \n{}'.format(prelim))
    # print('making template ...')
    # temp = tf.make_template(prelim, flist, 3)
    # temp.write_to('template.fits', 3)
    # print('template saved to: {}'.format(temp_name))

    deg = [0, 1, 2, 3, 4, 5, 6]
    sig = np.zeros_like(deg, dtype=np.float64)
    t = np.zeros_like(deg ,dtype=np.float64)
    for m in deg:
        # begin lap
        t0 = time.time()
        temp = sp.Spec(filename='template.fits')

        pre = pc.ProcessSpec()
        post = ppc.PostProcess()
        res = []
        for file in flist:
            S = sp.Spec(filename=file)
            SS = pre.run(S)

            T = tf.TFA(temp, SS, m)
            r = T.run()
            res.append(r)
            r.print()

        final_res = post.average(res)
        # sort by year
        final_res = final_res[np.argsort(final_res[:, 0])]
        # correct
        final_res[:, 1] -= final_res[0, 1]
        sig_mine = np.std(final_res[:, 1])
        # count lap time
        t1 = time.time()

        sig[m] = sig_mine
        t[m] = t1-t0

        print('[{}] finished with std={:.3f}, time={:.3f}'.format(m, sig_mine, t1-t0))
    plt.plot(deg, sig, linewidth=0.7, marker='.')
    plt.xlabel('Degree of normalization polynomial')
    plt.ylabel('Standard deviation')
    plt.grid(True)
    plt.savefig('StdVsDegree.png')

    plt.plot(deg, t, linewidth=0.7, marker='.')
    plt.xlabel('Degree of normalization polynomial')
    plt.ylabel('Time')
    plt.grid(True)
    plt.savefig('TimeVsDegree.png')
        
