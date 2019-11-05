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
    t0 = time.time()

    warnings.filterwarnings('ignore', category=RuntimeWarning)

    fpath = 'data/HARPS_Barnards_Star'
    ref_path = 'data/HARPS_Barnards_Star/reference.dat'
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
    post = ppc.PostProcess()
    res = []
    for file in flist:
        S = sp.Spec(filename=file)
        SS = pre.run(S)

        T = tf.TFA(temp, SS, 3)
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

    # Reference data
    ref = np.loadtxt(ref_path)
    sig_ref = np.std(ref[:, 1])

    # plot result
    plot_size = {'height_ratios': [2, 1]}
    props = props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw=plot_size)
    a0.plot(final_res[:, 0], final_res[:, 1], 
            label='my result', linewidth=0.7, marker='.')
    a0.plot(ref[:, 0], ref[:, 1], 
            label='reference', linewidth=0.7, marker='.')
    sig_text = '\n'.join((
                r'my $\sigma = {:.3f}$'.format(sig_mine),
                r'ref $\sigma = {:.3f}$'.format(sig_ref)))
    a0.text(0.6, 0.1, sig_text,
            transform=a0.transAxes, bbox=props)

    a0.grid(True)
    a1.plot(ref[:, 0], ref[:, 1]-final_res[:, 1],
            linewidth=0.7, marker='.')
    a1.grid(True)
    plt.xlabel('Julain Day')
    f.text(0.06, 0.5, 'Radial Velocity [m/s]',
        ha='center', va='center', rotation='vertical')
    a0.legend()
    plt.savefig('Average RV')

    print('Finished in {:.3f} seconds'.format(t1-t0))

