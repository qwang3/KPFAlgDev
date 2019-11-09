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

    f_path = 'data/HARPS_Barnards_Star'
    ref_path = 'data/HARPS_Barnards_Star/reference.dat'
    config_path = 'examples/debug.cfg'
    temp_path = 'template.fits'

    f_list = mc.findfiles(f_path, '_e2ds_A.fits')
    prelim = tf.prob_for_prelim(f_list)

    tfa_module = tf.TFAModule(config_file=config_path)
    temp = tfa_module.make_template(prelim, f_list)
    result = tfa_module.calc_velocity(temp, f_list)
    print(result)




    