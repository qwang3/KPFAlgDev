import sys, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
import time

import common.macro as mc
import common.primitive.rm_outlier as rmo


if __name__ == '__main__':

    warnings.filterwarnings('ignore', category=RuntimeWarning)
    RO = rmo.RemoveOutlier()

    # ref_path = 'data/HARPS_Barnards_Star_benchmark/reference.dat'
    # ref_path = 'ref.dat'
    ref_path = 'tau_ceti_ref.dat'

    result_file = 'result.csv'


    Df = pd.read_csv(result_file)
    time = Df['Julian Date'].values
    # RV values
    rv = Df['RV[km/s]'].values
    # Error values
    # rv = Df['Error'].values

    my_res = np.asarray([list(pair) for pair in sorted(zip(time, rv))])

    # For RV
    my_res[:, 1] *= -1
    year = np.divide(my_res[:, 0] - my_res[0, 0], 365.25) 
    offset = np.multiply(year, mc.SEC_ACC*1000)
    # my_res[:, 1] -= offset
    bad_mine = RO.sigma_clip(my_res[:, 1], 1)
    my_res[:, 1][bad_mine] = 0

    my_res[:, 1] -= np.mean(my_res[:, 1])
    sig_mine = np.std(my_res[:, 1])

    # Reference data
    ref = np.loadtxt(ref_path)
    ref = np.asarray([list(pair) for pair in sorted(zip(ref[:, 0], ref[:, 1]))])
    # ref[:, 1] -= offset
    bad_ref = RO.sigma_clip(ref[:, 1], 1)
    ref[:, 1][bad_ref] = 0

    ref[:, 1] -= np.mean(ref[:, 1])
    sig_ref = np.std(ref[:, 1])

    # plot result
    plot_size = {'height_ratios': [2, 1]}
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw=plot_size)
    a0.plot(my_res[:, 0], my_res[:, 1], 
            label=r'my result. $\sigma = {:.3f}$'.format(sig_mine), linewidth=0.7, marker='.')
    a0.plot(ref[:, 0], ref[:, 1], 
            label=r'CCF reference. $\sigma = {:.3f}$'.format(sig_ref), linewidth=0.7, marker='.')
    a0.grid(True)
    a0.tick_params(axis='x', bottom=False, labelbottom=False)
    a0.set_ylabel('Radial Velocity [m/s]')
    a1.plot(ref[:, 0], ref[:, 1]-my_res[:, 1],
            linewidth=0.7, marker='.', 
            label=r'Difference. $\sigma = {:.3f}$'.format(np.std(ref[:, 1]-my_res[:, 1])))
    a1.grid(True)
    a1.set_xlabel('Julain Date')
    a1.set_ylabel('Diff')

    a0.legend()
    a1.legend()
    plt.savefig('Average flat RV removed outlier', dpi=300)
    

