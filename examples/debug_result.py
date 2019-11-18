import sys, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
import time, os

import common.macro as mc
from tools import debugtool as dt


if __name__ == '__main__':

    warnings.filterwarnings('ignore', category=RuntimeWarning)

    star = 'tau_ceti'


    result_file = 'result.csv'
    # temp_path = 'template.fits'
    temp_path = 'template_tau_ceti.fits'

    dfs = mc.findfiles('debug', '.xlsx')
    DTs = []

    for files in dfs[0:3]:
        name = os.path.splitext(files)[0]
        DTs.append(dt.DebugTool(name, temp_path, star=star))

    for i, item in enumerate(DTs):
        print(item.name)
        item.plot_iteration_result(50, -1, 'plot {}'.format(i))
        item.plot_R_histogram(50, -1, 'histogram {}'.format(i))

    
