import numpy as np
import pandas as pd
import scipy.interpolate as ip
import matplotlib.pyplot as plt

import sys, os
import copy
import typing as tp

import common.macro as mc
from common.argument.spec import Spec
from common.primitive.process import ProcessSpec

class DebugTool:
    '''
    A class that is used to process the debug files, after the TFA routine 
    is ran in debug mode. Each DebugTool() instance is intended for processing
    one debug file
    '''

    def __init__(self, debug_file: str, temp_path: str, star='barnards_star'):
        '''
        constructor
        ''' 
        self.name = debug_file
        self.df = pd.ExcelFile(debug_file + '.xlsx')
        self.weights = np.genfromtxt(debug_file + '.dat', delimiter=',')
        self.order_res = []
        self.order_final_res = []
        for n in range(len(self.df.sheet_names)):
            final_res = self.df.parse(sheet_name=self.df.sheet_names[n], nrows=1, index_col=0)
            inter_res = self.df.parse(sheet_name=self.df.sheet_names[n], header =[0, 1], skiprows=3)
            self.order_res.append(inter_res)
            self.order_final_res.append(final_res)
        self.temp = Spec(filename=temp_path)
        epoch = os.path.basename(debug_file)
        if star == 'barnards_star':
            f_name = './data/HARPS_Barnards_Star_all/' + epoch + '.fits'
        elif star == 'tau_ceti':
            f_name = './data/HARPS_Tau_Ceti_all/' + epoch + '.fits'
        self.obs = Spec(filename=f_name)
    
        self.P = ProcessSpec()
        self.obs = self.P.run(self.obs)
        # print(self.order_final_res[0])
        # print(self.order_res[0])
    
    def _get_R(self, order: int, iteration: int) -> tp.Tuple:
        '''
        return the difference between obs and template,
        on the final iteration, normalized by mean
        '''
        # the excel sheet that stores the order
        sheet = self.df.sheet_names.index(str(order))
        # final result for alpha
        a = self.order_res[sheet]['alpha'].to_numpy()[iteration]

        obs = copy.deepcopy(self.obs)
        obs.shift(a, order)

        tlamb, tspec = self.temp.get_order(order)
        flamb, fspec = obs.get_order(order)
        w = self.weights[-1]
        
        ### plotting the result without weight
        # w = np.ones_like(w)

        lamb, w= mc.common_range(flamb, tlamb, w)
        tckF = ip.splrep(tlamb, tspec)
        tckf = ip.splrep(flamb,fspec)
        tF = np.multiply(w, ip.splev(lamb, tckF))
        tf = np.multiply(w, ip.splev(lamb, tckf))


        return tF, tf, lamb, w

    def get_all_failed_order(self) -> list:
        '''
        return a list of failed orders
        '''
        orders = []
        for result in self.order_final_res:
            if result['success'].values[0] == False:
                orders.append(result['success'].index[0])
        return orders
    
    def get_n_iteration(self, order: int) -> int:
        '''
        return the number of iterations a specific order went through
        '''
        sheet = self.df.sheet_names.index(str(order))
        return len(self.order_res[sheet].index) - 1

    def get_normalized_R(self, order: int, iteration: int) -> tp.Tuple:

        tF, tf ,lamb, w = self._get_R(order, iteration)
        tF /= np.mean(tF)
        tf/= np.mean(tf)
        return tF, tf, lamb, w
    
    def plot_R_histogram(self, order: int, iteration: int, name: str) -> None:
        '''
        get the difference from specified order and iteration, 
        and plot the histogram of that iteration
        '''
        tF, tf ,_, _ = self.get_normalized_R(order, iteration)
        R = (tF - tf)
        bins = np.linspace(np.amin(R), np.amax(R), 1000)
        hist = np.histogram(R, bins=bins, density=True)
        
        f, ax = plt.subplots()

        # ax.hist(hist)
        ax.plot(np.linspace(np.amin(R), np.amax(R), 999), hist[0])
        f.savefig(name, dpi=300)

    def plot_iteration_result(self, order: int, iteration: int, name: str) -> None:
        '''
        plot the intermeidate result of the solver 
        '''
        # the sheets are named by the order 
        # get the sheet index based on the given order
        tF, tf ,lamb, w = self.get_normalized_R(order, iteration)

        wd = {'height_ratios': [2, 1]}
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw=wd)
        comment1 = r'template'
        a0.plot(lamb, tF, color='r', linewidth=0.5, label=comment1)
        comment2 = r'observation '
        a0.plot(lamb, tf, color='b', linewidth=0.5, label=comment2)
        a0.set_ylabel('Observed and Template Spectrum')
        a0.tick_params(axis='x', bottom=False, labelbottom=False)

        commentd = r'Diff $\sigma=$ {:.6E}, $\chi^2=$ {:.6E}'.format(np.std(tF-tf), np.sum(np.square(tF-tf)))
        a1.plot(lamb, tF - tf, linewidth=0.5, label=commentd)
        a1.set_xlabel('Wavelength [nm]')
        a1.set_ylabel('Difference')

        a0.legend()
        a0.grid(True)
        a1.legend()
        a1.grid(True)
        f.savefig(name, dpi=300)







    
