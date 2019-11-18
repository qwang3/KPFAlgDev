import common.macro as mc
import common.primitive.rm_outlier as rm

import sys, os
import typing as tp
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def flatten(lst: np.ndarray):
    '''
    flatten a nested numpy array
    '''
    return sum( ([x] if not isinstance(x, np.ndarray) else flatten(x)
            for x in lst), [])

class TFAResult:
    # An Argument class used to store results from the 
    # template fitting algorithms 

    def __init__(self, header):
        '''
        constructor
        '''
        self.header = header
        self.res_df = pd.DataFrame(columns=self.header)

    def append(self, order, inter_res):
        '''
        add to the bottom of the result
        '''
        flat_res = flatten(inter_res)
        assert(len(flat_res) == len(self.res_df.columns))
        self.res_df.at[order] = flat_res



class TFADebugRes(TFAResult): 
    # an argument class that stores debug information 
    # and value for each order.
    # all significant parameters from each step of the
    # newton solver is recorded.
    # the result member is formatted as:
    # [a, da, err, converge, k, X2]

    def __init__(self, obs_name: str, temp_name: str, order: int, m: int):
        '''
        constructor
        '''
        self.obs_name = obs_name
        self.temp_name = temp_name
        self.order = order
        self.exit_msg = None

        # order specific values to be recorded
        # result is stored as pandas dataframe
        # add a header for each value of alpha 

        # top level header [alpha, dalpha, error...]
        top = ['alpha' for n in range(m+2)] + ['d_alpha' for n in range(m+2)]
        top += ['error', 'convergence', 'kappa', 'Chi^2']
        # sub level header [alpha[-1], alpha[0], ...]
        # alpha[-1] is alpha_v (for easy numbering)
        sub = ['alpha[{}]'.format(n-1) for n in range(m+2)] \
                + ['d_alpha[{}]'.format(n-1) for n in range(m+2)]
        sub += ['', '', '', ''] 
        #overall header
        header = [np.array(top), np.array(sub)]
        TFAResult.__init__(self, header)

        self.weight = None
        self.ran = False
    
    def log_exit(self, exit_msg: str) -> None:
        '''
        order process exited with a failed messge
        '''
        self.exit_msg = exit_msg
    
    def append_weight(self, weight: np.ndarray) -> None:
        '''
        add weight 
        '''
        if type(self.weight) == type(None):
            self.weight = weight
        else:
            self.weight = np.vstack([self.weight, weight])

    def record(self, xlsx_writer, w_path):
        '''
        record to an .xlsx file
        each sheet in this file represnet the result of a order
        the structure of the file is 
            exit message 
            table of data
        '''
        # name of the excel sheet we are writing to 
        sheet_name = str(self.order)
        self.final_res = pd.DataFrame(columns=['alpha[v]', 'Error', 'success', 'iteration', 'Message'])
        self.files = pd.DataFrame(columns=['name'])

        a = self.res_df['alpha']['alpha[-1]'].values[-1]
        e = self.res_df['error'].values[-1]
        iteration = len(self.res_df.index)

        # first row is header
        if self.exit_msg == None:
            # exit message never updated, so this order processed successfully
            msg = 'order computed successfully'
            success = True
        else:
            msg = self.exit_msg
            success = False
          
        # write all data to the sheet 
        self.final_res.at[self.order] = [a, e, success, iteration, msg]
        self.final_res.to_excel(xlsx_writer, sheet_name=sheet_name, startrow=0)
        self.res_df.to_excel(xlsx_writer, sheet_name=sheet_name, startrow=3)

        # write the weight to a .dat file
        np.savetxt(w_path, self.weight, delimiter=',')

class TFAOrderResult(TFAResult):
    # An Argument class used to store results from the 
    # template fitting algorithms 

    def __init__(self, m, jd):
        '''
        constructor
        initialize relevant result members
        '''
        
        # File specific information
        self.m = m
        self.julian_day = jd

        # order specific results from TFAs
        top = ['RV[km/s]']+ (m+2)*['alpha'] + ['Error', 'success', 'iteration']
        sub = []
        for i in range(m+2):
            sub += ['alpha[{}]'.format(i-1)]
        bottom = [''] + sub + ['', '', '']
        header = [np.array(top), np.array(bottom)]
        TFAResult.__init__(self, header)

        self.rmo = rm.RemoveOutlier()

    def append(self, order, inter_res):
        '''
        add to the bottom of the result
        '''
        # [a[0], e, s, i]
        flat_res = flatten(inter_res)
        flat_res = [(1-flat_res[0])*mc.C_SPEED] + flat_res
        assert(len(flat_res) == len(self.res_df.columns))
        self.res_df.at[order] = flat_res
    
    def average(self):
        '''
        flat average 
        '''
        #[RV, Error, s_rate, iteration]

        rv = np.mean(self.res_df['RV[km/s]'].to_numpy())
        error = np.mean(self.res_df['Error'].to_numpy())
        return [rv, error]
    
    def weighted_average(self):
        '''
        weighted average
        '''
        rv = self.res_df['RV[km/s]'].to_numpy()
        weight = np.divide(1, np.square(self.res_df['Error'].to_numpy()))
        
        mu_e = np.sum(np.multiply(rv, weight)) / np.sum(weight)
        error = np.mean(self.res_df['Error'].to_numpy())
        ret_val = [mu_e, error]

        return ret_val
    def write_to_final(self):
        '''

        '''
        date = self.julian_day.isot
        jd = self.julian_day.jd
        result = np.asarray(self.average())
        converge = np.mean(self.res_df['iteration'])
        success_rate = np.mean(self.res_df['success'])
        return np.asarray([date, jd, result, converge, success_rate])

class TFAFinalResult(TFAResult):
    
    def __init__(self):
        '''
        constructor
        '''

        header = ['Date', 'Julian Date', 'RV[km/s]', 'Error', 'converge', 'success_rate']
        TFAResult.__init__(self, header)

    def to_csv(self, f_path: str):
        '''
        write to a .csv file
        '''
        self.res_df.to_csv(f_path)
    
    def convert_to_ms(self):
        self.res_df['RV[km/s]'] *= 1000.0
        self.res_df['Error'] *= 1000.0
        self.res_df.rename(columns={'RV[km/s]': 'RV[m/s]'})