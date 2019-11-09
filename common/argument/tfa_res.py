import common.macro as mc

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

    def __init__(self, order: int, m: int):
        '''
        constructor
        '''

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

        self.ran = False
    
    def log_exit(self, exit_msg):
        '''
        order process exited with a failed messge
        '''
        self.exit_msg = exit_msg

    def record(self, xlsx_writer):
        '''
        record to an .xlsx file
        each sheet in this file represnet the result of a order
        the structure of the file is 
            exit message 
            table of data
        '''
        # name of the excel sheet we are writing to 
        sheet_name = 'order ' + str(self.order)

        # first row is header
        if self.exit_msg == None:
            # exit message never updated, so this order processed successfully
            msg = 'order computed successfully'
        else:
            msg = self.exit_msg
          
        # write all data to the sheet 
        self.res_df.to_excel(xlsx_writer, sheet_name=sheet_name, startrow=2)

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
        return np.mean(self.res_df[:].to_numpy())
    
    def weighted_average(self):
        '''
        weighted average
        '''
        rv = self.res_df['RV[km/s]'].to_numpy()
        weight = np.divide(1, np.square(self.res_df['Error']))
        mu_e = np.sum(np.multiply(rv, weight)) / np.sum(weight)
        error = np.sqrt(np.divide(1, np.sum(weight)))
        ret_val = [mu_e, error]

        return ret_val
    def write_to_final(self):
        '''

        '''
        date = self.julian_day.isot
        jd = self.julian_day.jd
        result = np.asarray(self.weighted_average())
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

    def record(self, f_path: str):
        '''
        write to a .csv file
        '''
        self.res_df.to_csv(f_path)
    
    def convert_to_ms(self):
        self.res_df['RV[km/s]'] *= 1000.0
        self.res_df['Error'] *= 1000.0
        self.res_df.rename(columns={'RV[km/s]': 'RV[m/s]'})