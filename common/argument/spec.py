import common.macro as mc

import copy
import typing as tp
from astropy.io import fits
import numpy as np
import scipy.ndimage as img

import matplotlib.pyplot as plt

class Spec:
    '''
    Argument object class that contains data from the echelle spectrum 
    Can be initialize to shape [72, 4096] of zeros by Spec()
    Can read from a .fits file by Spec(filename: str)
    can be inialized by 
    '''

    def __init__(self, data: mc.EchellePair_TYPE=None, 
                       filename: str=None) -> None:
        '''
        Constructor of the Spec Class 
        '''
        # a dictionary of flags, indicating how the class is initialized
        # certain method of this class can only be called when certain 
        # flag is set to True
        self.flag = {'from_file': False, 
                     'from_array': False}


        # The shape of wave and data must be the dimension 
        # defined by the global
        self._wave = np.zeros(mc.ECHELLE_SHAPE)
        self._spec = np.zeros(mc.ECHELLE_SHAPE)
        self.NOrder = mc.ECHELLE_SHAPE[0]
        self.NPixel = mc.ECHELLE_SHAPE[1]

        # If the data is generated from a .fits file, 
        # self.filename contains the .fits file destination 
        # self.header contains all headers of that  file 
        self.filename = None
        self.header = {}

        # Now initialize the class based on the given input
        if filename != None and data != None: 
            # error! Cannot read from both a file and a defined spectrum
            # --TODO-- 
            # Make this more informative
            print('what')
        elif filename != None:
            # a file is specified, so read data from that file 
            # prioritize data from file, so overlook any xy data 
            self.read_from(filename)
            self.flag['from_file'] = True
        elif data != None:
            # no filename is specified and a set of data is provided
            if data is not mc.EchellePair_TYPE: 
                # data must be 2xn numpy arrays
                msg = 'data must be of size 2xn numpy arrays (type mc.Echellepair_TYPE)'
                raise ValueError(msg)
            elif data[0].shape != data[1].shape:
                # size of wave data must be same as flux data
                msg = 'size of data[0] (wave) and data[1] must be same, \
                        but have size {}, {}'.format(
                            data[0].size, data[1].size) 
                raise ValueError(msg)
            else:
                # Success!
                self._wave = data[0]
                self._spec = data[1]
                self.NOrder, self.NPixel = self._wave.shape
                self.NPixel = mc.ECHELLE_SHAPE[1]
                self.flag['from_array'] = True
        else: 
            # in this case nothing is given 
            # we leave the data field blank 
            pass

    def __eq__(self, other: type) -> bool:
        '''
        Comparison == between two Spec Class
        return true if and only if both wavelength and flux 
        are exactly the same
        '''
        return np.logical_and(
            np.all(self._spec == other._spec),
            np.all(self._wave == other._wave)
        )

    # public methods
    def copy(self) -> type:
        '''
        returns a deep copy of self, with a new class ID
        '''
        dup = copy.deepcopy(self)
        return dup
    
    def get_order(self, order: int) -> mc.EchellePair_TYPE:
        '''
        returns a tuple of 2 np.ndarray representing 
        wave and flux of the specified order
        '''
        return (self._wave[order], self._spec[order])

    def read_from(self, fname: str, HDU: str='primary') -> None:
        ''' '''
        if fname.endswith('.fits') == False:
            # Can only read from .fits files
            msg = 'input files must be .fits files'
            raise IOError(msg)
        
        self.filename = fname
        with fits.open(fname) as hdu_list:
            # First record relevant header information 
            self.header = hdu_list[HDU].header
            self._spec = hdu_list[HDU].data
            self.NOrder, self.NPixel = self._spec.shape()
            # Generate wavelength values for each order
            for order in range(self.NOrder):
                self._wave[order] = self._gen_wave(order)

    def write_to(self, fname: str, deg: int) -> None:
        '''
        Take the current data and write to a .fits file
        '''
        if self.flag['from_file'] != True or self.flag['from_array'] != True:
            msg = 'Can only write to file when not empty!'
            raise ValueError(msg)
        if fname.endswith('.fits') == False:
            msg = 'Can only write to .fits files!'
            raise IOError(msg)

        # Initialize a new instance of HDU and save data to primary
        hdu = fits.PrimaryHDU(self._spec)
        hdu_header = hdu.header

        # Record relevat headers
        hdu_header.set('axis2', self.NOrder)
        hdu_header.set('axis1', self.NPixel)

        # degree of interpolation for wavelength
        deg_key = 'hierarch eso drs cal th deg ll'
        hdu_header.set(deg_key, deg)

        # Record polynomial interpolation results to headers
        for order in range(self.NOrder):
            c = np.polyfit(np.arange(self.NPixel), self._wave[order], deg)
            c = np.flip(c)
            for i, ci in enumerate(c):
                key = 'hierarch eso drs cal th coeff ll' + str((deg+1)*order+i)
                hdu_header.set(key, ci)

        hdul = fits.HDUList([hdu])
        hdul.writeto(fname, overwrite=True)

    def shift(self, a: mc.ALPHA_TYPE, order: int) -> None:
        ''' '''
        # Create flux normalization polynomials
        c = int(self._wave[order].size/2)
        am = np.flip(a)
        px = self._wave[order] - self._wave[order, c]
        norm = np.polyval(am, px)

        # alpha_v shift for spectrum 
        av = np.divide(1, a[0])

        # shift wavelength and flux accordingly 
        self._wave *= av
        self._spec *= norm

    def resample(self, resolution: float) -> None:
        ''' '''
        # img.zoom use Cubic spline interpolation on default
        for order in self.NOrder:
            self._wave[order] = img.zoom(self._wave[order], resolution)
            self._spec[order] = img.zoom(self._spec[order], resolution)
        self.NPixel *= resolution

    def plot(self, order: int, 
                   comment: str='', 
                   color: str='g') ->plt.Figure:
        ''' '''
        fig = plt.figure()
        plt.plot(self._wave[order], self._spec[order], 
                 label=comment,
                 color=color)
        plt.legend()
        return fig

    def shift_wave(self, a: float, order: int) -> None: 
        ''' 
        shift the wavelength of this spectrum only
        used in barycenter correction 
        '''
        self._wave[order] *= a

    # private helper functions:
    def _gen_wave(self, order: int) -> mc.EchelleData_TYPE:
        ''' generate wavelength for flux of specified order '''
        opower = self.header['eso drs cal th deg ll']
        a = np.zeros(opower+1)
        for i in range(0, opower+1, 1):
            keyi = 'eso drs cal th coeff ll' + str((opower+1)*order+i)
            a[i] = self.header[keyi]
        wave = np.polyval(
            np.flip(a),
            np.arange(self.NPixel, dtype=np.float64)
        )
        return wave
    
if __name__ == '__main__':
    pass