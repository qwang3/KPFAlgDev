#
#
import numpy as np

from astropy.io import fits
import common.macro as mc 

def get_ccf(ccf_path, dat_f_name):
    
    all_ccf = mc.findfiles(ccf_path, '_ccf_G2_A.fits')
    dat = np.zeros([len(all_ccf), 2])

    for i, ccf in enumerate(all_ccf): 
        print(ccf)
        with fits.open(ccf) as hdu:
            header = hdu['primary'].header
            RVC = header['HIERARCH ESO DRS CCF RVC']
            time = header['eso drs bjd']
            dat[i] = [time, RVC]
    
    # subtract mean out of radial velocity and convert to meters
    dat[:, 1] -= np.mean(dat[:, 1])
    dat[:, 1] *= 1000
    # [julian day, RV[m/s]]
    np.savetxt(dat_f_name, dat)

if __name__ == '__main__':
    ccf_path = 'data/HARPS_Tau_Ceti_CCF'
    dat_f_name = 'tau_ceti_ref.dat'
    get_ccf(ccf_path, dat_f_name)
