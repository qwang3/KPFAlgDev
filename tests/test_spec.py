import common.macro as mc
import common.argument.spec as sp

fpath = 'data/HARPS_Barnards_Star'
flist = mc.findfiles(fpath, '_e2ds_A.fits')

def test_io():

    # First test file:
    for file in flist:
        try:
            sp.Spec(filename=file)
        