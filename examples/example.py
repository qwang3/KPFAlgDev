import sys
import os

import common.macro as mc
import common.argument.spec as sp
import common.primitive.process as pc

if __name__ == '__main__':
    print(mc.C_SPEED)
    S = sp.Spec()
    SS = S.copy()
    P = pc.ProcessSpec(SS)
    SP = P.run()
    print(S)
    print(SS)
    print(SP)
    print(S == SS)
    print(S == SP)