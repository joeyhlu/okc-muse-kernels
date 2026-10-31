import sys
sys.path.insert(0, "<repo>/tools")
import vt16_patches as P, vcb_patches as C, ver_patches as V


def apply(s):
    return V.apply(C.apply(P.apply(s)))
