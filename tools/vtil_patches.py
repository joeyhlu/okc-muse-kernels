import pathlib, sys
sys.path.insert(0, "<repo>/tools")
import vt16_patches as P, vil_patches as I
def apply(s):
    return I.apply(P.apply(s))
