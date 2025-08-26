from curve_library import CurveList
import struct

def save_curve_list(filename, curvelist : CurveList):
    values = curvelist.getFullCurve()
    with open(filename, "wb") as f:
        for v in values:
            f.write(struct.pack(">d", v))