import os
import pydicom

dir = r'C:\Users\adamyarschenko\Documents\Research\Research_GitHub\Python Scripts'
os.chdir(dir)
from complexity_utilities import *

filepath = r'\\TBMPFS\Research\AY\Exported Plans\Plan1\RP.C039092.BreastJan2019.DIBH_LtBrt.dcm'
#os.path.exists(r'\\TBMPFS\Research')
#os.listdir(r'\\TBMPFS\Research\AY\Exported Plans\Plan1')

ds = pydicom.filereader.dcmread(filepath)

totalMU = get_total_beam_MU(ds,0)
print(totalMU)
ds[(300a, 5100)]
