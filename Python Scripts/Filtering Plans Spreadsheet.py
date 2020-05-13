import os
from pathlib import Path


file_location = os.path.dirname(os.path.abspath('Filtering Plans Spreadsheet.py'))
os.chdir(file_location)

os.path.abspath('Pivot Table of Data.xlsx')
file_location = os.path.dirname(os.path.abspath('Pivot Table of Data.xlsx'))
with open(os.path.abspath('Pivot Table of Data.xlsx')) as file:
    data = file.read()

Path.cwd()
os.chdir(Path(r'C:\Users\The Supreme Being\Documents\Research\Research\Python Scripts'))
