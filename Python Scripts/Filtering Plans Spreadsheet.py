import os

file_location = os.path.dirname(os.path.abspath('Filtering Plans Spreadsheet.py'))
os.chdir(file_location)

os.path.abspath('visvib_pami.pdf')
file_location = os.path.dirname(os.path.abspath('Pivot Table of Data.xlsx'))
with open(os.path.abspath('Pivot Table of Data.xlsx')) as file:
    data = file.read()

