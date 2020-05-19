import pandas as pd
from pathlib import Path
import os


dir_path = Path(r'N:\Research\DIR for QoL Study\H&N PRO Spreadsheets')
file_location = Path(dir_path, r'Anonymized Data (Complete - 66-70 Gy).csv')
#df = pd.read_excel(file_location, index_col=0)
df = pd.read_csv(file_location)
df['QoLID'].head(10)
