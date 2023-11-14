# import lib
import os
import glob
import pandas as pd
from pathlib import Path


# load all csv file in the path
def load_alldata(path="../data/"):
    all_files = sorted(glob.glob(path + "*.csv"))
    return load_data(all_files)

def load_data(csv_files): 
    all_data=pd.DataFrame()

    for file in csv_files:
        data = pd.read_csv(file, usecols=['p', 'gflops'], index_col='p')
        data.columns=[Path(file).stem + '_gflops']
        all_data = pd.concat([all_data, data], axis=1)

    return all_data
