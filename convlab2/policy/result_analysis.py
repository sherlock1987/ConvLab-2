# todo: analysis the file
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from argparse import ArgumentParser
import pandas as pd
from pandas.core.frame import DataFrame

# root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
parser = ArgumentParser()
parser.add_argument("--path", type=str, default="", help="path of analysis")
args = parser.parse_args()

res_path = args.path
# write this to xls with a name.
tiny = [0.619375, 0.65375, 0.6906249999999999, 0.7024999999999999, 0.723125, 0.725625, 0.7374999999999999, 0.739375, 0.75, 0.751875, 0.755, 0.75, 0.7531249999999999, 0.750625, 0.75, 0.753125, 0.7549999999999999, 0.75, 0.746875, 0.74375, 0.74125, 0.73625, 0.73875, 0.738125, 0.7381249999999999, 0.7368750000000001, 0.73625, 0.74, 0.7293749999999999, 0.7387500000000001]
data_df = DataFrame(tiny)
print(data_df)
writer = pd.ExcelWriter('test.xlsx')
data_df.to_excel(writer,'page_1', float_format='%.3f') # float_format 控制精度
writer.save()
