import sys, os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from argparse import ArgumentParser
import pandas as pd
from pandas.core.frame import DataFrame

parser = ArgumentParser()
# /home/raliegh/图片/ConvLab-2/convlab2/policy/ppo/idea4/save/idea4/res.txt
parser.add_argument("--log_res_path", type=str, default="", help="path of txt file")
args = parser.parse_args()

res_path = args.log_res_path
data = []
for line in open(res_path):
    if line[0] == "[":
        data_json = json.loads(line)
        data.append(data_json)
    else:
        pass
baseline_np = np.array(data)
mean = list(np.mean(baseline_np, axis = 0))
print((mean))
# start writing
f = open(args.log_res_path, 'a+')
f.write("Mean result:" + "\n")
f.write(str(mean) + "\n")
f.close()
# start write result to txt file.

