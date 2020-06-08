import numpy as np
import torch
import random
from torch import multiprocessing as mp
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dialog_agent.session import BiSession
from convlab2.dialog_agent.env import Environment
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.policy.rlmodule import Memory, Transition
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from pprint import pprint
import json
import matplotlib.pyplot as plt
import sys
import logging
import os
import datetime
import argparse
import os


a = [0.52, 0.57, 0.55, 0.58, 0.57, 0.55, 0.54, 0.58, 0.55, 0.53, 0.55, 0.53, 0.54, 0.55, 0.57, 0.53, 0.55, 0.54, 0.56, 0.56, 0.55, 0.54, 0.57, 0.56, 0.58, 0.6, 0.6, 0.55, 0.57, 0.57, 0.58, 0.58, 0.59, 0.57, 0.57, 0.57, 0.57, 0.57, 0.56, 0.58]
b = [0.52, 0.53, 0.54, 0.49, 0.44, 0.49, 0.46, 0.47, 0.44, 0.43, 0.42, 0.44, 0.47, 0.48, 0.49, 0.46, 0.45, 0.46, 0.45, 0.48, 0.46, 0.48, 0.49, 0.49, 0.49, 0.48, 0.45, 0.47, 0.43, 0.43, 0.43, 0.42, 0.42, 0.41, 0.42, 0.43, 0.44, 0.47, 0.45, 0.43]
# c = [0.69, 0.73, 0.76, 0.75, 0.71, 0.67, 0.68, 0.68, 0.68, 0.71, 0.71, 0.68, 0.7, 0.67, 0.7, 0.7, 0.64, 0.66, 0.65, 0.63]
# d = [0.45, 0.46, 0.47, 0.47, 0.44, 0.49, 0.51, 0.46, 0.46, 0.45, 0.43, 0.44, 0.44, 0.44, 0.41, 0.4, 0.41, 0.46, 0.39, 0.38]
axis = [i for i in range(len(a))]
plt.plot(axis, a)
plt.plot(axis, b)
# plt.plot(axis, c)
# plt.plot(axis, d)
my_y_ticks = np.arange(0, 1, 0.05)
plt.yticks(my_y_ticks)

# plt.legend(["pretrained RL","pretrained RL+reward","best RL","Baseline"], loc='upper right')
plt.legend(["MLE + idea3", "MLE"])
plt.xlabel('Number of Epoch')
plt.ylabel('Success rate')
plt.show()