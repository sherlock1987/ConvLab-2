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

a = [0.56, 0.59, 0.57, 0.6, 0.63, 0.62, 0.65, 0.68, 0.67, 0.7, 0.71, 0.72, 0.73, 0.71, 0.72, 0.72, 0.73, 0.76, 0.7, 0.74, 0.82, 0.77, 0.77, 0.76, 0.77, 0.73, 0.77, 0.71, 0.71, 0.7]
b = [0.59, 0.56, 0.62, 0.65, 0.66, 0.67, 0.7, 0.68, 0.7, 0.73, 0.74, 0.69, 0.74, 0.77, 0.73, 0.75, 0.71, 0.74, 0.74, 0.76, 0.77, 0.75, 0.73, 0.73, 0.69, 0.72, 0.7, 0.72, 0.74, 0.75]
c = [0.57, 0.6, 0.65, 0.64, 0.67, 0.7, 0.73, 0.77, 0.73, 0.72, 0.73, 0.73, 0.72, 0.7, 0.72, 0.72, 0.71, 0.73, 0.7, 0.71, 0.72, 0.74, 0.73, 0.74, 0.73, 0.75, 0.76, 0.76, 0.74, 0.77]
d = [0.57, 0.58, 0.61, 0.69, 0.67, 0.67, 0.71, 0.68, 0.68, 0.7, 0.74, 0.68, 0.73, 0.68, 0.68, 0.71, 0.71, 0.68, 0.76, 0.71, 0.7, 0.7, 0.69, 0.67, 0.69, 0.72, 0.68, 0.73, 0.69, 0.68]

# c = [0.69, 0.73, 0.76, 0.75, 0.71, 0.67, 0.68, 0.68, 0.68, 0.71, 0.71, 0.68, 0.7, 0.67, 0.7, 0.7, 0.64, 0.66, 0.65, 0.63]
# d = [0.45, 0.46, 0.47, 0.47, 0.44, 0.49, 0.51, 0.46, 0.46, 0.45, 0.43, 0.44, 0.44, 0.44, 0.41, 0.4, 0.41, 0.46, 0.39, 0.38]
axis = [i for i in range(len(a))]
plt.plot(axis, a)
plt.plot(axis, b)
plt.plot(axis, c)
plt.plot(axis, d)
my_y_ticks = np.arange(0.5, 1, 0.05)
plt.yticks(my_y_ticks)

plt.legend(["MLE","real","mask","domain"], loc='upper right')
plt.xlabel('Number of Epoch')
plt.ylabel('Success rate')
plt.show()