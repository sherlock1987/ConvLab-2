#!/bin/bash
"""
1. seed, the saving dir(seed)
1.1 write stuff to config file
2. evaluation.py ts write down the result into the same file
3. note the process_num
"""
num  1
echo '
	"batchsz": 32,
	"gamma": 0.99,
	"epsilon": 0.2,
	"tau": 0.95,
	"policy_lr": 0.0001,
	"value_lr": 0.00005,
	"save_dir": ${1},
	"log_dir": "log",
	"save_per_epoch": 1,
	"update_round": 5,
	"h_dim": 100,
	"hv_dim": 50,
	"load": "save/best"
' > /home/raliegh/图片/ConvLab-2/convlab2/policy/ppo/idea4/config.json
#for i in $(seq 1 2)
#do
#  {
#    python /home/raliegh/图片/ConvLab-2/convlab2/policy/ppo/train.py --save_dir ${i}
#    python test.py --save_dir ${i}
#  }&
#done
#wait