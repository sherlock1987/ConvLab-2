#!/bin/bash
# set random seed, save python file in different seed.
# run RL , and load reward model
# evaluate models in dir
time=$(date "+%Y-%m-%d--%H:%M:%S")
root=`pwd`
device=3
echo "${time}"

config_path=${root}/"convlab2/policy/ppo/idea4/config.json"
RL_path=${root}/"convlab2/policy/ppo/train.py"
load_path=${root}/"convlab2/policy/mle/multiwoz/best_mle"
Eval_path=${root}/"convlab2/policy/evaluate.py"
Anal_path=${root}/"convlab2/policy/result_analysis.py"
log_path=${root}/"convlab2/policy/ppo/idea4/save/"${time}/res.txt
for process_id in $(seq 1 8)
do
  {
    sub_save_path="save"/${time}/${process_id}
    complete_sub_save_path=${root}/"convlab2/policy/ppo/idea4/"${sub_save_path}
    new_RL_path=${root}/"convlab2/policy/ppo/train_${process_id}.py"
    sleep $[process_id*20]
    echo "Begin processing in ${sub_save_path}..., result is in ${log_path}"
    echo '{
	"batchsz": 32,
	"gamma": 0.99,
	"epsilon": 0.2,
	"tau": 0.95,
	"policy_lr": 0.0001,
	"value_lr": 0.00005,
	"save_dir": "'${sub_save_path}'",
	"log_dir": "log",
	"save_per_epoch": 1,
	"update_round": 5,
	"h_dim": 100,
	"hv_dim": 50,
	"load": "save/best"
}
'> ${config_path}
  sed -i '156d' ${RL_path}
  sed -i "156i seed=${process_id}" ${RL_path}
  cp ${RL_path} ${new_RL_path}
  CUDA_VISIBLE_DEVICES=${device} python ${new_RL_path} --load_path ${load_path} --load_path_reward_d ${root}/convlab2/policy/mle/idea4/GAN1/Dis/pretrain_D.mdl --load_path_reward_g ${root}/convlab2/policy/mle/idea4/GAN1/Gen/pretrain_G.mdl
  echo "${log_path}"
  python ${Eval_path} --model_name "PPO" --evluate_in_dir True --model_path_root ${complete_sub_save_path} --log_res_path ${log_path}
  }&
done
wait
sleep $[20]
python ${Anal_path} --log_res_path ${log_path}