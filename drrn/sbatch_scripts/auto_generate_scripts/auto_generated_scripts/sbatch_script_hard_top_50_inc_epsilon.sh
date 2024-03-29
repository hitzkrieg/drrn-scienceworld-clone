#!/bin/sh
#SBATCH --job-name=drrn_task_easy_limit_26_100k_hard_top_50_inc_epsilon
#SBATCH -o /project/pi_mccallum_umass_edu/hgolchha_umass_edu/drrn-scienceworld-fresh-install/drrn-scienceworld/drrn/logs_easy_limit_actions_by_pruner_hard_top_50_epsilon_inc_navigation_100k/drrn-task-26_%j.txt
#SBATCH --time=33:59:59
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH -d singleton    
#SBATCH  --exclude=ials-gpu028,ials-gpu031,ials-gpu032,ials-gpu014,ials-gpu016,ials-gpu018,ials-gpu025,ials-gpu026,ials-gpu027,ials-gpu004

export JAVA_HOME=/home/hgolchha_umass_edu/jre1.8.0_341
export PATH=$JAVA_HOME/bin:$PATH
# echo $SHELL
# conda init bash
# bash 
# source /home/hgolchha_umass_edu/anaconda3/bin/activate simcse

cd /home/hgolchha_umass_edu/SimCSE
python embedding_server.py     --model_name_or_path result/my-sup-bert-base-uncased-hard_neg_hf     --pooler_type cls     --max_length 32     --batch_size 32     --port 12345 &> embedding_server_log/embedding_server_log_hard_top_50_inc_epsilon_26.txt &
echo 'hello'
sleep 3m

cd /project/pi_mccallum_umass_edu/hgolchha_umass_edu/drrn-scienceworld-fresh-install/drrn-scienceworld/drrn

# source /home/hgolchha_umass_edu/anaconda3/bin/activate drrn1
conda run -n drrn1 python train-scienceworld-limit-actions-by-pruner.py --num_envs=8 --max_steps=100000 --task_idx=26  --priority_fraction=0.50 --memory_size=100000 --simplification_str=easy --env_step_limit=100 --eval_freq=1000 --eval_set=test --output_dir=logs_easy_limit_actions_by_pruner_hard_top_50_epsilon_inc_navigation_100k/drrn-task-26-test --historySavePrefix=logs_easy_limit_actions_by_pruner_hard_top_50_epsilon_inc_navigation_100k/drrn-task-26-test/drrn-task26-results-seed0-test --pruning_strategy=hard --threshold_strategy=top_k --starting_epsilon=0.1 --epsilon_schedule=increasing --no-prune_navigation_action