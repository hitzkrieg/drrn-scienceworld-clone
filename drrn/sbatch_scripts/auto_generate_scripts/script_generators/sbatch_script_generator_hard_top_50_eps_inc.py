import os

def generate_sbatch_file(java_dir, simcse_dir, drrn_dir, task_index, exp_dir, exp_name, gpus_to_exclude_str):
    sbatch_template = f"""#!/bin/sh
#SBATCH --job-name=drrn_task_easy_limit_{task_index}_100k_{exp_name}
#SBATCH -o {drrn_dir}/{exp_dir}/drrn-task-{task_index}_%j.txt
#SBATCH --time=33:59:59
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH -d singleton    
#SBATCH  --exclude={gpus_to_exclude_str}

export JAVA_HOME={java_dir}
export PATH=$JAVA_HOME/bin:$PATH
# echo $SHELL
# conda init bash
# bash 
# source /home/hgolchha_umass_edu/anaconda3/bin/activate simcse

cd {simcse_dir}
python embedding_server.py     --model_name_or_path result/my-sup-bert-base-uncased-hard_neg_hf     --pooler_type cls     --max_length 32     --batch_size 32     --port 12345 &> embedding_server_log/embedding_server_log_{exp_name}_{task_index}.txt &
echo 'hello'
sleep 3m

cd {drrn_dir}

# source /home/hgolchha_umass_edu/anaconda3/bin/activate drrn1
conda run -n drrn1 python train-scienceworld-limit-actions-by-pruner.py --num_envs=8 --max_steps=100000 --task_idx={task_index}  --priority_fraction=0.50 --memory_size=100000 --simplification_str=easy --env_step_limit=100 --eval_freq=1000 --eval_set=test --output_dir={exp_dir}/drrn-task-{task_index}-test --historySavePrefix={exp_dir}/drrn-task-{task_index}-test/drrn-task{task_index}-results-seed0-test --pruning_strategy=hard --threshold_strategy=top_k --starting_epsilon=0.1 --epsilon_schedule=increasing --no-prune_navigation_action"""
    return sbatch_template

# Task index you want to insert in the sbatch file
task_index = 26
gpus_to_exclude_str = 'ials-gpu028,ials-gpu031,ials-gpu032,ials-gpu014,ials-gpu016,ials-gpu018,ials-gpu025,ials-gpu026,ials-gpu027,ials-gpu004'
exp_dir = 'logs_easy_limit_actions_by_pruner_hard_top_50_epsilon_inc_navigation_100k'
exp_name = 'hard_top_50_inc_epsilon'
drrn_dir = os.environ.get('DRRN_HOME') # /project/pi_mccallum_umass_edu/hgolchha_umass_edu/drrn-scienceworld-fresh-install/drrn-scienceworld/drrn
simcse_dir = os.environ.get('SIMCSE_HOME') # /home/hgolchha_umass_edu/SimCSE
java_dir = os.environ.get('JAVA_HOME') #/home/hgolchha_umass_edu/jre1.8.0_341


sbatch_content = generate_sbatch_file(java_dir=java_dir, simcse_dir=simcse_dir, drrn_dir = drrn_dir, task_index=task_index, exp_dir=exp_dir, exp_name=exp_name, gpus_to_exclude_str=gpus_to_exclude_str)

output_file_dir = '/project/pi_mccallum_umass_edu/hgolchha_umass_edu/drrn-scienceworld-fresh-install/drrn-scienceworld/drrn/sbatch_scripts/auto_generate_scripts/auto_generated_scripts'
output_file_name = f'sbatch_script_{exp_name}.sh'
with open(os.path.join(output_file_dir, output_file_name), "w") as file:
    file.write(sbatch_content)

print(f"Sbatch script saved as {os.path.join(output_file_dir, output_file_name)} ")
