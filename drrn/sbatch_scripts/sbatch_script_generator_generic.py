def generate_sbatch_file(task_index, gpus_to_exclude_str, output_dir):
    sbatch_template = """#!/bin/sh
#SBATCH --job-name=drrn_task_easy_limit_{task_index}_100k_{exp_keyword}
#SBATCH -o /project/pi_mccallum_umass_edu/hgolchha_umass_edu/drrn-scienceworld-fresh-install/drrn-scienceworld/drrn/{output_dir}/drrn-task-{task_index}_%j.txt
#SBATCH --time=40:59:59
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH -d singleton    
#SBATCH --exclude={gpus_to_exclude_str}

export JAVA_HOME={java_path}
export PATH=$JAVA_HOME/bin:$PATH

cd {simcse_path}
python embedding_server.py     --model_name_or_path result/my-sup-bert-base-uncased-hard_neg_hf     --pooler_type cls     --max_length 32     --batch_size 32     --port 12345 &> embedding_server_log_hard_fixed_eps_{task_index}.txt &
echo 'hello'
sleep 3m

cd /project/pi_mccallum_umass_edu/hgolchha_umass_edu/drrn-scienceworld-fresh-install/drrn-scienceworld/drrn

conda run -n drrn1 python train-scienceworld-limit-actions-by-pruner.py --num_envs=8 --max_steps=100000 --task_idx={task_index}  --priority_fraction=0.50 --memory_size=100000 --simplification_str=easy --env_step_limit=100 --eval_freq=1000 --eval_set=test --output_dir={output_dir}/drrn-task-{task_index}-test --historySavePrefix={output_dir}/drrn-task-{task_index}-test/drrn-task{task_index}-results-seed0-test --pruning_strategy=hard --threshold_strategy=top_k --starting_epsilon=0.1 --epsilon_schedule=fixed --no-prune_navigation_action
    """
    return sbatch_template.format(task_index=task_index)

# Task index you want to insert in the sbatch file
task_index = 18
run_index = 1
output_dir = 'logs_easy_limit_actions_by_pruner_hard_top_50_epsilon_fixed_navigation_100k'
gpus_to_exclude_str = 'uri-gpu001,uri-gpu002,ials-gpu007'
java_path = '/home/hgolchha_umass_edu/jre1.8.0_341'
simcse_path = '/home/hgolchha_umass_edu/SimCSE'


sbatch_content = generate_sbatch_file(task_index)

# Save the generated sbatch content to a file
with open("sbatch_script_temp.sh", "w") as file:
    file.write(sbatch_content)
