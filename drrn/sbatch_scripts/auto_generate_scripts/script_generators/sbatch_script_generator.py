def generate_sbatch_file(task_index, exp_dir, exp_name, gpus_to_exclude_str):
    sbatch_template = f"""#!/bin/sh
#SBATCH --job-name=drrn_task_easy_limit_{task_index}_100k_soft_scaled_v2
#SBATCH -o /project/pi_mccallum_umass_edu/hgolchha_umass_edu/drrn-scienceworld-fresh-install/drrn-scienceworld/drrn/{exp_dir}/drrn-task-{task_index}_%j.txt
#SBATCH --time=40:59:59
#SBATCH --partition=gypsum-2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH -d singleton
#SBATCH --exclude={gpus_to_exclude_str}

export JAVA_HOME=/home/hgolchha_umass_edu/jre1.8.0_341
export PATH=$JAVA_HOME/bin:$PATH
# echo $SHELL
# conda init bash
# source /home/hgolchha_umass_edu/anaconda3/bin/activate simcse

cd /home/hgolchha_umass_edu/SimCSE
python embedding_server.py     --model_name_or_path result/my-sup-bert-base-uncased-hard_neg_hf     --pooler_type cls     --max_length 32     --batch_size 32     --port 12345 &> embedding_server_log_soft_{task_index}_{exp_name}.txt &
echo 'hello'
sleep 3m

cd /project/pi_mccallum_umass_edu/hgolchha_umass_edu/drrn-scienceworld-fresh-install/drrn-scienceworld/drrn/

# source /home/hgolchha_umass_edu/anaconda3/bin/activate drrn1
conda run -n drrn1 python train-scienceworld-limit-actions-by-pruner.py --num_envs=8 --max_steps=100000 --task_idx={task_index}  --priority_fraction=0.50 --memory_size=100000 --simplification_str=easy --env_step_limit=100 --eval_freq=1000 --eval_set=test --output_dir={exp_dir}/drrn-task-{task_index}-test --historySavePrefix={exp_dir}/drrn-task-{task_index}-test/drrn-task{task_index}-results-seed0-test    """
    return sbatch_template

# Task index you want to insert in the sbatch file
task_index = 0
gpus_to_exclude_str = 'ials-gpu003,ials-gpu004'

# Prepare some lists and dicts
exp_names = ['hard_top_50', 'hard_top_50_fixed_epsilon', 'hybrid_rescaled_top_50', 'hybrid_v2_rescaled_top_50', 'soft_v2_normalized', 'soft_v2']
exp_name_to_dir = {}


# Specify the experiment to perform
exp_name = 'hard_top_50' 
exp_dir = exp_name_to_dir[exp_name]





sbatch_content = generate_sbatch_file(task_index=task_index, gpus_to_exclude_str=gpus_to_exclude_str)

output_file_dir = '/project/pi_mccallum_umass_edu/hgolchha_umass_edu/drrn-scienceworld-fresh-install/drrn-scienceworld/drrn/sbatch_scripts/auto_generate_scripts/auto_generated_scripts'
output_file_name = 'sbatch_script_{}.sh'
with open(os.path.join(output_file_dir, output_file_name), "w") as file:
    file.write(sbatch_content)