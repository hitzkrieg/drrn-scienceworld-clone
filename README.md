# DRRN Agent with Pruning for ScienceWorld

This repository contains a reference implementation DRRN as mentioned in [Interactive Fiction Games: A Colossal Adventure](https://arxiv.org/abs/1909.05398), that has been modified for use with the [ScienceWorld](https://www.github.com/allenai/ScienceWorld) environment. 

Also, it has additional scripts to be used with a Pruner (a Language Model which learns an embedding space to align actions with task description)


# Quickstart

Install Dependencies:
```bash


# Create conda environment
conda create --name drrn1 python=3.8
conda activate drrn1
pip install -r requirements_new.txt
```

You can work in my directory if working from unity
```
cd /project/pi_mccallum_umass_edu/hgolchha_umass_edu/drrn-scienceworld-fresh-install/drrn-scienceworld/drrn/ 
```

Else clone
```
git clone https://github.com/hitzkrieg/drrn-scienceworld-clone.git
```



An example of training the DRRN model (using 8 threads, for 10k training steps, evaluating on dev every 1k steps):
```bash
cd drrn
python3 train-scienceworld.py --num_envs=8 --max_steps=10000 --task_idx=13 --simplification_str=easy --priority_fraction=0.50 --memory_size=100000 --env_step_limit=100 --eval_freq=1000 --eval_set=dev --historySavePrefix=drrn-task13-results-seed0-dev 
```
Here:
- **max_steps:** Maximum number of steps to train for (per environment thread)
- **num_envs:** The number of environment threads to simultaneously use during training (8 is a common number)
- **task_idx:** The ScienceWorld task index (0-29). *See **task list** below*
- **env_step_limit:** the maximum number of steps to run an environment for, before it times out and resets (100 typical)
- **eval_freq:** the number of steps between evaluations
- **eval_set:** which set to perform the evaluations on (dev or test)
- **historySavePrefix:** the filename prefix for saving the output history files, which contain full logs to calculate final scores, plot performance curves, examing action history, etc.
- **priority_fraction** and **memory_size**: Hyperparameters for the DRRN model (see paper for more information).

This configuration generally takes about 1-2 hours to run (to 10k steps).

## ScienceWorld Task List
```
TASK LIST: 
    0: 	                                                 task-1-boil  (30 variations)
    1: 	                        task-1-change-the-state-of-matter-of  (30 variations)
    2: 	                                               task-1-freeze  (30 variations)
    3: 	                                                 task-1-melt  (30 variations)
    4: 	             task-10-measure-melting-point-(known-substance)  (436 variations)
    5: 	           task-10-measure-melting-point-(unknown-substance)  (300 variations)
    6: 	                                     task-10-use-thermometer  (540 variations)
    7: 	                                      task-2-power-component  (20 variations)
    8: 	   task-2-power-component-(renewable-vs-nonrenewable-energy)  (20 variations)
    9: 	                                   task-2a-test-conductivity  (900 variations)
   10: 	             task-2a-test-conductivity-of-unknown-substances  (600 variations)
   11: 	                                          task-3-find-animal  (300 variations)
   12: 	                                    task-3-find-living-thing  (300 variations)
   13: 	                                task-3-find-non-living-thing  (300 variations)
   14: 	                                           task-3-find-plant  (300 variations)
   15: 	                                           task-4-grow-fruit  (126 variations)
   16: 	                                           task-4-grow-plant  (126 variations)
   17: 	                                        task-5-chemistry-mix  (32 variations)
   18: 	                task-5-chemistry-mix-paint-(secondary-color)  (36 variations)
   19: 	                 task-5-chemistry-mix-paint-(tertiary-color)  (36 variations)
   20: 	                             task-6-lifespan-(longest-lived)  (125 variations)
   21: 	         task-6-lifespan-(longest-lived-then-shortest-lived)  (125 variations)
   22: 	                            task-6-lifespan-(shortest-lived)  (125 variations)
   23: 	                               task-7-identify-life-stages-1  (14 variations)
   24: 	                               task-7-identify-life-stages-2  (10 variations)
   25: 	                       task-8-inclined-plane-determine-angle  (168 variations)
   26: 	             task-8-inclined-plane-friction-(named-surfaces)  (1386 variations)
   27: 	           task-8-inclined-plane-friction-(unnamed-surfaces)  (162 variations)
   28: 	                    task-9-mendellian-genetics-(known-plant)  (120 variations)
   29: 	                  task-9-mendellian-genetics-(unknown-plant)  (480 variations)
```

# Hardware requirements
This code generally runs best with at least num_threads+1 CPU cores (e.g. about 10 cores for an 8-thread environment).


# Known issues
- *Many threads*: If you are attempting to use a large number of threads (e.g. 20+), you may need to add an additional several-second delay after the threads spawn before the rest of the program runs.  (The ScienceWorld API already adds a 5 second delay, which handles small numbers of threads well.) 

- *Model saving with manys steps*: Very occassionally, on very long runs (generally 1M+ steps), the periodic pickling the model when saving checkpoints runs into issues and freezes.  The cause is unknown, but as a workaround the save has been wrapped in a timeout, so that if it takes longer than 2 minutes to save the model, the checkpoint is not saved and training continues.  Subsequent checkpoints usually save without issue.

- *Sometimes there are issues in logging (haven't debug yet)*

# Our Experiments

# Experiment Folders
There are the following directories, one for each experiment. Each directory has folders drrn-task-{idx}-test, idx=0-29. 
See make_dirs.py

1. `logs_easy_100k`: Rerunning the original baseline
2. `logs_easy_limit_actions_to_gold_100k` : DRRN agent with actions limited to actions from gold trajectory for that  variation
3. `logs_easy_limit_actions_to_gold_100k_v2`: DRRN agent with actions limited to actions from all gold actions of the task (across variations)
4. `logs_easy_limit_actions_to_gold_100k_v2_sparse`:  DRRN agent with actions limited to actions from all gold actions of the task (across variations), and without intermediate / negative rewards
5. `logs_easy_limit_actions_by_pruner_hard_top_50_epsilon_fixed_navigation_100k`: Hard pruner, topk k=50, epsilon=0.1 (fixed)
6. `logs_easy_limit_actions_by_pruner_hard_top_50_epsilon_inc_navigation_100k` :  Hard pruner, topk k=50, epsilon=0.1 (fixed). Has two more folders for reruns (_2 and _3)
7. `logs_easy_limit_actions_by_pruner_soft_100k`: Soft pruner, architecture v1, Has two more folders for reruns (_2 and _3)
8. `logs_easy_limit_actions_by_pruner_soft_scaled_100k`: Soft pruner, architecture v1, cosine_rescaling_factor=100
9. `logs_easy_limit_actions_by_pruner_soft_v2_100k`: Soft pruner, architecture v2, cosine_rescaling_factor=100
10. `logs_easy_limit_actions_by_pruner_soft_v2_normalized_100k`: Soft pruner, architecture v2, cosine_rescaling_factor=100, cosine scores normalized across actions
11. `logs_easy_limit_actions_by_pruner_hybrid_top_50_epsilon_inc_navigation_100k_`{idx}: hybrid pruner, soft pruner architecture = v1, cosine_rescaling_factor=100
12. `logs_easy_limit_actions_by_pruner_hybrid_v2_top_50_epsilon_inc_navigation_100k`: hybrid pruner, soft pruner architecture = v2, cosine_rescaling_factor=100

# Path to sbatch scripts
See `drrn/sbatch_scripts/reference_scripts` for the sbatch scripts used for the experiments
You need to change the task number at 7 locations (hopefully you don't miss out)

To reduce effort, I wrote Python scripts (for two experiments - hard fixed and inc epsilon) where you can edit some fields and generate the sbatch scripts automatically. You can create one for other experiments as well using it and script from reference_scripts. They are in the path `drrn/sbatch_scripts/auto_generate_scripts/script_generators`. The scripts get generated in the folder  `drrn/sbatch_scripts/auto_generate_scripts/auto_generated_scripts`







