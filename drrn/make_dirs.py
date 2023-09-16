"""
Initialize empty folders 
"""
import os

dir_to_create = 'logs_easy_limit_actions_by_pruner_soft_v2_normalized_100k'

if not os.path.exists(dir_to_create):
    os.mkdir(dir_to_create)

for i in range(30):
    os.mkdir(os.path.join(dir_to_create, f'drrn-task-{i}-test'))
