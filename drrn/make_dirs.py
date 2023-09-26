"""
Usage: python make_dirs.py <dir_to_create>
"""
import os
import sys

dir_to_create = sys.argv[1]

if not os.path.exists(dir_to_create):
    os.mkdir(dir_to_create)

for i in range(30):
    os.mkdir(os.path.join(dir_to_create, f'drrn-task-{i}-test'))
