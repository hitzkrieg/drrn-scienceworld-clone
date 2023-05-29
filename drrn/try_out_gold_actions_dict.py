import numpy as np 
from collections import defaultdict
import pickle
import json


# Load the gold actions set dict
filenameIn_cumulative = "/home/hgolchha_umass_edu/ScienceWorld/goldpaths/gold_action_set_cumulative.json"

transport_actions = [] 
new_actions = [] 

f = open(filenameIn_cumulative)
gold_actions_set_dict_cumulative = json.load(f)
print(gold_actions_set_dict_cumulative.keys())
for k in gold_actions_set_dict_cumulative.keys():
    gold_actions = gold_actions_set_dict_cumulative[k]
    for action in gold_actions:
        if action[:len('go to')] == 'go to':
            transport_actions.append(action)

for action in set(transport_actions):
    new_actions.append('teleport to' + action[len('go to'):])

