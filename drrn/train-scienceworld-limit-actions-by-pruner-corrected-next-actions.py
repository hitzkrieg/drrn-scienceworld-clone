"""
Usage:

python train-scienceworld-limit-actions-by-pruner-corrected-next-actions.py --num_envs=8 --max_steps=100000 --task_idx=10  --priority_fraction=0.50 --memory_size=100000 --simplification_str=easy --env_step_limit=100 --eval_freq=1000 --eval_set=test --output_dir=logs_easy_temp/drrn-task-10-test --historySavePrefix=logs_easy_temp/drrn-task-10-test/drrn-task10-results-seed0-test --embedding_server_port 12345 --threshold_strategy top_k --threshold_file threshold.json --pruning_strategy soft
This corrects the choice of next actions is sent to the DRRN loss. This should be the reduced set of actions after pruning. 

-- Protobuf issue and resolution:
    I had faced an issue with the protobuf versions. 
    While the embedding server was using protobuf version 4.23.0 (simcse env), this environment (drrn) has protobuf version 3.19.4.
    This was resulting in some error while trying to run this file. 

    I followed these instructions (modifying the answer from https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal):
    1. Install the protobuf version from simcse
    pip install --upgrade protobuf

    2. Copy builder.py from .../Lib/site-packages/google/protobuf/internal to another folder on your computer (let's say 'Documents')

    How to find this path?
    python -c "import site; print(site.getsitepackages())"

    3. Install a protobuf version that is compatible with your project (for me 3.19.4)
    pip install protobuf==3.19.4

    4. Copy builder.py from (let's say 'Documents') to Lib/site-packages/google/protobuf/internal
    5. Run your code


"""

import subprocess
import time
import math
import timeit
import torch
import logger
import argparse
from drrn import DRRN_Agent
from vec_env import VecEnv
import random
import json

from scienceworld import ScienceWorldEnv, BufferedHistorySaver
from vec_env import resetWithVariation, resetWithVariationDev, resetWithVariationTest, initializeEnv, sanitizeInfo, sanitizeObservation
from action_ranker import ActionScorer
import numpy as np 
import re

navigation_regex = re.compile(r"(go|teleport)\ to.*")


def configure_logger(log_dir):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    tb = logger.Logger(log_dir, [logger.make_output_format('tensorboard', log_dir),
                                 logger.make_output_format('csv', log_dir),
                                 logger.make_output_format('stdout', log_dir)])
    global log
    log = logger.log

def clean(strIn):
    charsToFilter = ['\t', '\n', '*', '-']
    for c in charsToFilter:
        strIn = strIn.replace(c, ' ')
    return strIn.strip()    

def get_navigation_action_positions(actions):
    """
    To do: Also need to add ids?
    """
    navigation_action_positions = [i for i, action in enumerate(actions) if re.search(navigation_regex, action)]
    return navigation_action_positions

def evaluate(agent, args, env_step_limit, bufferedHistorySaverEval, extraSaveInfo, action_ranker_obj, nb_episodes=10):    
    # Initialize a ScienceWorld thread for serial evaluation
    env = initializeEnv(threadNum = args.num_envs+10, args=args) # A threadNum (and therefore port) that shouldn't be used by any of the regular training workers

    scoresOut = []
    with torch.no_grad():
        
        for ep in range(nb_episodes):
            total_score = 0
            log("Starting evaluation episode {}".format(ep))   
            print("Starting evaluation episode " + str(ep) + " / " + str(nb_episodes))         
            extraSaveInfo['evalIdx'] = ep
            score = evaluate_episode(agent, env, env_step_limit, args.simplification_str, bufferedHistorySaverEval, extraSaveInfo, args.eval_set, args, action_ranker_obj)
            log("Evaluation episode {} ended with score {}\n\n".format(ep, score))
            total_score += score
            scoresOut.append(total_score)
            print("")

        avg_score = total_score / nb_episodes
        
        env.shutdown()
        
        return scoresOut, avg_score

def load_gold_action_set(taskIdx, variationIdx, simplification_str='easy'):
    gold_action_set = gold_actions_set_dict_cumulative[str(taskIdx)]
    return gold_action_set

def evaluate_episode(agent, env, env_step_limit, simplificationStr, bufferedHistorySaverEval, extraSaveInfo, evalSet, args, action_ranker_obj):
    step = 0
    done = False
    numSteps = 0
    ob = ""
    info = {}
    if (evalSet == "dev"):
        ob, info = resetWithVariationDev(env, simplificationStr)
        info = sanitizeInfo(info)
        ob = sanitizeObservation(ob, info)

    elif (evalSet == "test"):
        ob, info = resetWithVariationTest(env, simplificationStr)
        info = sanitizeInfo(info)
        ob = sanitizeObservation(ob, info)

    else:
        print("evaluate_episode: unknown evaluation set (expected 'dev' or 'test', found: " + str(evalSet) + ")")
        env.shutdown()

        exit(1)

    state = agent.build_state([ob], [info])[0]

    log('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']), clean(info['look'])))    
    while not done:
        #print("numSteps: " + str(numSteps))        
        # valid_acts = info['valid']
        # valid_acts = list(set(info['valid']).intersection(gold_action_set)) 

        valid_acts = info['valid']
        task_description = info['taskDesc']
        cosine_similarities, normalized_scores, desc_sorting_indices, size_of_pruned_set = action_ranker_obj.score_actions(valid_acts, task_description)

        random_number = np.random.uniform()

        if args.pruning_strategy in ['hard', 'hybrid'] and random_number > args.current_epsilon:
            if args.prune_navigation_action:
                # valid_acts
                valid_acts = np.asarray(valid_acts)[desc_sorting_indices][:size_of_pruned_set]
                cosine_similarities = np.asarray(cosine_similarities)[desc_sorting_indices][:size_of_pruned_set]
            else:
                navigation_action_positions = get_navigation_action_positions(valid_acts)
                reduced_actions_positions = np.asarray(list(set(navigation_action_positions).union(desc_sorting_indices[:size_of_pruned_set])))
                valid_acts = np.asarray(valid_acts)[reduced_actions_positions]
                cosine_similarities = np.asarray(cosine_similarities)[reduced_actions_positions]

        
        # Note: when args.pruning_strategy == 'hard', the values of normalized_scores, desc_sorting_indices have not been updated to a smaller length

        valid_ids = agent.encode(valid_acts)
        if args.pruning_strategy in ['soft', 'hybrid']:
            _, action_idx, action_values = agent.act(states = [state], poss_acts = [valid_ids], poss_acts_cosine_sim_scores = [cosine_similarities], sample=False)        
        else:
            _, action_idx, action_values = agent.act(states = [state], poss_acts = [valid_ids], sample=False)        

        action_idx = action_idx[0]
        action_values = action_values[0]
        action_str = valid_acts[action_idx]
        log('Action{}: {}, Q-Value {:.2f}'.format(step, action_str, action_values[action_idx].item()))        
        s = ''

        maxToDisplay = 10   # Max Q values to display, to limit the log size
        numDisplayed = 0
        for idx, (act, val) in enumerate(sorted(zip(valid_acts, action_values), key=lambda x: x[1], reverse=True), 1):
            s += "{}){:.2f} {} ".format(idx, val.item(), act)
            numDisplayed += 1
            if (numDisplayed > maxToDisplay):
                break
 
        log('Q-Values: {}'.format(s))
        ob, rew, done, info = env.step(action_str)
        info = sanitizeInfo(info)
        ob = sanitizeObservation(ob, info)

        
        log("Reward{}: {}, Score {}, Done {}".format(step, rew, info['score'], done))        
        step += 1
        log('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']), clean(info['look'])))
        state = agent.build_state([ob], [info])[0]        

        numSteps +=1        
        if (numSteps > env_step_limit):
            print("Maximum number of evaluation steps reached (" + str(env_step_limit) + ").")
            break    

    print("Completed one evaluation episode")
    # Save
    runHistory = env.getRunHistory()
    episodeIdx = str(extraSaveInfo['numEpisodes']) + "-" + str(extraSaveInfo['evalIdx'])
    bufferedHistorySaverEval.storeRunHistory(runHistory, episodeIdx, notes=extraSaveInfo)
    bufferedHistorySaverEval.saveRunHistoriesBufferIfFull(maxPerFile=extraSaveInfo['maxHistoriesPerFile'])
    print("Completed saving")


    return info['score']


def train(agent, envs, max_steps, update_freq, eval_freq, checkpoint_freq, log_freq, args, bufferedHistorySaverTrain, bufferedHistorySaverEval, action_ranker_obj):
    startTime = time.time()
    flush_cache_freq = 100

    #max_steps = int(math.ceil(max_steps / args.num_envs))
    numEpisodes = 0
    stepsFunctional = 0
    start1 = timeit.default_timer()


    # Reinit environments
    obs, infos = envs.reset()
    states = agent.build_state(obs, infos)

    valid_actions_list = [info['valid'] for info in infos]
    task_description_list = [info['taskDesc'] for info in infos]

    action_scores_list = map(action_ranker_obj.score_actions, valid_actions_list, task_description_list)
    cosine_similarities_list, normalized_scores_list, desc_sorting_indices_list, size_of_pruned_set_list = zip(*action_scores_list)

    assert len(cosine_similarities_list) == len(normalized_scores_list) == len(desc_sorting_indices_list) == len(size_of_pruned_set_list)

    navigation_action_positions_list = [get_navigation_action_positions(valid_actions) for valid_actions in valid_actions_list]
    reduced_actions_positions_list = [np.asarray(list(set(navigation_action_positions_list[i]).union(desc_sorting_indices_list[i][:size_of_pruned_set_list[i]]))) for i in range(len(navigation_action_positions_list))]

    random_number = np.random.uniform()

    if args.pruning_strategy in ['hard', 'hybrid'] and random_number > args.current_epsilon:
        # Prepare reduced lists and make sure the mapping for scores are also updated.
        if args.prune_navigation_action:
            valid_actions_list =  [np.asarray(valid_actions_list[i])[desc_sorting_indices_list[i]][:size_of_pruned_set_list[i]] for i in range(len(valid_actions_list))]
            cosine_similarities_list =  [np.asarray(cosine_similarities_list[i])[desc_sorting_indices_list[i]][:size_of_pruned_set_list[i]] for i in range(len(cosine_similarities_list))]
            normalized_scores_list =  [np.asarray(normalized_scores_list[i])[desc_sorting_indices_list[i]][:size_of_pruned_set_list[i]] for i in range(len(normalized_scores_list))]
        else:
            valid_actions_list =  [np.asarray(valid_actions_list[i])[reduced_actions_positions_list[i]] for i in range(len(valid_actions_list))]
            cosine_similarities_list =  [np.asarray(cosine_similarities_list[i])[reduced_actions_positions_list[i]] for i in range(len(cosine_similarities_list))]
            normalized_scores_list =  [np.asarray(normalized_scores_list[i])[reduced_actions_positions_list[i]] for i in range(len(normalized_scores_list))]


    # valid_ids = [agent.encode(info['valid']) for info in infos]

    valid_ids = [agent.encode(valid_actions) for valid_actions in valid_actions_list]


    for step in range(1, max_steps+1):
        stepsFunctional = step * envs.num_envs

        assert len(cosine_similarities_list) == len(valid_actions_list) == len(normalized_scores_list) == len(valid_ids)

        for i in range(len(valid_ids)):
            assert len(valid_ids[i]) == len(valid_actions_list[i])


        # Summary statistics
        print("-------------------")
        print("Step " + str(step))
        print("")
        end = timeit.default_timer()
        deltaTime = end - start1
        deltaTimeMins = deltaTime / 60
        print("Started at runtime: " + str(deltaTime) + " seconds  (" + str(deltaTimeMins) + " minutes)")
        print("")

        # Choose action(s)

        if args.pruning_strategy == 'hard':
            action_ids, action_idxs, _ = agent.act(states = states, poss_acts = valid_ids)
        # when args.pruning_strategy in ['soft', 'hybrid']
        else:
            action_ids, action_idxs, _ = agent.act(states = states, poss_acts = valid_ids, poss_acts_cosine_sim_scores= cosine_similarities_list)

        
        action_strs = [valid_actions[idx] for valid_actions, idx in zip(valid_actions_list, action_idxs)]        
        # Cosine sim of the action taken
        cosine_similarity_action_taken = [cosine_similarities[idx] for cosine_similarities, idx in zip(cosine_similarities_list, action_idxs)]

        # Perform the action(s) in the environment
        obs, rewards, dones, infos = envs.step(action_strs)
        # print("New variation Idxs")
        # print(new_variationIdxes)
        # print("---------------------------------")

        # Check for any completed episodes
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done:
                # # DEBUG
                # print(f"Done for env: {i}")
                # An episode has completed
                tb.logkv('EpisodeScore', info['score'])
                print("EPISODE SCORE: " + str(info['score']))
                print("EPISODE SCORE: " + str(info['score']) + " STEPS: " + str(step) + " STEPS (functional): " + str(stepsFunctional) + " EPISODES: " + str(numEpisodes))

                # Save the environment's history in the history logs
                runHistory = info['runHistory']
                bufferedHistorySaverTrain.storeRunHistory(runHistory, numEpisodes, notes={'step':step})
                bufferedHistorySaverTrain.saveRunHistoriesBufferIfFull(maxPerFile=args.maxHistoriesPerFile)

                numEpisodes += 1

        next_states = agent.build_state(obs, infos)
        next_valid_actions_list = [info['valid'] for info in infos] 
        next_task_description_list = [info['taskDesc'] for info in infos]

        next_action_scores_list = map(action_ranker_obj.score_actions, next_valid_actions_list, next_task_description_list)
        next_cosine_similarities_list, next_normalized_scores_list, next_desc_sorting_indices_list, next_size_of_pruned_set_list = zip(*next_action_scores_list)
    

        # While storing in the buffer use all the next actions even when args.pruning_strategy == 'hard'. This is because we are trying to estimate Q(s,a)
        next_valids = [agent.encode(valid_actions) for valid_actions in next_valid_actions_list]
        for state, act, act_cosine_sim, rew, next_state, valids,valids_cosine_sim, done in \
            zip(states, action_ids, cosine_similarity_action_taken, rewards, next_states, next_valids, next_cosine_similarities_list, dones):
            
            agent.observe(state, act, rew, next_state, valids, done, act_cosine_sim, valids_cosine_sim)

        # Now prune the next actions
        random_number = np.random.uniform()
        args.current_epsilon = ((1 - args.starting_epsilon)/(max_steps))*step + args.starting_epsilon 

        if args.pruning_strategy in ['hard', 'hybrid'] and random_number > args.current_epsilon:

            if args.prune_navigation_action:
                # Prepare reduced lists and make sure the mapping for scores are also updated.
                next_valid_actions_list =  [np.asarray(next_valid_actions_list[i])[next_desc_sorting_indices_list[i]][:next_size_of_pruned_set_list[i]] for i in range(len(next_valid_actions_list))]
                next_cosine_similarities_list =  [np.asarray(next_cosine_similarities_list[i])[next_desc_sorting_indices_list[i]][:next_size_of_pruned_set_list[i]] for i in range(len(next_cosine_similarities_list))]
                next_normalized_scores_list =  [np.asarray(next_normalized_scores_list[i])[next_desc_sorting_indices_list[i]][:next_size_of_pruned_set_list[i]] for i in range(len(next_normalized_scores_list))]
            
            else:
                # Prevent navigation actions from being pruned
                next_navigation_action_positions_list = [get_navigation_action_positions(next_valid_actions) for next_valid_actions in next_valid_actions_list]
                next_reduced_actions_positions_list = [np.asarray(list(set(next_navigation_action_positions_list[i]).union(next_desc_sorting_indices_list[i][:next_size_of_pruned_set_list[i]]))) for i in range(len(next_navigation_action_positions_list))]

                next_valid_actions_list =  [np.asarray(next_valid_actions_list[i])[next_reduced_actions_positions_list[i]] for i in range(len(next_valid_actions_list))]
                next_cosine_similarities_list =  [np.asarray(next_cosine_similarities_list[i])[next_reduced_actions_positions_list[i]] for i in range(len(next_cosine_similarities_list))]
                next_normalized_scores_list =  [np.asarray(next_normalized_scores_list[i])[next_reduced_actions_positions_list[i]] for i in range(len(next_normalized_scores_list))]

            next_valids = [agent.encode(valid_actions) for valid_actions in next_valid_actions_list]


        # Update x with next_x 
        states = next_states
        valid_ids = next_valids
        valid_actions_list = next_valid_actions_list
        task_description_list = next_task_description_list
        cosine_similarities_list = next_cosine_similarities_list
        normalized_scores_list = next_normalized_scores_list

        for i in range(len(valid_ids)):
            assert len(valid_ids[i]) == len(valid_actions_list[i])



        if step % log_freq == 0:            
            tb.logkv('Step', step)
            tb.logkv('StepsFunctional', step*envs.num_envs)
            tb.logkv("FPS", int((step*envs.num_envs)/(time.time()-startTime)))
            tb.logkv('numEpisodes', numEpisodes)
            tb.logkv('taskIdx', args.task_idx)
            tb.logkv('GPU_mem', agent.getMemoryUsage())

            print("*************************")
            print("Step:            " + str(step))
            print("StepsFunctional: " + str(step*envs.num_envs))
            print("FPS:             " + str( (step*envs.num_envs)/(time.time()-startTime)) )
            print("numEpisodes:     " + str(numEpisodes))
            print("taskIdx:         " + str(args.task_idx))
            print("GPU_mem:         " + str(agent.getMemoryUsage()))
            print("*************************")

        if step % update_freq == 0:            
            loss = agent.update()            
            if loss is not None:
                tb.logkv_mean('Loss', loss)

        if step % checkpoint_freq == 0:
            # Save model checkpoints
            agent.save("-steps" + str(stepsFunctional) + "-eps" + str(numEpisodes))

        if step % flush_cache_freq == 0:
            # Keep the GPU memory low
            agent.clearGPUCache()

        if step % eval_freq == 0:
            # Do the evaluation procedure
            extraSaveInfo = {'numEpisodes':numEpisodes, 'numSteps':step, 'stepsFunctional:':stepsFunctional, 'maxHistoriesPerFile':args.maxHistoriesPerFile}
            eval_scores, avg_eval_score = evaluate(agent, args, args.env_step_limit, bufferedHistorySaverEval, extraSaveInfo, action_ranker_obj)
            
            tb.logkv('EvalScore', avg_eval_score)
            tb.logkv('numEpisodes', numEpisodes)
            tb.dumpkvs()

            for eval_score in eval_scores:
                print("EVAL EPISODE SCORE: " + str(eval_score))
                print("EVAL EPISODE SCORE: " + str(eval_score) + " STEPS: " + str(step) + " STEPS: " + str(stepsFunctional) + " EPISODES: " + str(numEpisodes))

            obs, infos = envs.reset()
            states = agent.build_state(obs, infos)

            valid_actions_list = [info['valid'] for info in infos]
            task_description_list = [info['taskDesc'] for info in infos]

            action_scores_list = map(action_ranker_obj.score_actions, valid_actions_list, task_description_list)
            cosine_similarities_list, normalized_scores_list, desc_sorting_indices_list, size_of_pruned_set_list = zip(*action_scores_list)
            
            navigation_action_positions_list = [get_navigation_action_positions(valid_actions) for valid_actions in valid_actions_list]
            reduced_actions_positions_list = [np.asarray(list(set(navigation_action_positions_list[i]).union(desc_sorting_indices_list[i][:size_of_pruned_set_list[i]]))) for i in range(len(navigation_action_positions_list))]

            random_number = np.random.uniform()

            if args.pruning_strategy in ['hard', 'hybrid'] and random_number > args.current_epsilon:
                # Prepare reduced lists and make sure the mapping for scores are also updated.
                if args.prune_navigation_action:
                    valid_actions_list =  [np.asarray(valid_actions_list[i])[desc_sorting_indices_list[i]][:size_of_pruned_set_list[i]] for i in range(len(valid_actions_list))]
                    cosine_similarities_list =  [np.asarray(cosine_similarities_list[i])[desc_sorting_indices_list[i]][:size_of_pruned_set_list[i]] for i in range(len(cosine_similarities_list))]
                    normalized_scores_list =  [np.asarray(normalized_scores_list[i])[desc_sorting_indices_list[i]][:size_of_pruned_set_list[i]] for i in range(len(normalized_scores_list))]
                else:
                    valid_actions_list =  [np.asarray(valid_actions_list[i])[reduced_actions_positions_list[i]] for i in range(len(valid_actions_list))]
                    cosine_similarities_list =  [np.asarray(cosine_similarities_list[i])[reduced_actions_positions_list[i]] for i in range(len(cosine_similarities_list))]
                    normalized_scores_list =  [np.asarray(normalized_scores_list[i])[reduced_actions_positions_list[i]] for i in range(len(normalized_scores_list))]


            valid_ids = [agent.encode(valid_actions) for valid_actions in valid_actions_list]


    # Save anything left in history buffers
    bufferedHistorySaverTrain.saveRunHistoriesBufferIfFull(maxPerFile=args.maxHistoriesPerFile, forceSave=True)
    bufferedHistorySaverEval.saveRunHistoriesBufferIfFull(maxPerFile=args.maxHistoriesPerFile, forceSave=True)


    print("Training complete.")
    # Final save
    agent.save("-steps" + str(stepsFunctional) + "-eps" + str(numEpisodes))
    # Close environments
    envs.close_extras()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--spm_path', default='../spm_models/unigram_8k.model')
    parser.add_argument('--rom_path', default='zork1.z5')
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_envs', default=16, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--checkpoint_freq', default=500, type=int)
    parser.add_argument('--eval_freq', default=500, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--memory_size', default=5000000, type=int)
    parser.add_argument('--priority_fraction', default=0.0, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)

    parser.add_argument('--task_idx', default=0, type=int)    
    parser.add_argument('--maxHistoriesPerFile', default=1000, type=int)
    parser.add_argument('--historySavePrefix', default='saveout', type=str)

    parser.add_argument('--eval_set', default='dev', type=str)      # 'dev' or 'test'
    parser.add_argument('--simplification_str', default='', type=str)

    # Add arguments for the pruner
    parser.add_argument('--embedding_server_port', default=12345, type=int)
    parser.add_argument('--threshold_strategy', default='similarity_threshold', type = str) # Must be one of similarity_threshold, top_k, top_p 
    parser.add_argument('--threshold_file', default='threshold_file_similarity_threshold.json', type = str)
    parser.add_argument('--pruning_strategy', default='soft', type = str) # Must be one of 'soft', 'hard' or 'hybrid'. Will use epsilon exploration strategy with 'hard' pruning.  \
    parser.add_argument('--starting_epsilon', default=0.1, type = float) # Parameter which decides proportion of steps when actions are not pruned to enable agent to come out of fixed positions. 
    parser.add_argument('--epsilon_schedule', default='increasing', type = str) # Must be one of 'fixed', 'increasing' 
    parser.add_argument('--prune_navigation_action', action='store_true')
    parser.add_argument('--no-prune_navigation_action', dest='prune_navigation_action', action='store_false')
    return parser.parse_args()

def main():
    ## assert jericho.__version__ == '2.1.0', "This code is designed to be run with Jericho version 2.1.0."
    args = parse_args()
    # Add current_epsilon.  Will only be used by the hard pruner
    args.current_epsilon = args.starting_epsilon

    configure_logger(args.output_dir)    
    agent = DRRN_Agent(args)

    # Initialize a threaded wrapper for the ScienceWorld environment
    envs = VecEnv(args.num_envs, args)
    print("Sleeping for 10 secs")
    time.sleep(10)

    # Initialize the save buffers
    taskIdx = args.task_idx
    bufferedHistorySaverTrain = BufferedHistorySaver(filenameOutPrefix = args.historySavePrefix + "-task" + str(taskIdx) + "-train")
    bufferedHistorySaverEval = BufferedHistorySaver(filenameOutPrefix = args.historySavePrefix + "-task" + str(taskIdx) + "-eval")

    action_ranker_obj = ActionScorer(embedding_server_port = args.embedding_server_port, taskIdx = taskIdx, threshold_strategy = args.threshold_strategy, pruning_strategy = args.pruning_strategy, threshold_file = args.threshold_file)

    # Start training
    start = timeit.default_timer()


    train(agent, envs, args.max_steps, args.update_freq, args.eval_freq,
          args.checkpoint_freq, args.log_freq, args, bufferedHistorySaverTrain, bufferedHistorySaverEval, action_ranker_obj)

    end = timeit.default_timer()
    deltaTime = end - start
    deltaTimeMins = deltaTime / 60
    print("Runtime: " + str(deltaTime) + " seconds  (" + str(deltaTimeMins) + " minutes)")

    print("Rate: " + str(args.max_steps / deltaTime) + " steps/second")
    print("SimplificationStr: " + str(args.simplification_str))


def interactive_run(env):
    ob, info = env.reset()
    while True:
        print(clean(ob), 'Reward', reward, 'Done', done, 'Valid', info)
        ob, reward, done, info = env.step(input())
        info = sanitizeInfo(info)
        ob = sanitizeObservation(ob, info)

if __name__ == "__main__":
    main()
