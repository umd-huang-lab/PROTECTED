import pickle
import sqlite3
import copy

from policy_gradients.agent import Trainer
import numpy as np
import pandas as pd
import os
import copy
import random
import argparse
import json
import torch
from cox.store import Store
from run import add_common_parser_opts, override_json_params
from auto_LiRPA.eps_scheduler import LinearScheduler
import logging
from policy_gradients.exp3 import Exp3Alg

logging.disable(logging.INFO)


def get_parser():
    parser = argparse.ArgumentParser(description='Generate experiments to be run.')
    parser.add_argument('--config-path', type=str, default='configs/config_hopper_pa_atla_ppo.json', required=False,
                        help='json for this config')
    parser.add_argument('--out-dir-prefix', type=str, default='', required=False,
                        help='prefix for output log path')
    parser.add_argument('--exp-id', type=str, help='experiement id for testing', default='')
    parser.add_argument('--row-id', type=int, help='which row of the table to use', default=-1)
    parser.add_argument('--num-episodes', type=int, help='number of episodes for testing', default=1)
    parser.add_argument('--compute-kl-cert', action='store_true', help='compute KL certificate')
    parser.add_argument('--use-full-backward', action='store_true',
                        help='Use full backward LiRPA bound for computing certificates')
    parser.add_argument('--deterministic', action='store_true', help='disable Gaussian noise in action for evaluation')
    parser.add_argument('--noise-factor', type=float, default=1.0,
                        help='increase the noise (Gaussian std) by this factor.')
    parser.add_argument('--load-model', type=str, help='load a pretrained model file', default='')
    parser.add_argument('--seed', type=int, help='random seed', default=1234)
    # Other configs
    parser.add_argument('--iteration', type=int, help='number of iterations for our iterative methods', default=1)
    parser.add_argument('--ref-model-list', '--list', type=str, nargs='+')
    parser.add_argument('--attack-multiple-victims', action='store_true')
    parser.add_argument('--attack-exp3', action='store_true')
    parser.add_argument('--load-env', type=str)
    parser.add_argument('--results-log', type=str)
    # Switch configs
    parser.add_argument('--exp3-lr', type=float, default=5.0)
    parser.add_argument('--rounds', type=int, default=2000)
    parser.add_argument('--switch-type', type=str, choices=['period', 'prob'], default='period')
    parser.add_argument('--switch-interval', type=str, default=100)
    parser.add_argument('--switch-prob', type=float, default=0.1)

    parser = add_common_parser_opts(parser)

    return parser


def get_attack_seeds(rounds, switch_type, interval=100, prob=0.1):
    assert switch_type=='period' or switch_type=='prob'
    attack_seeds = np.zeros(rounds)
    if switch_type=='period':
        assert interval>0 and interval<=rounds
        attack_times = np.arange(0, rounds, 2*interval)
        for n in range(len(attack_times)-1):
            attack_seeds[switch_times[n]:switch_times[n]+interval] = 1.0
    else: # switch_type=='prob'
        assert prob>=0.0 and prob<=1.0
        intv = 100
        attack_seeds = (np.random.uniform(size=rounds)<=prob)
        switch_times = np.arange(0, rounds, intv)
        cache_num = 1.0
        for n in range(len(switch_times)-1):
            end_num = min(switch_times[n+1], total_rounds)
            attack_seeds[switch_times[n]:end_num] = cache_num
            cache_num = (cache_num+attack_seeds[switch_times[n+1]])%2
        attack_seeds[switch_times[-1]:total_rounds] = cache_num
    return attack_seeds


def get_current_attacks(params, seed):
    if seed==0:
        params['attack_method'] = None
        params['attack_eps'] = 0.0
    else:
        params['attack_method'] = "paadvpolicy"
    return params


def write_results(results_log, results_dict):
    with open(results_log, 'w') as wf:
        for exp in results_dict.keys():
            wf.write('Exp: {}\nmean: {}, std:{}, min:{}, max:{}\n'.format(exp,
                                                                          results_dict[exp]['mean'],
                                                                          results_dict[exp]['std'],
                                                                          results_dict[exp]['min'],
                                                                          results_dict[exp]['max']))


def main(params):
    override_params = copy.deepcopy(params)
    excluded_params = ['config_path', 'out_dir_prefix', 'num_episodes', 'row_id', 'exp_id',
                       'load_model', 'seed', 'deterministic', 'noise_factor', 'compute_kl_cert', 'use_full_backward',
                       'exp3_lr', 'rounds']

    # original_params contains all flags in config files that are overridden via command.
    for k in list(override_params.keys()):
        if k in excluded_params:
            del override_params[k]

    # Append a prefix for output path.
    if params['out_dir_prefix']:
        params['out_dir'] = os.path.join(params['out_dir_prefix'], params['out_dir'])
        print(f"setting output dir to {params['out_dir']}")

    if params['config_path']:
        # Load from a pretrained model using existing config.
        # First we need to create the model using the given config file.
        json_params = json.load(open(params['config_path']))

        params = override_json_params(params, json_params, excluded_params)

    if 'load_model' in params and params['load_model']:
        for k, v in zip(params.keys(), params.values()):
            assert v is not None, f"Value for {k} is None"

        # Create the agent from config file.
        p = Trainer.agent_from_params(params, store=None)
        print('Loading pretrained model', params['load_model'])
        pretrained_model = torch.load(params['load_model'])
        if 'policy_model' in pretrained_model:
            p.policy_model.load_state_dict(pretrained_model['policy_model'])
        if 'val_model' in pretrained_model:
            p.val_model.load_state_dict(pretrained_model['val_model'])
        if 'policy_opt' in pretrained_model:
            p.POLICY_ADAM.load_state_dict(pretrained_model['policy_opt'])
        if 'val_opt' in pretrained_model:
            p.val_opt.load_state_dict(pretrained_model['val_opt'])
        # Restore environment parameters, like mean and std.
        if 'envs' in pretrained_model:
            p.envs = pretrained_model['envs']
        for e in p.envs:
            e.normalizer_read_only = True
            e.setup_visualization(params['show_env'], params['save_frames'], params['save_frames_path'])
    else:
        # Load from experiment directory. No need to use a config.
        base_directory = params['out_dir']
        store = Store(base_directory, params['exp_id'], mode='r')
        if params['row_id'] < 0:
            row = store['final_results'].df
        else:
            checkpoints = store['checkpoints'].df
            row_id = params['row_id']
            row = checkpoints.iloc[row_id:row_id + 1]
        print("row to test: ", row)
        if params['cpu'] == None:
            cpu = False
        else:
            cpu = params['cpu']
        p, _ = Trainer.agent_from_data(store, row, cpu, extra_params=params, override_params=override_params,
                                       excluded_params=excluded_params)
        store.close()

    rewards = []

    print('Gaussian noise in policy:')
    print(torch.exp(p.policy_model.log_stdev))
    original_stdev = p.policy_model.log_stdev.clone().detach()
    if params['noise_factor'] != 1.0:
        p.policy_model.log_stdev.data[:] += np.log(params['noise_factor'])
    if params['deterministic']:
        print('Policy runs in deterministic mode. Ignoring Gaussian noise.')
        p.policy_model.log_stdev.data[:] = -100
    print('Gaussian noise in policy (after adjustment):')
    print(torch.exp(p.policy_model.log_stdev))

    num_episodes = params['num_episodes']
    all_rewards = []
    all_lens = []
    all_kl_certificates = []

    for i in range(num_episodes):
        print('Episode %d / %d' % (i + 1, num_episodes))
        ep_length, ep_reward, actions, action_means, states, kl_certificates = p.run_test(
            compute_bounds=params['compute_kl_cert'], use_full_backward=params['use_full_backward'],
            original_stdev=original_stdev)
        if i == 0:
            all_actions = actions.copy()
            all_states = states.copy()
        else:
            all_actions = np.concatenate((all_actions, actions), axis=0)
            all_states = np.concatenate((all_states, states), axis=0)
        if params['compute_kl_cert']:
            print('Epoch KL certificates:', kl_certificates)
            all_kl_certificates.append(kl_certificates)
        all_rewards.append(ep_reward)
        all_lens.append(ep_length)
        # Current step mean, std, min and max
        mean_reward, std_reward, min_reward, max_reward = np.mean(all_rewards), np.std(all_rewards), np.min(
            all_rewards), np.max(all_rewards)

    attack_dir = 'attack-{}-eps-{}'.format(params['attack_method'], params['attack_eps'])
    save_path = os.path.join(params['out_dir'], params['exp_id'], attack_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for name, value in [('actions', all_actions), ('states', all_states), ('rewards', all_rewards),
                        ('length', all_lens)]:
        with open(os.path.join(save_path, '{}.pkl'.format(name)), 'wb') as f:
            pickle.dump(value, f)
    print(params)
    with open(os.path.join(save_path, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    mean_reward, std_reward, min_reward, max_reward = np.mean(all_rewards), np.std(all_rewards), np.min(
        all_rewards), np.max(all_rewards)
    if params['compute_kl_cert']:
        print('KL certificates stats: mean: {}, std: {}, min: {}, max: {}'.format(np.mean(all_kl_certificates),
                                                                                  np.std(all_kl_certificates),
                                                                                  np.min(all_kl_certificates),
                                                                                  np.max(all_kl_certificates)))
    print('\n')
    print('all rewards:', all_rewards)
    print('rewards stats:\nmean: {}, std:{}, min:{}, max:{}'.format(mean_reward, std_reward, min_reward, max_reward))
    return mean_reward, std_reward, min_reward, max_reward


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.load_model:
        assert args.config_path, "Need to specificy a config file when loading a pretrained model."

    params = vars(args)
    if args.attack_exp3:
        meta_policy = np.load(os.path.basename(args.attack_advpolicy_network) + "/meta_policy.npy")
    results_dict = {}
    mean_list = []

    seed = params['seed']
    #torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # begin our algorithm
    total_rounds = params['rounds']
    attack_seed_list = get_attack_seeds(rounds=total_rounds, switch_type=params['switch_type'], 
                                        interval=params['switch_interval'], prob=params['switch_prob'])
    attack_seed_list[:total_rounds//4] = 1
    attack_seed_list[total_rounds//2:3*total_rounds//4] = 1

    exp3_alg = Exp3Alg(K=len(args.ref_model_list), T=total_rounds, normalizer=6000, lr=params['exp3_lr'])
    sample_list = np.zeros(total_rounds)
    weight_list = np.zeros((total_rounds, len(args.ref_model_list)))
    for t in range(total_rounds):
        # sample from bandit
        sampled_index = exp3_alg.sample()
        print("t: %d sampled_index: %d" % (t, sampled_index))
        victim_path = args.ref_model_list[sampled_index]

        params['load_model'] = victim_path
        try:
            # Swich attacks by update the params
            params_ = get_current_attacks(copy.deepcopy(params), attack_seed_list[t])
            mean_rew, _, _, _ = main(params_)
            # Update the bandit weights
            weight_list[t] = exp3_alg.get_policy()
            sample_list[t] = sampled_index
            exp3_alg.update(mean_rew, sampled_index)
        except:
            print('Test failed: {}')

        if t%10==0:
            df = pd.DataFrame(weight_list, columns=args.ref_model_list)
            df.to_csv('%s_weight.csv' % (params['results_log']), sep=',', index_label='serial')
    
    df = pd.DataFrame(weight_list, columns=args.ref_model_list)
    df.to_csv('%s_weight.csv' % (params['results_log']), sep=',', index_label='serial')
