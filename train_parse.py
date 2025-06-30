import os
import time
from datetime import datetime
import argparse

import torch
import numpy as np
from PPO import PPO
from Env import MultiDroneAoIEnv

import copy

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for PPO algorithm on RoboschoolWalker2d-v1")
    
    # Environment hyperparameters
    parser.add_argument('--env_name', type=str, default="A-MAPPO", 
                        help="Name of the gym environment")
    parser.add_argument('--has_continuous_action_space', action='store_true', 
                        help="Whether the action space is continuous (default: discrete)")
    parser.add_argument('--max_ep_len', type=int, default=1000, 
                        help="Maximum timesteps in one episode")
    parser.add_argument('--max_training_timesteps', type=int, default=int(3e6), 
                        help="Maximum total timesteps for training")
    parser.add_argument('--print_freq', type=int, default=None, 
                        help="Print average reward interval (in timesteps, default: max_ep_len * 10)")
    parser.add_argument('--log_freq', type=int, default=None, 
                        help="Log average reward interval (in timesteps, default: max_ep_len * 2)")
    parser.add_argument('--save_model_freq', type=int, default=int(1e5), 
                        help="Model saving frequency (in timesteps)")
    
    # !Continuous action space parameters 不用管
    parser.add_argument('--action_std', type=float, default=0.6, 
                        help="Starting std for action distribution (continuous action space)")
    parser.add_argument('--action_std_decay_rate', type=float, default=0.05, 
                        help="Decay rate for action std")
    parser.add_argument('--min_action_std', type=float, default=0.1, 
                        help="Minimum action std (stop decay when reached)")
    parser.add_argument('--action_std_decay_freq', type=int, default=int(2.5e5), 
                        help="Action std decay frequency (in timesteps)")
    
    # PPO hyperparameters
    parser.add_argument('--update_timestep', type=int, default=None, 
                        help="Policy update frequency (in timesteps, default: max_ep_len * 4)")
    parser.add_argument('--K_epochs', type=int, default=10, 
                        help="Number of epochs for policy update in one PPO update(每次更新使用多少次数据)")
    parser.add_argument('--eps_clip', type=float, default=0.2, 
                        help="Clip parameter for PPO")
    parser.add_argument('--gamma', type=float, default=0.97, 
                        help="Discount factor")
    parser.add_argument('--lr_actor', type=float, default=0.0003, 
                        help="Learning rate for actor network")
    parser.add_argument('--lr_critic', type=float, default=0.001, 
                        help="Learning rate for critic network")
    parser.add_argument('--random_seed', type=int, default=0, 
                        help="Random seed (0 = no random seed)")
    parser.add_argument('--entropy_ratio', type=float, default=float(0.01), 
                        help="Ratio of pre reward")
    parser.add_argument('--gae-lambda', type=float, default=0.97,
                        help='lambda parameter for GAE (default: 1.00)')
    parser.add_argument('--gae_flag', action='store_true',
                        help="Use gae?")
    
    # Logging and checkpointing
    parser.add_argument('--log_dir', type=str, default="PPO_logs", 
                        help="Directory for log files")
    parser.add_argument('--checkpoint_dir', type=str, default="PPO_preTrained", 
                        help="Directory for model checkpoints")
    parser.add_argument('--run_num_pretrained', type=int, default=0, 
                        help="Run number for pretrained model to prevent overwriting")

    # 环境参数
    parser.add_argument('--M', type=int, default=2, 
                        help="Num of UAVs")
    parser.add_argument('--N', type=int, default=1, 
                        help="Num of Bases")
    parser.add_argument('--K', type=int, default=20, 
                        help="Num of PoIs")
    parser.add_argument('--T', type=int, default=1800,
                        help="Time limits")
    parser.add_argument('--BS_back_times', type=int, default=5,
                        help="Time limits")
    parser.add_argument('--map_size', type=int, default=1000, 
                        help="Map size")
    parser.add_argument('--pre_reward_ratio', type=float, default=float(4), 
                        help="Ratio of pre reward")
    parser.add_argument('--buffer_punishment', action='store_true',
                        help="Whether the action space is continuous (default: discrete)")
    parser.add_argument('--punishment_value', type=float, default=float(4), 
                        help="Ratio of pre reward")
    parser.add_argument('--reward_scale_size', type=float, default=float(10000), 
                        help="Ratio of pre reward")
    parser.add_argument('--print_info', action='store_true',
                        help="print_info?")
    args = parser.parse_args()
    
    print(args.gae_flag, "#"*10)
    # Set default values for dependent parameters
    args.print_freq = args.print_freq if args.print_freq is not None else args.max_ep_len * 10
    args.log_freq = args.log_freq if args.log_freq is not None else args.max_ep_len * 2
    args.update_timestep = args.update_timestep if args.update_timestep is not None else args.max_ep_len * 4
    
    return args

################################### Training ###################################
def train():
    args = parse_args()
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = args.env_name
    has_continuous_action_space = args.has_continuous_action_space
    max_ep_len = args.max_ep_len
    max_training_timesteps = args.max_training_timesteps
    print_freq = args.print_freq
    log_freq = args.log_freq
    save_model_freq = args.save_model_freq
    action_std = args.action_std
    action_std_decay_rate = args.action_std_decay_rate
    min_action_std = args.min_action_std
    action_std_decay_freq = args.action_std_decay_freq

    ################ PPO hyperparameters ################
    update_timestep = args.update_timestep
    K_epochs = args.K_epochs
    eps_clip = args.eps_clip
    gamma = args.gamma
    lr_actor = args.lr_actor
    lr_critic = args.lr_critic
    random_seed = args.random_seed
    
   
    
    # tensorboard 文件名
    time = datetime.now().strftime("%Y%m%d-%H%M")
    tensorboard_dir = os.path.join(
        'logs',  f'{env_name}')
        # 'logs',  f'{env_name}-{time}')


    # !环境  
    env = MultiDroneAoIEnv(args.M, args.N, args.K, args.T, args.map_size, args=args)

    # state space dimension
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = args.run_num_pretrained

    directory = args.checkpoint_dir
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)



    ################# training procedure ################

    # initialize PPO agents
    ppo_agent = []
    for i in range(args.M):
        ppo_agent.append(PPO(i, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, tensorboard_dir, args.entropy_ratio, args.gae_lambda, args.gae_flag, action_std))

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    
    
    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = [0]*args.M
        done_tag = 0
        
        # 维护一个时间队列
        UAVs_actions_queue = np.ones(args.M)*-1
        
        # 维护一个掩码
        masks = np.ones((args.M, args.K+args.N))
        actions_set = [[] for i in range(args.M)]
        positions_set = [[] for i in range(args.M)]
        done_bool = [0 for i in range(args.M)]
        
        # select action with policy
        UAVs_timing_next = [0 for i in range(args.M)]
        
        non_base_count = 0
        
        # 开始一个episode
        for t in range(1, max_ep_len+1):

            for i in range(args.M):
                if UAVs_actions_queue[i] == -1:
                    
                    if non_base_count<100:
                        UAVs_actions_queue[i] = ppo_agent[i].select_action(env._get_obs(i), masks[i])
                        if UAVs_actions_queue[i] not in list(range(args.K, args.K+args.N)):
                            non_base_count += 1
                        else:    
                            non_base_count = 0
                    else:
                        UAVs_actions_queue[i] = ppo_agent[i].select_action(env._get_obs(i), masks[i], args.K+args.N-1)
                        non_base_count = 0
                        
                    UAVs_timing_next[i] += env.time_cost(i, int(UAVs_actions_queue[i]))
                    
                    # print(i, UAVs_actions_queue[i], non_base_count)
            for i in range(args.M):
                UAVs_timing_next[i] += done_bool[i]*100000

            # !超过args.T的就设置为足够大，不用再做动作了
            action_UAV = np.argmin(UAVs_timing_next)

            state, reward, done, masked = env.step(action_UAV, int(UAVs_actions_queue[action_UAV]))
            ppo_agent[action_UAV].buffer.masks.append(torch.FloatTensor(masks[action_UAV]).to(torch.device('cuda:0')))
            
            actions_set[action_UAV].append(int(UAVs_actions_queue[action_UAV]))
            positions_set[action_UAV].append(env.drone_position_now[action_UAV].copy())
            
            masks[action_UAV] = masked
            # masks[action_UAV] = np.ones(args.K+args.N)
            UAVs_actions_queue[action_UAV] = -1
            
            # saving reward and is_terminals
            ppo_agent[action_UAV].buffer.rewards.append(reward)
            ppo_agent[action_UAV].buffer.is_terminals.append(done)
            
            if done:
                done_tag += 1
                done_bool[action_UAV] = 1


            time_step += 1
            current_ep_reward[action_UAV] += reward

            # 所有人UAV都结束后，开始训练
            if done_tag == args.M:
                if args.M == 1:
                    print("Episode: {}, agent 1 len : {}, rewards 1: {}, agent 2 len: {}, rewards 2: {}".format(i_episode, len(ppo_agent[0].buffer.rewards), sum(ppo_agent[0].buffer.rewards), len(ppo_agent[0].buffer.rewards), sum(ppo_agent[0].buffer.rewards)))
                else:
                    # print("Episode: {}, agent 1 len : {}, rewards 1: {}, agent 2 len: {}, rewards 2: {}".format(i_episode, len(ppo_agent[0].buffer.rewards), sum(ppo_agent[0].buffer.rewards), len(ppo_agent[1].buffer.rewards), sum(ppo_agent[1].buffer.rewards)))
                    # 构建格式化字符串的开头
                    format_string = "Episode: {}"
                    values = [i_episode]
                    rewards_sum = 0
                    for i, agent in enumerate(ppo_agent, 1):  # 从 1 开始编号代理
                        format_string += ", Agent {} len: {}, rewards {}: {}"
                        values.extend([i, len(agent.buffer.rewards), i, sum(agent.buffer.rewards)])
                        rewards_sum += sum(agent.buffer.rewards)
                    format_string += " Sum of rewards: {}"
                    values.append(rewards_sum)
                    print(format_string.format(*values))
                    
                for i in range(args.M):
                    ppo_agent[i].update()

            
            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
                print("Episode : {} \t\t Timestep : {} \t\t Current epi reward : {}".format(i_episode, time_step, current_ep_reward))


                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            # !保存模型
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent[0].save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # !如果所有无人机都结束了
            if done_tag == args.M:
                break

        print_running_reward += current_ep_reward[0]
        print_running_episodes += 1

        log_running_reward += current_ep_reward[0]
        log_running_episodes += 1

        i_episode += 1
        
        
        # ! 测试
        if i_episode % 10 == 0:
            print(1)
            env.reset()
            masks = np.ones((args.M, args.K+args.N))
            UAVs_actions_queue = np.ones(args.M)*-1
            done_tag = 0
            test_rewards = [0]*args.M
            test_actions = [[[0, 0]] for i in range(args.M)]
            test_route = [[] for i in range(args.M)]
            
            
                    
            # 开始一个episode
            finish_step = 400
            for t in range(1, finish_step+1):

                # select action with policy
                UAVs_timing_next = env.drone_timing_now
                
                for i in range(args.M):
                    if UAVs_actions_queue[i] == -1:
                        UAVs_actions_queue[i] = ppo_agent[i].action_test(env._get_obs(i), masks[i])
                        UAVs_timing_next[i] += env.time_cost(i, int(UAVs_actions_queue[i]))

                # !超过args.T的就设置为足够大，不用再做动作了
                out_of_T = np.where(UAVs_timing_next >= args.T)
                UAVs_timing_next[out_of_T] = 100000
                action_UAV = np.argmin(UAVs_timing_next)
                
                state, reward, done, masked = env.step(action_UAV, int(UAVs_actions_queue[action_UAV]))
                masks[action_UAV] = masked
                test_rewards[action_UAV] += reward
                test_actions[action_UAV].append(list(env.get_pos(int(UAVs_actions_queue[action_UAV]))))
                test_route[action_UAV].append(int(UAVs_actions_queue[action_UAV]))
                UAVs_actions_queue[action_UAV] = -1
                
                if done:
                    done_tag += 1
                    
                
                if done_tag == args.M:
                # if finish_step == t:
                    ppo_agent[0].call_2_record(i_episode//10, test_rewards[0])
                    # env.visualize_routes(test_actions, "./test_figs")
                    env.visualize_routes(positions_set, "./test_figs")
                    break
                
            print(actions_set[0][:30])

            

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()