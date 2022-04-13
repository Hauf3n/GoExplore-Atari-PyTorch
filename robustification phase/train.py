import numpy as np
import argparse
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random
import os

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from Agent import PPO_Agent
from Backward_Manager import Backward_Manager
from Atari_Wrapper import Atari_Wrapper
import Environment_Runner as runner
from Environment_Runner_Demonstration import Env_Runner_Demonstrations
from Dataset import Batch_DataSet, Batch_DataSet_Demo

device = torch.device("cuda:0")
dtype = torch.float

def compute_advantage_and_value_targets(rewards, values, dones, gamma, lam):
    
    advantage_values = []
    old_adv_t = torch.tensor(0.0).to(device)
    
    value_targets = []
    old_value_target = values[-1]
    
    for t in reversed(range(len(rewards)-1)):
        
        if dones[t]:
            old_adv_t = torch.tensor(0.0).to(device)
        
        # ADV
        delta_t = rewards[t] + (gamma*(values[t+1])*int(not dones[t+1])) - values[t]
        
        A_t = delta_t + gamma*lam*old_adv_t
        advantage_values.append(A_t[0])
        
        old_adv_t = delta_t + gamma*lam*old_adv_t
        
        # VALUE TARGET
        value_target = rewards[t] + gamma*old_value_target*int(not dones[t+1])
        value_targets.append(value_target[0])
        
        old_value_target = value_target
    
    advantage_values.reverse()
    value_targets.reverse()
    
    return advantage_values, value_targets


def train(args):  
    
    # create folder to save networks, csv, hyperparameter
    folder_name = time.asctime(time.gmtime()).replace(" ","_").replace(":","_")
    os.mkdir(folder_name)
    
    # save the hyperparameters in a file
    f = open(f'{folder_name}/args.txt','w')
    for i in args.__dict__:
        f.write(f'{i},{args.__dict__[i]}\n')
    f.close()
    
    # arguments
    env_name = args.env
    num_stacked_frames = args.stacked_frames
    start_lr = args.lr 
    gamma = args.gamma
    lam = args.lam
    minibatch_size = args.minibatch_size
    T = args.T
    c1 = args.c1
    c2 = args.c2
    w_sil = args.w_sil
    w_sil_vf = args.w_sil_vf
    w_sil_ent = args.w_sil_ent
    actors = args.actors
    demonstration_actors = args.demonstration_actors
    start_eps = args.eps
    epochs = args.epochs
    total_steps = args.total_steps
    save_model_steps = args.save_model_steps

    # init
    
    # in/output    
    in_channels = num_stacked_frames
    num_actions = gym.make(f'{env_name}NoFrameskip-v4').env.action_space.n

    # network and optim
    agent = PPO_Agent(in_channels, num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=start_lr)
    
    # actors
    env_runners = []
    for actor in range(actors):

        raw_env = gym.make(f'{env_name}Deterministic-v4')
        env = Atari_Wrapper(raw_env, f'{env_name}NoFrameskip-v4', num_stacked_frames, use_add_done=args.lives)
        
        env_runners.append(runner.Env_Runner(env, agent, folder_name))
        
    # demonstration actors
    env_runners_demo = []
    for actor in range(demonstration_actors):

        raw_env = gym.make(f'{env_name}Deterministic-v4')
        env = Atari_Wrapper(raw_env, f'{env_name}Deterministic-v4', num_stacked_frames, use_add_done=args.lives)
        
        env_runners_demo.append(Env_Runner_Demonstrations(env, agent, gamma))
    
    # backward algorithm manager
    backward_manager = Backward_Manager(args)
    
    num_model_updates = 0

    start_time = time.time()
    while runner.cur_step < total_steps:
        
        # change lr and eps over time
        alpha = 1 - (runner.cur_step / total_steps)
        current_lr = start_lr * alpha
        current_eps = start_eps * alpha
        
        #set lr
        for g in optimizer.param_groups:
            g['lr'] = current_lr
        
        # get ppo actors data
        batch_obs, batch_actions, batch_adv, batch_v_t, batch_old_action_prob = None, None, None, None, None
    
        for env_runner in env_runners:
        
            demo_idx, demo = backward_manager.select_demo()
            demo_restore, demo_frame, _, demo_steps, desired_return = demo 
            data, performance, steps_to_reach_return = env_runner.run(demo_steps+50, demo_restore, demo_frame, desired_return)    
            obs, actions, rewards, dones, values, old_action_prob = data
            adv, v_t = compute_advantage_and_value_targets(rewards, values, dones, gamma, lam)
            
            # report ppo performance to backward manager
            save_network_request = backward_manager.report_performance(demo_idx, performance, steps_to_reach_return)
            # save network
            if save_network_request:
                torch.save(agent,f'{folder_name}/{env_name}-{runner.cur_step}.pt')
            
            # assemble data from the different ppo actors 
            batch_obs = torch.stack(obs[:-1]) if batch_obs == None else torch.cat([batch_obs,torch.stack(obs[:-1])])
            batch_actions = np.stack(actions[:-1]) if batch_actions is None else np.concatenate([batch_actions,np.stack(actions[:-1])])
            batch_adv = torch.stack(adv) if batch_adv == None else torch.cat([batch_adv,torch.stack(adv)])
            batch_v_t = torch.stack(v_t) if batch_v_t == None else torch.cat([batch_v_t,torch.stack(v_t)]) 
            batch_old_action_prob = torch.stack(old_action_prob[:-1]) if batch_old_action_prob == None else torch.cat([batch_old_action_prob,torch.stack(old_action_prob[:-1])])
    
        # load into dataset/loader
        dataset = Batch_DataSet(batch_obs, batch_actions, batch_adv, batch_v_t, batch_old_action_prob)
        dataloader = DataLoader(dataset, batch_size=minibatch_size, num_workers=0, shuffle=True, drop_last=True)
        
        # get demonstration actors data
        batch_obs_demo, batch_actions_demo, batch_returns_demo, batch_values_demo  = None, None, None, None
        for env_runner in env_runners_demo:
        
            _, demo = backward_manager.select_demo()
            demo_restore, demo_frame, demo_actions, _, _  = demo
            obs, actions, returns, _, values, _ = env_runner.run(demo_actions, demo_restore, demo_frame)
            
            # assemble data from the different demonstration actors
            batch_obs_demo =  torch.stack(obs[:-1]) if batch_obs_demo == None else torch.cat([batch_obs_demo,torch.stack(obs[:-1])])
            batch_actions_demo = np.stack(actions[:-1]) if batch_actions_demo is None else np.concatenate([batch_actions_demo,np.stack(actions[:-1])])
            batch_returns_demo = np.stack(returns[:-1]) if batch_returns_demo is None else np.concatenate([batch_returns_demo,np.stack(returns[:-1])])
            batch_values_demo = torch.stack(values[:-1]) if batch_values_demo == None else torch.cat([batch_values_demo,torch.stack(values[:-1])])
        
        # load demo into dataset/loader
        dataset_demo = Batch_DataSet_Demo(batch_obs_demo, batch_actions_demo, batch_returns_demo, batch_values_demo)
        dataloader_demo = DataLoader(dataset_demo, batch_size=minibatch_size, num_workers=0, shuffle=True, drop_last=True)
        
        # update
        for epoch in range(epochs):
            
            iter_dataloader = iter(dataloader)
            iter_dataloader_demo = iter(dataloader_demo)
             
            # sample minibatches
            for i in range(8):
                optimizer.zero_grad()
                
                # get data from loaders
                try:
                    batch = next(iter_dataloader)
                except StopIteration:
                    iter_dataloader = iter(dataloader)
                    batch = next(iter_dataloader)
                    
                try:    
                    batch_demo = next(iter_dataloader_demo)
                except StopIteration:
                    iter_dataloader_demo = iter(dataloader_demo)
                    batch_demo = next(iter_dataloader_demo)
                
                # get PPO data
                obs, actions, adv, v_target, old_action_prob = batch
                adv = adv.squeeze(1)
                
                # get SIL data
                obs_demo, actions_demo, returns_demo, old_values_demo = batch_demo
                old_values_demo = old_values_demo.squeeze(2).squeeze(1)
                returns_demo = returns_demo.to(dtype)
                
                # get policy and value function for PPO and SIL
                p, vf = agent(torch.cat([obs, obs_demo]))
                policy, policy_demo = torch.split(p, minibatch_size, dim=0)
                v, values_demo = torch.split(vf, minibatch_size, dim=0)
                
                ### PPO loss ####
                
                # normalize adv values
                adv = ( adv - torch.mean(adv) ) / ( torch.std(adv) + 1e-8)
                
                # get the correct policy actions
                pi = policy[range(minibatch_size),actions.long()]
                
                # probaility ratio r_t(theta)
                probability_ratio = pi / (old_action_prob + 1e-8)
                
                # compute CPI
                CPI = probability_ratio * adv
                # compute clip*A_t
                clip = torch.clamp(probability_ratio,1-current_eps,1+current_eps) * adv     
                
                # policy loss | take minimum
                L_CLIP = torch.mean(torch.min(CPI, clip))
                
                # value loss | mse
                L_VF = torch.mean(torch.pow(v - v_target,2))
                
                # policy entropy loss 
                S = torch.mean( - torch.sum(policy * torch.log(policy + 1e-8),dim=1))
                
                # PPO loss
                PPO_loss = - L_CLIP + c1 * L_VF - c2 * S
                
                ### SIL loss ###
                
                # get correct actions probs
                pi_demo = policy_demo[range(minibatch_size),actions_demo.long()]
                # alter values_demo shape
                values_demo = values_demo.squeeze(1)
                
                # compute SIL policy loss
                L_SIL_PG = torch.mean(- torch.log( pi_demo ) * torch.max(torch.tensor(0.0).to(device), returns_demo - old_values_demo))
                
                # compute SIL value loss
                L_SIL_VF = torch.mean(1/2 * (torch.max(torch.tensor(0.0).to(device), returns_demo - values_demo))**2)
                
                # compute SIL entropy loss
                S_SIL = torch.mean( - torch.sum(policy_demo * torch.log(policy_demo + 1e-8),dim=1))
                
                # SIL loss
                SIL_loss = L_SIL_PG + w_sil_vf * L_SIL_VF - w_sil_ent * S_SIL
                
                # SIL + PPO loss
                loss = PPO_loss +  w_sil * SIL_loss 
                loss.backward()
                optimizer.step()
        
        backward_manager.protocol_progress()    
        num_model_updates += 1
         
        # print time
        if runner.cur_step%50000 < T*actors:
            end_time = time.time()
            print(f'*** total steps: {runner.cur_step} | time(50K): {end_time - start_time} ***')
            start_time = time.time()
            
    env.close()
    
if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    
    # set hyperparameter
    
    args.add_argument('-lr', type=float, default=2.5e-4)
    args.add_argument('-env', default='MontezumaRevenge')
    args.add_argument('-lives', type=bool, default=False)
    args.add_argument('-stacked_frames', type=int, default=2)
    args.add_argument('-gamma', type=float, default=0.999)
    args.add_argument('-lam', type=float, default=0.95)
    args.add_argument('-eps', type=float, default=0.1)
    args.add_argument('-c1', type=float, default=0.5)
    args.add_argument('-c2', type=float, default=0.01)
    args.add_argument('-w_sil', type=float, default=0.1)#0.1)
    args.add_argument('-w_sil_vf', type=float, default=0.01)
    args.add_argument('-w_sil_ent', type=float, default=1e-5)
    args.add_argument('-minibatch_size', type=int, default=32)
    args.add_argument('-actors', type=int, default=32)
    args.add_argument('-demonstration_actors', type=int, default=2)
    args.add_argument('-T', type=int, default=150)
    args.add_argument('-epochs', type=int, default=4)
    args.add_argument('-total_steps', type=int, default=80000000)
    args.add_argument('-save_model_steps', type=int, default=100000)
    args.add_argument('-report', type=int, default=50000)
    args.add_argument('-backward_records_eval', type=int, default=30)
    args.add_argument('-backward_threshold', type=float, default=0.67)
    args.add_argument('-backward_steps', type=int, default=10)
    
    train(args.parse_args())
