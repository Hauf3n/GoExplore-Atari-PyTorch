import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import pickle
         
device = torch.device("cuda:0")
dtype = torch.float         
         
class Env_Runner_Demonstrations:
    # imitate a demo seq
    def __init__(self, env, agent, gamma):
        super().__init__()
        
        self.env = env
        self.agent = agent
        self.gamma = gamma
        
        self.ob = self.env.reset()        

    def run(self, demo_actions, restore, start_obs):
        self.env.reset()
        
        # set env and obs to restore position
        self.env.env.restore_full_state(restore)
        self.env.frame_stack = np.stack([self.env.preprocess_observation(start_obs) for i in range(self.env.k)])
        self.ob = np.stack([self.env.preprocess_observation(start_obs) for i in range(self.env.k)])
        
        obs = []
        actions = []
        rewards = []
        dones = []
        values = []
        action_prob = []
        
        for action in demo_actions:
            self.ob = torch.tensor(self.ob).to(device).to(dtype)
            policy, value = self.agent(self.ob.unsqueeze(0))
            
            
            obs.append(self.ob)
            actions.append(action)
            values.append(value.detach())
            action_prob.append(policy[0,action].detach())
            
            self.ob, r, done, info, additional_done = self.env.step(action)          
            if done: # environment reset, other add_dones are for learning purposes
                self.ob = self.env.reset()
                self.cur_demo = self.get_demonstration()
                
            
            rewards.append(r)
            dones.append(done or additional_done)
         
        # compute demonstration return
        R = []
        cur_R = 0
        for reward in reversed(rewards):
            cur_R = reward + self.gamma*cur_R
            R.append(cur_R)
        R.reverse() 
        
        return [obs, actions, R, dones, values, action_prob]