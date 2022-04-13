import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0")
dtype = torch.float

class Logger:
    
    def __init__(self, filename):
        self.filename = filename
        
        f = open(f"{self.filename}.csv", "w")
        f.close()
        
    def log(self, msg):
        f = open(f"{self.filename}.csv", "a+")
        f.write(f"{msg}\n")
        f.close()

cur_step = 0          
class Env_Runner:
    # agent env loop
    def __init__(self, env, agent, logger_folder):
        super().__init__()
        
        self.env = env
        self.agent = agent
        
        self.logger = Logger(f'{logger_folder}/training_info')
        self.logger.log("training_step, return")
        
        self.ob = self.env.reset()
        
    def run(self, steps, restore, start_obs, desired_return):
        global cur_step
        
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
        
        # track amount of timesteps to reach desired return
        steps_to_reach_desired_return = -1
        
        for step in range(steps):
            
            self.ob = torch.tensor(self.ob).to(device).to(dtype)
            policy, value = self.agent(self.ob.unsqueeze(0))
            action = self.agent.select_action(policy.detach().cpu().numpy()[0])
            
            obs.append(self.ob)
            actions.append(action)
            values.append(value.detach())
            action_prob.append(policy[0,action].detach())
            
            self.ob, r, done, info, additional_done = self.env.step(action)
            
            if self.env.Return == desired_return:
                if steps_to_reach_desired_return == -1:
                    steps_to_reach_desired_return = step
            
            if step == steps-1:
              self.logger.log(f'{cur_step+step},{self.env.Return}')
              
            if done: # real environment reset, other add_dones are for learning purposes
                self.ob = self.env.reset()
                if "return" in info:
                    self.logger.log(f'{cur_step+step},{info["return"]}')
            
            rewards.append(r)
            dones.append(done or additional_done)
            
        cur_step += steps
                                    
        return [obs, actions, rewards, dones, values, action_prob], self.env.Return, steps_to_reach_desired_return