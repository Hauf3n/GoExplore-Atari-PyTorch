import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import gym
import numpy as np
import pickle
import Environment_Runner as runner
from Atari_Wrapper import Atari_Wrapper
         
device = torch.device("cuda:0")
dtype = torch.float

class Backward_Demo:

    def __init__(self, demo, args):
        super().__init__()  
        self.args = args
        
        self.actions = demo[0]
        self.restores = demo[1]
        self.frames = demo[2]

        # save current backward position
        self.current_pos = len(self.actions) - args.backward_steps - 10
        self.demo_len = len(self.actions)
        
        # let the agent run T timesteps for this demo starting point
        # we want to decrease this value when ppo finds shorter traj than demo
        self.T = self.demo_len-self.current_pos
        
        # calculate the future collected reward for every pos to track performance
        # set env to first restore position and start calculating
        raw_env = gym.make(f'{args.env}Deterministic-v4')
        env = Atari_Wrapper(raw_env, f'{args.env}Deterministic-v4', args.stacked_frames, use_add_done=args.lives, clip=False)
        env.reset()
        env.env.restore_full_state(self.restores[0])
        
        rewards = []
        for action in self.actions:
            _, r, _, _, _ = env.step(action)
            rewards.append(r)
        
        self.perf_measure = []
        for i in range(len(self.actions)):
            self.perf_measure.append(np.sum(rewards[i::]))
        
        # track performance records for current demo position
        self.records = [] 
     
    def get(self):
    # get data for current position
        return (self.restores[self.current_pos], self.frames[self.current_pos], 
                self.actions[self.current_pos::], self.T, self.perf_measure[self.current_pos])
     
    def add_performance(self, perf_value):
    # ppo actor has to achieve the sum of rewards from the demo to count as successfull
        if perf_value >= self.perf_measure[self.current_pos]:
            self.records.append(1)
        else:
            self.records.append(0)
            
        return self.check_position()
        
    def calculate_env_steps(self, steps):
        # dont waste time with bad demos
        self.T = np.minimum(self.T, steps)
            
    def check_position(self):
    # we want to move the position closer to the start if the perfomance is good
        if len(self.records) < self.args.backward_records_eval:
            return False
            
        recent_records = self.records[- self.args.backward_records_eval::] 
        if np.sum(recent_records)/self.args.backward_records_eval >= self.args.backward_threshold:
            # move position closer to start
            self.current_pos = np.maximum(0, self.current_pos - self.args.backward_steps)
            # reset records
            self.records = []
            # reset T and wait for lower value 
            self.T = self.demo_len-self.current_pos
            
            print(f'Moving positions! Position: {self.current_pos}/{self.demo_len} | Performance NEW/OLD {self.perf_measure[self.current_pos]}/{self.perf_measure[self.current_pos+self.args.backward_steps]}')
            
            if self.current_pos == 0:
                # save network
                return True
            
        return False
                    

class Logger:
    
    def __init__(self, filename):
        self.filename = filename
        
        f = open(f"{self.filename}.csv", "w")
        f.close()
        
    def log(self, msg):
        f = open(f"{self.filename}.csv", "a+")
        f.write(f"{msg}\n")
        f.close()
        
class Backward_Manager:
    
    def __init__(self, args):
        super().__init__()  
        
        self.demonstrations = []
        # load demonstrations
        filenames = os.listdir(os.getcwd())
        for filename in filenames:
            if 'demonstration' in filename:
                f = open(filename, 'rb')
                demonstration = pickle.load(f)
                f.close()
                
                self.demonstrations.append(demonstration)
                print("demo len: ", len(demonstration[0]))
                
        print(f'found {len(self.demonstrations)} demonstrations.')
        self.num_demonstrations = len(self.demonstrations)
        
        self.backward_demos = []
        for demo in self.demonstrations:
            self.backward_demos.append(Backward_Demo(demo, args))
            
        self.logger = Logger(f'position_info')
        s = 'training_step'
        for i in range(len(self.demonstrations)):
            s += f',{i}'
        self.logger.log(s)
    
    def select_demo(self):
    # select a demo   
        idx = np.random.choice(range(self.num_demonstrations),1)[0]
        return (idx, self.backward_demos[idx].get())
    
    def report_performance(self, idx, value, steps_to_reach_desired_return):
        # add a record for a demo
        
        save_network_request = self.backward_demos[idx].add_performance(value) 
        
        # we dont want to waste time, attempt to decrease T
        if steps_to_reach_desired_return > 0:
            self.backward_demos[idx].calculate_env_steps(steps_to_reach_desired_return)
        
        return save_network_request
    
    def protocol_progress(self):
        s = f'{runner.cur_step}'
        for i in range(len(self.demonstrations)):
            s += f',{self.backward_demos[i].current_pos}'
        self.logger.log(s)