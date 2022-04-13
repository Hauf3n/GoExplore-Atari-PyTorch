import numpy as np
import os
import cv2
import gym

class Atari_Wrapper(gym.Wrapper):
    
    def __init__(self, env, k):
        super(Atari_Wrapper, self).__init__(env)
        self.k = k
        self.env = env
        
    def reset(self):
    
        ob = self.env.reset() 
        return ob
    
    
    def step(self, action): 
        # do k frameskips, same action for every intermediate frame
        
        reward = 0
        # k frame skips or end of episode
        for i in range(self.k):
            
            ob, r, done, info = self.env.step(action)
            
            # add reward
            reward += r
            
            if done: # env done
                break
                
        return ob, reward, done, info
        
    def restore_full_state(self, restore):
        self.env.restore_full_state(restore)