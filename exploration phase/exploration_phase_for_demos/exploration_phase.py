import numpy as np
import cv2
import gym
import random
import matplotlib.pyplot as plt
import multiprocessing
import pickle

from Atari_Wrapper import Atari_Wrapper

# parameter
env_name = "MontezumaRevengeDeterministic-v4"
downscale_features = (8,11,8)

# downscale rgb img to a key representation
def make_representation(frame):
    h, w, p = downscale_features
    greyscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(greyscale_img, (h,w))
    resized_img_pix_threshold = ((resized_img/255.0) * p).reshape(-1).astype(int)
    return tuple(resized_img_pix_threshold)

class Cell():
    # item in archive
    
    def __init__(self, idx, restore, frame, key, score=-np.inf, traj_len=np.inf):
    
        self.visits = 0
        
        self.restore = restore
        self.idx = idx
        self.key = key
        self.score = score
        self.traj_len = traj_len
        self.frame = frame
        
class Archive():
    def __init__(self):
        # key | cell
        self.cells = {}
        
    def __iter__(self):
        return iter(self.cells)
    
    def init_archive(self, start_info):
        self.cells = {}
        # start cell
        self.cells[start_info[2]] = Cell(start_info[3],start_info[0],start_info[1],
                                         start_info[2], score=0, traj_len=0)
        # DONE cell
        self.cells[None] = Cell(start_info[3]+1, None, None, None)
        
        
class Env_Actor():
    # suggested paper actor - random action repeating actor
    # sample from bernoulli distribution with p = 1/mean for action repetition
    
    def __init__(self, env, mean_repeat=7):
        self.num_actions = env.action_space.n
        self.mean_repeat = mean_repeat
        self.env = env
        
        self.current_action = self.env.action_space.sample()
        self.repeat_action = np.random.geometric(1 / self.mean_repeat)
        
    def get_action(self):
        
        if self.repeat_action > 0:
            self.repeat_action -= 1
            return self.current_action
            
        self.current_action = self.env.action_space.sample()
        self.repeat_action = np.random.geometric(1 / self.mean_repeat) - 1
        return self.current_action
    
class Env_Runner():
    # agent env loop
    def __init__(self, env_name):
        self.env_name = env_name
        self.raw_env = gym.make(env_name).unwrapped
        self.actor = Env_Actor(self.raw_env)
        self.env = Atari_Wrapper(self.raw_env, 2)
        self.env.seed(0)
        
    def run(self, start_cell, max_steps=100):
        
        # init to selected state
        self.env.restore_full_state(start_cell.restore)
            
        traj_elemtents = []
        step = 0
        done = False
        while not done and step < max_steps:
            
            # collect data
            action = self.actor.get_action()
            frame, reward, d, _ = self.env.step(action)
            restore = self.env.env.clone_full_state()
            
            # save data
            traj_element = (make_representation(frame), frame, action, reward, d, restore)
            traj_elemtents.append(traj_element)
            
            if d:
                done = True
            step += 1
            
        return traj_elemtents
    
class CellSeletor():
    # select starting cells
    
    def __init__(self, archive):
        self.archive = archive
        
    def select_cells(self, amount):
        keys = []
        weights = []
        for key in self.archive.cells:
            if key == None: # done cell
                weights.append(0.0)
            else:
                weights.append(1/(np.sqrt(self.archive.cells[key].visits)+1))
            keys.append(key)
            
        indexes = np.random.choice(range(len(weights)),size=amount,p=weights/np.sum(weights))
        
        selected_cells = []
        for i in indexes:
            if self.archive.cells[keys[i]].key != None:
                selected_cells.append(self.archive.cells[keys[i]])
        return selected_cells
        
# multiprocessing method
def run(start_cell): 
    env_runner = Env_Runner(env_name)
    traj = env_runner.run(start_cell)
    return traj

# handle overlap conflict in archive
def better_cell_than_current(current_cell, new_score, new_traj_len):
    return ((current_cell.score < new_score) or 
        (current_cell.score == new_score and current_cell.traj_len > new_traj_len))

def main(filename='experience.data'):

    # init cell archive
    idx_counter = 0
    env_tmp = gym.make(env_name).unwrapped
    env_tmp.seed(0)
    start_s = env_tmp.reset()
    start_restore = env_tmp.clone_full_state()
    start_cell_info = [start_restore, start_s, make_representation(start_s), idx_counter]
    archive = Archive()
    archive.init_archive(start_cell_info)
    idx_counter += 2
    
    # init selector
    selector = CellSeletor(archive)

    iteration = 0
    best_score = -np.inf
    best_traj_len = np.inf
    # remember max score cell idx
    max_score_key = None
       
    # save the experience for demonstrations
    total_exp_actions = []
    total_exp_cell_idx = []
    total_exp_traj_len = []
    
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    steps = 0
    while archive.cells[max_score_key].score < 2500: #while True:
        
        # get data
        start_cells = selector.select_cells(int(multiprocessing.cpu_count() * 4)) 
        result = pool.map(run, start_cells)
        
        for traj, start_cell in zip(result,start_cells): # iterate all generated trajs
            
            # save experience of current trajectory
            exp_actions = []
            exp_cell_idx = [start_cell.idx]
            exp_traj_len = [start_cell.traj_len]
            
            steps += len(traj)
            # compute score and traj len for current pos
            cur_score = start_cell.score
            cur_traj_len = start_cell.traj_len

            seen_keys = []
            
            for i,traj_element in enumerate(traj):
                key, frame, action, reward, done, restore = traj_element

                if done:
                    key = None
                    restore = None

                cur_score += reward
                cur_traj_len += 1

                if key in archive.cells: # replace cell or not
                    new_is_better = better_cell_than_current(archive.cells[key], cur_score, cur_traj_len)

                    if new_is_better:
                        # transform existing cell
                        cell = archive.cells[key]
                        cell.visits = 0
                        cell.restore = restore
                        cell.score = cur_score
                        cell.traj_len = cur_traj_len
                        cell.frame = frame
                        cell.idx = idx_counter                      
                        
                        idx_counter += 1

                        if key == None:
                            pass

                else: # add new cell
                    new_cell = Cell(idx_counter, restore, frame, key, score=cur_score, traj_len=cur_traj_len)
                    archive.cells[key] = new_cell                  
                        
                    idx_counter += 1

                if cur_score > best_score or (cur_score == best_score and cur_traj_len < best_traj_len):
                    best_score = cur_score
                    best_traj_len = cur_traj_len
                    max_score_key = key
                    print("new MAX: ",best_score, " len:", best_traj_len)
                    if best_score == 700:
                        return

                if key not in seen_keys:
                    archive.cells[key].visits += 1
                    seen_keys.append(key)
                    
                # add experience
                exp_actions.append(action)
                exp_cell_idx.append(archive.cells[key].idx)
                exp_traj_len.append(archive.cells[key].traj_len)
            
            # collect experience
            total_exp_actions.append(exp_actions)
            total_exp_cell_idx.append(exp_cell_idx)
            total_exp_traj_len.append(exp_traj_len)
            
        iteration += 1
        
    # save the experience
    f = open(filename, 'wb')
    pickle.dump((total_exp_actions,total_exp_cell_idx, total_exp_traj_len, archive.cells[max_score_key].idx),f)
    f.close()

if __name__ == "__main__":
    for i in range(10):
        main(f'experience{i}.data')

