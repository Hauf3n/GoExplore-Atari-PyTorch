import numpy as np
import cv2
import gym
import random
import pickle
import imageio

from Atari_Wrapper import Atari_Wrapper

#env
env_name = "MontezumaRevengeDeterministic-v4"
# render demonstration
render = True

# build tree for navigation - used for correct concat of actions trajs

class Node:
# simple node class

    def __init__(self, idx):
        self.childs = {}
        self.parents = []
        self.idx = idx
        
    def add_child(self, idx, action_traj):
        if idx in self.childs:
            return
            if len(action_traj) < len(self.childs[idx]):
                self.childs[idx] = action_traj
        else:
            self.childs[idx] = action_traj
            
    def add_parent(self, idx):
        if idx not in self.parents:
            self.parents.append(idx)
            
class Tree:
# simple tree class

    def __init__(self):
        self.nodes = {}
        
    def add_node(self, idx):
        if idx not in self.nodes:
            self.nodes[idx] = Node(idx)
            
    def add_edge(self, parent_idx, child_idx, action_traj):
        self.nodes[parent_idx].add_child(child_idx, action_traj)
        self.nodes[child_idx].add_parent(parent_idx)



def main(filename='experience.data'):

    # load experience
    f = open(f'{filename}.data','rb')
    total_exp_actions, total_exp_cell_idx, total_exp_traj_len, max_idx = pickle.load(f)
    f.close()
    
    tree = Tree()
    
    # seen indexes 
    seen = [0]
    # idx cell traj len
    idx_traj_len = {}
    idx_traj_len[0] = 0

    # build tree
    for actions, idxs, traj_len in zip(total_exp_actions,total_exp_cell_idx,total_exp_traj_len):
        
        start_idx = idxs[0]
        seen_idxs = [start_idx]
        for i, current_idx in enumerate(idxs):
            
            if current_idx in seen_idxs:
                continue
                
            # cell with current_idx first time in the data
            
            seen_idxs.append(current_idx)
            
            # add navigation path
            
            action_traj = actions[0:i]
            
            tree.add_node(start_idx)
            tree.add_node(current_idx)
            
            if current_idx not in seen:
                tree.add_edge(start_idx, current_idx, action_traj)
                seen.append(current_idx)
                idx_traj_len[current_idx] = traj_len[i]  


    # generate the demonstration from start state to desired state
    # build demonstration backward from desired state
    
    demonstration = []
    start_idx = 0
    current_idx = max_idx

    seen = []
    while current_idx != start_idx:
        cur_node = tree.nodes[current_idx]
        parent_idxs = cur_node.parents
        
        # choose parent with shortest traj len
        chosen_parent_idx = None
        chosen_parent_traj_len = np.inf
        for p_idx in parent_idxs:
            if chosen_parent_traj_len > idx_traj_len[p_idx]:
                chosen_parent_idx = p_idx
                chosen_parent_traj_len = idx_traj_len[p_idx]
        
        # obtain action traj from parent to child
        actions = tree.nodes[chosen_parent_idx].childs[current_idx]
        current_idx = chosen_parent_idx
        
        demonstration =  actions + demonstration
    
    print("demonstration length: ",len(demonstration))
    
    # get and save the restores of the demonstration for backward algorithm 
    env = gym.make(env_name).unwrapped
    env.seed(0)
    env = Atari_Wrapper(env, 2)
    frame = env.reset()
    
    demo_restores = [env.env.clone_full_state()]
    demo_frames = [frame]
    
    for action in demonstration:
    
        frame, _, done, _ = env.step(action)
        demo_frames.append(frame)
        demo_restores.append(env.env.clone_full_state())
        
        if done:
            break
            
    env.close()
    
    # save the demonstration data
    f = open(f'demonstration_{filename}.data', 'wb')
    pickle.dump((demonstration, demo_restores, demo_frames),f)
    f.close()
    
    # render?
    if not render:
        return
    # create video
    w = imageio.get_writer(f'{filename}.mp4', format='FFMPEG', mode='I', fps=10, quality=9)
    for frame in demo_frames:
        w.append_data(frame.astype(np.uint8))
    w.close()

if __name__ == "__main__":
    for i in range(5):
        main(f'experience{i}')