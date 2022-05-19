# GoExplore-Atari-PyTorch
 Implementation of [First return, then explore](https://www.nature.com/articles/s41586-020-03157-9) (Go-Explore) by Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O. Stanley, Jeff Clune. The result is a neural network policy that reaches a score of 2500 on the Atari environment MontezumaRevenge.<br><br>
# Content 
1. Exploration Phase with demonstration generation
2. Robustification Phase (PPO + SIL + Backward algorithm)
# Results
1. Exploration Phase demos with 2500 score <br>
![1](https://github.com/Hauf3n/GoExplore-Atari-PyTorch/blob/main/.media/demo0.gif)
![2](https://github.com/Hauf3n/GoExplore-Atari-PyTorch/blob/main/.media/demo1.gif)
![4](https://github.com/Hauf3n/GoExplore-Atari-PyTorch/blob/main/.media/demo3.gif)
![5](https://github.com/Hauf3n/GoExplore-Atari-PyTorch/blob/main/.media/demo4.gif)<br><br>
2. Robustification Phase with 2500 score <br>
![9](https://github.com/Hauf3n/GoExplore-Atari-PyTorch/blob/main/.media/2500_5_runs.gif)<br><br>
3. Robustification Phase backward algorithm progress<br>
![10](https://github.com/Hauf3n/GoExplore-Atari-PyTorch/blob/main/.media/starting_pos_2500.jpg)
<br><br>
4. Robustification Phase path<br>
![11](https://github.com/Hauf3n/GoExplore-Atari-PyTorch/blob/main/.media/montezuma_backward_demo.jpg)
<br><br>
