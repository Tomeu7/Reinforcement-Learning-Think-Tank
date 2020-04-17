In this Folder the Monte Carlo methods are implemented.

Monte Carlo methods estimate the value functions directly from the return of the episodes.

The following algorithms are implemented:

+ On-policy Monte Carlo Control.

Starting with an estimate of Q-values and a visit number for each state we update the Q-values in the following way:

0. <img src="https://render.githubusercontent.com/render/math?math=G=0, \ \ \ \ \tau \sim (S_0, A_0, R_1, ...)">
1. <img src="https://render.githubusercontent.com/render/math?math=From \ \ \ \ \ t=T-1 \ \ \ \ \ to \ \ \ \ \ 0 "> 
2. <img src="https://render.githubusercontent.com/render/math?math=G = \gamma G %2B R_{t%2B1}">
3. <img src="https://render.githubusercontent.com/render/math?math=N(s,a) = 1 \ \ \ \or \ \ \ \N(s,a) = N(s,a) %2B 1 \ \ \ \(depending \ \ \ \ on \ \ \ \ the \ \ \ \ strategy)">  
4. <img src="https://render.githubusercontent.com/render/math?math=Q(s,a) = Q(s,a) %2B 1/N(s,a)(G - Q(s,a))"> 

Results:

![MC](../docs/Monte_Carlo.png)

References:

1. Sutton and Barto. Introduction to Reinforcement Learning Second Edition.