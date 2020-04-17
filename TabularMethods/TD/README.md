In this Folder the Temporal Difference methods are implemented.

All methods are implemented via classes temporal_difference_methods.py. The agent class is a class that chooses over the methods.

Temporal Difference methods change their estimation of the value functions in the following way (from 1):

- <img src="https://render.githubusercontent.com/render/math?math=V(S_t) \longleftarrow V(S_t) %2B \alpha[R_{t%2B1} %2B \gamma V(S_{t%2B1})]">


The estimation of state S_t is changed according to the obtained reward and the estimation of the following state.

Temporal difference methods have on-policy (methods that learn the same policy which they use to act) and off-policy methods (methods that learn a different policy than the one used to act).

The following algorithms are implemented:

+ Sarsa (on-policy)

Sarsa (State Action Reward State Action) updates the action value function in the following way:

1. <img src="https://render.githubusercontent.com/render/math?math=a \longleftarrow \pi(s)">

2. <img src="https://render.githubusercontent.com/render/math?math=r,s^' \longleftarrow env.step(a)">

3. <img src="https://render.githubusercontent.com/render/math?math=a^' \longleftarrow \pi(s^')">

4. <img src="https://render.githubusercontent.com/render/math?math=Q(s,a) \longleftarrow Q(S,A) %2B \alpha [r %2B \gamma Q(s^',a^') - Q(s,a)]">

We are updating the estimation of the Q-values following the policy as a' comes from the policy.

+ Q-learning

Q-learning works a little bit different from SARSA because we learn not from the exploratory policy but from the policy that maximizes the Q-values.

1. <img src="https://render.githubusercontent.com/render/math?math=a \longleftarrow \pi(s)">

2. <img src="https://render.githubusercontent.com/render/math?math=r,s^' \longleftarrow env.step(a)">

1. <img src="https://render.githubusercontent.com/render/math?math=Q(s,a) \longleftarrow Q(S,A) %2B \alpha [r %2B \gamma max_a Q(s^',a) - Q(s,a)]">


+ Double Q-learning

The max operator of Q-learning tends to overestimate the Q-values. For that Double Q-learning was invented.

The idea is to consider two Q-values Q_1(s,a) and Q_2(s,a) and act according to the sum of both.

Then at the time of update we choose one of the matrices according with 0.5 probability each and updating according to:

1. <img src="https://render.githubusercontent.com/render/math?math=Q_1(s,a) \longleftarrow Q_1(S,A) %2B \alpha [r %2B \gamma Q_2(s^', argmax_a Q_1(s^',a)) - Q_2(s,a)]">

+ Expected Sarsa

Expected Sarsa considers the learning as in Q-learning but using a expected value instead of a maximum. It can be both on-policy and off-policy.

1. <img src="https://render.githubusercontent.com/render/math?math=Q(s,a) \longleftarrow Q(S,A) %2B \alpha [r %2B \gamma \E_\pi [Q(s^', a)] - Q(s,a)]">

Results:

![TD](https://github.com/Tomeu7/Reinforcement-Learning-Think-Tank/blob/master/docs/Monte_Carlo.png)

References:

1. Sutton and Barto. Introduction to Reinforcement Learning Second Edition.