import numpy as np
import gym

env = gym.make("CartPole-v0") 

NUM_STEPS = 100 # maximum length of episode
# Alg settings:
N_ITER = 10 # number of iterations of CEM
BATCH_SIZE = 50 # number of samples per batch
ELITE_FRAC = 0.2 # fraction of samples used as elite set


# Initialize mean and standard deviation
dim_theta = (env.observation_space.shape[0]+1) * env.action_space.n
theta_mean = np.zeros(dim_theta)
theta_std = np.ones(dim_theta)


def make_policy(theta):
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    W = theta[0 : n_states * n_actions].reshape(n_states, n_actions)
    b = theta[n_states * n_actions : None].reshape(1, n_actions)
    
    def policy_fn(state):
        y = state.dot(W) + b
        action = y.argmax()
        return action
    return policy_fn

def run_episode(theta, num_steps=NUM_STEPS, render=False):
    total_reward = 0
    state = env.reset()
    policy = make_policy(theta)
    for t in range(num_steps):
        a = policy(state)
        state, reward, done, _ = env.step(a)
        total_reward += reward
        if render and t%3==0: env.render()
        if done: break
    return total_reward

# Now, for the algorithms
for it in range(N_ITER):
    '''
    Your code goes here:
        # Sample parameter vectors
        # Get elite parameters
        # Update theta_mean, theta_std
        # 100-episode average

    '''
    scores = [run_episode(theta_mean) for _ in range(100)]
    
    if np.mean(scores)>200:
        print("I won!")
        break
    print("\r Iteration {} / {}. Mean reward f: {}. Max reward: {}".format(it,N_ITER,np.mean(scores), np.max(scores)), end="")
    run_episode(theta_mean, NUM_STEPS, render=False)

# Run your final estimate
run_episode(theta_mean,render=True)

'''
YOUR TASK:
    - Implement CEM and NES. To be sure it works, you can use them first to
    compute the maximum of a function (as, we are treating RL as black-box 
    optimization problem).

    - Implement REINFORCE. You can build in from your implementation of Montecarlo. 
    Try it first in FrozenLake, then move to some more complicated environment. 

    - BONUS: Can you implement REINFORCE with continuous actions? For instance, for MountainCar-Continuous.



'''