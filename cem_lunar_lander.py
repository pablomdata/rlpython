import numpy as np
import gym

# Task settings:
env = gym.make("LunarLander-v2") # Change as needed

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
    # Sample parameter vectors
    thetas = np.random.normal(theta_mean, theta_std, (BATCH_SIZE,dim_theta))
    rewards = [run_episode(theta) for theta in thetas]
    # Get elite parameters
    n_elite = int(BATCH_SIZE * ELITE_FRAC)
    elite_inds = np.argsort(rewards)[BATCH_SIZE - n_elite:BATCH_SIZE]
    elite_thetas = [thetas[i] for i in elite_inds]
    # Update theta_mean, theta_std
    theta_mean = np.mean(elite_thetas,axis=0)
    theta_std = np.std(elite_thetas,axis=0)

    # 100-episode average
    scores = [run_episode(theta_mean) for _ in range(100)]
    if np.mean(scores)>200:
        print("I won!")
        break
    print("\r Iteration {} / {}. Mean reward f: {}. Max reward: {}".format(it,N_ITER,np.mean(scores), np.max(scores)), end="")
    run_episode(theta_mean, NUM_STEPS, render=False)
run_episode(theta_mean,render=True)