# https://gym.openai.com/envs/#box2d
# you need pybox2d installed:
#  https://github.com/pybox2d/pybox2d/blob/master/INSTALL.md

import numpy as np
import gym
from sklearn.neural_network import MLPRegressor

class Estimator:
    def __init__(self,model):
        self.model = model

    def predict(self,s,a):
        x = [np.concatenate([s,[a]])]
        return self.model.predict(x)

    def update(self,s,a,td_target):
        x = [np.concatenate([s,[a]])]
        y = [td_target]
        self.model.fit(x,y)
    
def epsilon_greedy_policy(estimator, epsilon, actions):
    """Creates an epsilon-greedy policy using the estimator"""
    
    def policy_fn(state):
        if np.random.rand()>epsilon:
            action = np.argmax([estimator.predict(state,a) for a in actions])
        else:
            action = np.random.choice(actions)
        return int(action)
    return policy_fn


env = gym.make("LunarLander-v2")

'''
Lunar lander has 4 actions and 8 states
'''

n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
actions = range(n_actions)

initial_train_data = [np.concatenate([env.reset(),[a]]) for a in actions]
model = MLPRegressor(hidden_layer_sizes=[128,256],   warm_start=True)
model.fit(initial_train_data,actions)

estimator = Estimator(model)

gamma = 0.99

n_episodes = 100

score = []    
for j in range(n_episodes):
    done = False
    state = env.reset()
    policy = epsilon_greedy_policy(estimator, \
                                   epsilon=10./(j+1), actions = actions )
    
    ### Generate sample episode
    while not done:
        #env.render()
        action = policy(state)
        new_state, reward, done, _ =  env.step(action)
        new_action = policy(new_state)
                       
        #Calculate the td_target
        if done:
            td_target = reward
        else:
            new_q_val = np.max([estimator.predict(state,a) for a in actions])
            td_target = reward + gamma * new_q_val
        
        estimator.update(state,action,td_target)
        state = new_state
            
        if done:
            if len(score) < 100:
                score.append(reward)
            else:
                score[j % 100] = reward
            print("\rEpisode {} / {}.\
             Avg score: {}".format(j+1, \
             n_episodes, np.mean(score)), end="")

env.close()