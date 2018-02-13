import numpy as np
import gym
import gym_maze

class Estimator:
    def __init__(self, n_states, n_actions):
        np.random.seed(1)
        self.w = np.ones((n_states,n_actions))
        self.alpha = 0.01
        
    def predict(self,s):
        return np.matmul(s,self.w)
    
    def update(self,s,a, td_target):
        error = td_target-self.predict(s)[a]
        self.w += self.alpha*error
        
    
def epsilon_greedy_policy(estimator, epsilon, actions):
    """Creates an epsilon-greedy policy using the estimator"""
    
    def policy_fn(state):
        if np.random.rand()>epsilon:
            action = np.argmax(estimator.predict(state))
        else:
            action = np.random.choice(actions)
        return int(action)
    return policy_fn


env = gym.make("maze-random-5x5-v0")
#env = gym.make("CartPole-v0")
estimator = Estimator(env.observation_space.shape[0], env.action_space.n)

gamma = 1

n_episodes = 100


actions = range(env.action_space.n)

score = []    
for j in range(n_episodes):
    done = False
    state = env.reset()
    policy = epsilon_greedy_policy(estimator, \
                                   epsilon=100./(j+1), actions = actions )
    
    
    ### Generate sample episode
    while not done:
        env.render()
        action = policy(state)
        new_state, reward, done, _ =  env.step(action)
        new_action = policy(new_state)
                       
        #Calculate the td_target
        if done:
            td_target = reward
        else:
            new_q_val = estimator.predict(new_state)[new_action]
            td_target = reward + gamma * new_q_val
        
        estimator.update(state,action, td_target)    
        state, action = new_state, new_action
            
        if done:
            if reward == 1:
                print("I'm free!!")
                break
            if len(score) < 100:
                score.append(reward)
            else:
                score[j % 100] = reward
            print("\rEpisode {} / {}.\
             Avg score: {}".format(j+1, \
             n_episodes, np.mean(score)), end="")

env.close()