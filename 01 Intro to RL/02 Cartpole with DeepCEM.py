import seaborn as sns
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm
import numpy as np

class DeepCEM():
    def __init__(self, initial_state, n_actions, clf=MLPClassifier(hidden_layer_sizes=(20,20),
                              activation='tanh',
                              warm_start=True, #keep progress between .fit(...) calls
                              max_iter=1 #make only 1 iteration on each .fit(...)
                             )):
        self.clf = clf
        
        #initialize clf
        clf.fit([initial_state]*n_actions,range(n_actions))
        
    def policy(self, state):
        probs = self.clf.predict_proba([state])[0]
        action = np.random.choice(n_actions, p=probs)
        return action

    def train(self,n_iter):
        n_episodes = 100 
        percentile = 75
        
        for i in tqdm(range(n_iter)):
            #generate new episodes
            episodes = [play_episode(self) for _ in range(n_episodes)]

            batch_states,batch_actions,batch_rewards = map(np.array,zip(*episodes))

            reward_threshold = np.percentile(batch_rewards,70)
            idxs = [i for i in range(len(batch_rewards)) if batch_rewards[i]>reward_threshold]

            elite_states, elite_actions = np.concatenate(batch_states[idxs],axis=0), np.concatenate(batch_actions[idxs],axis=0)
            self.clf.fit(elite_states, elite_actions)


            # CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials. 
            # Reward of +1 for every time step the pole does not fall

            if np.mean(batch_rewards)> 195:
                print("You Win! You can stop training now with the 'Stop' button.")


# In[5]:

def play_episode(agent, max_iter=1000,render=False):
    states,actions = [],[]
    total_reward = 0
    state = env.reset()
    for _ in range(max_iter):
        # choose the action according to the policy
        action = agent.policy(state)
        new_state,reward,done,info = env.step(action)
        if render:
            env.render()
        # record sessions
        states.append(state)
        actions.append(action)
        total_reward+=reward
        state = new_state
        if done: 
            break
    return states,actions,total_reward

env = gym.make("CartPole-v0").env  #if you see "<classname> has no attribute .env", remove .env or update gym
env.reset()
n_actions = env.action_space.n

agent = DeepCEM(initial_state=env.reset(), n_actions=2)
agent.train(n_iter=100)
states, actions, rewards= play_episode(agent, render=True)

