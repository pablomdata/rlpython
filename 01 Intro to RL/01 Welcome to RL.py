
# # Frozen Lake - Introduction to OpenAI Gym

# If you haven't please install the library OpenAI gym (pip install gym)
from tqdm import tqdm
import gym

#create a single game instance
env = gym.make("FrozenLake-v0")

#start new game
env.reset()

# Display the game state
env.render()

# Here, **S** is the initial state, and your aim is to reach **G**, without falling into the holes, **H**. The squares marked with **F** are frozen, which means you can step on them.
# 
# 
# **Note:** The environment is non-deterministic, you can slip in the ice and end up in a different state.

# ## How to use the environment?
# 
# - **reset()** returns the initial state / first observation.
# - **render()** returns the current environment state. 
# - **step(a)** returns what happens after action a:
#     - *new observation*: the new state.
#     - *reward*: the reward corresponding to that action in that state.
#     - *is done*: binary flag, True if the game is over. 
#     - *info*: Some auxiliary stuff, which we can ignore now.
#     
# 
print("The initial state: ", env.reset())
print(" and it looks like: ")
env.render()


print("Now let's take an action: ")
new_state, reward, done, _ = env.step(1)
env.render()


idx_to_action = {
    0:"<", #left
    1:"v", #down
    2:">", #right
    3:"^" #up
}


# ## Defining a policy
# 
# - The environment has a 4x4 grid of states. 
# - For each state there are 4 possible actions. 
# 
# A **policy** is a function from states to actions. It tells us what we should do on each state. In this case, any array of size 16x4 is a (deterministic) policy.
# 
# We can implement policies as dictionaries. 


import numpy as np
n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize random_policy:
def init_random_policy():
    random_policy  = {}
    for state in range(n_states):
        random_policy[state] = np.random.choice(n_actions)
    return random_policy
    


random_policy = init_random_policy()


def evaluate(env, policy, max_episodes=100): 
    tot_reward = 0
    for ep in range(max_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        
        # Reward per episode
        while not done:
            action = policy[state]
            new_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = new_state
            if done:
                tot_reward += ep_reward
    return tot_reward/(ep+1)
    

best_policy = None
best_score = -float('inf')

# Random search
for i in tqdm(range(10000)):
    policy = init_random_policy()
    score = evaluate(env,policy,100)
    if score > best_score:
        best_score = score
        best_policy = policy
    if i%1000 == 0:
        print("Current best score:", best_score)
print("Best policy:")
print(best_policy)


# Let's see the policy in action
def play(env,policy, render=False):
    s = env.reset()
    d = False
    while not d:
        a = policy[s]
        print("*"*10)
        print("State: ",s)
        print("Action: ",idx_to_action[a])
        s, r, d, _ = env.step(a)
        if render:
            env.render()
        if d:
            print(r)


def print_policy(policy):
    lake = "SFFFFHFHFFFHHFFG"
    arrows = [idx_to_action[policy[i]] 
               if lake[i] in 'SF' else '*' for i in range(n_states)]
    for i in range(0,16,4):
        print(''.join(arrows[i:i+4]))

print_policy(best_policy)

play(env,best_policy)


# # Your turn:
# 
# Let's try something more interesting than random search. 
# 
# - Implement Cross Entropy Method.
# 
# - Extra Credit: Evolution Strategies (https://blog.openai.com/evolution-strategies/)
