import gym
import numpy as np
#import gym_maze

#env = gym.make("maze-random-10x10-plus-v0")
#env = gym.make("MountainCar-v0")

env = gym.make("BipedalWalker-v2")
#env = gym.make("LunarLander-v2")
actions = range(env.action_space.n)

state = env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
