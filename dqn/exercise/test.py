import torch
import time
import matplotlib.pyplot as plt
from dqn_agent import Agent
import gym

env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

agent = Agent(state_size=8, action_size=4, seed=0)


# load the weights from file
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

# checkpoint = torch.load(pathname, map_location=map_location)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth',map_location=map_location))

for i in range(100):
    state = env.reset()
    for j in range(20000):
        action = agent.act(state)
        env.render(mode='rgb_array')
        state, reward, done, _ = env.step(action)
        time.sleep(0.01)
        if done:
            break 
env.close()