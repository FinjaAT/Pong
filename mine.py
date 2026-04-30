import pong_rl_environment as envi
import deepQ_agent as ag
import random

env = envi.pong_environment(render=True)
agent = ag.my_agent(8,3)

positiondata = env.give_start_state()

while True:
    
    actionrightpaddle = agent.get_action(positiondata)

    state = positiondata

    positiondata, reward,rewardleft,done,running = env.one_step(actionrightpaddle)

    agent.memory.append((state, actionrightpaddle, reward, positiondata, done))

    agent.train()

    if not running:
        break