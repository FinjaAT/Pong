from collections import deque
import numpy as np
import pong_rl_environment as envi
import deepQ_agent as ag
import random

history_len = 10
env = envi.pong_environment(render=True)
agent = ag.my_agent(8 * history_len, 3)

positiondata = env.give_start_state()
history = deque([positiondata] * history_len, maxlen=history_len)

while True:
    state = np.concatenate(history)
    actionrightpaddle = agent.get_action(state)

    next_positiondata, reward, rewardleft, done, running = env.one_step(actionrightpaddle)

    history.append(next_positiondata)
    next_state = np.concatenate(history)

    agent.memory.append((state, actionrightpaddle, reward, next_state, done))
    agent.train()

    if not running:
        break