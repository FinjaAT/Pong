import time
import numpy as np
import tensorflow as tf
from deepQ_agent import my_agent

agent = my_agent(80, 3, loadmodel=False, trainme=False)
state = np.zeros((80,), dtype=np.float32)
for _ in range(5):
    agent.predict_fn(tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)).numpy()
start = time.time()
for _ in range(20):
    agent.predict_fn(tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)).numpy()
end = time.time()
print('avg predict_fn ms', (end - start) / 20 * 1000)
print('done')
