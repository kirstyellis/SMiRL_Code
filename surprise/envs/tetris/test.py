from surprise.envs.tetris.tetris import TetrisEnv
import numpy as np
import matplotlib.pyplot as plt

env = TetrisEnv()
env.reset()
done = False

for i in range(100):
    if not done:
        obs, rew, done, info= env.step(np.random.randint(12), record=True)
    else:
        break
    print(info)
    env.render()