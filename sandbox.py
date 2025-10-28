from custom_env import DrivingEnv, Sim, SAC
from time import perf_counter
env = DrivingEnv("./tracks/miami_optimized.png",
                         wheelbase=1.25, max_speed=10,
                         lad=25, pp_max_speed=9,
                         dt=1/60)

model = SAC.load("wandb/latest-run/files/model.zip")

def iterate():
    start = perf_counter()
    obs, info = env.reset()
    action, states = model.predict(obs, deterministic=True)
    obs, reward, terminated, *_ = env.step(action)
    fps = 1/(perf_counter() - start)
    return fps, obs, reward, terminated, info

a = []
for i in range(100):
    a.append(iterate()[0])

import numpy as np
print(np.mean(a), np.median(a), np.std(a), np.min(a), np.max(a))