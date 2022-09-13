import os
# move to parent directory so all file references work
os.chdir("../")

from hyperparameter_tuner import Results
from MachineLearning.q_learning import DeepQLearning
from App.world import Map, World
from App.AppScreens.ai_car_simulation import run_ai_car_simulation
import matplotlib.pyplot as plt
import numpy as np
import pygame
import global_settings as gs

gs.Q_LEARNING_SETTINGS["TRAINING"] = False

results = Results.load("lr and target and discount net finer 2")

default_map = Map.load_map("10.09;00.53")[0]

lrs = []
ncs = []
drs = []
rewards = []

# sort by dr to prevent plotting scatter over each other
for r in sorted(list(results), key=lambda x: x.discount_rate):
    r: DeepQLearning

    world = World(None, default_map)
    world.ai_car.controller.q_learning = r
    screen = pygame.display.set_mode((gs.WIDTH, gs.HEIGHT))

    print(r.mean_rewards, r.learning_rate, r.network_copy_steps)
    run_ai_car_simulation(screen, world, 3, False)

    lrs.append(np.log10(r.learning_rate))
    ncs.append(r.network_copy_steps)
    drs.append(300 if r.discount_rate==0.95 else 100)
    rewards.append(r.mean_rewards)

plt.scatter(lrs, ncs, s=drs, c=rewards, cmap="autumn")
plt.show()
