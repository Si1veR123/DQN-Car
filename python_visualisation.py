import numpy as np
import random
np.random.seed(1)
random.seed(1)


import pygame
pygame.init()

from App.world import World, Map
from SocketCommunication.tcp_socket import LocalTCPSocket

from App.AppScreens.ai_car_simulation import run_ai_car_simulation
from App.AppScreens.map_selection import run_map_selection

import global_settings as gs


def main_app(map_override=None, background=False):
    """
    :param map_override: a map name to use instead of showing map selection screen, defaults to None
    :param background: run the PyGame window in the background, or use dimensions from settings
    :return: DeepQLearning object(s) with learned parameters
    """
    # Create Window
    # if in background, make size (1, 1)
    dim = (1, 1) if background else (gs.WIDTH, gs.HEIGHT)
    screen = pygame.display.set_mode(dim)

    socket = LocalTCPSocket(gs.PORT) if gs.USE_UNREAL_SOCKET else None

    if map_override is None:
        selected_map = run_map_selection(screen)
    else:
        selected_map = Map.load_map(map_override)

    # ================================================= MAP BUILDER ====================================================

    world = World(socket, selected_map)
    world.replicate_map_spawn()

    # ================================================ GAME LOOP =======================================================
    run_ai_car_simulation(screen, world)

    # show reward graphs, error graphs and save models, IF TRAINING
    # return the relevant DQN models
    controller = world.ai_car.controller
    try:
        if gs.get_q_learning_settings("gas")["TRAINING"]:
            controller.gas_q_learning.reward_graph()
            controller.gas_q_learning.save_model("gas")
        if gs.get_q_learning_settings("steer")["TRAINING"]:
            controller.steer_q_learning.reward_graph()
            controller.steer_q_learning.save_model("steer")
        return controller.gas_q_learning, controller.steer_q_learning
    except AttributeError:
        # using combined, not individual
        if gs.get_q_learning_settings("combined")["TRAINING"]:
            controller.q_learning.reward_graph()
            controller.q_learning.error_graph(color="red")
            controller.q_learning.save_model("combined")
        return controller.q_learning


if __name__ == '__main__':
    main_app()
