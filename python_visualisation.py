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


def main_app(map_override=None, background=False, end_at_min_epsilon=False, verbose=2):
    """
    :param map_override: a map name to use instead of showing map selection screen, defaults to None
    :param background: run the PyGame window in the background, or use dimensions from settings
    :param end_at_min_epsilon: close training window when min epsilon is reached
    :return: DeepQLearning object with learned parameters
    """
    params = [map_override, background, end_at_min_epsilon]
    if any(params) and not all(params):
        raise TypeError("Run in background without required parameters. Impossible to choose map/end.")

    # Create Window
    # if in background, use a surface to simulate screen
    screen = pygame.Surface((1, 1)) if background else pygame.display.set_mode((gs.WIDTH, gs.HEIGHT))

    socket = LocalTCPSocket(gs.PORT) if gs.USE_UNREAL_SOCKET else None

    # ================================================= MAP BUILDER ====================================================
    if map_override is None:
        selected_map = run_map_selection(screen)
    else:
        selected_map = Map.load_map(map_override)[0]

    world = World(socket, selected_map)
    world.replicate_map_spawn()

    # ================================================ GAME LOOP =======================================================
    run_ai_car_simulation(screen, world, end_at_min_epsilon=end_at_min_epsilon, verbose=verbose)

    # show reward graphs, error graphs and save models, IF TRAINING
    # return the DQN model
    controller = world.ai_car.controller
    if gs.Q_LEARNING_SETTINGS["TRAINING"] and not background:
        controller.q_learning.reward_graph()
        controller.q_learning.error_graph(color="red")
        controller.q_learning.save_model("combined")
    return controller.q_learning


if __name__ == '__main__':
    if input("Background: (y)") == "y":
        q = main_app(background=True, map_override="10.09;00.53", end_at_min_epsilon=True)
        q.reward_graph()
        q.error_graph(color="red")
        q.save_model("combined")
    else:
        main_app()
