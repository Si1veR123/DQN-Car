import pygame
pygame.init()

from App.world import World
from SocketCommunication.tcp_socket import LocalTCPSocket

from App.AppScreens.ai_car_simulation import run_ai_car_simulation
from App.AppScreens.map_selection import run_map_selection

import global_settings as gs


# Create Window
screen = pygame.display.set_mode((gs.WIDTH, gs.HEIGHT))

socket = LocalTCPSocket(gs.PORT) if gs.USE_UNREAL_SOCKET else None

selected_map = run_map_selection(screen)

# ================================================= MAP BUILDER ========================================================

world = World(socket, selected_map)

world.replicate_map_spawn()

# ================================================ GAME LOOP ===========================================================
run_ai_car_simulation(screen, world)

world.ai_car.controller.q_learning.reward_graph()
world.ai_car.controller.q_learning.save_model()
