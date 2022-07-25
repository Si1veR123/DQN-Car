import pygame
pygame.init()

from App.world import World
from SocketCommunication.tcp_socket import LocalTCPSocket

from App.AppScreens.map_builder import run_map_builder
from App.AppScreens.ai_car_simulation import run_ai_car_simulation

import global_settings as gs


# Create Window
screen = pygame.display.set_mode((gs.WIDTH, gs.HEIGHT))

# Create World Object
dimensions = (gs.WIDTH//gs.GRID_SIZE_PIXELS+1, gs.HEIGHT//gs.GRID_SIZE_PIXELS+1)
world = World(None, dimensions)


# ================================================= MAP BUILDER ========================================================

# if loading map, it is loaded when creating world object. if not, fill in map by running map builder
if not gs.LOAD_MAP:
    run_map_builder(screen, world, gs.FPS)


world.socket = LocalTCPSocket(gs.PORT) if gs.USE_UNREAL_SOCKET else None

# ================================================ GAME LOOP ===========================================================
run_ai_car_simulation(screen, world, gs.FPS)

world.ai_car.controller.q_learning.reward_graph()
world.ai_car.controller.q_learning.save_model()
