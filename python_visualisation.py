import pygame
pygame.init()

from App.world import World
from SocketCommunication.tcp_socket import LocalTCPSocket

from App.AppScreens.app_helper_functions import draw_background
from App.AppScreens.map_builder import run_map_builder

from view_filters import AiVisOnly, NoAiVis

from global_settings import *

# ======================================================================================================================


# Create Window
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Create World Object
dimensions = (WIDTH//GRID_SIZE_PIXELS+1, HEIGHT//GRID_SIZE_PIXELS+1)
print(dimensions)
world = World(None, dimensions)


# ================================================= MAP BUILDER ========================================================

# if loading map, it is loaded when creating world object. if not, fill in map by running map builder
if not LOAD_MAP:
    run_map_builder(screen, world, FPS)

# ======================================================================================================================

world.socket = LocalTCPSocket(PORT) if USE_UNREAL_SOCKET else None

# ================================================ GAME LOOP ===========================================================
world.initiate_cars()

# view_filters.FILTERS.append(AiVisOnly())

clock = pygame.time.Clock()
run = True
while run:
    game_time = pygame.time.get_ticks() / 1000
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    draw_background(screen)

    # Update all placeables
    world.map.blit_grid(screen, game_time)
    world.ai_car.trace_all_rays(world, screen)

    world.car_collision()
    world.update_cars(VELOCITY_CONSTANT)

    world.blit_cars(screen)

    if world.ai_car.controller.ai_dead:
        world.initiate_cars()

    pygame.display.update()
# ======================================================================================================================

world.ai_car.controller.q_learning.reward_graph()
world.ai_car.controller.q_learning.save_model(r"E:\EPQ\PythonAI\saved_models\\")
