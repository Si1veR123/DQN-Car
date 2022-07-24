from App.world import World
from App.placeable import SolidRoadPath, SolidBlock
from global_settings import *
from App.AppScreens.app_helper_functions import draw_background
from App.generate_roads import generate_roads
import pygame
import numpy as np
import datetime


def run_map_builder(screen, world: World, fps=None):
    clock = pygame.time.Clock()
    run = True
    while run:
        game_time = pygame.time.get_ticks() / 1000
        if fps is not None:
            clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        draw_background(screen)

        mouse_grid = (
            round((pygame.mouse.get_pos()[0]) // GRID_SIZE_PIXELS),
            round((pygame.mouse.get_pos()[1]) // GRID_SIZE_PIXELS)
        )

        current_block = world.map.grid[mouse_grid[1]][mouse_grid[0]]
        if type(current_block) == SolidRoadPath:
            world.map.blit_grid(screen, game_time)
            pygame.display.update()
            continue

        clicked = pygame.mouse.get_pressed()[0]

        if clicked:
            world.spawn_item(SolidRoadPath(COL_PLACED_ROAD, GRID_SIZE_PIXELS, game_time), mouse_grid)
        else:
            world.spawn_item(SolidBlock(COL_MOUSE_HIGHLIGHT, GRID_SIZE_PIXELS), mouse_grid)

        world.map.blit_grid(screen, game_time)

        pygame.display.update()

        if not clicked:
            # replace block after drawing the highlight
            world.spawn_item(SolidBlock(COL_BACKGROUND, GRID_SIZE_PIXELS), mouse_grid)

    # matrix of grid, where 1 indicates a painted road and 0 is empty
    painted_roads = [[1 if type(x) == SolidRoadPath else 0 for x in y] for y in world.map.grid]

    dimensions = np.array(world.map.grid).shape[::-1]
    print(dimensions)
    world.map.reset_grid(dimensions)

    # create roads from painted roads
    roads = generate_roads(painted_roads, GRID_SIZE_PIXELS)
    for row_num, row in enumerate(roads):
        for col_num, col in enumerate(row):
            if col is not None:
                world.spawn_item(col, (col_num, row_num))

    save = input("Save Map (y/n):") == "y"
    if save:
        world.map.save_map(r"E:\EPQ\PythonAI\saved_maps\\" + datetime.datetime.now().strftime("%d.%m;%H.%M"))
