"""
This screen allows the user to draw on the grid. It is converted to roads when the window is quit.
The roads are written to the Map object of the given World.
This can optionally be saved to file.
"""

from App.world import Map
from App.placeable import SolidRoadPath, SolidBlock
from App.AppScreens.app_helper_functions import draw_background
from App.generate_roads import generate_roads
from typing import Tuple
import pygame
import numpy as np
import global_settings as gs


def run_map_builder(screen) -> Tuple[Map, bool]:
    # returns Map object, and boolean 'saved'
    grid_dimensions = (gs.WIDTH // gs.GRID_SIZE_PIXELS + 1, gs.HEIGHT // gs.GRID_SIZE_PIXELS + 1)

    fps = gs.FPS

    map_obj = Map(grid_dimensions)

    clock = pygame.time.Clock()
    run = True
    while run:
        # total game time in seconds
        game_time = pygame.time.get_ticks() / 1000

        # limit fps
        clock.tick(fps)

        # move on from drawing if quit window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        draw_background(screen)

        # calculate the current grid position that the mouse is in
        mouse_grid = (
            round((pygame.mouse.get_pos()[0]) // gs.GRID_SIZE_PIXELS),
            round((pygame.mouse.get_pos()[1]) // gs.GRID_SIZE_PIXELS)
        )

        # get the current placeable object beneath the cursor
        current_block = map_obj.grid[mouse_grid[1]][mouse_grid[0]]

        l_clicked, r_clicked = pygame.mouse.get_pressed()[0], pygame.mouse.get_pressed()[2]

        # if grid position is already selected, move to next frame
        if type(current_block) == SolidRoadPath and not r_clicked:
            map_obj.blit_grid(screen, game_time)
            pygame.display.update()
            continue

        if l_clicked:
            # lock in current grid position to be a road
            map_obj.spawn_item_local(SolidRoadPath(gs.COL_PLACED_ROAD, gs.GRID_SIZE_PIXELS, game_time), mouse_grid)
        else:
            # color the current grid position with highlighted color
            map_obj.spawn_item_local(SolidBlock(gs.COL_MOUSE_HIGHLIGHT, gs.GRID_SIZE_PIXELS), mouse_grid)

        map_obj.blit_grid(screen, game_time)

        pygame.display.update()

        if not l_clicked:
            # replace block after drawing the highlight
            map_obj.spawn_item_local(SolidBlock(gs.COL_BACKGROUND, gs.GRID_SIZE_PIXELS), mouse_grid)

    # matrix of grid, where 1 indicates a painted road and 0 is empty
    painted_roads = [[1 if type(x) == SolidRoadPath else 0 for x in y] for y in map_obj.grid]

    dimensions = np.array(map_obj.grid).shape[::-1]

    map_obj.reset_grid(dimensions)

    # create roads from painted roads, and spawn them in
    roads = generate_roads(painted_roads, gs.GRID_SIZE_PIXELS)

    for row_num, row in enumerate(roads):
        for col_num, col in enumerate(row):
            if col not in (None, np.nan):
                map_obj.spawn_item_local(col, (col_num, row_num))

    draw_background(screen, grid=False)
    map_obj.blit_grid(screen, pygame.time.get_ticks() / 1000)
    pygame.display.update()

    save = input("Save Map (y/n):") == "y"
    if save:
        # save map to path
        map_obj.save_map(screen)

    return map_obj, save
