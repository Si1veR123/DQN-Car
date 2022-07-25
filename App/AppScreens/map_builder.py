"""
This screen allows the user to draw on the grid. It is converted to roads when the window is quit.
The roads are written to the Map object of the given World.
This can optionally be saved to file.
"""

from App.world import World
from App.placeable import SolidRoadPath, SolidBlock
from App.AppScreens.app_helper_functions import draw_background
from App.generate_roads import generate_roads
import pygame
import numpy as np
import global_settings as gs


def run_map_builder(screen, world: World, fps=None):
    clock = pygame.time.Clock()
    run = True
    while run:
        # total game time in seconds
        game_time = pygame.time.get_ticks() / 1000

        # limit fps if given
        if fps is not None:
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
        current_block = world.map.grid[mouse_grid[1]][mouse_grid[0]]

        # if grid position is already selected, move to next frame
        if type(current_block) == SolidRoadPath:
            world.map.blit_grid(screen, game_time)
            pygame.display.update()
            continue

        clicked = pygame.mouse.get_pressed()[0]

        if clicked:
            # lock in current grid position to be a road
            world.spawn_item(SolidRoadPath(gs.COL_PLACED_ROAD, gs.GRID_SIZE_PIXELS, game_time), mouse_grid)
        else:
            # color the current grid position with highlighted color
            world.spawn_item(SolidBlock(gs.COL_MOUSE_HIGHLIGHT, gs.GRID_SIZE_PIXELS), mouse_grid)

        world.map.blit_grid(screen, game_time)

        pygame.display.update()

        if not clicked:
            # replace block after drawing the highlight
            world.spawn_item(SolidBlock(gs.COL_BACKGROUND, gs.GRID_SIZE_PIXELS), mouse_grid)

    # matrix of grid, where 1 indicates a painted road and 0 is empty
    painted_roads = [[1 if type(x) == SolidRoadPath else 0 for x in y] for y in world.map.grid]

    dimensions = np.array(world.map.grid).shape[::-1]

    world.map.reset_grid(dimensions)

    # create roads from painted roads, and spawn them in
    roads = generate_roads(painted_roads, gs.GRID_SIZE_PIXELS)
    for row_num, row in enumerate(roads):
        for col_num, col in enumerate(row):
            if col is not None:
                world.spawn_item(col, (col_num, row_num))

    save = input("Save Map (y/n):") == "y"
    if save:
        # save map to path
        world.map.save_map()
