"""
Miscellaneous functions that are used in multiple app screens
"""

import global_settings as gs
import pygame
import view_filters


def draw_background(screen, grid=True):
    # Background is drawn by world's placeables, but this is beneath
    screen.fill(gs.COL_BACKGROUND)

    # Only draw grid if the filter allows it
    if not view_filters.can_show_type("grid") or not grid:
        return

    # Draw Grid
    [pygame.draw.line(screen, gs.COL_GRID, (x * gs.GRID_SIZE_PIXELS, 0), (x * gs.GRID_SIZE_PIXELS, gs.HEIGHT)) for x in
        range((gs.WIDTH // round(gs.GRID_SIZE_PIXELS)) + 1)]
    [pygame.draw.line(screen, gs.COL_GRID, (0, y * gs.GRID_SIZE_PIXELS), (gs.WIDTH, y * gs.GRID_SIZE_PIXELS)) for y in
        range((gs.HEIGHT // round(gs.GRID_SIZE_PIXELS)) + 1)]