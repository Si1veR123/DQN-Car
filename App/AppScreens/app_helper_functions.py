from global_settings import *
import pygame
import view_filters


def draw_background(screen):
    # Background is drawn by world's placeables, but this is beneath
    screen.fill(COL_BACKGROUND)

    if not view_filters.can_show_type("grid"):
        return

    # Draw Grid
    [pygame.draw.line(screen, COL_GRID, (x * GRID_SIZE_PIXELS, 0), (x * GRID_SIZE_PIXELS, HEIGHT)) for x in
        range((WIDTH // round(GRID_SIZE_PIXELS)) + 1)]
    [pygame.draw.line(screen, COL_GRID, (0, y * GRID_SIZE_PIXELS), (WIDTH, y * GRID_SIZE_PIXELS)) for y in
        range((HEIGHT // round(GRID_SIZE_PIXELS)) + 1)]