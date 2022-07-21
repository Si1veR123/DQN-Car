import numpy as np
import pygame


def collision_filter(screen, world, grid_size):
    for x in range(384):
        x = 5 * x
        for y in range(216):
            y = 5 * y
            grid_box = np.array((x // world.grid_size, y // world.grid_size))
            placeable = world.grid[grid_box[1]][grid_box[0]]
            colour = (255, 0, 0) if placeable.overlap((np.array((x, y))) % world.grid_size, grid_size) else (
            0, 0, 255)
            pygame.draw.rect(screen, colour, pygame.rect.Rect((x, y), (3, 3)))
