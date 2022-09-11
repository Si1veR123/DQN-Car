"""
Runs the main window, which simulates the ai car on the map, specified in the given World
"""

from App.AppScreens.app_helper_functions import draw_background
from App.world import World
import global_settings as gs
import numpy as np
import pygame
import view_filters


def collision_filter(screen, world):
    for x in range(384):
        x = 5 * x
        for y in range(216):
            y = 5 * y
            grid_box = np.array((x // gs.GRID_SIZE_PIXELS, y // gs.GRID_SIZE_PIXELS))
            placeable = world.map.grid[grid_box[1]][grid_box[0]]
            colour = (255, 0, 0) if placeable.overlap((np.array((x, y))) % gs.GRID_SIZE_PIXELS) else (0, 0, 255)
            pygame.draw.rect(screen, colour, pygame.rect.Rect((x, y), (3, 3)))


def run_ai_car_simulation(screen, world: World):
    world.initiate_cars()

    default_font = pygame.font.Font(pygame.font.get_default_font(), 24)

    fps = gs.FPS

    episode_frames = 0

    # dont show collision by default
    view_filters.FILTERS.append(view_filters.NoCollision())

    if gs.TRAINING:
        view_filters.FILTERS.append(view_filters.FastTrainingView())

    clock = pygame.time.Clock()
    run = True
    while run:
        # game time in seconds
        game_time = pygame.time.get_ticks() / 1000
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        draw_background(screen)

        # Update and draw all placeables
        world.map.blit_grid(screen, game_time)

        if view_filters.can_show_type("collision"):
            collision_filter(screen, world)

        # Trace all rays for ai car, which updates the Q learning state
        world.ai_car.trace_all_rays(world, screen)

        # Move all cars
        world.update_cars()

        # Check if crashed
        world.car_collision()

        # AI controller has things to do after moving and checking collision
        world.ai_car.controller.end_of_frame()

        # Draw all cars to screen
        world.blit_cars(screen)

        world.blit_ai_action(screen)

        # If AI car has crashed, reset all cars
        if world.ai_car.controller.ai_dead or (episode_frames >= gs.MAX_EPISODE_FRAMES and gs.TRAINING):

            world.initiate_cars()
            episode_frames = 0

        fps_text = default_font.render(str(int(clock.get_fps())), False, (255, 255, 255))
        screen.blit(fps_text, (0, 0))

        speed_text = default_font.render(str(round(world.ai_car.controller.velocity, 2)), False, (255, 255, 255))
        screen.blit(speed_text, (200, 0))

        episode_frames += 1
        pygame.display.update()

    view_filters.FILTERS.clear()
