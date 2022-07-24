from SocketCommunication.tcp_socket import LocalTCPSocket
from World.placeable import Placeable, SolidBlock
from World.car import AICar
from World.road import StraightRoad
import numpy as np
import pygame
import typing
import math
import random
import view_filters
import pickle
from global_settings import *


class Map:
    def __init__(self, grid_dimensions):
        self.grid = []
        self.reset_grid(grid_dimensions)

    def reset_grid(self, grid_dimensions):
        self.grid = [[SolidBlock(COL_BACKGROUND, GRID_SIZE_PIXELS) for _ in range(grid_dimensions[0])] for _ in range(grid_dimensions[1])]

    def blit_grid(self, screen: pygame.surface.Surface, game_time):
        for row_num, row in enumerate(self.grid):
            for col_num, item in enumerate(row):

                item.tick(game_time)

                if item.image is not None:
                    # offset by 1 to allow for 1px grid lines
                    if view_filters.can_show_type(item.name_id):
                        screen.blit(item.image, (math.floor(col_num * GRID_SIZE_PIXELS) + 1, math.floor(row_num * GRID_SIZE_PIXELS) + 1))


class World:
    def __init__(self,
                 socket: typing.Union[LocalTCPSocket, None],
                 grid_dimensions: tuple,
                 ):
        self.socket = socket

        self.ai_car = AICar("ai_car")
        self.npc_cars = []

        self.map = Map(grid_dimensions)

        self.start_grid_location = (0, 0)
        self.start_rotation = 0

    @property
    def all_cars(self):
        return self.npc_cars + [self.ai_car]

    def spawn_item(self, item: Placeable, grid_pos):
        self.map.grid[grid_pos[1]][grid_pos[0]] = item

        if item.replicate_spawn:
            try:
                self.socket.send_ue_data("spawn", item.name_id, {"gridx": grid_pos[0], "gridy": grid_pos[1]})
            except AttributeError:
                # socket is None, not in use
                pass

    def initiate_cars(self):
        spawn_pos = []
        for row_num, row in enumerate(self.map.grid):
            for col_num, col in enumerate(row):
                if type(col) == StraightRoad:
                    spawn_pos.append(((col_num, row_num), col.direction))

        random.shuffle(spawn_pos)

        ai_loc = spawn_pos[0][0]
        ai_rot = spawn_pos[0][1]
        self.ai_car.controller.location = (np.array(ai_loc) * GRID_SIZE_PIXELS) + np.array((GRID_SIZE_PIXELS / 2, GRID_SIZE_PIXELS / 2))
        self.ai_car.controller.rotation = random.choice([ai_rot * 90, (ai_rot*90)-180])

        for car_num, car in enumerate(self.all_cars):
            car_num = car_num + 1

            try:
                loc, rot = spawn_pos[car_num][0], spawn_pos[car_num][1]
                car.controller.location = (np.array(loc) * GRID_SIZE_PIXELS) + np.array((GRID_SIZE_PIXELS / 2, GRID_SIZE_PIXELS / 2))
                car.controller.rotation = rot * 90
            except IndexError:
                print("Couldnt spawn: ", car.object_name)

            car.reset_state()

    def update_cars(self, velocity_constant):
        for car in self.all_cars:
            car.controller.update_transform(velocity_constant)

    def blit_cars(self, screen):
        for car in self.all_cars:
            car.tick()
            if view_filters.can_show_type("car"):
                car.draw_car(screen)

    def car_collision(self):
        for car in self.all_cars:
            overlaps = []
            corners = car.get_corners()

            for corner in corners:
                grid_pos = (corner // GRID_SIZE_PIXELS).astype(int).tolist()

                try:
                    placeable = self.map.grid[grid_pos[1]][grid_pos[0]]
                    overlaps.append(placeable.overlap(corner % GRID_SIZE_PIXELS, GRID_SIZE_PIXELS))
                except IndexError:
                    pass

            if not all(overlaps):
                try:
                    self.npc_cars.remove(car)
                except ValueError:
                    # ai car dead
                    self.ai_car.controller.ai_dead = True
