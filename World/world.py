from SocketCommunication.tcp_socket import LocalTCPSocket
from World.placeable import Placeable, SolidBlock
from World.car import AICar
from World.road import StraightRoad
import numpy as np
import pygame
import typing
import math
import random


class World:
    def __init__(self, socket: typing.Union[LocalTCPSocket, None], grid_size, grid_dimensions, background_colour):
        self.socket = socket

        self.ai_car = AICar("ai_car")
        self.npc_cars = []

        self.grid = []
        self.grid_size = grid_size

        self.reset_grid(background_colour, grid_dimensions)

        self.start_grid_location = (0, 0)
        self.start_rotation = 0

    @property
    def all_cars(self):
        return self.npc_cars + [self.ai_car]

    def reset_grid(self, background_colour, grid_dimensions):
        self.grid = [[SolidBlock(background_colour, self.grid_size) for _ in range(grid_dimensions[0])] for _ in range(grid_dimensions[1])]

    def spawn_item(self, item: Placeable, grid_pos):
        self.grid[grid_pos[1]][grid_pos[0]] = item

        if item.replicate_spawn:
            try:
                self.socket.send_ue_data("spawn", item.name_id, {"gridx": grid_pos[0], "gridy": grid_pos[1]})
            except AttributeError:
                # socket is None, not in use
                pass

    def blit_grid(self, screen: pygame.surface.Surface, game_time):
        for row_num, row in enumerate(self.grid):
            for col_num, item in enumerate(row):

                item.tick(game_time)

                if item.image is not None:
                    # offset by 1 to allow for 1px grid lines
                    screen.blit(item.image, (math.floor(col_num * self.grid_size) + 1, math.floor(row_num * self.grid_size) + 1))

    def initiate_cars(self):
        spawn_pos = []
        for row_num, row in enumerate(self.grid):
            for col_num, col in enumerate(row):
                if type(col) == StraightRoad:
                    spawn_pos.append(((col_num, row_num), col.direction))

        random.shuffle(spawn_pos)

        ai_loc = spawn_pos[0][0]
        ai_rot = spawn_pos[0][1]
        self.ai_car.controller.location = (np.array(ai_loc) * self.grid_size) + np.array((self.grid_size / 2, self.grid_size / 2))
        self.ai_car.controller.rotation = random.choice([ai_rot * 90, (ai_rot*90)-180])

        for car_num, car in enumerate(self.all_cars):
            car_num = car_num + 1

            try:
                loc, rot = spawn_pos[car_num][0], spawn_pos[car_num][1]
                car.controller.location = (np.array(loc) * self.grid_size) + np.array((self.grid_size / 2, self.grid_size / 2))
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
            car.draw_car(screen)

    def car_collision(self):
        for car in self.all_cars:
            overlaps = []
            corners = car.get_corners()

            for corner in corners:
                grid_pos = (corner // self.grid_size).astype(int).tolist()

                try:
                    placeable = self.grid[grid_pos[1]][grid_pos[0]]
                    overlaps.append(placeable.overlap(corner % self.grid_size, self.grid_size))
                except IndexError:
                    pass

            if not all(overlaps):
                try:
                    self.npc_cars.remove(car)
                except ValueError:
                    # ai car dead
                    self.ai_car.controller.ai_dead = True
