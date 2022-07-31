from SocketCommunication.tcp_socket import LocalTCPSocket
from App.placeable import Placeable, SolidBlock
from App.car import AICar
from App.road import StraightRoad
import pygame
import numpy as np
import datetime
import typing
import math
import random
import view_filters
import pickle
import global_settings as gs
import os


class Map:
    """
    The Map object stores the map and handles drawing the grid, saving and loading.
    Referenced in the World object.
    """
    def __init__(self, grid_dimensions):
        self.grid = []
        self.reset_grid(grid_dimensions)

    def reset_grid(self, grid_dimensions):
        # Sets entire grid to SolidBlock with colour of background
        self.grid = [[SolidBlock(gs.COL_BACKGROUND, gs.GRID_SIZE_PIXELS) for _ in range(grid_dimensions[0])] for _ in range(grid_dimensions[1])]

    def blit_grid(self, screen, game_time):
        # Tick and draw each Placeable

        for row_num, row in enumerate(self.grid):
            for col_num, item in enumerate(row):

                item.tick(game_time)

                if item.image is not None:
                    # offset by 1 to allow for 1px grid lines
                    # check if item type can be shown
                    if view_filters.can_show_type(item.name_id):
                        screen.blit(item.image, (math.floor(col_num * gs.GRID_SIZE_PIXELS) + 1, math.floor(row_num * gs.GRID_SIZE_PIXELS) + 1))

    def spawn_item_local(self, item: Placeable, grid_pos):
        self.grid[grid_pos[1]][grid_pos[0]] = item

    def save_map(self, screen):
        name = datetime.datetime.now().strftime("%d.%m;%H.%M")
        pygame.image.save_extended(screen, gs.SAVED_MAPS_ROOT + name + ".png")

        # save map by pickling the grid array
        with open(gs.SAVED_MAPS_ROOT + name, "wb") as file:
            pickle.dump(self.grid, file)
            file.flush()
        print("Saved map and image.")

    @classmethod
    def load_map(cls, name):
        # load map by unpickling the grid array
        print("LOADED:", name)
        with open(gs.SAVED_MAPS_ROOT + name, "rb") as file:
            grid_loaded = pickle.load(file)

        # create new class with grid set as loaded grid
        map = cls(np.array(grid_loaded).shape[::-1])
        map.grid = grid_loaded

        image = pygame.image.load_extended(gs.SAVED_MAPS_ROOT + name + ".png")

        return map, image


class World:
    """
    The World object stores and handles all data for the current state of the level.
        e.g. map, simulating cars, drawing everything to screen
    """
    def __init__(self,
                 socket: typing.Union[LocalTCPSocket, None],
                 map: Map
                 ):
        # socket used to communicate with Unreal Engine
        self.socket = socket

        self.ai_car = AICar("ai_car")
        self.npc_cars = []  # TODO: fully implement and test npc cars. currently untested

        self.map = map

    @property
    def all_cars(self):
        return self.npc_cars + [self.ai_car]

    def replicate_map_spawn(self):
        for row_num, row in enumerate(self.map.grid):
            for col_num, col in enumerate(row):
                if col.replicate_spawn:
                    try:
                        self.socket.send_ue_data("spawn", col.name_id, {"gridx": col_num, "gridy": row_num})
                    except AttributeError:
                        # socket is None, not in use
                        pass

    def initiate_cars(self):
        """
        Move all cars to initial (random) positions
        Tell each car to reset state
        """

        # find all possible spawn places (straight roads)
        spawn_pos = []
        for row_num, row in enumerate(self.map.grid):
            for col_num, col in enumerate(row):
                if type(col) == StraightRoad:
                    # tuple of possible spawn location and road direction
                    spawn_pos.append(((col_num, row_num), col.direction))

        random.shuffle(spawn_pos)

        ai_loc = spawn_pos[0][0]
        ai_rot = spawn_pos[0][1]
        self.ai_car.controller.location = (np.array(ai_loc) * gs.GRID_SIZE_PIXELS) + np.array((gs.GRID_SIZE_PIXELS / 2, gs.GRID_SIZE_PIXELS / 2))
        # rotate so car faces either backwards or forwards, randomly
        self.ai_car.controller.rotation = random.choice([ai_rot * 90, (ai_rot*90)-180])

        self.ai_car.reset_state()

        # initiate npc cars
        for car_num, car in enumerate(self.npc_cars):
            try:
                loc, rot = spawn_pos[car_num][0], spawn_pos[car_num][1]
                car.controller.location = (np.array(loc) * gs.GRID_SIZE_PIXELS) + np.array((gs.GRID_SIZE_PIXELS / 2, gs.GRID_SIZE_PIXELS / 2))
                car.controller.rotation = rot * 90
            except IndexError:
                print("Couldnt spawn: ", car.object_name)

            car.reset_state()

    def update_cars(self, velocity_constant):
        # update each car's controller
        for car in self.all_cars:
            car.controller.update_transform(velocity_constant)

    def blit_cars(self, screen):
        # Tick and draw each car
        for car in self.all_cars:
            car.tick()
            if view_filters.can_show_type("car"):
                car.draw_car(screen)

    def car_collision(self):
        """
        Check every car's collision, and if it has crashed, by checking if corners overlap roads
        """
        for car in self.all_cars:
            overlaps = []
            corners = car.get_corners()

            for corner in corners:
                grid_pos = (corner // gs.GRID_SIZE_PIXELS).astype(int).tolist()

                try:
                    placeable = self.map.grid[grid_pos[1]][grid_pos[0]]
                    overlaps.append(placeable.overlap(corner % gs.GRID_SIZE_PIXELS))
                except IndexError:
                    pass

            # if not all corners on roads
            if not all(overlaps):
                try:
                    # remove npc car from list. if ValueError raised, isn't an npc car
                    self.npc_cars.remove(car)
                except ValueError:
                    # ai car dead
                    self.ai_car.controller.ai_dead = True

    def blit_ai_action(self, screen):
        width, height = screen.get_size()

        left_rect = pygame.rect.Rect(10, height-60, 50, 50)
        down_rect = pygame.rect.Rect(70, height-60, 50, 50)
        right_rect = pygame.rect.Rect(130, height-60, 50, 50)
        up_rect = pygame.rect.Rect(70, height-120, 50, 50)

        action = self.ai_car.controller.current_action

        normal_col = (150, 150, 150)
        action_col = (255, 255, 255)

        pygame.draw.rect(screen, action_col if action == 1 else normal_col, down_rect)
        pygame.draw.rect(screen, action_col if action == 2 else normal_col, up_rect)
        pygame.draw.rect(screen, action_col if action == 3 else normal_col, left_rect)
        pygame.draw.rect(screen, action_col if action == 4 else normal_col, right_rect)
