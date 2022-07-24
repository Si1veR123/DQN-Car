from SocketCommunication.replicate import ReplicatedTransform
from MachineLearning.autonomous_driving_controller import AutonomousDrivingController
from World.car_controller import PlayerController
from misc_funcs import rotate_vector_acw
from World.placeable import Placeable

import global_settings

import numpy as np
import pygame
import view_filters


class Car(ReplicatedTransform):
    def __init__(self, car_name, controller):
        super().__init__("car", car_name)
        self.controller = controller

        # default is 400px, so scale
        self.car_size = 40
        self.car_image = pygame.transform.scale(pygame.image.load("image_assets/car.png"), (self.car_size, self.car_size))

        self.x_bound = 8
        self.y_bound = 18

    def tick(self):
        pass

    def reset_state(self):
        pass

    def draw_car(self, screen: pygame.surface.Surface):
        rotated = pygame.transform.rotate(self.car_image, self.controller.rotation)

        new_rect = rotated.get_rect(center=self.controller.location)

        screen.blit(rotated, new_rect)

    def get_data_to_replicate(self):
        return self.controller.location, self.controller.rotation, self.controller.scale

    def get_corners(self):
        rel_tl = rotate_vector_acw((-self.x_bound, -self.y_bound), -self.controller.rotation)
        rel_tr = rotate_vector_acw((self.x_bound, -self.y_bound), -self.controller.rotation)
        rel_bl = rotate_vector_acw((-self.x_bound, self.y_bound), -self.controller.rotation)
        rel_br = rotate_vector_acw((self.x_bound, self.y_bound), -self.controller.rotation)

        loc = np.array(self.controller.location)

        return [rel_tl + loc, rel_tr + loc, rel_bl + loc, rel_br + loc]


class AICar(Car):
    def __init__(self, car_name):
        self.ray_angle_range = 80
        self.ray_count = 9
        self.ray_distance = 300
        self.ray_check_frequency = 5

        super().__init__(car_name, AutonomousDrivingController(self.ray_count))

        self.ray_offset = np.array((0, self.controller.wheel_distance * .7))

    def reset_state(self):
        # NEW EPISODE
        self.controller.steering_angle = 0
        self.controller.velocity = self.controller.start_velocity

        if global_settings.TRAINING:  # global settings
            self.controller.q_learning.decay_exploration_probability()
            self.controller.q_learning.train()

        self.controller.distance_travelled = 0

        self.controller.ai_dead = False

    def trace_all_rays(self, world, screen):
        offset = np.array(rotate_vector_acw(self.ray_offset, -self.controller.rotation))
        ray_start = np.array(self.controller.location) + offset

        for ray_number in range(self.ray_count):
            relative_ray_angle = ((self.ray_angle_range * 2) / (self.ray_count - 1)) * ray_number - self.ray_angle_range

            ray_angle = relative_ray_angle + self.controller.rotation

            ray_end = ray_start + rotate_vector_acw((0, self.ray_distance), -ray_angle)

            self.controller.state[ray_number] = self.ray_trace(ray_start, ray_end, world, screen)

    def ray_trace(self, start, end, world, screen=None):
        # start and end are numpy arrays, 2 length
        # trace an individual ray by testing points at ray_check_frequency
        # if screen is provided, draws lines to screen

        length = self.ray_distance
        for f in range(self.ray_distance//self.ray_check_frequency):
            ray_point = start + ((end - start)/self.ray_distance * (f * self.ray_check_frequency))

            ray_grid_box = (ray_point // global_settings.GRID_SIZE_PIXELS).astype(int).tolist()

            try:
                placeable = world.map.grid[ray_grid_box[1]][ray_grid_box[0]]
                placeable: Placeable
            except IndexError:
                continue

            # not overlapping road
            if not placeable.overlap(ray_point % global_settings.GRID_SIZE_PIXELS, global_settings.GRID_SIZE_PIXELS):
                length = f * self.ray_check_frequency
                break

        if screen is not None and view_filters.can_show_type("ai_rays"):
            pygame.draw.line(screen, (255, 155, 155), start, (((end-start)/self.ray_distance)*length)+start)

        if screen is not None and view_filters.can_show_type("ai_ray_collisions"):
            pygame.draw.circle(screen, (255, 155, 155), (((end-start)/self.ray_distance)*length)+start, 4)

        return length


class PlayerCar(Car):
    def __init__(self, car_name):
        super().__init__(car_name, PlayerController())

    def reset_state(self):
        self.controller.steering_angle = 0
        self.controller.velocity = 0

        self.controller.location = (5, 5)

        self.controller.rotation = 0

