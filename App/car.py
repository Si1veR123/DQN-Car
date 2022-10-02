from SocketCommunication.replicate import ReplicatedTransform
from MachineLearning.autonomous_driving_controller import AutonomousDrivingControllerCombined
from App.car_controller import PlayerController
from misc_funcs import rotate_vector_acw
from App.placeable import Placeable

import global_settings as gs

import numpy as np
import pygame
import view_filters


class Car(ReplicatedTransform):
    """
    The Car object is... the car.
    It handles the drawing, collision etc. of cars in the world.
    Since it can be replicated to the Unreal Engine world, it inherits from ReplicatedTransform.
    The controller handles the movement.
    """
    def __init__(self, car_name, controller):
        super().__init__("car", car_name, "loc")
        self.controller = controller

        # default is 400px, so scale
        self.car_size = 40 * gs.SF
        self.car_image = pygame.transform.scale(pygame.image.load("image_assets/car.png"), tuple([self.car_size]*2))

        self.x_bound = 8 * gs.SF
        self.y_bound = 18 * gs.SF

    def tick(self):
        pass

    def reset_state(self):
        pass

    def draw_car(self, screen: pygame.surface.Surface):
        # rotate the car by the rotation given by its controller
        rotated = pygame.transform.rotate(self.car_image, self.controller.rotation)

        # since rotating changes the center of the surface, get a rect which is re-centered
        new_rect = rotated.get_rect(center=self.controller.location)

        # draw the car, using the new_rect to correctly position the surface
        screen.blit(rotated, new_rect)

    def get_data_to_replicate(self):
        return self.controller.location, self.controller.rotation, self.controller.scale

    def get_corners(self):
        """
        :return: a list of 4 corners in screen space, of the car
        """
        rel_tl = rotate_vector_acw((-self.x_bound, -self.y_bound), -self.controller.rotation)
        rel_tr = rotate_vector_acw((self.x_bound, -self.y_bound), -self.controller.rotation)
        rel_bl = rotate_vector_acw((-self.x_bound, self.y_bound), -self.controller.rotation)
        rel_br = rotate_vector_acw((self.x_bound, self.y_bound), -self.controller.rotation)

        loc = np.array(self.controller.location)

        return [rel_tl + loc, rel_tr + loc, rel_bl + loc, rel_br + loc]


class AICar(Car):
    """
    AICar handles the extra features needed for DQ Learning, such as ray tracing.
    AutonomousDrivingController handles the DQN training and actions.
    """
    def __init__(self, car_name):
        self.ray_angle_range = 80
        self.ray_count = 9
        self.ray_distance = 300 * gs.SF
        self.ray_check_frequency = 6 * gs.SF

        super().__init__(car_name, AutonomousDrivingControllerCombined(self.ray_count + 1))  # add 1 for velocity state

        self.ray_offset = np.array((0, self.controller.wheel_distance * 0.7))

    def reset_state(self):
        pass

    def trace_all_rays(self, world, screen):
        # The rotated offset from center of car to ray start position
        offset = np.array(rotate_vector_acw(self.ray_offset, -self.controller.rotation))
        ray_start = np.array(self.controller.location) + offset

        for ray_number in range(self.ray_count):
            # gets the angle relative to 0, that the current ray_number should be at
            relative_ray_angle = ((self.ray_angle_range * 2) / (self.ray_count - 1)) * ray_number - self.ray_angle_range

            # convert to angle relative to car forward
            ray_angle = relative_ray_angle + self.controller.rotation

            # rotate a vector, which is parallel to y, clockwise by the ray angle. This vector, added to start is the end location
            ray_end = ray_start + rotate_vector_acw((0, self.ray_distance), -ray_angle)

            # trace this ray for collisions, then set as state for AI
            self.controller.state[ray_number] = self.ray_trace(ray_start, ray_end, world, screen)*(1/gs.SF)  # state is as if no scaling is applied, so reverse scaling
        self.controller.state[self.ray_count] = self.controller.velocity

    def ray_trace(self, start, end, world, screen=None):
        # start and end are numpy arrays, 2 length
        # trace an individual ray by testing points at ray_check_frequency
        # if screen is provided, draws lines to screen

        # initiate length to maximum
        length = self.ray_distance

        for f in range(int(self.ray_distance//self.ray_check_frequency)):
            # iterate over equal distances along the ray
            # this is the point currently testing
            ray_point = start + ((end - start)/self.ray_distance * (f * self.ray_check_frequency))

            # find the grid box that the currently testing point is in
            ray_grid_box = (ray_point // gs.GRID_SIZE_PIXELS).astype(int).tolist()

            try:
                # retrieve placeable that the currently testing point is over
                placeable = world.map.grid[ray_grid_box[1]][ray_grid_box[0]]
                placeable: Placeable
            except IndexError:
                # point out of map
                continue

            # not overlapping road (collided with road)
            if not placeable.overlap(ray_point % gs.GRID_SIZE_PIXELS):
                # collision point has been found. set length and break.
                length = f * self.ray_check_frequency
                break

        # draw the ray's line
        if screen is not None and view_filters.can_show_type("ai_rays"):
            pygame.draw.line(screen, (255, 155, 155), start, (((end-start)/self.ray_distance)*length)+start)

        # draw a circle at collision point
        if screen is not None and view_filters.can_show_type("ai_ray_collisions"):
            pygame.draw.circle(screen, (255, 155, 155), (((end-start)/self.ray_distance)*length)+start, 4)

        return length


class PlayerCar(Car):
    """
    A car that can be controlled by the player
    """
    def __init__(self, car_name):
        super().__init__(car_name, PlayerController())

    def reset_state(self):
        # TODO: spawn at random place, make this method on controller
        self.controller.steering_angle = 0
        self.controller.velocity = 0

        self.controller.location = (5, 5)

        self.controller.rotation = 0

