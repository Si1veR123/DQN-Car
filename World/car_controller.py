import numpy as np
import pygame
from misc_funcs import rotate_vector_acw


class CarController:
    def __init__(self):
        self.location = (500, 500)
        self.rotation = 90
        self.scale = (0, 0)

    def update_transform(self, time):
        """
        :param time: time since last update
        :return:
        """
        raise NotImplementedError


class CarControllerRoadFollow(CarController):
    # controller moves by setting new location and rotation
    def __init__(self, road):
        super().__init__()
        self.road = road

    def update_transform(self, time):
        pass


class CarControllerKinematic(CarController):
    # kinematic controller moves using steering angle, and velocity
    def __init__(self):
        super().__init__()

        self.steering_angle = 0  # -90 - 90 relative to car rotation
        self.velocity = 0
        self.acceleration = 0

        self.wheel_distance = 13

    def update_transform(self, velocity_constant):
        if self.steering_angle:
            radius = self.wheel_distance / np.sin(self.steering_angle * np.pi / 180)  # radius of steering circle
        else:
            radius = 9999999999999999  # big number to simulate infinity

        angular_change = (self.velocity*velocity_constant*5) / radius  # 10 makes it more realistic
        self.rotation += angular_change

        self.location += rotate_vector_acw(np.array((0, self.velocity * velocity_constant)), -self.rotation)

        self.velocity += self.acceleration * velocity_constant


class PlayerController(CarControllerKinematic):
    def update_transform(self, velocity_constant):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            self.velocity = 10
        elif keys[pygame.K_DOWN]:
            self.velocity = -10
        elif keys[pygame.K_RIGHT]:
            self.steering_angle = -80
        elif keys[pygame.K_LEFT]:
            self.steering_angle = 80
        else:
            self.steering_angle = 0

        super().update_transform(velocity_constant)
