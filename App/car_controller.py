import numpy as np
import pygame
from misc_funcs import rotate_vector_acw
import global_settings as gs


class CarController:
    """
    CarController provides a car with location, rotation and scale
    """
    def __init__(self):
        self.location = (0, 0)
        self.rotation = 0
        self.scale = (0, 0)

    def update_transform(self, time):
        """
        :param time: time since last update
        :return:
        """
        raise NotImplementedError


class CarControllerRoadFollow(CarController):
    """
    Controls a car by following the road's path
    """
    def __init__(self, road):
        super().__init__()
        self.road = road

    def update_transform(self, time):
        raise NotImplementedError


class CarControllerKinematic(CarController):
    """
    Kinematic controller provides location and rotation by calculating the turning circle,
    using steering angle, velocity and acceleration
    """
    def __init__(self):
        super().__init__()

        self.steering_angle = 0  # -90 - 90 relative to car rotation
        self.velocity = 0  # distance per frame, scaled by VELOCITY CONSTANT
        self.acceleration = 0

        self.wheel_distance = 13 * (gs.GRID_SIZE_PIXELS/60)

    def update_transform(self, velocity_constant):
        # if steering angle is 0, dont calculate as sin(0)=0
        if self.steering_angle:
            radius = self.wheel_distance / np.sin(self.steering_angle * np.pi / 180)  # radius of steering circle
        else:
            radius = 9999999999999999  # big number to simulate infinity

        angular_change = (self.velocity*velocity_constant*8) / radius  # 8 changes steering intensity
        self.rotation += angular_change

        # move forward by the velocity rotated clockwise by rotation
        self.location += rotate_vector_acw(np.array((0, self.velocity * velocity_constant)), -self.rotation)

        self.velocity += self.acceleration


class PlayerController(CarControllerKinematic):
    """
    Controls the car by arrow keys
    """
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
