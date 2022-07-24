import numpy as np
import pygame
import view_filters
from misc_funcs import rotate_vector_acw
from App.placeable import Placeable


class Road(Placeable):
    road_images = []


class StraightRoad(Road):
    try:
        road_images = [pygame.image.load("image_assets/straight_road.png")]
    except FileNotFoundError:
        road_images = []

    def __init__(self, direction, grid_size):
        self.direction = direction  # 0 or 1, 1 is horizontal
        super().__init__("straight_road", grid_size)

    def create_grid_image(self):
        return pygame.transform.rotate(self.road_images[0], 90*self.direction)

    def overlap(self, relative_point, grid_size):
        return True


def circle(x, radius):
    a = np.square(radius) - np.square(x-radius)
    if a < 0:
        return np.nan
    return np.sqrt(a)


def circle_collision(relative_point, grid_location, grid_size, rotation, width):
    """
    :param relative_point: relative point to grid location
    :param grid_location: grid location to make relative point relative to the top right
    :param grid_size: size of grid square
    :param rotation: rotation of curved road
    :param width: width of entire curved road (in grid squares)
    :return: true or false if point is outside of road
    """

    relative_point = rotate_vector_acw(relative_point - grid_size/2, -rotation) + grid_size/2

    new_point = np.array(relative_point) + np.array(grid_location) * grid_size

    outer_circle = new_point[1] > (width*grid_size-circle(new_point[0], width*grid_size))  # true if on road (below outer circle)

    inner_circle_height = width*grid_size - circle(new_point[0] - grid_size, (width-1)*grid_size)

    inner_circle = True  # default to true ( < inner circle), then if valid, and greater than, change to false
    if not np.isnan(inner_circle_height):
        inner_circle = new_point[1] < inner_circle_height

    return inner_circle and outer_circle


class CurvedRoad(Road):

    try:
        road_images = [
            pygame.image.load("image_assets/curve_road_1.png"),
            pygame.image.load("image_assets/curve_road_2.png"),
            pygame.image.load("image_assets/curve_road_3.png"),
            pygame.image.load("image_assets/curve_road_4.png")
        ]
    except FileNotFoundError:
        road_images = []

    def __init__(self, rotation, section, grid_size):
        self.rotation = rotation  # rotation is 0-3
        self.section = section  # 0-3, as curved road is 4 grid spaces
        self.width = 2

        super().__init__("curved_road_" + str(section), grid_size)

    def create_grid_image(self):
        return pygame.transform.rotate(self.road_images[self.section], -self.rotation*90)

    @property
    def grid_location(self):
        return self.section % self.width, self.section//self.width

    def overlap(self, relative_point, grid_size):
        # relative point is relative to top left of section

        rotation = self.rotation * 90

        return circle_collision(relative_point, self.grid_location, grid_size, rotation, self.width)


class LargeCurvedRoad(CurvedRoad):
    try:
        road_images = [
            None,
            "image_assets/large_curve_road/section_1.png",
            "image_assets/large_curve_road/section_2.png",
            "image_assets/large_curve_road/section_3.png",
            "image_assets/large_curve_road/section_4.png",
            "image_assets/large_curve_road/section_5.png",
            "image_assets/large_curve_road/section_6.png",
            "image_assets/large_curve_road/section_7.png",
            "image_assets/large_curve_road/section_8.png",
            "image_assets/large_curve_road/section_9.png",
            None,
            None,
            "image_assets/large_curve_road/section_12.png",
            "image_assets/large_curve_road/section_13.png",
            None,
            None
        ]

        # really bad code for opening each image in the list unless its none
        road_images = [pygame.image.load(path) if path is not None else None for path in road_images]

    except FileNotFoundError:
        road_images = []

    def __init__(self, rotation, section, grid_size):
        super().__init__(rotation, section, grid_size)

        self.width = 4

