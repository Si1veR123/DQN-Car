import numpy as np
import pygame
import global_settings as gs
from misc_funcs import rotate_vector_acw
from App.placeable import Placeable


class Road(Placeable):
    # list of possible images for this road type
    road_images = []


class StraightRoad(Road):
    """Simple, straight road"""
    try:
        road_images = [pygame.image.load("image_assets/straight_road.png")]
    except FileNotFoundError:
        road_images = []

    def __init__(self, direction, grid_size):
        self.direction = direction  # 0 or 1, 1 is horizontal
        super().__init__("straight_road", grid_size)

    def create_grid_image(self):
        # Rotate image based on direction
        return pygame.transform.rotate(self.road_images[0], 90*self.direction)

    def overlap(self, relative_point):
        # Any point on this block overlaps road
        return True


def circle(x, radius):
    """
    Equation for positive y values of a circle, starting at 0,0
    sqrt(r^2 - (x-r)^2)
    :param x: x in circle equaion
    :param radius: circle radius

    :returns: positive y value at x, or NAN if invalid
    """
    a = np.square(radius) - np.square(x-radius)
    if a < 0:
        return np.nan
    return np.sqrt(a)


def circle_collision_road(relative_point, grid_location, grid_size, rotation, width):
    """
    :param relative_point: relative point to individual grid square location
    :param grid_location: grid location, to make relative_point relative to entire road section
    :param grid_size: size of grid square
    :param rotation: rotation of curved road
    :param width: width of entire curved road (in grid squares)
    :return: true or false if point is outside of road
    """

    """
    rotate relative point about center of grid square. then we can use relative point ignoring rotation.
    relative point can then be processed as if the road is like this
    ░░░░░░░░█████
    ░░░░░███░░░░░
    ░░░██░░░░░░░░
    ░██░░░░░░████
    ██░░░░░██░░░░
    █░░░░██░░░░░░
    """
    relative_point = rotate_vector_acw(relative_point - grid_size/2, -rotation) + grid_size/2

    # point relative to top left of curved road
    new_point = np.array(relative_point) + np.array(grid_location) * grid_size

    outer_circle = new_point[1] > (width*grid_size-circle(new_point[0], width*grid_size))  # true if on road (below outer circle)

    inner_circle_height = width*grid_size - circle(new_point[0] - grid_size, (width-1)*grid_size)

    inner_circle = True  # default to true ( < inner circle), then if valid, and greater than, change to false
    if not np.isnan(inner_circle_height):
        inner_circle = new_point[1] < inner_circle_height

    return inner_circle and outer_circle


class CurvedRoad(Road):
    """
    Allows curved roads
    Each instance of CurvedRoad is a single grid spot
    Multiple grid spots form a curved road
    'section' is used to show which part of the curved road an instance is
    Circles are used to calculate collision
    """
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
        self.width = 2  # grid space width of curved road

        super().__init__("curved_road_" + str(section), grid_size)

    def create_grid_image(self):
        return pygame.transform.rotate(self.road_images[self.section], -self.rotation*90)

    @property
    def grid_location(self):
        """
        :return: Grid location tuple, relative to top left of curved road pattern
        """
        return self.section % self.width, self.section//self.width

    def overlap(self, relative_point):
        # relative point is relative to top left of section

        rotation = self.rotation * 90

        return circle_collision_road(relative_point, self.grid_location, gs.GRID_SIZE_PIXELS, rotation, self.width)


class LargeCurvedRoad(CurvedRoad):
    """
    Larger curved road, width of 4
    """
    try:
        # None means the block is empty
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

