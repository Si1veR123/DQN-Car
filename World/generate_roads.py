from World.road import *
from World.placeable import SolidBlock
import numpy as np
import misc_funcs

# list of tuples
# [0] of tuple is the 2d matrix for the pattern
# [1] of the tuple is a list of data for each block in the pattern


# if type is None dont change block
# 'lock' causes other patterns to be unable to overwrite the current pattern if placed, and no other patterns can match
# in the same place
patterns = [
    (
        [[1, 1]], [
            {"centre": (0, 0), "lock": False, "type": StraightRoad, "parameters": {"direction": 1}},
            {"centre": (1, 0), "lock": False, "type": StraightRoad, "parameters": {"direction": 1}}
        ]
     ),
    (
        [[1],
         [1]], [
            {"centre": (0, 0), "lock": False, "type": StraightRoad, "parameters": {"direction": 0}},
            {"centre": (0, 1), "lock": False, "type": StraightRoad, "parameters": {"direction": 0}}
        ]
     ),
    (
        [[1, 1],
         [1, 0]], [
            {"centre": (0, 0), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 0, "section": 0}},
            {"centre": (1, 0), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 0, "section": 1}},
            {"centre": (0, 1), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 0, "section": 2}},
            {"centre": (1, 1), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 0, "section": 3}}
        ]
     ),

    (
        [[1, 1],
         [0, 1]], [
            {"centre": (0, 0), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 1, "section": 2}},
            {"centre": (1, 0), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 1, "section": 0}},
            {"centre": (0, 1), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 1, "section": 3}},
            {"centre": (1, 1), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 1, "section": 1}}
        ]
     ),

    (
        [[0, 1],
         [1, 1]], [
            {"centre": (0, 0), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 2, "section": 3}},
            {"centre": (1, 0), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 2, "section": 2}},
            {"centre": (0, 1), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 2, "section": 1}},
            {"centre": (1, 1), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 2, "section": 0}}
        ]
     ),

    (
        [[1, 0],
         [1, 1]], [
            {"centre": (0, 0), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 3, "section": 1}},
            {"centre": (1, 0), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 3, "section": 3}},
            {"centre": (0, 1), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 3, "section": 0}},
            {"centre": (1, 1), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 3, "section": 2}}
        ]
     ),
    (
        [[1, 1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0]], [
            {"centre": (0, 0), "lock": False, "type": None},
            {"centre": (1, 0), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 1}},
            {"centre": (2, 0), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 2}},
            {"centre": (3, 0), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 3}},
            {"centre": (0, 1), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 4}},
            {"centre": (1, 1), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 5}},
            {"centre": (2, 1), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 6}},
            {"centre": (3, 1), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 7}},
            {"centre": (0, 2), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 8}},
            {"centre": (1, 2), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 9}},
            {"centre": (2, 2), "lock": False, "type": None},
            {"centre": (3, 2), "lock": False, "type": None},
            {"centre": (0, 3), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 12}},
            {"centre": (1, 3), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 13}},
            {"centre": (2, 3), "lock": False, "type": None},
            {"centre": (3, 3), "lock": False, "type": None},
        ]
    ),
    (
        [[1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1]], [
            {"centre": (2, 0), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 1, "section": 12}},
            {"centre": (3, 0), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 1, "section": 8}},
            {"centre": (4, 0), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 1, "section": 4}},
            {"centre": (5, 0), "lock": False, "type": None},
            {"centre": (2, 1), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 1, "section": 13}},
            {"centre": (3, 1), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 1, "section": 9}},
            {"centre": (4, 1), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 1, "section": 5}},
            {"centre": (5, 1), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 1, "section": 1}},
            {"centre": (2, 2), "lock": False, "type": None},
            {"centre": (3, 2), "lock": False, "type": None},
            {"centre": (4, 2), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 1, "section": 6}},
            {"centre": (5, 2), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 1, "section": 2}},
            {"centre": (2, 3), "lock": False, "type": None},
            {"centre": (3, 3), "lock": False, "type": None},
            {"centre": (4, 3), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 1, "section": 7}},
            {"centre": (5, 3), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 1, "section": 3}},
        ]
    ),
    (
        [[0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1],
         [1, 1, 1, 1, 1, 1]], [
            {"centre": (2, 2), "lock": False, "type": None},
            {"centre": (3, 2), "lock": False, "type": None},
            {"centre": (4, 2), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 2, "section": 13}},
            {"centre": (5, 2), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 2, "section": 12}},
            {"centre": (2, 3), "lock": False, "type": None},
            {"centre": (3, 3), "lock": False, "type": None},
            {"centre": (4, 3), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 2, "section": 9}},
            {"centre": (5, 3), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 2, "section": 8}},
            {"centre": (2, 4), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 2, "section": 7}},
            {"centre": (3, 4), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 2, "section": 6}},
            {"centre": (4, 4), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 2, "section": 5}},
            {"centre": (5, 4), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 2, "section": 4}},
            {"centre": (2, 5), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 2, "section": 3}},
            {"centre": (3, 5), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 2, "section": 2}},
            {"centre": (4, 5), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 2, "section": 1}},
            {"centre": (5, 5), "lock": False, "type": None},
        ]
    ),
    (
        [[1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1]], [
            {"centre": (0, 2), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 3, "section": 3}},
            {"centre": (1, 2), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 3, "section": 7}},
            {"centre": (2, 2), "lock": False, "type": None},
            {"centre": (3, 2), "lock": False, "type": None},
            {"centre": (0, 3), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 3, "section": 2}},
            {"centre": (1, 3), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 3, "section": 6}},
            {"centre": (2, 3), "lock": False, "type": None},
            {"centre": (3, 3), "lock": False, "type": None},
            {"centre": (0, 4), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 3, "section": 1}},
            {"centre": (1, 4), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 3, "section": 5}},
            {"centre": (2, 4), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 3, "section": 9}},
            {"centre": (3, 4), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 3, "section": 13}},
            {"centre": (0, 5), "lock": False, "type": None},
            {"centre": (1, 5), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 3, "section": 4}},
            {"centre": (2, 5), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 3, "section": 8}},
            {"centre": (3, 5), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 3, "section": 12}},
        ]
    ),
]


def get_pattern_data(pattern):
    """
    :param pattern: pattern matrix
    :return: data for this pattern, using the patterns list
    """
    for p in patterns:
        if np.array_equal(np.array(p[0]), np.array(pattern)):
            return p[1]
    return []


def generate_roads(drawn_roads, grid_size):
    """
    uses patterns to convert drawn roads to road placeables
    :param drawn_roads: 2D matrix of 1s and 0s
    :return: matrix of None or road placeable
    """

    shape = np.array(drawn_roads).shape
    roads = [[None for _ in range(shape[1])] for _ in range(shape[0])]

    for pattern in map(lambda x: x[0], patterns):
        pattern_data = get_pattern_data(pattern)
        locations = search_pattern(drawn_roads, pattern)

        for location in locations:
            for block in pattern_data:
                block: dict  # prevents warnings
                block_location = (block["centre"][0] + location[0], block["centre"][1] + location[1])

                if block["lock"]:
                    # if lock, prevent other blocks being placed here by setting that it was never drawn in
                    drawn_roads[block_location[1]][block_location[0]] = 0

                if block["type"] is not None:

                    parameters = block["parameters"]
                    parameters["grid_size"] = grid_size
                    new_road = block["type"](**parameters)

                    roads[block_location[1]][block_location[0]] = new_road
                else:
                    roads[block_location[1]][block_location[0]] = None

    return roads


def search_pattern(drawn_roads, pattern):
    """
    :param drawn_roads: full matrix of 0s and 1s, where 1 is a drawn road
    :param pattern: matrix of pattern to search for
    :param current_roads: currently placed roads
    :return: list of locations where the pattern occurs in the drawn roads. location is the top right of pattern.
    """
    pattern_size = np.array(pattern).shape[::-1]
    locations = []
    for row_num, row in enumerate(drawn_roads):
        for col_num in range(len(row)):

            start = np.array([col_num, row_num])

            compare_pattern = misc_funcs.cut_matrix(drawn_roads, start, start + pattern_size)
            if compare_pattern is None:
                continue

            if np.array_equal(np.array(compare_pattern), np.array(pattern)):
                locations.append((col_num, row_num))

    return locations
