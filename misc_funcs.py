import numpy as np


def stdfrm(m, p):
    # standard form
    return m*(10**p)


def rotate_vector_acw(vector, angle):
    """
    :param vector: a 2 length numpy array
    :param angle: an anti clockwise angle
    :return: the vector rotated anticlockwise by angle
    """
    angle = angle * np.pi / 180

    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    return rotation.dot(vector)


def cut_matrix(matrix, start, end):
    """
    Given a 2d list (matrix)
    Returns a matrix cut from a top left position (start)
        to a bottom right position (end)
    """

    new_matrix = []
    difference = (end[0]-start[0], end[1]-start[1])

    for x in range(difference[1]):
        row = []
        for y in range(difference[0]):
            try:
                row.append(matrix[start[1]+x][start[0]+y])
            except IndexError:
                return None

        new_matrix.append(row)

    return new_matrix


def color_lerp(color1, color2, alpha):
    color1 = np.array(color1)
    color2 = np.array(color2)
    return alpha*(color2-color1)+color1
