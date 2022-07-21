import pygame
import typing


class Placeable:
    def __init__(self, image: typing.Union[pygame.surface.Surface, None], name_id, grid_size):
        # name_id is what the placeable is called, not unique

        # scale to 2 pixels less than grid size to allow for 1px grid lines
        self.image = pygame.transform.scale(image, (grid_size - 1, grid_size - 1))
        self.name_id = name_id

        self.replicate_spawn = True

    def tick(self, game_time):
        pass

    def overlap(self, relative_point, grid_size):
        return False


class Obstacle(Placeable):
    def __init__(self, type, image, grid_size):
        super().__init__(image, "obstacle", grid_size)
        self.type = type


class SolidBlock(Placeable):
    def __init__(self, colour, grid_size):
        surface = pygame.surface.Surface((100, 100))
        surface.fill(colour)
        super().__init__(surface, "solid", grid_size)
        self.replicate_spawn = False


class SolidRoadPath(SolidBlock):
    # in map builder, placed on confirmed road places
    def __init__(self, colour, grid_size, game_time):
        super().__init__(colour, grid_size)
        self.start_time = game_time

    def tick(self, game_time):
        # fade in road
        time_since_spawn = game_time - self.start_time
        animated = min(255, time_since_spawn*2500)

        if not animated >= 500:
            # 500 to ensure it is set at least once at 255
            self.image.set_alpha(animated)
