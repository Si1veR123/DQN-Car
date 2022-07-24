import pygame
import typing
import global_settings


class Placeable:
    def __init__(self, name_id, grid_size):
        # name_id is what the placeable is called, not unique

        # scale to 2 pixels less than grid size to allow for 1px grid lines
        try:
            self.image = pygame.transform.scale(self.create_grid_image(), (grid_size - 1, grid_size - 1))
        except ValueError:
            self.image = None

        self.name_id = name_id

        self.replicate_spawn = True

    def create_grid_image(self):
        raise NotImplementedError

    def tick(self, game_time):
        pass

    def overlap(self, relative_point, grid_size):
        return False

    # Custom Pickling
    def __getstate__(self):
        state = self.__dict__.copy()

        del state["image"]

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # before this is called, all attributes that are used by this method should be set to prevent errors
        self.image = pygame.transform.scale(self.create_grid_image(), tuple([global_settings.GRID_SIZE_PIXELS - 1]*2))


class SolidBlock(Placeable):
    def __init__(self, colour, grid_size):
        self.colour = colour
        super().__init__("solid", grid_size)
        self.replicate_spawn = False

    def create_grid_image(self):
        surface = pygame.surface.Surface((100, 100))
        surface.fill(self.colour)
        return surface


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
