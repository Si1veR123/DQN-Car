import pygame
import global_settings as gs


class Placeable:
    """
    A base class for anything that will be in the map's grid.
    """
    def __init__(self, name_id, grid_size):

        # scale to 2 pixels less than grid size to allow for 1px grid lines
        try:
            self.image = pygame.transform.scale(self.create_grid_image(), (grid_size - 1, grid_size - 1))
        except ValueError:
            self.image = None

        # name_id is what the Placeable is called, not unique to the instance
        # isn't static as some child classes have multiple name_ids for different sections, e.g. curved roads
        self.name_id = name_id

        # whether spawn is replicated to Unreal Engine
        self.replicate_spawn = True

    def create_grid_image(self):
        """
        Uses self attributes to return a Surface, which is the Placeables image on the grid
        :return:
        """
        raise NotImplementedError

    def tick(self, game_time):
        """
        Runs every frame
        :param game_time: game time in seconds
        """
        pass

    def overlap(self, relative_point):
        """
        Whether a car can overlap at this point.
        :param relative_point: pixel location relative to top left of the grid location
        :return: bool, true if relative point is allowed for a car e.g. road
        """
        return False

    # Custom Pickling
    def __getstate__(self):
        """Called when pickling"""

        # get all attributes
        state = self.__dict__.copy()
        # don't save image, as Surface can't be pickled
        del state["image"]
        return state

    def __setstate__(self, state):
        """Called when unpickling"""

        # reload state to self
        self.__dict__.update(state)

        # reload image using the method, and resize it
        self.image = pygame.transform.scale(self.create_grid_image(), tuple([gs.GRID_SIZE_PIXELS - 1]*2))


class SolidBlock(Placeable):
    """Solid colour block"""
    def __init__(self, colour, grid_size):
        self.colour = colour
        super().__init__("solid", grid_size)
        self.replicate_spawn = False

    def create_grid_image(self):
        # Solid colour surface
        surface = pygame.surface.Surface((1, 1))
        surface.fill(self.colour)
        return surface


class SolidRoadPath(SolidBlock):
    """
    In map builder, placed on confirmed road places
    Only difference from normal solid block, is that this fades in
    """
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
