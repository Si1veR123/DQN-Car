"""
Classes used to filter which elements should be displayed on screen

Types:

ai_rays
ai_ray_collisions

grid

collision

q_target_change
"""

FILTERS = []


def can_show_type(draw_type):
    bools = []

    for f in FILTERS:
        bools.append(f.show_type(draw_type))

    return all(bools)


class Filter:
    def show_type(self, type):
        pass


class AllowedFilter(Filter):
    allowed = []

    def show_type(self, draw_type):
        return draw_type in self.allowed


class BlockedFilter(Filter):
    blocked = []

    def show_type(self, draw_type):
        return draw_type not in self.blocked


class NoCollision(BlockedFilter):
    blocked = ["collision"]


class NoAiVis(BlockedFilter):
    blocked = ["ai_rays", "ai_ray_collisions"]


class AiVisOnly(AllowedFilter):
    def __init__(self, rays=False):
        self.allowed = ["ai_ray_collisions", "car"]
        if rays:
            self.allowed.append("ai_rays")


class FastTrainingView(AllowedFilter):
    """
    Allows faster training by not drawing entire grid
    """
    def __init__(self, q_target_change=True):
        self.allowed = ["ai_rays", "car"]
        if q_target_change:
            self.allowed.append("q_target_change")


class NoQTargetChange(BlockedFilter):
    blocked = ["q_target_change"]
