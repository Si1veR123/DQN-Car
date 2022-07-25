"""
Classes used to filter which elements should be displayed on screen

Types:

ai_rays
ai_ray_collisions

grid

collision
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
    allowed = ["ai_ray_collisions", "car"]
