import pygame, time
pygame.init()

from World.car import PlayerCar
from World.world import World
from World.placeable import SolidBlock, SolidRoadPath
from World.generate_roads import generate_roads
from SocketCommunication.tcp_socket import LocalTCPSocket


FPS = 0
PORT = 5656
GRID_SIZE_PIXELS = 60

HEIGHT = 1081
WIDTH = 1921

USE_UNREAL_SOCKET = False

VELOCITY_CONSTANT = .4

# COLOURS
COL_BACKGROUND = (97, 139, 74)
COL_GRID = (37, 60, 47)
COL_MOUSE_HIGHLIGHT = (109, 163, 77)
COL_PLACED_ROAD = (84, 86, 86)

# Create Window
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Create World Object
dimensions = (WIDTH//GRID_SIZE_PIXELS+1, HEIGHT//GRID_SIZE_PIXELS+1)

world = World(None, GRID_SIZE_PIXELS, dimensions, COL_BACKGROUND)


def draw_background(screen):
    # Background is drawn by world's placeables, but this is beneath
    screen.fill(COL_BACKGROUND)

    # Draw Grid
    [pygame.draw.line(screen, COL_GRID, (x * GRID_SIZE_PIXELS, 0), (x * GRID_SIZE_PIXELS, HEIGHT)) for x in
        range((WIDTH // round(GRID_SIZE_PIXELS)) + 1)]
    [pygame.draw.line(screen, COL_GRID, (0, y * GRID_SIZE_PIXELS), (WIDTH, y * GRID_SIZE_PIXELS)) for y in
        range((HEIGHT // round(GRID_SIZE_PIXELS)) + 1)]


# Map builder
mouse_grid = (0, 0)
clock = pygame.time.Clock()
run = True
first_block = True
while run:
    game_time = pygame.time.get_ticks() / 1000
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    draw_background(screen)

    mouse_grid = (
        round((pygame.mouse.get_pos()[0]) // GRID_SIZE_PIXELS),
        round((pygame.mouse.get_pos()[1]) // GRID_SIZE_PIXELS)
    )

    current_block = world.grid[mouse_grid[1]][mouse_grid[0]]
    if type(current_block) == SolidRoadPath:
        world.blit_grid(screen, game_time)
        pygame.display.update()
        continue

    clicked = pygame.mouse.get_pressed()[0]

    if clicked:
        world.spawn_item(SolidRoadPath(COL_PLACED_ROAD, GRID_SIZE_PIXELS, game_time), mouse_grid)
        if first_block:
            world.start_grid_location = mouse_grid
            world.start_rotation = 0
            first_block = False
    else:
        world.spawn_item(SolidBlock(COL_MOUSE_HIGHLIGHT, GRID_SIZE_PIXELS), mouse_grid)

    world.blit_grid(screen, game_time)

    pygame.display.update()

    if not clicked:
        # replace block after drawing the highlight
        world.spawn_item(SolidBlock(COL_BACKGROUND, GRID_SIZE_PIXELS), mouse_grid)

# matrix of grid, where 1 indicates a painted road and 0 is empty
painted_roads = [[1 if type(x) == SolidRoadPath else 0 for x in y] for y in world.grid]

world.reset_grid(COL_BACKGROUND, dimensions)

# create roads from painted roads
roads = generate_roads(painted_roads, GRID_SIZE_PIXELS)
for row_num, row in enumerate(roads):
    for col_num, col in enumerate(row):
        if col is not None:
            world.spawn_item(col, (col_num, row_num))

world.socket = LocalTCPSocket(PORT) if USE_UNREAL_SOCKET else None

# Game Loop
world.initiate_cars()

clock = pygame.time.Clock()
run = True
while run:
    game_time = pygame.time.get_ticks() / 1000
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    draw_background(screen)

    # Update all placeables
    world.blit_grid(screen, game_time)
    world.ai_car.trace_all_rays(world, screen)

    world.car_collision()
    world.update_cars(VELOCITY_CONSTANT)

    world.blit_cars(screen)

    if world.ai_car.controller.ai_dead:
        world.initiate_cars()

    pygame.display.update()

world.ai_car.controller.q_learning.reward_graph()
world.ai_car.controller.q_learning.save_model(r"E:\EPQ\PythonAI\saved_models\\")
