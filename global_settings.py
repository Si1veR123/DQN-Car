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

# Q Learning
TRAINING = False

LEARNING_RATE = 0.00001
DISCOUNT_RATE = 0.95
EXPLORATION_PROBABILITY = 1 if TRAINING else 0
EXPLORATION_DECAY = 0.005
TARGET_NET_COPY_STEPS = 5000
