from misc_funcs import stdfrm


SAVED_MAPS_ROOT = r"saved_maps/"
SAVED_MODELS_ROOT = r"saved_models/"

# ======== WINDOW SETTINGS ========
FPS = 0  # 0 for unlimited
GRID_SIZE_PIXELS = 30  # 30 for low res
HEIGHT = 541  # 541 for low res
WIDTH = 961  # 961 for low res
SF = GRID_SIZE_PIXELS/60  # scale factor


# ======== SOCKET SETTINGS ========
PORT = 5656
USE_UNREAL_SOCKET = False
MESSAGE_LENGTH = 60  # padded to this length with @


# ======== COLOURS ========
COL_BACKGROUND = (97, 139, 74)
COL_GRID = (37, 60, 47)
COL_MOUSE_HIGHLIGHT = (109, 163, 77)
COL_PLACED_ROAD = (84, 86, 86)


# ======== MISC ========
FREE_ROAM = False  # no collision


# ======== Deep Q Learning ========
LOAD_MODEL = None
MAX_EPISODE_FRAMES = 4000

Q_LEARNING_SETTINGS = {
    "TRAINING": True,

    "TRAINING_FRAME_SKIP": 2,  # 1 is every frame, 2 is every 2 frames etc (can make predicting future reward easier due to less states)

    "LEARNING_RATE": stdfrm(3, -8),

    "GD_MOMENTUM": 0.9,

    "DISCOUNT_RATE": 0.999,

    "EPSILON_PROBABILITY": 1,
    "EPSILON_DECAY": 0.00004,
    "EPSILON_MIN": 0.1,

    "TARGET_NET_COPY_STEPS": 17000,
    "TRAIN_AMOUNT": 0.8,

    "BUFFER_LENGTH": 4000
}

# backward compatibility
TRAINING = Q_LEARNING_SETTINGS["TRAINING"]
