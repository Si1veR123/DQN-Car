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
LOAD_MODEL = "combined_model_02.10;17.16_0"
MAX_EPISODE_FRAMES = 4000

Q_LEARNING_SETTINGS = {
    "TRAINING": True,

    "MOVEMENT_PER_FRAME": 3,  # 1 is normal, 2 is double etc (can make predicting future reward easier due to less states)

    "LEARNING_RATE": stdfrm(1, -5),

    "GD_MOMENTUM": 0.03,

    "DISCOUNT_RATE": 0.95,

    "EPSILON_PROBABILITY": 0.15,
    "EPSILON_DECAY": 0.0001,
    "EPSILON_MIN": 0.05,

    "TARGET_NET_COPY_STEPS": 5000,
    "TRAIN_AMOUNT": 0.9,

    "BUFFER_LENGTH": 17000
}

# backward compatibility
TRAINING = Q_LEARNING_SETTINGS["TRAINING"]
