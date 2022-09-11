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
LOAD_MODEL_STEER = None
LOAD_MODEL_GAS = None
LOAD_MODEL_COMBINED = None
MAX_EPISODE_FRAMES = 4000

# either use separate DQN for steering and gas, or one combined DQN
COMBINED_MODELS = True

Q_LEARNING_SETTINGS = {
    "TRAINING": True,

    "LEARNING_RATE_STEER": 0.00000000000001,
    "LEARNING_RATE_GAS": 0.00000000000001,
    "LEARNING_RATE_COMBINED": 1e-010,

    "GD_MOMENTUM": 0.9,

    "DISCOUNT_RATE": 0.999,

    "EXPLORATION_PROBABILITY": 1,
    "EXPLORATION_DECAY": 0.00003,
    "EXPLORATION_MIN": 0.1,

    "TARGET_NET_COPY_STEPS": 50000,
    "TRAIN_AMOUNT": 0.6,

    "BUFFER_LENGTH": 4000
}

# ======== MORE SETTINGS CODE ========

# if any training settings are True, program is training a model
TRAINING = any([val for setting, val in Q_LEARNING_SETTINGS.items() if setting.startswith("TRAINING")])


def get_q_learning_settings(suffix: str):
    """
    Given a suffix e.g. STEER, returns all settings for that Q Learning model
    If setting doesn't end in STEER or other suffix, it is assumed to be used for all suffixes
    """
    settings = {}
    suffix = "_" + suffix.upper()

    for setting in Q_LEARNING_SETTINGS.keys():
        # settings ending in suffix
        if setting.endswith(suffix):
            settings[setting.replace(suffix, "")] = Q_LEARNING_SETTINGS[setting]

    for setting in Q_LEARNING_SETTINGS.keys():
        # settings that dont end in any suffix
        setting_root = "_".join(setting.split("_")[:-1])
        if setting_root not in settings.keys():
            settings[setting] = Q_LEARNING_SETTINGS[setting]

    return settings
