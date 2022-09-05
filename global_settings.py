SAVED_MAPS_ROOT = r"saved_maps/"
SAVED_MODELS_ROOT = r"saved_models/"

FPS = 0  # 0 for unlimited
PORT = 5656
GRID_SIZE_PIXELS = 30  # 30 for low res

HEIGHT = 541  # 541 for low res
WIDTH = 961  # 961 for low res

SF = GRID_SIZE_PIXELS/60

USE_UNREAL_SOCKET = False
MESSAGE_LENGTH = 50  # padded to this length with @

VELOCITY_CONSTANT = 1  # change the speed of car movement globally

FREE_ROAM = False  # no collision

# COLOURS
COL_BACKGROUND = (97, 139, 74)
COL_GRID = (37, 60, 47)
COL_MOUSE_HIGHLIGHT = (109, 163, 77)
COL_PLACED_ROAD = (84, 86, 86)

# Q Learning
LOAD_MODEL_STEER = None
LOAD_MODEL_GAS = None
LOAD_MODEL_COMBINED = "combined_model_05.09;13.30_654"

MAX_EPISODE_FRAMES = 4000

# either use separate DQN for steering and gas, or one combined DQN
COMBINED_MODELS = True

Q_LEARNING_SETTINGS = {
    "TRAINING_COMBINED": True,
    "TRAINING_STEER": True,
    "TRAINING_GAS": True,

    "LEARNING_RATE_STEER": 0.00000000000001,
    "LEARNING_RATE_GAS": 0.00000000000001,
    "LEARNING_RATE_COMBINED": 0.0000000,

    "GD_MOMENTUM": 0.0,

    "DISCOUNT_RATE": 0.99,

    "EXPLORATION_PROBABILITY_COMBINED": 0.5,
    "EXPLORATION_PROBABILITY_STEER": 0.5,
    "EXPLORATION_PROBABILITY_GAS": 1,

    "EXPLORATION_DECAY_COMBINED": 0.0001,
    "EXPLORATION_DECAY_STEER": 0.00004,
    "EXPLORATION_DECAY_GAS": 0.00004,

    "TARGET_NET_COPY_STEPS": 5000,
    "TRAIN_AMOUNT": 0.8,

    "BUFFER_LENGTH": 4000
}


# MORE SETTINGS CODE

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
