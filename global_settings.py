SAVED_MAPS_ROOT = r"saved_maps/"
SAVED_MODELS_ROOT = r"saved_models/"

FPS = 0  # 0 for unlimited
PORT = 5656
GRID_SIZE_PIXELS = 60  # 30 for low res

HEIGHT = 1081  # 541 for low res
WIDTH = 1921  # 961 for low res

SF = GRID_SIZE_PIXELS/60

USE_UNREAL_SOCKET = False
MESSAGE_LENGTH = 50  # padded to this length with @

VELOCITY_CONSTANT = 1.5  # change the speed of car movement globally

FREE_ROAM = False  # no collision

MAX_EPISODE_FRAMES = 3000

# COLOURS
COL_BACKGROUND = (97, 139, 74)
COL_GRID = (37, 60, 47)
COL_MOUSE_HIGHLIGHT = (109, 163, 77)
COL_PLACED_ROAD = (84, 86, 86)

# Q Learning
LOAD_MODEL_STEER = "steer_model_26.08;17.45_983"
LOAD_MODEL_GAS = None

Q_LEARNING_SETTINGS = {
    "TRAINING_STEER": True,
    "TRAINING_GAS": True,

    "LEARNING_RATE": 0.00000000005,

    "DISCOUNT_RATE": 0.95,

    "EXPLORATION_PROBABILITY_STEER": 0.5,
    "EXPLORATION_PROBABILITY_GAS": 1,

    "EXPLORATION_DECAY_STEER": 0.0001,
    "EXPLORATION_DECAY_GAS": 0.0001,

    "TARGET_NET_COPY_STEPS": 6000,
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
