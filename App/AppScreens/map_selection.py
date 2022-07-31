import os
import pygame
import global_settings as gs
from App.world import Map
from App.AppScreens.map_builder import run_map_builder

X_GAP = 50
Y_GAP = 133.5
X_SIZE = 324
Y_SIZE = 182

WIDTH = 5


def get_grid_pos(row, col):
    # gap between is 50
    # map select width is 324
    x = X_GAP + (X_SIZE + X_GAP)*row

    # height is 182
    # gap is 133.5
    y = Y_GAP + (Y_GAP + Y_SIZE)*col

    return x, y


def get_selected_map(mouse_pos):
    from_origin = mouse_pos[0] - X_GAP, mouse_pos[1] - Y_GAP
    x_leftover = from_origin[0] % (X_SIZE + X_GAP)
    y_leftover = from_origin[1] % (Y_SIZE + Y_GAP)
    if x_leftover < X_SIZE and y_leftover < Y_SIZE:
        x_grid = from_origin[0] // (X_SIZE + X_GAP)
        y_grid = from_origin[1] // (Y_SIZE + Y_GAP)
        return int(x_grid), int(y_grid)
    return None


def get_new_maps(pre_loaded_maps):
    maps = os.listdir(gs.SAVED_MAPS_ROOT)

    maps_to_delete = []
    for num, m in enumerate(maps):
        if m.endswith(".png") or m in pre_loaded_maps:
            maps_to_delete.append(m)

    maps = list(filter(lambda x: x not in maps_to_delete, maps))

    loaded_maps_dict = {}
    for m in maps:
        loaded_map, loaded_image = Map.load_map(m)
        resized = pygame.transform.scale(loaded_image, (X_SIZE, Y_SIZE))
        loaded_maps_dict[m] = (loaded_map, resized)

    return loaded_maps_dict


def run_map_selection(screen: pygame.Surface):
    editor_image = pygame.image.load("image_assets/editor_box.png")
    editor_image = pygame.transform.scale(editor_image, (X_SIZE, Y_SIZE))

    loaded_maps_dict = get_new_maps([])
    loaded_names = list(loaded_maps_dict.keys())
    loaded_maps = list(loaded_maps_dict.values())

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        screen.fill(gs.COL_BACKGROUND)

        editor_pos = get_grid_pos(0, 0)
        screen.blit(editor_image, editor_pos)

        for m_num, (m, m_image) in enumerate(loaded_maps):
            m_num = m_num + 1  # first slot is for map creator
            col = int(m_num // WIDTH)
            row = int(m_num % WIDTH)
            x, y = get_grid_pos(row, col)

            screen.blit(m_image, (x, y))

        if pygame.mouse.get_pressed()[0]:
            mouse_pos = pygame.mouse.get_pos()
            grid_pos = get_selected_map(mouse_pos)

            if grid_pos is not None:

                index = grid_pos[0] + WIDTH*grid_pos[1]
                if index == 0:
                    created_map, saved = run_map_builder(screen)
                    if not saved:
                        return created_map
                    # load maps that havent been loaded
                    new_loaded_maps_dict = get_new_maps(loaded_names)
                    new_loaded_names = list(new_loaded_maps_dict.keys())
                    new_loaded_maps = list(new_loaded_maps_dict.values())

                    loaded_names = loaded_names + new_loaded_names
                    loaded_maps = loaded_maps + new_loaded_maps

                else:
                    try:
                        return list(map(lambda x: x[0], loaded_maps))[index-1]  # first is map creator so -1
                    except IndexError:
                        pass

        elif pygame.mouse.get_pressed()[2]:
            mouse_pos = pygame.mouse.get_pos()
            grid_pos = get_selected_map(mouse_pos)
            if grid_pos is not None:
                index = grid_pos[0] + WIDTH*grid_pos[1]
                if index != 0:
                    index -= 1

                    try:
                        map_name = loaded_names[index]
                    except IndexError:
                        map_name = None
                    if map_name is not None and input("Delete map? (y/n):") == "y":
                        Map.delete_map(map_name)
                        del loaded_names[index]
                        del loaded_maps[index]

        pygame.display.update()
