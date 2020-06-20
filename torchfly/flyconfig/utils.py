import os

def split_config_path(config_path):
    if config_path is None or config_path == "":
        return None, None
    root, ext = os.path.splitext(config_path)

    if ext in (".yaml", ".yml"):
        # assuming dir/config.yaml form
        config_file = os.path.basename(config_path)
        config_dir = os.path.dirname(config_path)
    else:
        # assuming dir form without a config file.
        config_file = None
        config_dir = config_path

    if config_dir == "":
        config_dir = None

    if config_file == "":
        config_file = None
    return config_dir, config_file

# def init_flyconfig(config_path):
#     config_dir, config_file = split_config_path(config_path)