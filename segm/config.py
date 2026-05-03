import yaml
from pathlib import Path

import os


_PATHS_CONFIG = None


def get_project_root():
    return str(Path(__file__).parent.parent)


def load_config():
    return yaml.load(
        open(Path(__file__).parent / "config.yml", "r"), Loader=yaml.FullLoader
    )


def load_paths():
    global _PATHS_CONFIG
    if _PATHS_CONFIG is None:
        paths_file = Path(get_project_root()) / "configs" / "paths.yaml"
        _PATHS_CONFIG = yaml.load(open(paths_file, "r"), Loader=yaml.FullLoader)
    return _PATHS_CONFIG


def resolve_path(relative_path):
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(get_project_root(), relative_path)


def check_os_environ(key, use):
    if key not in os.environ:
        raise ValueError(
            f"{key} is not defined in the os variables, it is required for {use}."
        )


def dataset_dir():
    paths = load_paths()
    if paths.get('dataset_dir') and paths['dataset_dir'] != '/path/to/ImageNet':
        return paths['dataset_dir']
    check_os_environ("DATASET", "data loading")
    return os.environ["DATASET"]
