import errno
import json
import os
from typing import IO, Any, Tuple

import yaml
from easydict import EasyDict as edict

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class TextLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        with open(self.log_path, "w") as f:
            f.write("")

    def log(self, log):
        with open(self.log_path, "a+") as f:
            f.write(log + "\n")


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(
        os.path.join(loader._root, loader.construct_scalar(node))
    )
    extension = os.path.splitext(filename)[1].lstrip(".")

    with open(filename, "r") as f:
        if extension in ("yaml", "yml"):
            return yaml.load(f, Loader)
        elif extension in ("json",):
            return json.load(f)
        else:
            return "".join(f.readlines())


def get_config(config_path):
    yaml.add_constructor("!include", construct_include, Loader)
    with open(os.path.join(ROOT_PATH, config_path), "r") as stream:
        config = yaml.load(stream, Loader=Loader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config


def create_checkpoint_folder(args, opts) -> Tuple[str, str]:
    """
    Function that creates a checkpoint folder on the specified checkpoint folder.
    Creates checkpoint_folder/data_name/

     Parameters:
    - args (Namespace): A namespace object containing command line arguments.
    - opts (Namespace): A namespace object containing additional options.
    Returns:
    - Tuple[str, str]: A tuple containing the data name extracted from the processed data path and the absolute path
      to the created checkpoint directory.

    """
    normalized_path = os.path.normpath(args.processed_data_root)
    data_name = os.path.basename(normalized_path)
    check_point_path = os.path.join(ROOT_PATH, opts.checkpoint, data_name, args.foldername)
    try:
        os.makedirs(check_point_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError(
                "Unable to create checkpoint directory:", check_point_path
            )

    return data_name, check_point_path
